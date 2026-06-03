"""Temporal dislocation detector for NegRisk baskets. DETECTION ONLY.

A static snapshot cannot tell a real arb from structural missing-mass (see README: with
top-of-book only, 1 - mass ≥ edge always). The discriminator is *time*:

  - a structural / phantom event sits at a STABLE depressed ask_sum (its gap below $1 is
    just unlisted-outcome probability — flat forever);
  - a genuine, capturable dislocation is a TRANSIENT drop of ask_sum below the event's own
    baseline that then recovers.

This module reads the per-cycle snapshot log (`scan.py --snapshot ...`), builds a robust
baseline per event (median / MAD, outlier-resistant), and flags dips: contiguous runs where
ask_sum falls a robust z-score below baseline. Each dip reports its depth (how much cheaper
than normal the basket got) and duration (how many cycles it lasted) — duration × poll
interval is the lifetime a non-latency trader would have had to act. That lifetime, not a
single frame, is the real go/no-go.

stdlib only.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from collections import defaultdict


@dataclass
class Dip:
    request_id: str
    title: str
    baseline_ask: float     # event's robust-median ask_sum (its "normal" level)
    min_ask: float          # lowest ask_sum reached during the dip
    depth: float            # baseline_ask - min_ask (transient cheapness vs. normal)
    n_points: int           # consecutive snapshots below threshold (× interval = lifetime)
    start_ts: str
    end_ts: str
    robust_sigma: float     # 1.4826 * MAD of the event's ask_sum series

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))


def load_snapshots(path: str) -> list[dict]:
    """Read a JSONL snapshot log (one NegRiskSnapshot per line)."""
    out: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def group_by_event(snaps: list[dict]) -> dict[str, list[dict]]:
    """Bucket snapshots by request_id, each list sorted by timestamp."""
    by: dict[str, list[dict]] = defaultdict(list)
    for s in snaps:
        by[s["request_id"]].append(s)
    for rs in by.values():
        rs.sort(key=lambda r: r["ts"])
    return dict(by)


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def robust_stats(xs: list[float]) -> tuple[float, float]:
    """(median, robust_sigma) where robust_sigma = 1.4826 * MAD. Outlier-resistant, so a
    real dip doesn't inflate the baseline it's being measured against."""
    med = _median(xs)
    mad = _median([abs(x - med) for x in xs])
    return med, 1.4826 * mad


def detect_dips(
    snaps: list[dict],
    *,
    k: float = 4.0,
    min_points: int = 12,
    min_depth: float = 0.005,
) -> list[Dip]:
    """Flag transient drops of ask_sum below each event's robust baseline.

    A point is "below" when ask_sum < baseline - max(k * robust_sigma, min_depth); the
    absolute `min_depth` floor stops near-zero-variance series from firing on rounding noise.
    Consecutive below-points are merged into one dip episode. Events with fewer than
    `min_points` snapshots are skipped (not enough history for a trustworthy baseline).

    Returns dips sorted by depth, deepest first.
    """
    dips: list[Dip] = []
    for rid, rs in group_by_event(snaps).items():
        if len(rs) < min_points:
            continue
        asks = [r["ask_sum"] for r in rs]
        baseline, sigma = robust_stats(asks)
        threshold = baseline - max(k * sigma, min_depth)
        title = rs[-1].get("title", rid)

        i = 0
        while i < len(rs):
            if asks[i] >= threshold:
                i += 1
                continue
            j = i
            while j < len(rs) and asks[j] < threshold:
                j += 1
            episode = rs[i:j]
            ep_asks = asks[i:j]
            mn = min(ep_asks)
            dips.append(Dip(
                request_id=rid,
                title=title,
                baseline_ask=round(baseline, 6),
                min_ask=round(mn, 6),
                depth=round(baseline - mn, 6),
                n_points=len(episode),
                start_ts=episode[0]["ts"],
                end_ts=episode[-1]["ts"],
                robust_sigma=round(sigma, 6),
            ))
            i = j
    dips.sort(key=lambda d: d.depth, reverse=True)
    return dips


def summarize(snaps: list[dict]) -> str:
    """Human-readable per-event baseline table — a quick 'where are we' over the log."""
    by = group_by_event(snaps)
    lines = [f"{len(snaps)} snapshots across {len(by)} events"]
    if snaps:
        ts = [s["ts"] for s in snaps]
        lines.append(f"span: {min(ts)}  ->  {max(ts)}")
    lines.append("")
    lines.append(f"{'event':40} {'pts':>4} {'base_ask':>8} {'min_ask':>8} {'dip':>6} {'maxN':>4}")
    rows = []
    for rid, rs in by.items():
        asks = [r["ask_sum"] for r in rs]
        base, _ = robust_stats(asks)
        mn = min(asks)
        rows.append((base - mn, rs[-1].get("title", rid), len(rs), base, mn,
                     max(r["legs"] for r in rs)))
    for dip, title, pts, base, mn, n in sorted(rows, reverse=True)[:30]:
        lines.append(f"{title[:40]:40} {pts:>4} {base:>8.3f} {mn:>8.3f} {dip:>6.3f} {n:>4}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse
    p = argparse.ArgumentParser(description="NegRisk temporal dislocation detector (read-only).")
    p.add_argument("snapshot_log", help="JSONL written by scan.py --snapshot")
    p.add_argument("-k", type=float, default=4.0, help="robust z-score threshold for a dip.")
    p.add_argument("--min-points", type=int, default=12, help="min snapshots/event for a baseline.")
    p.add_argument("--min-depth", type=float, default=0.005, help="min ask_sum drop to count (USD).")
    p.add_argument("--summary", action="store_true", help="print the per-event baseline table.")
    args = p.parse_args(argv)

    snaps = load_snapshots(args.snapshot_log)
    if args.summary:
        print(summarize(snaps))
        print()
    dips = detect_dips(snaps, k=args.k, min_points=args.min_points, min_depth=args.min_depth)
    if not dips:
        print("no transient dips detected — every crossing event sits flat at its baseline "
              "(structural missing-mass, not a capturable dislocation).")
        return 0
    print(f"{len(dips)} dip episode(s), deepest first "
          f"(depth = how far ask_sum fell below the event's normal level):\n")
    for d in dips:
        print(f"  depth={d.depth * 100:5.2f}c  base={d.baseline_ask:.3f} -> min={d.min_ask:.3f}  "
              f"{d.n_points:>3} pts  {d.start_ts}..{d.end_ts}  {d.title[:44]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
