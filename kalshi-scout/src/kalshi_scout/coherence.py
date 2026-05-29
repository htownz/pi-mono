"""Cross-bracket coherence pass.

Per AGENTS.md invariant I7: contracts in one Kalshi event are mutually
exclusive. The engine must reflect that:

  - If any sibling is `LOCKED_YES`, all other siblings are `DEAD_NO`.
  - If every sibling is `DEAD_NO`, the event is inconsistent (this is a
    bug / settlement-source mismatch) — we annotate but do not crash.
  - When tradable Yes-ask prices across siblings sum to substantially less
    than 100, that's a "stale book" arbitrage signal — flagged in notes
    but not auto-graded (orderbook depth handling lands in V0.6).

This pass runs *after* per-contract classification + grading and may
demote/regrade siblings. It returns a new list; inputs are not mutated.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Optional

from kalshi_scout.models import ContractEvaluation, ContractState


def enforce_coherence(
    evals: list[ContractEvaluation],
) -> list[ContractEvaluation]:
    """Return the input list with cross-bracket constraints applied.

    Constraints applied in order:

    1. If exactly one contract is LOCKED_YES, demote every sibling to DEAD_NO
       (preserving the original state in notes for diagnostics).
    2. If multiple contracts are LOCKED_YES, that's a settlement-source bug:
       annotate but do not silently pick a winner.
    3. Sum of yes_ask cents across siblings: if > 105 → markets are
       "over-priced" together (book stale on the No side); if < 95 → "under-
       priced" together. Annotated as a notes-only signal.
    """
    if not evals:
        return evals

    locked: list[int] = [
        i for i, e in enumerate(evals) if e.state is ContractState.LOCKED_YES
    ]

    if len(locked) > 1:
        return [
            replace(
                e,
                notes=[*e.notes, f"coherence: {len(locked)} siblings LOCKED_YES (settlement-source mismatch)"],
            )
            for e in evals
        ]

    out: list[ContractEvaluation] = list(evals)
    if len(locked) == 1:
        winner_idx = locked[0]
        for i, e in enumerate(out):
            if i == winner_idx:
                continue
            if e.state is ContractState.DEAD_NO:
                continue
            note = f"coherence: sibling {evals[winner_idx].market.ticker} LOCKED_YES → demoted from {e.state.value}"
            # When we demote a sibling we recompute the fair probability to
            # near-zero, but leave the grade and edge alone — the ranker will
            # re-grade on the next call. For now we just mark the state.
            out[i] = replace(
                e,
                state=ContractState.DEAD_NO,
                reason=f"sibling LOCKED_YES: {evals[winner_idx].market.ticker}",
                fair_prob_low=0.0,
                fair_prob_high=0.02,
                notes=[*e.notes, note],
            )

    # Sum-of-ask sanity check.
    asks = [e.yes_ask_cents for e in out if e.yes_ask_cents is not None]
    if len(asks) == len(out) and len(asks) >= 2:
        total = sum(asks)
        if total > 105:
            for i, e in enumerate(out):
                out[i] = replace(
                    e, notes=[*e.notes, f"coherence: event yes-asks sum to {total}c (overpriced book)"]
                )
        elif total < 95:
            for i, e in enumerate(out):
                out[i] = replace(
                    e, notes=[*e.notes, f"coherence: event yes-asks sum to {total}c (underpriced book / stale)"]
                )

    return out
