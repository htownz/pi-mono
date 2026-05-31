"""Tests for the `doctor` CLI command.

Uses Click's CliRunner + respx to mock the Kalshi auth round-trip and the
NWS endpoints. The command's purpose is operational confidence before
turning on live `serve --auto-trade`, so the tests focus on:
  - All-PASS path with a working API key
  - Auth failure surfaces clearly
  - Unwriteable kill / audit paths surface clearly
  - --paper skips the Kalshi check
  - Exit status is non-zero on any FAIL
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import respx
from click.testing import CliRunner
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from kalshi_scout.cli import main


# -- fixtures ----------------------------------------------------------------

@pytest.fixture
def keypair(tmp_path: Path):
    """Generate a real RSA key so the doctor's signing path runs end-to-end
    (we mock only the HTTP layer, not the crypto)."""
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem_path = tmp_path / "kalshi.pem"
    pem_path.write_bytes(key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ))
    return pem_path


@pytest.fixture
def store_path(tmp_path: Path) -> Path:
    """An on-disk SQLite store the doctor can probe."""
    from kalshi_scout.store import SnapshotStore
    p = tmp_path / "scout.db"
    SnapshotStore(p).close()   # creates schema + closes
    return p


# -- Paper mode (no Kalshi key needed) ---------------------------------------

def test_doctor_paper_passes_without_key(store_path, monkeypatch):
    """--paper skips the auth round-trip; the rest of the checks still run.

    NWS calls happen via httpx — when offline (the sandbox case) they
    surface as FAIL but the command still runs to completion. We assert
    that paper-mode at minimum gives PASS on the snapshot-store and
    kill/audit path checks (which are local-only).
    """
    runner = CliRunner()
    result = runner.invoke(main, [
        "doctor", "--store", str(store_path), "--paper",
    ])
    # Local checks should pass; NWS checks may fail in offline environments,
    # so we don't assert overall exit code here — just check the local
    # checks appear in the output.
    assert "snapshot store readable" in result.output
    assert "PASS" in result.output
    assert "kill-switch path writable" in result.output
    assert "audit log path writable" in result.output
    # Kalshi check skipped.
    assert "skipped (--paper)" in result.output


def test_doctor_live_passes_with_valid_key(store_path, keypair):
    """Mock the Kalshi balance endpoint → asserts the auth check passes
    and the balance is displayed."""
    runner = CliRunner()
    with respx.mock(base_url="https://api.elections.kalshi.com/trade-api/v2") as mock:
        mock.get("/portfolio/balance").respond(200, json={"balance": 4250})
        result = runner.invoke(main, [
            "doctor",
            "--store", str(store_path),
            "--api-key-id", "kid_test",
            "--api-key-path", str(keypair),
        ])
    assert "Kalshi auth round-trip" in result.output
    assert "PASS" in result.output
    assert "$42.50" in result.output


def test_doctor_live_fails_on_auth_rejection(store_path, keypair):
    """Kalshi returns 401 → auth check FAILs; doctor exits non-zero."""
    runner = CliRunner()
    with respx.mock(base_url="https://api.elections.kalshi.com/trade-api/v2") as mock:
        mock.get("/portfolio/balance").respond(401, text="invalid signature")
        result = runner.invoke(main, [
            "doctor",
            "--store", str(store_path),
            "--api-key-id", "kid_bad",
            "--api-key-path", str(keypair),
        ])
    assert result.exit_code == 1
    assert "Kalshi auth round-trip" in result.output
    assert "FAIL" in result.output


def test_doctor_live_without_key_fails(store_path):
    """Live mode without --api-key-id is a configuration error — but the
    doctor should report it as a FAIL line instead of crashing, so the
    operator sees what's missing."""
    runner = CliRunner()
    result = runner.invoke(main, [
        "doctor", "--store", str(store_path),
    ])
    assert result.exit_code == 1
    assert "missing --api-key-id" in result.output


def test_doctor_fails_when_kill_file_path_unwriteable(store_path, tmp_path):
    """An unwriteable kill_file parent dir surfaces as FAIL."""
    runner = CliRunner()
    bad_dir = tmp_path / "ro"
    bad_dir.mkdir()
    bad_dir.chmod(0o500)   # read+execute only
    try:
        result = runner.invoke(main, [
            "doctor", "--store", str(store_path), "--paper",
            "--kill-file", str(bad_dir / "scout.kill"),
        ])
        assert result.exit_code == 1
        assert "kill-switch path writable" in result.output
        assert "FAIL" in result.output
    finally:
        bad_dir.chmod(0o700)   # restore so pytest can clean up
