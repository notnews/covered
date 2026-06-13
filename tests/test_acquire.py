"""Tests for the acquire helpers that don't require network/token."""

from covered import acquire


def test_sha256_bytes_is_stable() -> None:
    assert acquire.sha256_bytes(b"hello") == (
        "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    )
