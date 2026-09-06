"""Unit tests for the AWS KMS reference promotion signer/verifier.

The real KMS round-trip needs a cloud account + a real asymmetric key (bucket-③, not on CI). Here we
inject a deterministic fake KMS client and lock the signing LOGIC: the wire format, ARN key ids (which
contain ``:``), the promotion-report integration, and every reject path — plus the boto3-missing guard.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.robot.grasping.calibration import model_promotion as mp
from src.robot.grasping.calibration.signing import (
    AWS_KMS_SCHEME,
    AwsKmsSigner,
    AwsKmsVerifier,
    parse_signature,
)


class _FakeKms:
    """Deterministic stand-in for a boto3 KMS client (asymmetric Sign/Verify)."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    @staticmethod
    def _raw(key_id: str, message: bytes) -> bytes:
        return hashlib.sha256(key_id.encode("utf-8") + b"|" + message).digest()

    def sign(self, *, KeyId: str, Message: bytes, MessageType: str, SigningAlgorithm: str) -> dict:
        self.calls.append("sign")
        assert MessageType == "RAW"
        return {"Signature": self._raw(KeyId, Message)}

    def verify(
        self, *, KeyId: str, Message: bytes, Signature: bytes, MessageType: str, SigningAlgorithm: str
    ) -> dict:
        self.calls.append("verify")
        return {"SignatureValid": Signature == self._raw(KeyId, Message)}


ARN = "arn:aws:kms:eu-central-1:123456789012:key/abcd-1234"


def test_sign_then_verify_roundtrip_with_arn_key_id() -> None:
    kms = _FakeKms()
    signer = AwsKmsSigner(ARN, client=kms)
    verifier = AwsKmsVerifier(client=kms)

    sig = signer.sign(b"chain-sha-payload")

    # Wire format <scheme>:<key_id>:<b64>; the ARN (with ':') survives the round-trip.
    parsed = parse_signature(sig)
    assert parsed is not None
    scheme, key_id, _raw = parsed
    assert scheme == AWS_KMS_SCHEME
    assert key_id == ARN
    assert verifier.verify(b"chain-sha-payload", sig) is True


def test_verify_rejects_tampered_payload_wrong_scheme_and_none() -> None:
    kms = _FakeKms()
    signer = AwsKmsSigner("key-1", client=kms)
    verifier = AwsKmsVerifier(client=kms)
    sig = signer.sign(b"payload")

    assert verifier.verify(b"different-payload", sig) is False   # payload tamper
    assert verifier.verify(b"payload", "ed25519-v1:key-1:AAAA") is False  # wrong scheme
    assert verifier.verify(b"payload", "none") is False          # unsigned
    assert verifier.verify(b"payload", "garbage") is False        # malformed


def test_signer_plugs_into_promotion_report(tmp_path: Path) -> None:
    # Minimal artifact the promotion report chains over.
    (tmp_path / "model.json").write_text(json.dumps({"m": 1}), encoding="utf-8")
    (tmp_path / "manifest.json").write_text(json.dumps({"n": 2}), encoding="utf-8")
    # The stub has to be a PLAUSIBLE stub since 2026-09-05: `verify_promotion` now also refuses a
    # report whose validation slice is not the plan-locked one, and whose recorded metrics contradict
    # its own recorded bounds. `kind="unit", seed=1, n_attempts=1` with a metrics dict carrying no
    # `log_loss` was fine while nothing looked at either. This test is about SIGNING, so the rest of
    # the report is made honest rather than the gate made lenient.
    verdict = mp.PromotionVerdict(
        verdict="pass", metrics={"brier": 0.05, "log_loss": 0.2},
        thresholds=mp.PromotionThresholds(),
    )
    slice_ = mp.ValidationSlice(
        kind="synthetic_split", seed=mp.PROMOTION_VALIDATION_SEED,
        n_attempts=mp.PROMOTION_VALIDATION_ATTEMPTS, dataset_sha256="0" * 64,
    )

    kms = _FakeKms()
    report = mp.build_promotion_report(
        tmp_path, verdict, slice_, promoted_at="2026-01-01T00:00:00Z",
        promoted_by="tester", signer=AwsKmsSigner("key-1", client=kms),
    )

    assert report.signature.startswith(AWS_KMS_SCHEME + ":")
    # verify_promotion accepts the KMS signature over the chain SHA.
    mp.write_promotion_report(report, tmp_path)
    ok, reasons = mp.verify_promotion(tmp_path, verifier=AwsKmsVerifier(client=kms), require_signature=True)
    assert ok is True, reasons


def test_boto3_missing_raises_actionable_import_error() -> None:
    # No client injected + boto3 absent -> a clear, actionable ImportError (bare-env behaviour).
    try:
        import boto3  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="boto3"):
            AwsKmsSigner("key-1").sign(b"x")
    else:
        pytest.skip("boto3 is installed; cannot exercise the missing-boto3 guard")
