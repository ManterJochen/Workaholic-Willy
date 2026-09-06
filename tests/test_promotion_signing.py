"""The optional Ed25519 promotion-signing seam.

The promotion gate's trust root is, and remains, the SHA-256 attestation chain. This adds an OPTIONAL
signature layer that a deployment with a real key-management story can enable by injecting a Signer (build)
and a Verifier (verify). No committed key ships, so the default is unchanged (signature='none', the SHA
chain is the trust). These tests pin the Ed25519 round-trip, tamper detection, the build/verify
integration, and the backward-compatible default.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from src.robot.grasping.calibration import model_promotion as mp
from src.robot.grasping.calibration.signing import (
    SIGNATURE_NONE,
    encode_signature,
    generate_ed25519_keypair,
    parse_signature,
)

try:
    import cryptography  # noqa: F401

    _HAS_CRYPTO = True
except ImportError:  # pragma: no cover - environment-dependent
    _HAS_CRYPTO = False

_SKIP_REASON = "requires the optional 'cryptography' dependency (requirements/signing.txt)"

REPO_ROOT = Path(__file__).resolve().parents[1]
COMMITTED_ARTIFACT = REPO_ROOT / "assets" / "models" / "success_probability" / "v1"


def _copy_artifact(dst_root: Path) -> Path:
    dst = dst_root / "v1"
    shutil.copytree(COMMITTED_ARTIFACT, dst)
    return dst


@unittest.skipUnless(_HAS_CRYPTO, _SKIP_REASON)
class SigningModuleTests(unittest.TestCase):
    def test_ed25519_round_trip(self) -> None:
        signer, verifier = generate_ed25519_keypair("k1")
        payload = b"chain-sha-deadbeef"
        sig = signer.sign(payload)
        self.assertTrue(sig.startswith("ed25519-v1:k1:"))
        self.assertTrue(verifier.verify(payload, sig))

    def test_tampered_payload_fails(self) -> None:
        signer, verifier = generate_ed25519_keypair("k1")
        sig = signer.sign(b"original")
        self.assertFalse(verifier.verify(b"tampered", sig))

    def test_tampered_signature_fails(self) -> None:
        signer, verifier = generate_ed25519_keypair("k1")
        sig = signer.sign(b"p")
        self.assertFalse(verifier.verify(b"p", sig[:-4] + "AAAA"))

    def test_wrong_key_fails(self) -> None:
        signer, _ = generate_ed25519_keypair("k1")
        _, other_verifier = generate_ed25519_keypair("k2")
        sig = signer.sign(b"p")
        self.assertFalse(other_verifier.verify(b"p", sig))

    def test_none_and_malformed_never_verify(self) -> None:
        _, verifier = generate_ed25519_keypair("k1")
        self.assertFalse(verifier.verify(b"p", SIGNATURE_NONE))
        self.assertFalse(verifier.verify(b"p", "garbage"))
        self.assertIsNone(parse_signature("none"))
        self.assertIsNone(parse_signature("a:b"))  # too few parts
        self.assertIsNone(parse_signature("a:b:!!! not base64 !!!"))

    def test_wrong_scheme_rejected(self) -> None:
        _, verifier = generate_ed25519_keypair("k1")
        # A well-formed signature in a different scheme must not verify against the Ed25519 verifier.
        bogus = encode_signature("rsa-v9", "k1", b"\x00\x01\x02")
        self.assertFalse(verifier.verify(b"p", bogus))


@unittest.skipUnless(_HAS_CRYPTO, _SKIP_REASON)
class PromotionSigningIntegrationTests(unittest.TestCase):
    def _build(self, signer=None):
        verdict, _m, validation = mp.evaluate_for_promotion(COMMITTED_ARTIFACT)
        return mp.build_promotion_report(
            COMMITTED_ARTIFACT,
            verdict,
            validation,
            promoted_at="2030-01-01T00:00:00Z",
            promoted_by="r10-test",
            signer=signer,
        )

    def test_signed_report_verifies(self) -> None:
        signer, verifier = generate_ed25519_keypair("k1")
        report = self._build(signer)
        self.assertTrue(report.signature.startswith("ed25519-v1:k1:"))
        # the signature is over the chain SHA (artifact_sha256)
        self.assertTrue(
            verifier.verify(report.artifact_sha256.encode("utf-8"), report.signature)
        )
        with tempfile.TemporaryDirectory() as td:
            tmp = _copy_artifact(Path(td))
            mp.write_promotion_report(report, tmp)
            ok, reasons = mp.verify_promotion(tmp, verifier=verifier)
            self.assertTrue(ok, msg=str(reasons))

    def test_wrong_verifier_fails(self) -> None:
        signer, _ = generate_ed25519_keypair("k1")
        _, wrong = generate_ed25519_keypair("k2")
        report = self._build(signer)
        with tempfile.TemporaryDirectory() as td:
            tmp = _copy_artifact(Path(td))
            mp.write_promotion_report(report, tmp)
            ok, reasons = mp.verify_promotion(tmp, verifier=wrong)
            self.assertFalse(ok)
            self.assertTrue(any("signature_invalid" in r for r in reasons))

    def test_no_verifier_falls_back_to_sha_chain(self) -> None:
        # Backward-compatible default: a signed report still verifies on the SHA chain alone.
        signer, _ = generate_ed25519_keypair("k1")
        report = self._build(signer)
        with tempfile.TemporaryDirectory() as td:
            tmp = _copy_artifact(Path(td))
            mp.write_promotion_report(report, tmp)
            ok, reasons = mp.verify_promotion(tmp)
            self.assertTrue(ok, msg=str(reasons))

    def test_unsigned_default_is_none(self) -> None:
        report = self._build(signer=None)
        self.assertEqual(report.signature, SIGNATURE_NONE)

    def test_require_signature_rejects_unsigned(self) -> None:
        _, verifier = generate_ed25519_keypair("k1")
        report = self._build(signer=None)
        with tempfile.TemporaryDirectory() as td:
            tmp = _copy_artifact(Path(td))
            mp.write_promotion_report(report, tmp)
            ok, reasons = mp.verify_promotion(
                tmp, verifier=verifier, require_signature=True
            )
            self.assertFalse(ok)
            self.assertTrue(
                any("signature_required_but_absent" in r for r in reasons)
            )


if __name__ == "__main__":
    unittest.main()
