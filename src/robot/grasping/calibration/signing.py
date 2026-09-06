"""Promotion signing seam.

The trust root of the promotion gate is, and remains, the SHA-256 attestation chain that
:mod:`model_promotion` describes. This module adds an optional signature layer above it: a deployment
with a real key-management story, a cloud KMS, an HSM or a CI signing identity, injects a
:class:`Signer` to sign the promotion chain SHA when building a report, and a :class:`Verifier` to
check it when verifying one.

This repository ships no committed key, so the signature stays ``"none"`` and no trust root is
implied that does not exist. Faking PKI in a repository with no key-management story would be
theatre: the SHA chain is the trust here, and the signature is a seam to plug into. The reference
Ed25519 implementation imports ``cryptography`` lazily, and the :class:`Signer` and
:class:`Verifier` Protocols carry no crypto dependency at all, so a caller who plugs in their own
signer pulls in nothing. The reference cloud-KMS implementation,
:class:`AwsKmsSigner` with :class:`AwsKmsVerifier`, signs and verifies through the asymmetric AWS KMS
operations behind a lazy ``boto3``; another provider, GCP KMS or Azure Key Vault, is the same
Protocol with that SDK.

Signature wire format: ``"<scheme>:<key_id>:<base64 raw-signature>"`` or ``"none"``. Encoding everything
in the existing ``signature`` string keeps ``PROMOTION_SCHEMA_VERSION`` unchanged (no golden regen) and
keeps the report self-describing about which key signed it.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Optional, Protocol

from src.robot.grasping.constants import (
    CALIBRATION_SIGNING_LOG_FILE,
    create_grasping_logger,
)

#: Key loads and KMS round-trips are the two things an operator has to be able to
#: reconstruct after a failed verification. Key IDs are logged (they are
#: identifiers, published with the trust root); key material and raw signature
#: bytes never are.
logger = create_grasping_logger("CalibrationSigning", CALIBRATION_SIGNING_LOG_FILE)


SIGNATURE_NONE: str = "none"
ED25519_SCHEME: str = "ed25519-v1"


class Signer(Protocol):
    """Signs the promotion chain SHA. A deployment's KMS/HSM/CI signer implements this Protocol."""

    key_id: str
    scheme: str

    def sign(self, payload: bytes) -> str: ...


class Verifier(Protocol):
    """Verifies a promotion signature against the chain SHA."""

    scheme: str

    def verify(self, payload: bytes, signature: str) -> bool: ...


def encode_signature(scheme: str, key_id: str, raw: bytes) -> str:
    """Encode a raw signature as ``<scheme>:<key_id>:<base64>``."""

    return f"{scheme}:{key_id}:{base64.b64encode(raw).decode('ascii')}"


def parse_signature(signature: str) -> Optional[tuple[str, str, bytes]]:
    """Parse ``<scheme>:<key_id>:<b64>`` into ``(scheme, key_id, raw)``; ``None`` for ``"none"``/malformed input (never raises).

    The ``key_id`` may itself contain ``:``, an AWS KMS ARN for instance, so the scheme is split off
    the front and the base64 signature off the back, and everything between them is the key id.
    """

    if not signature or signature == SIGNATURE_NONE or ":" not in signature:
        return None
    scheme, rest = signature.split(":", 1)
    if ":" not in rest:
        return None
    key_id, b64 = rest.rsplit(":", 1)
    if not scheme or not key_id:
        return None
    try:
        raw = base64.b64decode(b64, validate=True)
    except Exception:  # noqa: BLE001 (a malformed base64 body is simply not a valid signature)
        return None
    return scheme, key_id, raw


class Ed25519Signer:
    """Reference Ed25519 signer. Lazily requires the optional ``cryptography`` dependency."""

    scheme: str = ED25519_SCHEME

    def __init__(self, private_key: Any, key_id: str) -> None:
        self._private_key = private_key
        self.key_id = key_id

    def sign(self, payload: bytes) -> str:
        raw = self._private_key.sign(payload)
        return encode_signature(self.scheme, self.key_id, raw)


class Ed25519Verifier:
    """Reference Ed25519 verifier. Lazily requires the optional ``cryptography`` dependency."""

    scheme: str = ED25519_SCHEME

    def __init__(self, public_key: Any) -> None:
        self._public_key = public_key

    def verify(self, payload: bytes, signature: str) -> bool:
        parsed = parse_signature(signature)
        if parsed is None:
            return False
        scheme, _key_id, raw = parsed
        if scheme != self.scheme:
            logger.warning(
                "Ed25519 verify refused: signature scheme %r != %r", scheme, self.scheme
            )
            return False
        from cryptography.exceptions import InvalidSignature

        try:
            self._public_key.verify(raw, payload)
        except InvalidSignature:
            logger.warning("Ed25519 verify refused: invalid signature for key %s", _key_id)
            return False
        return True


AWS_KMS_SCHEME: str = "aws-kms-v1"


def _boto3_kms_client(region_name: str | None) -> Any:
    """Build a real AWS KMS client (lazy ``boto3``). Deployments inject ``client=`` instead in tests."""
    try:
        import boto3  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover (exercised only on bare envs)
        raise ImportError(
            "AwsKmsSigner/AwsKmsVerifier require boto3, which requirements.txt pins, "
            "or inject a client=."
        ) from exc
    return boto3.client("kms", region_name=region_name)


class AwsKmsSigner:
    """Reference cloud-KMS :class:`Signer`, backed by the asymmetric Sign operation of AWS KMS.

    The trust root of the deployment is a KMS key it controls. ``boto3`` and the client are both lazy
    or injected, so importing this module needs neither. ``key_id`` is the KMS key id or ARN of an
    asymmetric SIGN_VERIFY key. Another cloud, GCP KMS or Azure Key Vault, is the same
    :class:`Signer` Protocol with that SDK; this one is the AWS reference. The KMS round trip needs a
    real account, while the signing logic, meaning the wire format and the error handling, works
    against an injected client.
    """

    scheme: str = AWS_KMS_SCHEME

    def __init__(
        self,
        key_id: str,
        *,
        signing_algorithm: str = "ECDSA_SHA_256",
        client: Any | None = None,
        region_name: str | None = None,
    ) -> None:
        self.key_id = key_id
        self._algorithm = signing_algorithm
        self._client = client
        self._region = region_name

    def _kms(self) -> Any:
        if self._client is None:
            self._client = _boto3_kms_client(self._region)
        return self._client

    def sign(self, payload: bytes) -> str:
        # A network round trip to the KMS of the deployment, worth a line because a
        # hung or misrouted signer is otherwise indistinguishable from a slow build.
        logger.info(
            "KMS sign requested: key=%s algorithm=%s", self.key_id, self._algorithm
        )
        resp = self._kms().sign(
            KeyId=self.key_id,
            Message=payload,
            MessageType="RAW",
            SigningAlgorithm=self._algorithm,
        )
        return encode_signature(self.scheme, self.key_id, resp["Signature"])


class AwsKmsVerifier:
    """Reference cloud-KMS :class:`Verifier`, backed by the Verify operation of AWS KMS, which holds the public key."""

    scheme: str = AWS_KMS_SCHEME

    def __init__(
        self,
        *,
        signing_algorithm: str = "ECDSA_SHA_256",
        client: Any | None = None,
        region_name: str | None = None,
        key_id: str | None = None,
    ) -> None:
        self._algorithm = signing_algorithm
        self._client = client
        self._region = region_name
        self._key_id = key_id  # optional pin; else the key id embedded in the signature is used

    def _kms(self) -> Any:
        if self._client is None:
            self._client = _boto3_kms_client(self._region)
        return self._client

    def verify(self, payload: bytes, signature: str) -> bool:
        parsed = parse_signature(signature)
        if parsed is None:
            return False
        scheme, key_id, raw = parsed
        if scheme != self.scheme:
            logger.warning(
                "KMS verify refused: signature scheme %r != %r", scheme, self.scheme
            )
            return False
        try:
            resp = self._kms().verify(
                KeyId=self._key_id or key_id,
                Message=payload,
                MessageType="RAW",
                Signature=raw,
                SigningAlgorithm=self._algorithm,
            )
        except Exception as exc:  # noqa: BLE001 (any KMS/transport error means "not a valid signature")
            # Swallowed by contract; without this line a KMS outage looks exactly
            # like a forged signature.
            logger.error(
                "KMS verify failed for key %s: %s", self._key_id or key_id, exc
            )
            return False
        valid = bool(resp.get("SignatureValid", False))
        if not valid:
            logger.warning("KMS verify refused: signature invalid for key %s", self._key_id or key_id)
        return valid


def generate_ed25519_keypair(key_id: str) -> tuple[Ed25519Signer, Ed25519Verifier]:
    """Generate an ephemeral Ed25519 keypair for development. It is not a committed trust root."""

    from cryptography.hazmat.primitives.asymmetric import ed25519

    private_key = ed25519.Ed25519PrivateKey.generate()
    logger.info("Generated EPHEMERAL Ed25519 keypair key_id=%s (dev/test only)", key_id)
    return (
        Ed25519Signer(private_key, key_id),
        Ed25519Verifier(private_key.public_key()),
    )


def load_ed25519_signer(private_key_pem_path: str | Path, key_id: str) -> Ed25519Signer:
    """Load a signer from a PEM private key on disk. A cloud-KMS signer implements :class:`Signer` directly instead."""

    from cryptography.hazmat.primitives.serialization import load_pem_private_key

    data = Path(private_key_pem_path).read_bytes()
    private_key = load_pem_private_key(data, password=None)
    logger.info(
        "Loaded Ed25519 signer key_id=%s from %s", key_id, Path(private_key_pem_path)
    )
    return Ed25519Signer(private_key, key_id)


def load_ed25519_verifier(public_key_pem_path: str | Path) -> Ed25519Verifier:
    """Load a verifier from a PEM public key on disk, which is the published trust root of the deployment."""

    from cryptography.hazmat.primitives.serialization import load_pem_public_key

    data = Path(public_key_pem_path).read_bytes()
    public_key = load_pem_public_key(data)
    logger.info("Loaded Ed25519 verifier (trust root) from %s", Path(public_key_pem_path))
    return Ed25519Verifier(public_key)


__all__ = [
    "AWS_KMS_SCHEME",
    "ED25519_SCHEME",
    "SIGNATURE_NONE",
    "AwsKmsSigner",
    "AwsKmsVerifier",
    "Ed25519Signer",
    "Ed25519Verifier",
    "Signer",
    "Verifier",
    "encode_signature",
    "generate_ed25519_keypair",
    "load_ed25519_signer",
    "load_ed25519_verifier",
    "parse_signature",
]
