from __future__ import annotations

import io
import os
from functools import lru_cache
from pathlib import Path

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt


MAGIC = b"OTENC001"
SALT_SIZE = 16
NONCE_SIZE = 12
SOURCE_PURPOSE = b"onetrainer/source/v1"
CACHE_PURPOSE = b"onetrainer/cache/v1"

_configured_secret: str | bytes | None = None


def configure_encryption_key(secret: str | bytes | None) -> None:
    """Configure the process-local key used for encrypted dataset/cache reads."""
    global _configured_secret
    _configured_secret = secret if secret not in ("", b"") else None


def _secret_bytes(secret: str | bytes | None) -> bytes:
    value = _configured_secret if secret is None else secret
    if value is None or value == "" or value == b"":
        raise RuntimeError(
            "Encrypted OneTrainer data was opened without an encryption key. "
            "Enter the dataset/cache encryption key or set OT_DATASET_ENCRYPTION_KEY_B64."
        )
    return value if isinstance(value, bytes) else value.encode("utf-8")


def _purpose_bytes(purpose: str | bytes) -> bytes:
    return purpose if isinstance(purpose, bytes) else purpose.encode("utf-8")


@lru_cache(maxsize=32)
def _derive_key(secret: bytes, salt: bytes, purpose: bytes) -> bytes:
    # Purpose is included in the KDF input as well as the AEAD associated data so
    # source files and cache files cannot be substituted for each other.
    kdf = Scrypt(salt=salt, length=32, n=2**14, r=8, p=1)
    return kdf.derive(secret + b"\0" + purpose)


def _encrypt_blob(
        plaintext: bytes,
        secret: str | bytes,
        purpose: str | bytes,
        *,
        salt: bytes | None = None,
) -> tuple[bytes, bytes]:
    salt = os.urandom(SALT_SIZE) if salt is None else bytes(salt)
    if len(salt) != SALT_SIZE:
        raise ValueError(f"encryption salt must be {SALT_SIZE} bytes")
    nonce = os.urandom(NONCE_SIZE)
    purpose_bytes = _purpose_bytes(purpose)
    key = _derive_key(_secret_bytes(secret), salt, purpose_bytes)
    ciphertext = AESGCM(key).encrypt(nonce, plaintext, MAGIC + purpose_bytes)
    return MAGIC + salt + nonce + ciphertext, salt


def _decrypt_blob(
        payload: bytes,
        secret: str | bytes | None,
        purpose: str | bytes,
) -> bytes:
    if not payload.startswith(MAGIC):
        return payload
    header_size = len(MAGIC) + SALT_SIZE + NONCE_SIZE
    if len(payload) <= header_size:
        raise RuntimeError("Encrypted OneTrainer file is truncated.")
    offset = len(MAGIC)
    salt = payload[offset:offset + SALT_SIZE]
    offset += SALT_SIZE
    nonce = payload[offset:offset + NONCE_SIZE]
    ciphertext = payload[offset + NONCE_SIZE:]
    purpose_bytes = _purpose_bytes(purpose)
    key = _derive_key(_secret_bytes(secret), salt, purpose_bytes)
    try:
        return AESGCM(key).decrypt(nonce, ciphertext, MAGIC + purpose_bytes)
    except InvalidTag as exc:
        raise RuntimeError(
            "Could not decrypt OneTrainer data. The encryption key is wrong or the file is corrupted."
        ) from exc


def is_encrypted_file(path: str | os.PathLike[str]) -> bool:
    try:
        with open(path, "rb") as handle:
            return handle.read(len(MAGIC)) == MAGIC
    except FileNotFoundError:
        return False


class EncryptedReader(io.BytesIO):
    """Readable in-memory view of an encrypted OneTrainer file."""

    def __init__(
            self,
            path: str | os.PathLike[str],
            purpose: str | bytes = SOURCE_PURPOSE,
            *,
            secret: str | bytes | None = None,
    ) -> None:
        payload = Path(path).read_bytes()
        super().__init__(_decrypt_blob(payload, secret, purpose))
        self.name = str(path)


def open_source_binary(path: str | os.PathLike[str]):
    """Open a dataset source file, transparently decrypting it when needed."""
    if is_encrypted_file(path):
        return EncryptedReader(path, SOURCE_PURPOSE)
    return open(path, "rb")


def encrypt_file(
        source: str | os.PathLike[str],
        destination: str | os.PathLike[str],
        secret: str | bytes,
        *,
        salt: bytes | None = None,
        purpose: str | bytes = SOURCE_PURPOSE,
) -> bytes:
    payload, salt = _encrypt_blob(Path(source).read_bytes(), secret, purpose, salt=salt)
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    destination_path.write_bytes(payload)
    return salt


def decrypt_file(
        source: str | os.PathLike[str],
        destination: str | os.PathLike[str],
        secret: str | bytes,
        *,
        purpose: str | bytes = SOURCE_PURPOSE,
) -> None:
    plaintext = _decrypt_blob(Path(source).read_bytes(), secret, purpose)
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    destination_path.write_bytes(plaintext)
