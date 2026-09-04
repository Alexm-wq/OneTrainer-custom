from __future__ import annotations

import hashlib
import io
import os
import struct
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt


MAGIC = b"OTENC001"
LEGACY_MAGIC = b"OTENCV1\x00"
SALT_SIZE = 16
NONCE_SIZE = 12
SOURCE_PURPOSE = b"onetrainer/source/v1"
CACHE_PURPOSE = b"onetrainer/cache/v1"

# Legacy OTENCV1 header:
# magic[8] + salt[16] + nonce_prefix[8] + chunk_size[u32be] + purpose_digest[16]
_LEGACY_HEADER = struct.Struct(">8s16s8sI16s")
_LEGACY_RECORD = struct.Struct(">I")
_LEGACY_TAG_SIZE = 16

_configured_secret: str | bytes | None = None


class EncryptionError(RuntimeError):
    """Raised when authenticated OneTrainer data cannot be decrypted."""


def configure_encryption_key(secret: str | bytes | None) -> None:
    """Configure the process-local key used for encrypted dataset/cache reads."""
    global _configured_secret
    _configured_secret = secret if secret not in ("", b"") else None
    _derive_key.cache_clear()
    _legacy_derive_key.cache_clear()


def _secret_bytes(secret: str | bytes | None) -> bytes:
    value = _configured_secret if secret is None else secret
    if value is None or value == "" or value == b"":
        raise RuntimeError(
            "Encrypted OneTrainer data was opened without an encryption key. "
            "Configure the dataset/cache key first."
        )
    return value if isinstance(value, bytes) else value.encode("utf-8")


def _purpose_bytes(purpose: str | bytes) -> bytes:
    return purpose if isinstance(purpose, bytes) else purpose.encode("utf-8")


@lru_cache(maxsize=64)
def _derive_key(secret: bytes, salt: bytes, purpose: bytes) -> bytes:
    kdf = Scrypt(salt=salt, length=32, n=2**14, r=8, p=1)
    return kdf.derive(secret + b"\0" + purpose)


@lru_cache(maxsize=64)
def _legacy_derive_key(secret: bytes, salt: bytes, purpose_digest: bytes) -> bytes:
    return Scrypt(
        salt=salt + purpose_digest,
        length=32,
        n=2**15,
        r=8,
        p=1,
    ).derive(secret)


def _encrypt_blob(
        plaintext: bytes,
        secret: str | bytes | None,
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


def _decrypt_current_blob(
        payload: bytes,
        secret: str | bytes | None,
        purpose: str | bytes,
) -> bytes:
    header_size = len(MAGIC) + SALT_SIZE + NONCE_SIZE
    if len(payload) <= header_size:
        raise EncryptionError("Encrypted OneTrainer file is truncated.")
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
        raise EncryptionError(
            "Could not decrypt OneTrainer data. The encryption key is wrong or the file is corrupted."
        ) from exc


def _legacy_purpose_digest(purpose: str | bytes) -> bytes:
    return hashlib.sha256(_purpose_bytes(purpose)).digest()[:16]


def _decrypt_legacy_blob(
        payload: bytes,
        secret: str | bytes | None,
        purpose: str | bytes,
) -> bytes:
    if len(payload) < _LEGACY_HEADER.size:
        raise EncryptionError("Truncated OTENCV1 header.")
    header = payload[:_LEGACY_HEADER.size]
    magic, salt, nonce_prefix, chunk_size, stored_purpose = _LEGACY_HEADER.unpack(header)
    if magic != LEGACY_MAGIC:
        raise EncryptionError("Not an OTENCV1 file.")
    expected_purpose = _legacy_purpose_digest(purpose)
    if stored_purpose != expected_purpose:
        raise EncryptionError("OTENCV1 file has the wrong purpose/context.")

    aes = AESGCM(_legacy_derive_key(_secret_bytes(secret), salt, stored_purpose))
    output = bytearray()
    offset = _LEGACY_HEADER.size
    chunk_index = 0
    while offset < len(payload):
        if offset + _LEGACY_RECORD.size > len(payload):
            raise EncryptionError("Truncated OTENCV1 chunk header.")
        length_bytes = payload[offset:offset + _LEGACY_RECORD.size]
        plaintext_length = _LEGACY_RECORD.unpack(length_bytes)[0]
        offset += _LEGACY_RECORD.size
        if plaintext_length > chunk_size:
            raise EncryptionError("Invalid OTENCV1 chunk length.")
        ciphertext_length = plaintext_length + _LEGACY_TAG_SIZE
        end = offset + ciphertext_length
        if end > len(payload):
            raise EncryptionError("Truncated OTENCV1 chunk.")
        ciphertext = payload[offset:end]
        offset = end

        index_bytes = chunk_index.to_bytes(4, "big")
        nonce = nonce_prefix + index_bytes
        aad = header + index_bytes + length_bytes
        try:
            output.extend(aes.decrypt(nonce, ciphertext, aad))
        except InvalidTag as exc:
            raise EncryptionError(
                "Could not authenticate OTENCV1 data. The encryption key is wrong or the file was modified."
            ) from exc
        chunk_index += 1

    if chunk_index == 0:
        raise EncryptionError("OTENCV1 file contains no authenticated chunks.")
    return bytes(output)


def decrypt_bytes(
        payload: bytes,
        secret: str | bytes | None = None,
        purpose: str | bytes = SOURCE_PURPOSE,
) -> bytes:
    if payload.startswith(MAGIC):
        return _decrypt_current_blob(payload, secret, purpose)
    if payload.startswith(LEGACY_MAGIC):
        return _decrypt_legacy_blob(payload, secret, purpose)
    return payload


def is_encrypted_file(path: str | os.PathLike[str]) -> bool:
    try:
        with open(path, "rb") as handle:
            magic = handle.read(8)
        return magic in {MAGIC, LEGACY_MAGIC}
    except FileNotFoundError:
        return False


class EncryptedReader(io.BytesIO):
    """Readable in-memory view of current or legacy encrypted OneTrainer data."""

    def __init__(
            self,
            path: str | os.PathLike[str],
            purpose: str | bytes = SOURCE_PURPOSE,
            *,
            secret: str | bytes | None = None,
    ) -> None:
        super().__init__(decrypt_bytes(Path(path).read_bytes(), secret, purpose))
        self.name = str(path)


def open_source_binary(path: str | os.PathLike[str]):
    """Open plaintext, OTENC001, or legacy OTENCV1 source data."""
    if is_encrypted_file(path):
        return EncryptedReader(path, SOURCE_PURPOSE)
    return open(path, "rb")


def read_source_text(path: str | os.PathLike[str], encoding: str = "utf-8") -> str:
    with open_source_binary(path) as source:
        return source.read().decode(encoding)


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
    plaintext = decrypt_bytes(Path(source).read_bytes(), secret, purpose)
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    destination_path.write_bytes(plaintext)


def secure_torch_save(
        value: Any,
        path: str | os.PathLike[str],
        *,
        encrypted: bool = False,
        purpose: str | bytes = CACHE_PURPOSE,
        salt: bytes | None = None,
) -> bytes | None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    torch.save(value, buffer)
    data = buffer.getvalue()
    used_salt = None
    if encrypted:
        data, used_salt = _encrypt_blob(data, None, purpose, salt=salt)
    temp = destination.with_name(destination.name + f".tmp-{os.getpid()}")
    temp.write_bytes(data)
    os.replace(temp, destination)
    return used_salt


def secure_torch_load(
        path: str | os.PathLike[str],
        *,
        purpose: str | bytes = CACHE_PURPOSE,
        map_location: Any = "cpu",
        desired_encryption: bool | None = None,
) -> Any:
    source = Path(path)
    encrypted = is_encrypted_file(source)
    if desired_encryption is not None and encrypted != bool(desired_encryption):
        raise EncryptionError(
            f"Cache encryption state mismatch for {source}: expected encrypted={bool(desired_encryption)}"
        )
    data = source.read_bytes()
    if encrypted:
        data = decrypt_bytes(data, None, purpose)
    return torch.load(io.BytesIO(data), weights_only=False, map_location=map_location)
