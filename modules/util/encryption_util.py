import base64
import binascii
import os

from modules.util.config.TrainConfig import TrainConfig

from mgds.crypto import configure_encryption_key


KEY_ENVIRONMENT_VARIABLE = "OT_DATASET_ENCRYPTION_KEY_B64"


def _key_from_environment() -> str:
    encoded = os.environ.get(KEY_ENVIRONMENT_VARIABLE, "")
    if encoded == "":
        return ""
    try:
        return base64.b64decode(encoded, validate=True).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError) as exc:
        raise RuntimeError(
            f"{KEY_ENVIRONMENT_VARIABLE} is not valid base64-encoded UTF-8"
        ) from exc


def configure_data_encryption(
        config: TrainConfig,
        *,
        require_key: bool = True,
) -> None:
    enabled = config.dataset_encryption_enabled or config.cache_encryption_enabled
    key = config.secrets.dataset_encryption_key
    if enabled and key == "":
        key = _key_from_environment()
        config.secrets.dataset_encryption_key = key

    if enabled and key == "" and require_key:
        raise RuntimeError(
            "Dataset/cache encryption is enabled, but no encryption key was entered. "
            "Enter it in the Data tab or set OT_DATASET_ENCRYPTION_KEY_B64."
        )

    configure_encryption_key(key if enabled and key != "" else None)
