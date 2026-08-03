from typing import Any

from modules.util.config.BaseConfig import BaseConfig
from modules.util.config.CloudConfig import CloudSecretsConfig


class SecretsConfig(BaseConfig):
    huggingface_token: str
    dataset_encryption_key: str
    cloud: CloudSecretsConfig

    def __init__(self, data: list[(str, Any, type, bool)]):
        super().__init__(data)

    @staticmethod
    def default_values() -> 'SecretsConfig':
        data = []

        # name, default value, data type, nullable
        data.append(("huggingface_token", "", str, False))
        # Session-only: TopBarController deliberately removes this from
        # secrets.json before writing it.
        data.append(("dataset_encryption_key", "", str, False))

        # cloud
        cloud = CloudSecretsConfig.default_values()
        data.append(("cloud", cloud, CloudSecretsConfig, False))

        return SecretsConfig(data)
