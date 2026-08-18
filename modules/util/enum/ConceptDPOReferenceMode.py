from enum import Enum


class ConceptDPOReferenceMode(Enum):
    """Per-concept override for the frozen DPO reference."""

    DEFAULT = "DEFAULT"
    BASE_MODEL = "BASE_MODEL"
    CURRENT_ADAPTER_SNAPSHOT = "CURRENT_ADAPTER_SNAPSHOT"
    CURRENT_ADAPTER_SNAPSHOT_CPU = "CURRENT_ADAPTER_SNAPSHOT_CPU"

    def __str__(self):
        return self.value
