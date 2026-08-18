from enum import Enum


class DPORefMode(Enum):
    NEW_ADAPTER = "NEW_ADAPTER"
    EXISTING_ADAPTER = "EXISTING_ADAPTER"
    EXISTING_ADAPTER_CPU = "EXISTING_ADAPTER_CPU"
    EMA_ADAPTER = "EMA_ADAPTER"

    def __str__(self):
        return self.value
