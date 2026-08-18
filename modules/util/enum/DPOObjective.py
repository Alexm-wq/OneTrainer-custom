from enum import Enum


class DPOObjective(Enum):
    SIGMOID = 'SIGMOID'
    IPO = 'IPO'
    ANCHORED_REJECT = 'ANCHORED_REJECT'
    LINEAR = 'LINEAR'
    BALANCED_REJECT = 'BALANCED_REJECT'

    def __str__(self):
        return self.value
