from enum import Enum


class DPOObjective(Enum):
    SIGMOID = 'SIGMOID'
    ANCHORED_REJECT = 'ANCHORED_REJECT'
    IPO = 'IPO'

    def __str__(self):
        return self.value
