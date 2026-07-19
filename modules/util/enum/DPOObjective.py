from enum import Enum


class DPOObjective(Enum):
    SIGMOID = 'SIGMOID'
    IPO = 'IPO'
    ANCHORED_REJECT = 'ANCHORED_REJECT'

    def __str__(self):
        return self.value
