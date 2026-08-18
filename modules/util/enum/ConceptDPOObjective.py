from enum import Enum


class ConceptDPOObjective(Enum):
    """Per-concept DPO objective selection.

    DEFAULT inherits the objective selected in the global RLHF tab.  Only the
    explicit SIGMOID override is exposed for now so future objectives can be
    added without changing the concept config field again.
    """

    DEFAULT = "DEFAULT"
    SIGMOID = "SIGMOID"

    def __str__(self):
        return self.value
