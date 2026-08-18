from modules.model.MageFlowModel import MageFlowModel
from modules.modelSaver.mixin.LoRAModelSaverMixin import LoRAModelSaverMixin


class MageFlowLoRASaver(LoRAModelSaverMixin):
    def __init__(self):
        super().__init__()

    def _get_state_dict(self, model: MageFlowModel) -> dict:
        state_dict = {}
        if model.transformer_lora is not None:
            state_dict |= model.transformer_lora.state_dict()
        return state_dict
