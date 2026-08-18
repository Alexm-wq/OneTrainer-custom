from modules.model.MageFlowModel import MageFlowModel
from modules.modelLoader.mixin.LoRALoaderMixin import LoRALoaderMixin
from modules.util.ModelNames import ModelNames


class MageFlowLoRALoader(LoRALoaderMixin):
    def __init__(self):
        super().__init__()

    def load(self, model: MageFlowModel, model_names: ModelNames):
        return self._load(model, model_names)
