from modules.model.PRXPixelModel import PRXPixelModel
from modules.modelLoader.mixin.LoRALoaderMixin import LoRALoaderMixin
from modules.util.ModelNames import ModelNames


class PRXPixelLoRALoader(LoRALoaderMixin):
    def __init__(self):
        super().__init__()

    def load(self, model: PRXPixelModel, model_names: ModelNames):
        return self._load(model, model_names)
