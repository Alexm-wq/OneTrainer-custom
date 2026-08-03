from modules.model.PRXPixelModel import PRXPixelModel
from modules.modelLoader.GenericLoRAModelLoader import make_lora_model_loader
from modules.modelLoader.prxPixel.PRXPixelLoRALoader import PRXPixelLoRALoader
from modules.modelLoader.prxPixel.PRXPixelModelLoader import PRXPixelModelLoader
from modules.util.enum.ModelType import ModelType


PRXPixelLoRAModelLoader = make_lora_model_loader(
    model_spec_map={ModelType.PRX_PIXEL: "resources/sd_model_spec/prx-pixel-lora.json"},
    model_class=PRXPixelModel,
    model_loader_class=PRXPixelModelLoader,
    embedding_loader_class=None,
    lora_loader_class=PRXPixelLoRALoader,
)
