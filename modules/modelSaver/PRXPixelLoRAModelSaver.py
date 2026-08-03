from modules.model.PRXPixelModel import PRXPixelModel
from modules.modelSaver.GenericLoRAModelSaver import make_lora_model_saver
from modules.modelSaver.prxPixel.PRXPixelLoRASaver import PRXPixelLoRASaver
from modules.util.enum.ModelType import ModelType


PRXPixelLoRAModelSaver = make_lora_model_saver(
    ModelType.PRX_PIXEL,
    model_class=PRXPixelModel,
    lora_saver_class=PRXPixelLoRASaver,
    embedding_saver_class=None,
)
