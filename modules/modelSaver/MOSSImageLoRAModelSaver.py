from modules.model.MOSSImageModel import MOSSImageModel
from modules.modelSaver.moss.MOSSImageLoRASaver import MOSSImageLoRASaver
from modules.modelSaver.GenericLoRAModelSaver import make_lora_model_saver
from modules.util.enum.ModelType import ModelType

MOSSImageLoRAModelSaver = make_lora_model_saver(
    ModelType.MOSS_IMAGE,
    model_class=MOSSImageModel,
    lora_saver_class=MOSSImageLoRASaver,
    embedding_saver_class=None,
)
