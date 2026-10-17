from modules.model.Flux3Model import Flux3Model
from modules.modelSaver.flux3.Flux3LoRASaver import Flux3LoRASaver
from modules.modelSaver.GenericLoRAModelSaver import make_lora_model_saver
from modules.util.enum.ModelType import ModelType

Flux3LoRAModelSaver = make_lora_model_saver(
    ModelType.FLUX_3_KLEIN,
    model_class=Flux3Model,
    lora_saver_class=Flux3LoRASaver,
    embedding_saver_class=None,
)
