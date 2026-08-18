from modules.model.MageFlowModel import MageFlowModel
from modules.modelSaver.GenericLoRAModelSaver import make_lora_model_saver
from modules.modelSaver.mageFlow.MageFlowLoRASaver import MageFlowLoRASaver
from modules.util.enum.ModelType import ModelType

MageFlowLoRAModelSaver = make_lora_model_saver(
    ModelType.MAGE_FLOW,
    model_class=MageFlowModel,
    lora_saver_class=MageFlowLoRASaver,
    embedding_saver_class=None,
)
