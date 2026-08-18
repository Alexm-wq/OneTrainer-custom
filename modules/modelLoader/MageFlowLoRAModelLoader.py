from modules.model.MageFlowModel import MageFlowModel
from modules.modelLoader.GenericLoRAModelLoader import make_lora_model_loader
from modules.modelLoader.mageFlow.MageFlowLoRALoader import MageFlowLoRALoader
from modules.modelLoader.mageFlow.MageFlowModelLoader import MageFlowModelLoader
from modules.util.enum.ModelType import ModelType

MageFlowLoRAModelLoader = make_lora_model_loader(
    model_spec_map={ModelType.MAGE_FLOW: "resources/sd_model_spec/mage-flow-lora.json"},
    model_class=MageFlowModel,
    model_loader_class=MageFlowModelLoader,
    embedding_loader_class=None,
    lora_loader_class=MageFlowLoRALoader,
)
