from modules.model.MageFlowModel import MageFlowModel
from modules.modelLoader.GenericFineTuneModelLoader import make_fine_tune_model_loader
from modules.modelLoader.mageFlow.MageFlowModelLoader import MageFlowModelLoader
from modules.util.enum.ModelType import ModelType

MageFlowFineTuneModelLoader = make_fine_tune_model_loader(
    model_spec_map={ModelType.MAGE_FLOW: "resources/sd_model_spec/mage-flow.json"},
    model_class=MageFlowModel,
    model_loader_class=MageFlowModelLoader,
    embedding_loader_class=None,
)
