from modules.model.MageFlowModel import MageFlowModel
from modules.modelSaver.GenericFineTuneModelSaver import make_fine_tune_model_saver
from modules.modelSaver.mageFlow.MageFlowModelSaver import MageFlowModelSaver
from modules.util.enum.ModelType import ModelType

MageFlowFineTuneModelSaver = make_fine_tune_model_saver(
    ModelType.MAGE_FLOW,
    model_class=MageFlowModel,
    model_saver_class=MageFlowModelSaver,
    embedding_saver_class=None,
)
