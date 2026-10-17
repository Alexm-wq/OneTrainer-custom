from modules.model.MOSSImageModel import MOSSImageModel
from modules.modelSaver.moss.MOSSImageModelSaver import MOSSImageModelSaver
from modules.modelSaver.GenericFineTuneModelSaver import make_fine_tune_model_saver
from modules.util.enum.ModelType import ModelType

MOSSImageFineTuneModelSaver = make_fine_tune_model_saver(
    ModelType.MOSS_IMAGE,
    model_class=MOSSImageModel,
    model_saver_class=MOSSImageModelSaver,
    embedding_saver_class=None,
)
