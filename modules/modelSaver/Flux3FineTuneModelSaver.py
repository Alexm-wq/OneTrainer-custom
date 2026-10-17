from modules.model.Flux3Model import Flux3Model
from modules.modelSaver.flux3.Flux3ModelSaver import Flux3ModelSaver
from modules.modelSaver.GenericFineTuneModelSaver import make_fine_tune_model_saver
from modules.util.enum.ModelType import ModelType

Flux3FineTuneModelSaver = make_fine_tune_model_saver(
    ModelType.FLUX_3_KLEIN,
    model_class=Flux3Model,
    model_saver_class=Flux3ModelSaver,
    embedding_saver_class=None,
)
