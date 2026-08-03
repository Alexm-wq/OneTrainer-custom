import os
import traceback

from modules.model.PRXPixelModel import PRXPixelModel
from modules.modelLoader.mixin.HFModelLoaderMixin import HFModelLoaderMixin
from modules.util.config.TrainConfig import QuantizationConfig
from modules.util.enum.ModelType import ModelType
from modules.util.ModelNames import ModelNames
from modules.util.ModelWeightDtypes import ModelWeightDtypes

import torch

from diffusers import FlowMatchEulerDiscreteScheduler, GGUFQuantizationConfig, PRXTransformer2DModel
from transformers import AutoTokenizer, Qwen3VLTextModel


class PRXPixelModelLoader(HFModelLoaderMixin):
    def __init__(self):
        super().__init__()

    def __load_diffusers(
            self,
            model: PRXPixelModel,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            transformer_model_name: str,
            quantization: QuantizationConfig,
    ):
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, subfolder="tokenizer")
        noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(base_model_name, subfolder="scheduler")

        text_encoder = self._load_transformers_sub_module(
            Qwen3VLTextModel,
            weight_dtypes.text_encoder,
            weight_dtypes.fallback_train_dtype,
            base_model_name,
            "text_encoder",
        )

        if transformer_model_name:
            transformer = PRXTransformer2DModel.from_single_file(
                transformer_model_name,
                config=base_model_name,
                subfolder="transformer",
                torch_dtype=torch.bfloat16 if weight_dtypes.transformer.torch_dtype() is None
                else weight_dtypes.transformer.torch_dtype(),
                quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16)
                if weight_dtypes.transformer.is_gguf() else None,
            )
            transformer = self._convert_diffusers_sub_module_to_dtype(
                transformer, weight_dtypes.transformer, weight_dtypes.train_dtype, quantization,
            )
        else:
            transformer = self._load_diffusers_sub_module(
                PRXTransformer2DModel,
                weight_dtypes.transformer,
                weight_dtypes.train_dtype,
                base_model_name,
                "transformer",
                quantization,
            )

        model.model_type = model_type
        model.tokenizer = tokenizer
        model.noise_scheduler = noise_scheduler
        model.text_encoder = text_encoder
        model.transformer = transformer

    def __load_internal(
            self,
            model: PRXPixelModel,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            transformer_model_name: str,
            quantization: QuantizationConfig,
    ):
        if not os.path.isfile(os.path.join(base_model_name, "meta.json")):
            raise Exception("not an internal model")
        self.__load_diffusers(
            model, model_type, weight_dtypes, base_model_name, transformer_model_name, quantization,
        )

    def load(
            self,
            model: PRXPixelModel,
            model_type: ModelType,
            model_names: ModelNames,
            weight_dtypes: ModelWeightDtypes,
            quantization: QuantizationConfig,
    ):
        stacktraces = []
        loaders = [self.__load_internal, self.__load_diffusers]

        for loader in loaders:
            try:
                loader(
                    model,
                    model_type,
                    weight_dtypes,
                    model_names.base_model,
                    model_names.transformer_model,
                    quantization,
                )
                return
            except Exception:
                stacktraces.append(traceback.format_exc())

        for stacktrace in stacktraces:
            print(stacktrace)
        raise Exception("could not load PRX Pixel model: " + model_names.base_model)
