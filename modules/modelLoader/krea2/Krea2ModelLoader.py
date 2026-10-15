import logging
import os
import time
import traceback

from modules.model.Krea2Model import Krea2Model
from modules.modelLoader.mixin.HFModelLoaderMixin import HFModelLoaderMixin
from modules.util.config.TrainConfig import QuantizationConfig
from modules.util.convert_util import convert, reverse_conversion
from modules.util.enum.ModelType import ModelType
from modules.util.ModelNames import ModelNames
from modules.util.ModelWeightDtypes import ModelWeightDtypes

import torch

from diffusers import (
    AutoencoderKLQwenImage,
    FlowMatchEulerDiscreteScheduler,
    Krea2Transformer2DModel,
)
from diffusers.loaders.single_file_utils import load_single_file_checkpoint
from transformers import Qwen2Tokenizer, Qwen3VLModel

import accelerate

logger = logging.getLogger(__name__)


class Krea2ModelLoader(
    HFModelLoaderMixin,
):
    def __init__(self):
        super().__init__()

    def _load_transformer_checkpoint(
            self,
            model: Krea2Model,
            base_model_name: str,
            transformer_model_name: str,
            weight_dtypes: ModelWeightDtypes,
            quantization: QuantizationConfig,
    ):
        if weight_dtypes.transformer.is_gguf():
            raise ValueError("Krea 2 transformer overrides require unquantized safetensors, not GGUF. "
                             "Select bfloat16, or let OneTrainer quantize the weights after loading.")

        started = time.perf_counter()
        config = Krea2Transformer2DModel.load_config(base_model_name, subfolder="transformer")
        # Avoid allocating a second, randomly initialized 12.8B model beside the checkpoint.
        with accelerate.init_empty_weights():
            transformer = Krea2Transformer2DModel.from_config(config)

        state_dict = load_single_file_checkpoint(transformer_model_name)
        # Accept native Krea/Kroma and Diffusers transformer-only exports, including the
        # common ComfyUI prefixes. Never drop unmatched weights from a mixed checkpoint.
        for prefix in ("model.diffusion_model.", "diffusion_model.", "transformer."):
            if state_dict and all(key.startswith(prefix) for key in state_dict):
                state_dict = {key.removeprefix(prefix): value for key, value in state_dict.items()}
                break
        if "first.weight" in state_dict:
            state_dict = convert(state_dict, reverse_conversion(model.checkpoint_diffusers_to_original()))

        # Cast one tensor at a time, releasing the original reference. In particular Kroma
        # Base ships in FP32; don't retain a complete FP32 copy when training in BF16.
        dtype = weight_dtypes.transformer.torch_dtype() or torch.bfloat16
        for key in state_dict:
            state_dict[key] = state_dict[key].to(dtype=dtype)
        # Strict assignment checks every name and shape and materializes the meta model.
        transformer.load_state_dict(state_dict, strict=True, assign=True)
        del state_dict
        transformer = self._convert_diffusers_sub_module_to_dtype(
            transformer, weight_dtypes.transformer, weight_dtypes.train_dtype, quantization,
        )
        logger.info("Loaded Krea 2 transformer override %s in %.2fs", transformer_model_name,
                    time.perf_counter() - started)
        return transformer

    def __load_internal(
            self,
            model: Krea2Model,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            transformer_model_name: str,
            vae_model_name: str,
            quantization: QuantizationConfig,
    ):
        if os.path.isfile(os.path.join(base_model_name, "meta.json")):
            self.__load_diffusers(
                model, model_type, weight_dtypes, base_model_name, transformer_model_name, vae_model_name, quantization,
            )
        else:
            raise Exception("not an internal model")

    def __load_diffusers(
            self,
            model: Krea2Model,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            transformer_model_name: str,
            vae_model_name: str,
            quantization: QuantizationConfig,
    ):
        tokenizer = Qwen2Tokenizer.from_pretrained(
            base_model_name,
            subfolder="tokenizer",
        )

        noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            base_model_name,
            subfolder="scheduler",
        )

        text_encoder = self._load_transformers_sub_module(
            Qwen3VLModel,
            weight_dtypes.text_encoder,
            weight_dtypes.fallback_train_dtype,
            base_model_name,
            "text_encoder",
        )

        if vae_model_name: #TODO simplify
            vae = self._load_diffusers_sub_module(
                AutoencoderKLQwenImage,
                weight_dtypes.vae,
                weight_dtypes.train_dtype,
                vae_model_name,
            )
        else:
            vae = self._load_diffusers_sub_module(
                AutoencoderKLQwenImage,
                weight_dtypes.vae,
                weight_dtypes.train_dtype,
                base_model_name,
                "vae",
            )

        if transformer_model_name:
            transformer = self._load_transformer_checkpoint(
                model, base_model_name, transformer_model_name, weight_dtypes, quantization,
            )
        else:
            transformer = self._load_diffusers_sub_module(
                Krea2Transformer2DModel,
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
        model.vae = vae
        model.transformer = transformer

    def __load_safetensors(
            self,
            model: Krea2Model,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            transformer_model_name: str,
            vae_model_name: str,
            quantization: QuantizationConfig,
    ):
        raise NotImplementedError("Loading of single file Krea 2 models not supported. Use the diffusers model instead. Optionally, transformer-only safetensor files can be loaded by overriding the transformer.")

    def load( #TODO share code between models
            self,
            model: Krea2Model,
            model_type: ModelType,
            model_names: ModelNames,
            weight_dtypes: ModelWeightDtypes,
            quantization: QuantizationConfig,
    ):
        stacktraces = []

        try:
            self.__load_internal(
                model, model_type, weight_dtypes, model_names.base_model, model_names.transformer_model, model_names.vae_model, quantization,
            )
            return
        except Exception:
            stacktraces.append(traceback.format_exc())

        try:
            self.__load_diffusers(
                model, model_type, weight_dtypes, model_names.base_model, model_names.transformer_model, model_names.vae_model, quantization,
            )
            return
        except Exception:
            stacktraces.append(traceback.format_exc())

        try:
            self.__load_safetensors(
                model, model_type, weight_dtypes, model_names.base_model, model_names.transformer_model, model_names.vae_model, quantization,
            )
            return
        except Exception:
            stacktraces.append(traceback.format_exc())

        for stacktrace in stacktraces:
            print(stacktrace)
        raise Exception("could not load model: " + model_names.base_model)
