import os

from modules.model.MOSSImageModel import MOSSImageModel
from modules.modelLoader.GenericFineTuneModelLoader import make_fine_tune_model_loader
from modules.modelLoader.GenericLoRAModelLoader import make_lora_model_loader
from modules.modelLoader.mixin.HFModelLoaderMixin import HFModelLoaderMixin
from modules.modelLoader.mixin.LoRALoaderMixin import LoRALoaderMixin
from modules.module.moss.pipeline_moss_image import MOSSImageScheduler, MOSSImageTransformer
from modules.util.enum.DataType import DataType
from modules.util.enum.ModelType import ModelType

import torch

from diffusers import AutoencoderKLFlux2
from transformers import Qwen2Tokenizer, Qwen3_5Model

from safetensors.torch import load_file


class MOSSImageModelLoader(HFModelLoaderMixin):
    def load(self, model, model_type, model_names, weight_dtypes, quantization):
        # The upstream forward explicitly casts activations to weight.dtype.
        # Integer/float8 weights therefore require a dedicated quantization integration.
        for name in ("transformer", "text_encoder"):
            if getattr(weight_dtypes, name) not in (DataType.FLOAT_32, DataType.BFLOAT_16):
                raise ValueError(f"MOSS {name} currently supports FLOAT_32 or BFLOAT_16 weights.")
        base = model_names.base_model
        override = model_names.transformer_model
        if override and os.path.isfile(override):
            config = MOSSImageTransformer.load_config(base, subfolder="transformer")
            # meta + assign avoids constructing a second 5B model in CPU RAM.
            with torch.device("meta"):
                transformer = MOSSImageTransformer.from_config(config)
            transformer.load_state_dict(load_file(override), strict=True, assign=True)
            transformer.to(dtype=weight_dtypes.transformer.torch_dtype())
        else:
            transformer = MOSSImageTransformer.from_pretrained(
                override or base, subfolder=None if override else "transformer",
                torch_dtype=weight_dtypes.transformer.torch_dtype(),
            )
        # Accelerate's direct tensor placement can bypass the upstream RoPE load
        # hook. Rebuild the derived frequencies in fp32 immediately after loading.
        transformer.rope.float()
        model.transformer = transformer
        model.tokenizer = Qwen2Tokenizer.from_pretrained(base, subfolder="tokenizer")
        model.text_encoder = Qwen3_5Model.from_pretrained(
            base, subfolder="text_encoder", dtype=weight_dtypes.text_encoder.torch_dtype(),
        ).eval()
        model.vae = AutoencoderKLFlux2.from_pretrained(
            model_names.vae_model or base, subfolder=None if model_names.vae_model else "vae",
            torch_dtype=torch.float32,
        ).eval()
        model.noise_scheduler = MOSSImageScheduler.from_pretrained(base, subfolder="scheduler")
        model.model_type = model_type


class MOSSImageLoRALoader(LoRALoaderMixin):
    def _legacy_conversion(self, model):
        return None

    def load(self, model, model_names):
        return self._load(model, model_names)


MOSSImageLoRAModelLoader = make_lora_model_loader(
    model_spec_map={ModelType.MOSS_IMAGE: "resources/sd_model_spec/moss-image-lora.json"},
    model_class=MOSSImageModel, model_loader_class=MOSSImageModelLoader,
    lora_loader_class=MOSSImageLoRALoader, embedding_loader_class=None,
)
MOSSImageFineTuneModelLoader = make_fine_tune_model_loader(
    model_spec_map={ModelType.MOSS_IMAGE: "resources/sd_model_spec/moss-image.json"},
    model_class=MOSSImageModel, model_loader_class=MOSSImageModelLoader,
    embedding_loader_class=None,
)
