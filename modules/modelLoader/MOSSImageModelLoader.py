import os

from modules.model.MOSSImageModel import MOSSImageModel
from modules.modelLoader.GenericFineTuneModelLoader import make_fine_tune_model_loader
from modules.modelLoader.GenericLoRAModelLoader import make_lora_model_loader
from modules.modelLoader.mixin.HFModelLoaderMixin import HFModelLoaderMixin
from modules.modelLoader.mixin.LoRALoaderMixin import LoRALoaderMixin
from modules.module.moss.pipeline_moss_image import MOSSImageScheduler, MOSSImageTransformer
from modules.util.enum.ModelType import ModelType

import torch

from diffusers import AutoencoderKLFlux2
from diffusers.loaders.single_file_utils import load_single_file_checkpoint
from transformers import Qwen2Tokenizer, Qwen3_5Model


class MOSSImageModelLoader(HFModelLoaderMixin):
    def load(self, model, model_type, model_names, weight_dtypes, quantization):
        for name in ("transformer", "text_encoder"):
            dtype = getattr(weight_dtypes, name)
            if dtype.is_gguf():
                raise ValueError(
                    f"MOSS {name} expects native safetensors or a Diffusers directory, not GGUF. "
                    "Select INT8, NF4 or FP8 to quantize during loading."
                )
        base = model_names.base_model
        override = model_names.transformer_model
        storage_dtype = weight_dtypes.transformer.torch_dtype() or torch.bfloat16
        is_single_file = override and (
            os.path.isfile(override) or override.lower().split("?")[0].endswith(".safetensors")
        )
        if is_single_file:
            config = MOSSImageTransformer.load_config(base, subfolder="transformer")
            with torch.device("meta"):
                transformer = MOSSImageTransformer.from_config(config)
            state = load_single_file_checkpoint(override)
            for prefix in ("model.diffusion_model.", "diffusion_model.", "transformer."):
                if state and all(key.startswith(prefix) for key in state):
                    state = {key.removeprefix(prefix): value for key, value in state.items()}
                    break
            for key in state:
                state[key] = state[key].to(storage_dtype)
            transformer.load_state_dict(state, strict=True, assign=True)
            del state
        else:
            transformer = MOSSImageTransformer.from_pretrained(
                override or base, subfolder=None if override else "transformer",
                torch_dtype=storage_dtype,
            )
        transformer = self._convert_diffusers_sub_module_to_dtype(
            transformer, weight_dtypes.transformer, weight_dtypes.train_dtype, quantization,
        )
        # Accelerate's direct tensor placement can bypass the upstream RoPE load
        # hook. Rebuild the derived frequencies in fp32 immediately after loading.
        transformer.rope.float()
        model.transformer = transformer
        model.tokenizer = Qwen2Tokenizer.from_pretrained(base, subfolder="tokenizer")
        text_encoder = Qwen3_5Model.from_pretrained(
            base, subfolder="text_encoder",
            dtype=weight_dtypes.text_encoder.torch_dtype() or torch.bfloat16,
        ).eval()
        # Native Transformers loading preserves Qwen3.5's derived buffers and
        # checkpoint conversion. Reuse OneTrainer's quantizers after loading.
        model.text_encoder = self._convert_transformers_sub_module_to_dtype(
            text_encoder, weight_dtypes.text_encoder, weight_dtypes.fallback_train_dtype,
        )
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
