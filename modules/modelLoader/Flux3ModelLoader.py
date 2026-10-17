import os

from modules.model.Flux3Model import Flux3Model
from modules.modelLoader.GenericFineTuneModelLoader import make_fine_tune_model_loader
from modules.modelLoader.GenericLoRAModelLoader import make_lora_model_loader
from modules.modelLoader.mixin.HFModelLoaderMixin import HFModelLoaderMixin
from modules.modelLoader.mixin.LoRALoaderMixin import LoRALoaderMixin
from modules.util.enum.ModelType import ModelType

import torch

from diffusers import AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler, Flux3Transformer2DModel
from diffusers.loaders.single_file_utils import (
    convert_flux3_transformer_checkpoint_to_diffusers,
    load_single_file_checkpoint,
)
from transformers import AutoProcessor, Qwen3VLModel


class Flux3ModelLoader(HFModelLoaderMixin):
    def load(self, model, model_type, model_names, weight_dtypes, quantization):
        for name in ("transformer", "text_encoder"):
            if getattr(weight_dtypes, name).is_gguf():
                raise ValueError(
                    f"FLUX.3 {name}: select floating-point safetensors and INT8/NF4/FP8 quantization. "
                    "Native BFL rotated FP8/NVFP4 packages and GGUF are not interchangeable with these weights."
                )
        base, override = model_names.base_model, model_names.transformer_model
        storage_dtype = (
            weight_dtypes.train_dtype.torch_dtype()
            if weight_dtypes.transformer.is_quantized() else weight_dtypes.transformer.torch_dtype()
        ) or torch.bfloat16
        single_file = override and (
            os.path.isfile(override) or override.lower().split("?")[0].endswith(".safetensors")
        )
        if single_file:
            config = Flux3Transformer2DModel.load_config(base, subfolder="transformer")
            state = load_single_file_checkpoint(override)
            for prefix in ("model.diffusion_model.", "diffusion_model.", "transformer."):
                if state and all(key.startswith(prefix) for key in state):
                    state = {key.removeprefix(prefix): value for key, value in state.items()}
                    break
            if any(key.startswith("emb_in.") for key in state):
                state = convert_flux3_transformer_checkpoint_to_diffusers(state)
            config["flow_map"] = any(key.startswith("target_timestep_embedder.") for key in state)
            with torch.device("meta"):
                transformer = Flux3Transformer2DModel.from_config(config)
            state = {key: value.to(storage_dtype) for key, value in state.items()}
            transformer.load_state_dict(state, strict=True, assign=True)
        else:
            transformer = Flux3Transformer2DModel.from_pretrained(
                override or base, subfolder=None if override else "transformer", torch_dtype=storage_dtype,
            )
        model.transformer = self._convert_diffusers_sub_module_to_dtype(
            transformer, weight_dtypes.transformer, weight_dtypes.train_dtype, quantization,
        )
        # Preserve the reference's fp32 QK norm scales after OT weight conversion.
        for name, module in model.transformer.named_modules():
            if name.endswith(("norm_q", "norm_k")):
                module.float()
        # BFL keeps the tokenizer/processor beside the text encoder, not in tokenizer/.
        model.processor = AutoProcessor.from_pretrained(base, subfolder="text_encoder")
        model.tokenizer = model.processor.tokenizer
        model.tokenizer.chat_template = model.processor.chat_template
        model.tokenizer.padding_side = "right"
        text_dtype = (
            weight_dtypes.fallback_train_dtype.torch_dtype()
            if weight_dtypes.text_encoder.is_quantized() else weight_dtypes.text_encoder.torch_dtype()
        ) or torch.bfloat16
        encoder = Qwen3VLModel.from_pretrained(base, subfolder="text_encoder", dtype=text_dtype).eval()
        model.text_encoder = self._convert_transformers_sub_module_to_dtype(
            encoder, weight_dtypes.text_encoder, weight_dtypes.fallback_train_dtype, quantization,
        )
        model.vae = AutoencoderKLFlux2.from_pretrained(
            model_names.vae_model or base, subfolder=None if model_names.vae_model else "vae",
            torch_dtype=torch.float32,
        ).eval()
        # A unit-shift 1000-bin grid describes the training loss. Sampling uses
        # the separate BFL alpha=16 schedule; training applies its shift once.
        model.noise_scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000, shift=1.0, use_dynamic_shifting=False,
        )
        model.model_type = model_type


class Flux3LoRALoader(LoRALoaderMixin):
    def _legacy_conversion(self, model):
        return None

    def load(self, model, model_names):
        return self._load(model, model_names)


Flux3LoRAModelLoader = make_lora_model_loader(
    model_spec_map={ModelType.FLUX_3_KLEIN: "resources/sd_model_spec/flux3-klein-lora.json"},
    model_class=Flux3Model, model_loader_class=Flux3ModelLoader,
    lora_loader_class=Flux3LoRALoader, embedding_loader_class=None,
)
Flux3FineTuneModelLoader = make_fine_tune_model_loader(
    model_spec_map={ModelType.FLUX_3_KLEIN: "resources/sd_model_spec/flux3-klein.json"},
    model_class=Flux3Model, model_loader_class=Flux3ModelLoader, embedding_loader_class=None,
)
