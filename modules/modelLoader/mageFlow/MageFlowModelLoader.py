from __future__ import annotations

import os

from modules.model.MageFlowModel import MageFlowModel
from modules.util.config.TrainConfig import QuantizationConfig
from modules.util.enum.ModelType import ModelType
from modules.util.ModelNames import ModelNames
from modules.util.ModelWeightDtypes import ModelWeightDtypes


class MageFlowModelLoader:
    """Load Microsoft's official diffusers-style Mage repository lazily."""

    @staticmethod
    def _require_mage():
        try:
            from mage_flow.pipeline import load_from_repo
            return load_from_repo
        except ImportError as exc:
            raise RuntimeError(
                "Mage-Flow is part of OneTrainer's cuda13 Pixi environment. "
                "From the OneTrainer repository run: pixi install -e cuda13. "
                "Then launch OneTrainer with: pixi run -e cuda13 ui."
            ) from exc

    @staticmethod
    def _resolve_attention_backend() -> tuple[str, str]:
        """Select Mage's native FA4 backend when available, otherwise SDPA.

        Mage upstream defaults to FlashAttention2. On CUDA 13/Blackwell our
        environment installs the CuTeDSL FlashAttention4 package instead. The
        first value controls Mage's shared packed-varlen attention shim; the
        second is the Hugging Face Qwen3-VL attention implementation used while
        constructing the frozen text encoder.
        """
        try:
            from flash_attn.cute import flash_attn_varlen_func  # noqa: F401
            return "flash4", "flash_attention_4"
        except ImportError:
            return "sdpa", "sdpa"

    def load(
            self,
            model: MageFlowModel,
            model_type: ModelType,
            model_names: ModelNames,
            weight_dtypes: ModelWeightDtypes,
            quantization: QuantizationConfig,
    ):
        if model_type != ModelType.MAGE_FLOW:
            raise ValueError(f"MageFlowModelLoader cannot load model type {model_type}")
        if not model_names.base_model:
            raise ValueError("Mage-Flow requires a base model directory or Hugging Face repository id")

        load_from_repo = self._require_mage()
        mage_attn_backend, hf_attn_impl = self._resolve_attention_backend()

        # Upstream Mage constructs ModelConfig without an attn_type override, so
        # it defaults to flash2. Force the Qwen3-VL constructor independently,
        # then switch Mage's shared packed attention shim after construction.
        previous_hf_attn_impl = os.environ.get("VF_HF_ATTN_IMPL")
        os.environ["VF_HF_ATTN_IMPL"] = hf_attn_impl
        try:
            official = load_from_repo(model_names.base_model, device="cpu")
        finally:
            if previous_hf_attn_impl is None:
                os.environ.pop("VF_HF_ATTN_IMPL", None)
            else:
                os.environ["VF_HF_ATTN_IMPL"] = previous_hf_attn_impl

        from mage_flow.models.modules._attn_backend import set_attn_backend
        set_attn_backend(mage_attn_backend)
        print(
            f"[Mage-Flow] attention backend={mage_attn_backend} "
            f"qwen_attn_implementation={hf_attn_impl}"
        )

        if model_names.transformer_model:
            from safetensors.torch import load_file
            override = model_names.transformer_model
            if os.path.isdir(override):
                single = os.path.join(override, "diffusion_pytorch_model.safetensors")
                if os.path.isfile(single):
                    override = single
            if not os.path.isfile(override):
                raise FileNotFoundError(f"Mage transformer override not found: {override}")
            state = load_file(override, device="cpu")
            missing, unexpected = official.transformer.load_state_dict(state, strict=False, assign=True)
            if unexpected:
                raise RuntimeError(f"Unexpected Mage transformer keys: {unexpected[:8]}")
            if missing:
                print(f"WARNING: Mage transformer override has {len(missing)} missing keys")

        model.model_type = model_type
        model.base_model_name = model_names.base_model
        model.tokenizer = official.txt_enc.tokenizer
        model.noise_scheduler = official.scheduler
        model.text_encoder_wrapper = official.txt_enc
        model.text_encoder = official.txt_enc.hf_module
        model.vae = official.vae
        model.transformer = official.transformer
        model.official_model = official

        transformer_dtype = weight_dtypes.transformer.torch_dtype()
        vae_dtype = weight_dtypes.vae.torch_dtype()
        text_dtype = weight_dtypes.text_encoder.torch_dtype()
        if transformer_dtype is not None:
            model.transformer.to(dtype=transformer_dtype)
        if vae_dtype is not None:
            model.vae.to(dtype=vae_dtype)
        if text_dtype is not None:
            model.text_encoder_wrapper.to(dtype=text_dtype)
