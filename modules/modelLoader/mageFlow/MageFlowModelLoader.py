from __future__ import annotations

import os

from modules.model.MageFlowModel import MageFlowModel
from modules.modelLoader.mixin.HFModelLoaderMixin import HFModelLoaderMixin
from modules.util.config.TrainConfig import QuantizationConfig
from modules.util.enum.ModelType import ModelType
from modules.util.ModelNames import ModelNames
from modules.util.ModelWeightDtypes import ModelWeightDtypes
from modules.util.quantization_util import is_quantized_parameter

import torch
from torch import nn


class MageFlowModelLoader(HFModelLoaderMixin):
    """Load Microsoft's official diffusers-style Mage repository lazily."""

    def __init__(self):
        super().__init__()

    @staticmethod
    def _require_mage():
        try:
            import mage_flow.pipeline as mage_pipeline
            return mage_pipeline
        except ImportError as exc:
            raise RuntimeError(
                "Mage-Flow is part of OneTrainer's cuda13 Pixi environment. "
                "From the OneTrainer repository run: pixi install -e cuda13. "
                "Then launch OneTrainer with: pixi run -e cuda13 ui."
            ) from exc

    @staticmethod
    def _resolve_attention_backend() -> tuple[str, str]:
        """Select a safe Mage packed-attention backend.

        ``OT_MAGE_ATTN_BACKEND`` can explicitly select ``sdpa`` or ``flash4``.
        Without an override, consumer Blackwell (SM120/SM121) uses SDPA. FA4's
        CuTeDSL varlen path has had native-process crashes and kernel failures on
        these architectures, and Mage uses varlen attention for both the Qwen
        conditioning encoder and the image transformer. A Python ImportError
        probe therefore is not a sufficient runtime-safety test on RTX 50-series.

        Datacenter Hopper/Blackwell keeps the previous FA4-if-importable policy.
        The second return value is the Hugging Face Qwen attention implementation
        used for non-packed Qwen forwards; packed conditioning still goes through
        Mage's shared backend shim.
        """
        override = os.environ.get("OT_MAGE_ATTN_BACKEND", "").strip().lower()
        if override:
            if override in {"sdpa", "sdp", "torch_sdpa"}:
                print("[Mage-Flow] OT_MAGE_ATTN_BACKEND=sdpa")
                return "sdpa", "sdpa"
            if override in {"flash4", "fa4", "flash_attention_4"}:
                try:
                    from flash_attn.cute import flash_attn_varlen_func  # noqa: F401
                except ImportError as exc:
                    raise RuntimeError(
                        "OT_MAGE_ATTN_BACKEND=flash4 was requested, but "
                        "flash_attn.cute is not importable."
                    ) from exc
                print("[Mage-Flow] OT_MAGE_ATTN_BACKEND=flash4")
                return "flash4", "flash_attention_4"
            raise ValueError(
                "OT_MAGE_ATTN_BACKEND must be one of: sdpa, flash4"
            )

        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            if major == 12:
                print(
                    f"[Mage-Flow] CUDA capability sm_{major}{minor}: using SDPA "
                    "for packed attention by default. Set "
                    "OT_MAGE_ATTN_BACKEND=flash4 to explicitly test FA4."
                )
                return "sdpa", "sdpa"

        try:
            from flash_attn.cute import flash_attn_varlen_func  # noqa: F401
            return "flash4", "flash_attention_4"
        except ImportError:
            return "sdpa", "sdpa"

    @staticmethod
    def _quantization_summary(module: nn.Module, requested_dtype) -> tuple[int, int]:
        """Return eligible Linear count and count backed by OT quantization."""
        total = 0
        quantized = 0
        extra_quantized_types = {
            "LinearA8",
            "LinearGGUFA8",
            "GGUFLinear",
        }
        for child in module.modules():
            is_linear = isinstance(child, nn.Linear) or child.__class__.__name__ in extra_quantized_types
            if not is_linear:
                continue
            total += 1
            try:
                is_quantized = any(
                    is_quantized_parameter(child, parameter_name)
                    for parameter_name in child._parameters
                )
            except (AttributeError, TypeError):
                is_quantized = False
            if is_quantized or child.__class__.__name__ in extra_quantized_types:
                quantized += 1
        print(
            f"[Mage-Flow] {module.__class__.__name__} "
            f"weight_dtype={requested_dtype} linear_layers={total} "
            f"quantized_weight_layers={quantized}"
        )
        return total, quantized

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

        mage_pipeline = self._require_mage()
        mage_attn_backend, hf_attn_impl = self._resolve_attention_backend()

        # Upstream load_from_repo() constructs ModelConfig without attn_type,
        # which otherwise selects its default "flash2" before the text encoder
        # is created. Force the backend at the actual ModelConfig construction
        # point so BOTH Mage's packed-varlen attention and Qwen3-VL are created
        # consistently. Keep the HF env override as a belt-and-suspenders
        # override for the Qwen constructor, then restore all upstream globals.
        original_model_config = mage_pipeline.ModelConfig
        previous_hf_attn_impl = os.environ.get("VF_HF_ATTN_IMPL")

        def model_config_with_attention(*args, **kwargs):
            kwargs["attn_type"] = mage_attn_backend
            return original_model_config(*args, **kwargs)

        mage_pipeline.ModelConfig = model_config_with_attention
        os.environ["VF_HF_ATTN_IMPL"] = hf_attn_impl
        try:
            official = mage_pipeline.load_from_repo(model_names.base_model, device="cpu")
        finally:
            mage_pipeline.ModelConfig = original_model_config
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

        # Mage's official loader returns fully materialized torch modules. Run
        # those modules through OneTrainer's normal conversion pass so configured
        # weight dtypes and quantized Linear replacements are applied before the
        # setup phase calls quantize_layers(). Pass the user's QuantizationConfig
        # to BOTH transformer and text encoder so layer filtering/SVD settings are
        # not silently dropped for Qwen.
        official.transformer = self._convert_diffusers_sub_module_to_dtype(
            official.transformer,
            weight_dtypes.transformer,
            weight_dtypes.train_dtype,
            quantization,
        )
        official.txt_enc.hf_module = self._convert_transformers_sub_module_to_dtype(
            official.txt_enc.hf_module,
            weight_dtypes.text_encoder,
            weight_dtypes.fallback_train_dtype,
            quantization,
        )
        official.vae = self._convert_diffusers_sub_module_to_dtype(
            official.vae,
            weight_dtypes.vae,
            weight_dtypes.train_dtype,
        )

        self._quantization_summary(official.transformer, weight_dtypes.transformer)
        _, te_quantized = self._quantization_summary(
            official.txt_enc.hf_module,
            weight_dtypes.text_encoder,
        )
        if weight_dtypes.text_encoder.is_quantized() and te_quantized == 0:
            raise RuntimeError(
                "Mage text encoder is configured with quantized weight dtype "
                f"{weight_dtypes.text_encoder}, but zero Qwen Linear layers were "
                "replaced. Check the Quantization layer filter; it may exclude "
                "all text-encoder layer names."
            )

        model.model_type = model_type
        model.base_model_name = model_names.base_model
        model.mage_attention_backend = mage_attn_backend
        model.tokenizer = official.txt_enc.tokenizer
        model.noise_scheduler = official.scheduler
        model.text_encoder_wrapper = official.txt_enc
        model.text_encoder = official.txt_enc.hf_module
        model.vae = official.vae
        model.transformer = official.transformer
        model.official_model = official
