from __future__ import annotations

import os

from modules.model.MageFlowModel import MageFlowModel
from modules.util.config.TrainConfig import QuantizationConfig
from modules.util.enum.ModelType import ModelType
from modules.util.ModelNames import ModelNames
from modules.util.ModelWeightDtypes import ModelWeightDtypes

import torch


class MageFlowModelLoader:
    """Load Microsoft's official diffusers-style Mage repository.

    ``mage_flow`` is deliberately imported lazily. OneTrainer can therefore be
    used for every existing model without installing Mage's optional package.
    Install Mage with ``--no-deps`` in the existing pixi environment so its
    metadata cannot silently replace OneTrainer's CUDA/PyTorch stack.
    """

    @staticmethod
    def _require_mage():
        try:
            from mage_flow.pipeline import load_from_repo
            return load_from_repo
        except ImportError as exc:
            raise RuntimeError(
                "Mage-Flow support requires Microsoft's official mage_flow package. "
                "Install it in the active OneTrainer environment with: "
                "python -m pip install --no-deps 'git+https://github.com/microsoft/Mage.git'"
            ) from exc

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
        official = load_from_repo(model_names.base_model, device="cpu")

        # Optional denoiser override follows OneTrainer's normal transformer_model
        # convention. This is useful for a raw fine-tuned transformer safetensor.
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

        # The official loader constructs BF16 modules. Respect an explicit
        # OneTrainer component dtype when it is an ordinary torch dtype; later
        # setup/quantization remains handled by OneTrainer.
        transformer_dtype = weight_dtypes.transformer.torch_dtype()
        vae_dtype = weight_dtypes.vae.torch_dtype()
        text_dtype = weight_dtypes.text_encoder.torch_dtype()
        if transformer_dtype is not None:
            model.transformer.to(dtype=transformer_dtype)
        if vae_dtype is not None:
            model.vae.to(dtype=vae_dtype)
        if text_dtype is not None:
            model.text_encoder_wrapper.to(dtype=text_dtype)
