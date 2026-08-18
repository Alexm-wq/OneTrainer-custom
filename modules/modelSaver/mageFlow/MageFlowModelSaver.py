from __future__ import annotations

import os
import shutil
from pathlib import Path

from modules.model.MageFlowModel import MageFlowModel
from modules.modelSaver.mixin.DtypeModelSaverMixin import DtypeModelSaverMixin
from modules.util.enum.ModelFormat import ModelFormat

import torch
from safetensors.torch import save_file


class MageFlowModelSaver(DtypeModelSaverMixin):
    """Persist Mage in the same diffusers-style repository layout Microsoft ships."""

    def __init__(self):
        super().__init__()

    @staticmethod
    def _source_repo(model: MageFlowModel) -> str:
        source = model.base_model_name
        if not source:
            raise RuntimeError("Mage model has no remembered base repository")
        if os.path.isdir(source):
            return os.path.realpath(source)
        from huggingface_hub import snapshot_download
        return snapshot_download(repo_id=source)

    def _transformer_state(self, model: MageFlowModel, dtype: torch.dtype | None):
        state = self._convert_state_dict_dtype(model.transformer.state_dict(), dtype)
        self._convert_state_dict_to_contiguous(state)
        return state

    def _save_repo(self, model: MageFlowModel, destination: str, dtype: torch.dtype | None):
        source = self._source_repo(model)
        destination = os.path.realpath(destination)
        os.makedirs(destination, exist_ok=True)

        # Preserve model_index/config/scheduler/VAE/text assets from the official
        # base repository, then replace only the trainable diffusion transformer.
        for entry in os.listdir(source):
            src = os.path.join(source, entry)
            dst = os.path.join(destination, entry)
            if os.path.realpath(src) == destination:
                continue
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True, symlinks=False)
            else:
                shutil.copy2(src, dst)

        transformer_dir = os.path.join(destination, "transformer")
        os.makedirs(transformer_dir, exist_ok=True)
        # Remove stale sharded weights/index files before writing the canonical
        # single-file transformer understood by Mage's load_from_repo().
        for name in os.listdir(transformer_dir):
            if name.startswith("diffusion_pytorch_model") and name.endswith((".safetensors", ".json")):
                os.remove(os.path.join(transformer_dir, name))
        state = self._transformer_state(model, dtype)
        save_file(
            state,
            os.path.join(transformer_dir, "diffusion_pytorch_model.safetensors"),
            self._create_safetensors_header(model, state),
        )

    def _save_transformer(self, model: MageFlowModel, destination: str, dtype: torch.dtype | None):
        state = self._transformer_state(model, dtype)
        os.makedirs(Path(destination).parent.absolute(), exist_ok=True)
        save_file(state, destination, self._create_safetensors_header(model, state))

    def save(
            self,
            model: MageFlowModel,
            output_model_format: ModelFormat,
            output_model_destination: str,
            dtype: torch.dtype | None,
    ):
        match output_model_format:
            case ModelFormat.DIFFUSERS:
                self._save_repo(model, output_model_destination, dtype)
            case ModelFormat.ORIGINAL_TRANSFORMER:
                self._save_transformer(model, output_model_destination, dtype)
            case ModelFormat.INTERNAL:
                self._save_repo(model, output_model_destination, None)
            case _:
                raise NotImplementedError(f"Unsupported Mage output format: {output_model_format}")
