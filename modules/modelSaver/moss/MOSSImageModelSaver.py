import os.path
import shutil
import json
from pathlib import Path

from modules.model.MOSSImageModel import MOSSImageModel
from modules.module.moss import pipeline_moss_image
from modules.modelSaver.mixin.DtypeModelSaverMixin import DtypeModelSaverMixin
from modules.util.convert_util import convert
from modules.util.enum.ModelFormat import ModelFormat

import torch

from safetensors.torch import save_file


class MOSSImageModelSaver(
    DtypeModelSaverMixin,
):
    def __init__(self):
        super().__init__()

    def __save_diffusers(
            self,
            model: MOSSImageModel,
            destination: str,
            dtype: torch.dtype | None,
    ):
        # Copy the model to cpu by first moving the original model to cpu. This preserves some VRAM.
        pipeline = model.create_pipeline()
        pipeline.to("cpu")
        save_pipeline = self._copy_pipeline_to_dtype(pipeline, dtype, pipeline.tokenizer)

        os.makedirs(Path(destination).absolute(), exist_ok=True)
        # Preserve the original fp32 VAE instead of rounding its weights to output_dtype.
        save_pipeline.vae = model.vae
        # Save explicit components: Diffusers' library discovery does not know
        # OneTrainer's nested custom module package.
        save_pipeline.transformer.save_pretrained(os.path.join(destination, "transformer"))
        save_pipeline.scheduler.save_pretrained(os.path.join(destination, "scheduler"))
        save_pipeline.text_encoder.save_pretrained(os.path.join(destination, "text_encoder"))
        save_pipeline.tokenizer.save_pretrained(os.path.join(destination, "tokenizer"))
        model.vae.save_pretrained(os.path.join(destination, "vae"))
        shutil.copyfile(pipeline_moss_image.__file__, os.path.join(destination, "pipeline_moss_image.py"))
        shutil.copyfile(
            os.path.join(os.path.dirname(pipeline_moss_image.__file__), "LICENSE"),
            os.path.join(destination, "MOSS-LICENSE"),
        )
        # Portable custom-component names, independent of OneTrainer's package layout.
        index_path = os.path.join(destination, "model_index.json")
        index = {
            "_class_name": "MOSSImagePipeline",
            "text_encoder": ["transformers", "Qwen3_5Model"],
            "tokenizer": ["transformers", "Qwen2Tokenizer"],
            "vae": ["diffusers", "AutoencoderKLFlux2"],
        }
        index["transformer"] = ["pipeline_moss_image", "MOSSImageTransformer"]
        index["scheduler"] = ["pipeline_moss_image", "MOSSImageScheduler"]
        with open(index_path, "w", encoding="utf-8") as handle:
            json.dump(index, handle, indent=2)
            handle.write("\n")

        if dtype is not None:
            del save_pipeline

    def __save_safetensors(
            self,
            model: MOSSImageModel,
            destination: str,
            dtype: torch.dtype | None,
    ):
        state_dict = model.transformer.state_dict()
        state_dict = convert(state_dict, model.checkpoint_diffusers_to_original())

        save_state_dict = self._convert_state_dict_dtype(state_dict, dtype)
        self._convert_state_dict_to_contiguous(save_state_dict)

        os.makedirs(Path(destination).parent.absolute(), exist_ok=True)

        save_file(save_state_dict, destination, self._create_safetensors_header(model, save_state_dict))

    def __save_internal(
            self,
            model: MOSSImageModel,
            destination: str,
    ):
        self.__save_diffusers(model, destination, None)

    def save(
            self,
            model: MOSSImageModel,
            output_model_format: ModelFormat,
            output_model_destination: str,
            dtype: torch.dtype | None,
    ):
        match output_model_format:
            case ModelFormat.DIFFUSERS:
                self.__save_diffusers(model, output_model_destination, dtype)
            case ModelFormat.LEGACY_SAFETENSORS | ModelFormat.ORIGINAL_TRANSFORMER:
                self.__save_safetensors(model, output_model_destination, dtype)
            case ModelFormat.INTERNAL:
                self.__save_internal(model, output_model_destination)
            case _:
                raise NotImplementedError(f"Unsupported output format: {output_model_format}")
