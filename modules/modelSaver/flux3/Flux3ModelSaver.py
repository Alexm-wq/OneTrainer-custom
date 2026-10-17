import json
import os.path
from pathlib import Path

from modules.model.Flux3Model import Flux3Model
from modules.modelSaver.mixin.DtypeModelSaverMixin import DtypeModelSaverMixin
from modules.module.quantized.LinearA8 import LinearA8
from modules.module.quantized.mixin.QuantizedLinearMixin import QuantizedLinearMixin
from modules.util.convert_util import convert
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.quantization_util import get_unquantized_weight

import torch

from diffusers import Flux3Transformer2DModel
from transformers import Qwen3VLModel

from safetensors.torch import save_file


class Flux3ModelSaver(
    DtypeModelSaverMixin,
):
    def __init__(self):
        super().__init__()

    @staticmethod
    def _plain_state_dict(module, dtype=None):
        """Portable floating weights, including quantized frozen text encoders."""
        state = module.state_dict()
        for name, layer in module.named_modules():
            if isinstance(layer, (QuantizedLinearMixin, LinearA8)):
                prefix = name.replace(".checkpoint.", ".") + "."
                for key in [key for key in state if key.startswith(prefix)]:
                    del state[key]
                compute_dtype = dtype or getattr(layer, "compute_dtype", None) or torch.float32
                state[prefix + "weight"] = get_unquantized_weight(layer, compute_dtype, torch.device("cpu"))
                if layer.bias is not None:
                    state[prefix + "bias"] = layer.bias.detach()
        return {
            key: value.detach().to(device="cpu", dtype=dtype if value.is_floating_point() else None).contiguous()
            for key, value in state.items()
        }

    def __save_diffusers(
            self,
            model: Flux3Model,
            destination: str,
            dtype: torch.dtype | None,
    ):
        # Let conductors release their device allocations before exporting.
        model.to(torch.device("cpu"))
        with torch.device("meta"):
            transformer = Flux3Transformer2DModel.from_config(model.transformer.config)
            text_encoder = Qwen3VLModel(model.text_encoder.config)
        transformer.load_state_dict(self._plain_state_dict(model.transformer, dtype), strict=True, assign=True)
        text_encoder.load_state_dict(self._plain_state_dict(model.text_encoder, dtype), strict=True, assign=True)

        os.makedirs(Path(destination).absolute(), exist_ok=True)
        # Preserve the original fp32 VAE instead of rounding its weights to output_dtype.
        # Save explicit components: Diffusers' library discovery does not know
        # OneTrainer's nested custom module package.
        transformer.save_pretrained(os.path.join(destination, "transformer"))
        model.noise_scheduler.save_pretrained(os.path.join(destination, "scheduler"))
        text_encoder.save_pretrained(os.path.join(destination, "text_encoder"))
        model.processor.save_pretrained(os.path.join(destination, "text_encoder"))
        model.processor.save_pretrained(os.path.join(destination, "tokenizer"))
        model.vae.save_pretrained(os.path.join(destination, "vae"))
        # Components can be loaded independently by OT and Diffusers; the
        # standalone sampler does not require an optional prompt-upsample model.
        index = {
            "_class_name": "Flux3DistilledModularPipeline" if model.transformer.config.flow_map else "Flux3ModularPipeline",
            "transformer": ["diffusers", "Flux3Transformer2DModel"],
            "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
            "text_encoder": ["transformers", "Qwen3VLModel"],
            "tokenizer": ["transformers", "AutoProcessor"],
            "vae": ["diffusers", "AutoencoderKLFlux2"],
        }
        with open(os.path.join(destination, "modular_model_index.json"), "w", encoding="utf-8") as handle:
            json.dump(index, handle, indent=2)
            handle.write("\n")


    def __save_safetensors(
            self,
            model: Flux3Model,
            destination: str,
            dtype: torch.dtype | None,
    ):
        model.transformer_to(torch.device("cpu"))
        state_dict = self._plain_state_dict(model.transformer, dtype)
        state_dict = convert(state_dict, model.checkpoint_diffusers_to_original())

        save_state_dict = self._convert_state_dict_dtype(state_dict, dtype)
        self._convert_state_dict_to_contiguous(save_state_dict)

        os.makedirs(Path(destination).parent.absolute(), exist_ok=True)

        save_file(save_state_dict, destination, self._create_safetensors_header(model, save_state_dict))

    def __save_internal(
            self,
            model: Flux3Model,
            destination: str,
    ):
        self.__save_diffusers(model, destination, None)

    def save(
            self,
            model: Flux3Model,
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
