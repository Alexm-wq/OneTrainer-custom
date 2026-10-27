"""ComfyUI node that applies a OneTrainer T-LoRA with its timestep-dependent ranks (MOSS Image).

ComfyUI's LoRA loaders merge every rank into the weights, so a T-LoRA trained in OneTrainer would
use all ranks at every noise level. This loader runs the LoRA beside each layer instead and keeps
only the ranks OneTrainer trained for the current noise level, like OneTrainer's own samples.
"""
import logging

import comfy.lora
import comfy.lora_convert
import comfy.model_management
import comfy.utils
import comfy.weight_adapter
import folder_paths
from comfy.patcher_extension import PatcherInjection, WrappersMP

from .tlora_core import MaskedLoRA, Schedule, file_min_rank


class OneTrainerTLoRALoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "lora_name": (folder_paths.get_filename_list("loras"),),
            "strength_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
            "min_rank": ("INT", {
                "default": 0, "min": 0, "max": 4096,
                "tooltip": "LoRA ranks used at pure noise. 0 reads the value OneTrainer stored in the file, else 1.",
            }),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "loaders"
    DESCRIPTION = "Applies a OneTrainer T-LoRA with fewer ranks the noisier the input, as during training."

    def load(self, model, lora_name, strength_model, min_rank):
        lora, metadata = comfy.utils.load_torch_file(
            folder_paths.get_full_path_or_raise("loras", lora_name), safe_load=True, return_metadata=True,
        )
        if min_rank == 0:
            min_rank = file_min_rank(metadata)
            if min_rank is None:
                logging.warning("OneTrainer T-LoRA: no T-LoRA min rank in the file, using 1.")
                min_rank = 1

        key_map = comfy.lora.model_lora_keys_unet(model.model, {})
        loaded = comfy.lora.load_lora(comfy.lora_convert.convert_lora(lora), key_map)
        if not loaded:
            raise ValueError(f"OneTrainer T-LoRA: no layer of this model matches {lora_name}.")

        patched = model.clone()
        schedule = Schedule(min_rank)
        hooks = {}
        for key, adapter in loaded.items():
            if not isinstance(adapter, comfy.weight_adapter.LoRAAdapter) or any(w is not None for w in adapter.weights[3:5]):
                raise ValueError(f"OneTrainer T-LoRA: {key} is not a plain LoRA (DoRA, LoHa, LoKr and diffs are not supported).")
            up, down, alpha = adapter.weights[:3]
            rank = down.shape[0]
            if min_rank > rank:
                raise ValueError(f"OneTrainer T-LoRA: min rank {min_rank} exceeds the LoRA rank {rank}.")
            weight_key, (dim, start, size) = key if isinstance(key, tuple) else (key, (0, None, None))
            if dim != 0:
                raise ValueError(f"OneTrainer T-LoRA: {weight_key} is split along dimension {dim}.")
            module_name = weight_key.removesuffix(".weight")
            if module_name not in hooks:
                hooks[module_name] = MaskedLoRA(patched.get_model_object(module_name), schedule)
            scale = (alpha / rank if alpha is not None else 1.0) * strength_model
            hooks[module_name].add(up.flatten(1), down.flatten(1), scale, start, size)

        def inject(model_patcher):
            device = comfy.model_management.get_torch_device()
            for hook in hooks.values():
                hook.inject(device)

        def eject(model_patcher):
            for hook in hooks.values():
                hook.eject()

        patched.set_injections(f"onetrainer_tlora_{id(schedule)}", [PatcherInjection(inject=inject, eject=eject)])
        patched.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, f"onetrainer_tlora_{id(schedule)}", schedule.wrapper)
        logging.info(f"OneTrainer T-LoRA: {len(hooks)} layers, min rank {min_rank}.")
        return (patched,)


NODE_CLASS_MAPPINGS = {"OneTrainerTLoRALoader": OneTrainerTLoRALoader}
NODE_DISPLAY_NAME_MAPPINGS = {"OneTrainerTLoRALoader": "Load T-LoRA (OneTrainer)"}
