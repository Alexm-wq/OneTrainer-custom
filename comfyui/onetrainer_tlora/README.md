# OneTrainer T-LoRA loader for ComfyUI

A LoRA trained in OneTrainer with **T-LoRA** uses fewer ranks the noisier the input. ComfyUI's
normal LoRA loaders merge all ranks into the weights, so at high noise they also apply ranks that
were only trained on low noise. This node applies the LoRA the way OneTrainer trains and samples it.

## Install

Copy this folder (`onetrainer_tlora`) into `ComfyUI/custom_nodes/` and restart ComfyUI. It needs
no extra packages.

## Use

Replace the LoRA loader with **Load T-LoRA (OneTrainer)** (category `loaders`):

- `model`: the MOSS Image model from ComfyUI's native loaders.
- `lora_name`: the LoRA from `models/loras`. Save it from OneTrainer in the **Comfy** LoRA format;
  the Kohya and Original formats work as well.
- `strength_model`: as in ComfyUI's LoRA loader.
- `min_rank`: `0` reads T-LoRA Min Rank from the file (OneTrainer writes it when T-LoRA is on);
  otherwise the given value. A LoRA trained with **T-LoRA Gradient Only** is a plain LoRA: use
  ComfyUI's normal LoRA loader for it; at `0` this node refuses such a file.

At each sampling step the node takes the step's noise level sigma and uses the first
`floor((1 - sigma) * (rank - min_rank)) + min_rank` ranks, as OneTrainer does.

## How it works and limits

- The LoRA runs beside each layer at every forward instead of being merged, which costs one small
  low-rank matmul per layer (a few percent with rank 32). It works with quantized base weights,
  since the base weights are not touched.
- Plain LoRA only, no DoRA/LoHa/LoKr. Text-encoder LoRA weights are not applied by this node.
- Reference latents (edit models) use the noise level of the image they belong to.
- Tested against OneTrainer's training code with a miniature MOSS model on the CPU (same output at
  every noise level). The ComfyUI integration itself (model patcher, sampler hand-over, ComfyUI's
  `TorchCompileModel`) has not been run in ComfyUI yet.
