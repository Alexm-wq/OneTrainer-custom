# FLUX.3 Image Klein

Select **FLUX.3 Image Klein** in the model picker. Load a preset from **FLUX.3 Klein**.
Use `black-forest-labs/flux-3-image-klein-base` as Base Model for training.
Accept the model's Hugging Face license and supply your access token if required.

## Presets

The default LoRA preset uses BF16, rank/alpha 32, LR 5e-5, 1024 resolution,
batch 1 with accumulation 4, a frozen Qwen3-VL encoder, and image/text caching.
Separate FP8, INT8, NF4, INT8 with offloading, full fine-tune and mixed DPO presets
are included. These are starting points, not measured optimal hyperparameters.

Mixed DPO exposes independent normal/DPO/chosen weights (defaults 1/1/0.1).
Layer filters cover attention/MLP, attention only, MLP only, image blocks,
text refinement and the full transformer. The default covers image/text mode
blocks and joint blocks; unused reference-only blocks receive no T2I gradient.

## Conditioning and sampling

Training follows the Diffusers FLUX.3 LoRA reference's shifted logit-normal
distribution (shift 6.93). Sampling follows the BFL implementation: 50 Euler
steps, CFG 4, alpha 16, fp32 integration and VAE, and bf16-rounded CPU noise.
Sampling alpha is separately adjustable. These two shifts serve different purposes.

Qwen3-VL-4B hidden layers 4/8/12/16/20/24/28/32 are concatenated. Each prompt is
right-padded to a multiple of 80, capped at the configured sequence length
(default 512 in presets; maximum 4096). Padding positions use position id 1.
The DiT attends to those padding embeddings. Only unused cache tail values
are removed; heterogeneous prompt windows are forwarded separately to preserve
conditioning. Image positional coordinates are scaled by 0.5.

The sampler also accepts one optional reference image. The distilled
`black-forest-labs/flux-3-image-klein` checkpoint can be sampled with CFG 1 and
16 steps, using its target-timestep input. Standard LoRA/full training uses
the Base checkpoint: a distilled flow-map needs a different training objective.

BFL's optional 9B prompt upsampler is separate from the encoder. This integration
uses your captions/prompts as entered, without automatic rewriting or an extra
9B model. For reference-style generation quality, enter an expanded prompt.

## Features and formats

- LoRA, DoRA and other existing PEFT methods; full DiT fine-tuning.
- Masked loss, validation, mixed RLHF and resume through the shared trainer.
- Multi-resolution image cache, text cache, caption dropout and CEP.
- Transformer and frozen-encoder INT8/NF4/FP8 quantization; transformer W8A8/SVD options.
- SDPA, Flash Attention and cuDNN through Diffusers' backend selection.
- Gradient checkpointing, block compilation, layer/activation offloading.
  Reference-only blocks are checkpointed but excluded from the T2I offload
  schedule because they do not execute during T2I training.
- Original or Diffusers transformer safetensors overrides, local or HF URL.
- Diffusers/component full export and internal backup; original BFL transformer
  export; Diffusers, Original, Comfy and Kohya adapter namespaces.
  External loaders must support FLUX.3 and the selected PEFT algorithm.
- Model, training and sampling controls in both CTK and PySide6 UIs.

The encoder stays frozen, as in the training reference. Native BFL rotated-FP8,
NVFP4 packages and GGUF are not the same format as OneTrainer's on-load
quantization. Use the floating-point checkpoint as the source for OT quantization.
VAE weights and execution remain FP32.

## Validation scope

Targeted tests use small randomly initialized real transformers, Qwen3-VL and
FLUX.2 VAE components. They compare velocities and gradients against pinned
BFL code, cache conditioning, quantized PEFT backward passes, mixed losses,
serialization, sampling and both UIs. No pretrained weights are downloaded by
CI. CUDA kernel performance and full-size image quality still need a GPU run.

References:
- [BFL reference](https://github.com/black-forest-labs/flux-3-image-klein/tree/bfc8889a4497f846416f253a3972a1941ac698b9)
- [Base checkpoint](https://huggingface.co/black-forest-labs/flux-3-image-klein-base)
- [Diffusers training reference](https://github.com/huggingface/diffusers/blob/b800f828f3951c4d206e154411bcbfa693cd322e/examples/dreambooth/train_dreambooth_lora_flux3_klein.py)
