# MOSS Image

Select **MOSS Image** and load the **moss-image LoRA** preset. Base model:
`OpenMOSS-Team/MOSS-Image-Base`. Downloads occur on the machine running OneTrainer.
The bundled integration uses the upstream model implementation from Hugging Face revision
`1ceede80badddd6a7812069619edaa5633587192`; it does not execute arbitrary remote Python.

## Training

- LoRA and full DiT fine-tuning; Qwen3.5 stays frozen.
- Start with BF16 DiT/text weights, FP32 VAE, rank/alpha 32, batch 1,
  accumulation 4, learning rate 5e-5 and 1024 resolution.
- The `attn-mlp` filter selects attention and feed-forward projections in all
  42 main DiT blocks (`layers.`). `full` also includes text refiners, input/output
  projections and timestep modulation.
- Image/text disk caching and multi-resolution caching use the shared OneTrainer
  path, including dataset-change invalidation. Cached latents are raw VAE
  posterior means. Patchification and batch-normalization scaling happen at training time.
- Prompts use the official chat template with thinking enabled. Padding is removed
  before each Qwen3.5 forward; the returned mask is preserved through caching.
- The training time is 1..1000, matching sigma exactly. The fixed training shift is
  1 by default; inference shift 5 is a separate upstream scheduler setting.
- The mixed DPO preset uses Normal=1, DPO=1, Chosen=0.1 and learning rate 1e-5.
  These are starting values, not a measured optimum for MOSS.
- Gradient checkpointing is supported. Quantized weights and layer/activation
  offloading are not yet supported: upstream casts inputs to weight dtype and
  executes variable-length samples separately. Unsupported settings fail explicitly.
  Whole-component moves between CPU and GPU during caching/sampling remain available.
- The sequence length preset is 512 to reduce cache size; increase it for long
  layout prompts. Sampling uses the training length when invoked during training.

## Sampling

The default is 1024, Euler 50 steps, CFG 3, shift 5, with main DiT layer 22 skipped
only in the negative CFG branch. **negative skip layers** accepts comma-separated
zero-based indices; clear it to disable skipping. Noise, CFG, Euler updates and VAE
decoding stay FP32. The scheduler follows the upstream nearest-training-grid rule.

## Checkpoints

Use the full HF repository or a Diffusers directory as the base model. A local
transformer-only safetensors checkpoint or transformer directory/repository can
be selected as the Transformer override. Single-file overrides must contain the
native MOSS transformer keys without a prefix. VAE overrides must be compatible
FLUX.2 VAE directories/repositories. Keep the supplied small decoder as the default.

LoRA output supports Diffusers, Original, Comfy and Kohya key formats. Comfy keys
match the official MOSS custom node model layout. This does not imply every
third-party LoRA loader supports MOSS. Full-model directory exports include the
custom pipeline and its Apache-2.0 license; transformer-only exports are supported.
For external Diffusers loading, select the exported `pipeline_moss_image.py` as the
custom pipeline. OneTrainer loads its own known implementation.

## Validation

GitHub Actions runs CPU tests with randomly initialized miniature MOSS transformers:
training/backward, gradient checkpointing, mixed RLHF gradients, LoRA format
roundtrips, strict transformer override loading, text padding/dropout, latent
normalization and exact sampling parity with the bundled official pipeline.
The existing multi-resolution, Krea checkpoint and mixed RLHF suites also run.
These tests do not download model weights and do not replace a full GPU
training run or visual quality assessment with the 5B checkpoint.

Source: https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base
Upstream code and ComfyUI integration: https://github.com/OpenMOSS/MOSS-Image
