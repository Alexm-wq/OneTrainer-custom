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
- Presets sample logit-normal timesteps with noising bias 1.09 and weight 0 (standard
  deviation 1), close to the final pretraining interval (sigma = sigmoid(N(1.09, 0.95)),
  median about 0.75). Its 0.95 cutoff and 1% uniform mixture are not reproduced.
- The mixed DPO preset uses Normal=1, DPO=1, Chosen=0.1 and learning rate 1e-5.
  These are starting values, not a measured optimum for MOSS.
- INT8, NF4, FP8 weights, W8A8/A8 and SVDQuant use OneTrainer's shared quantizers.
  Transformer quantization has its own layer filter, independent of the training
  filter. The frozen text encoder can also use INT8/NF4/FP8.
- Gradient checkpointing, transformer layer/activation offloading, frozen text
  encoder layer offloading and block compilation are connected. Blocks now run
  once per batch; attention still uses each prompt's actual length. Offloading
  trained layers requires checkpointing. Full fine-tuning additionally retains
  the shared optimizer/fused-backward requirements.
- Attention can use SDPA (default), Flash Attention or PyTorch cuDNN. The latter
  two require a compatible GPU/runtime; their availability is not implied by a
  CPU test. BF16 is the default compute dtype; FP16, FP32 and TF32 are available.
- Additional filters: `attention`, `mlp`, `text-refiner`, and `full`.
  INT8/NF4/FP8 LoRA presets quantize the main DiT blocks and the text encoder.
  The INT8 Offload preset starts with 50% layer offload and activation offload;
  adjust to available VRAM and host RAM. These are starting configurations, not
  measured capacity guarantees. The full fine-tune preset uses floating weights.
- The sequence length preset is 512 to reduce cache size; increase it for long
  layout prompts. Sampling uses the training length when invoked during training.

## Sampling

The default is 1024, Euler 50 steps, CFG 3, shift 5, with main DiT layer 22 skipped
only in the negative CFG branch. **negative skip layers** accepts comma-separated
zero-based indices; clear it to disable skipping. Noise, CFG, Euler updates and VAE
decoding stay FP32. The scheduler follows the upstream nearest-training-grid rule.

## Checkpoints

Use the full HF repository or a Diffusers directory as the base model. A local or Hugging Face URL
transformer-only safetensors checkpoint or transformer directory/repository can
be selected as the Transformer override. Single-file overrides must contain the
native MOSS transformer keys; uniform `model.diffusion_model.`, `diffusion_model.`
or `transformer.` prefixes are accepted. GGUF is not a supported MOSS format. VAE overrides must be compatible
FLUX.2 VAE directories/repositories. Keep the supplied small decoder as the default.

LoRA output supports Diffusers, Original, Comfy and Kohya key formats. Comfy keys
match the official MOSS custom node model layout. This does not imply every
third-party LoRA loader supports MOSS. Full-model directory exports include the
custom pipeline and its Apache-2.0 license; transformer-only exports are supported.
For external Diffusers loading, select the exported `pipeline_moss_image.py` as the
custom pipeline. OneTrainer loads its own known implementation. Quantized components
are dequantized for portable full exports; full DiT training requires floating weights.

## Validation

The feature workflow runs only the new quantization, offloading, backend/compilation,
export and UI checks with randomly initialized miniature components. It covers
real CPU INT8/NF4/FP8 LoRA forward/backward, quantization filters, INT8 device-state
bookkeeping, Qwen encoding, portable export/reload, variable prompt lengths and
checkpoint recomputation. Both actual UI backends build the new controls/presets.
CUDA layer/activation transfer checks are included but skipped on the CPU runner.
No pretrained weights are downloaded. A full GPU training run, CUDA attention
kernels and visual quality with the 5B checkpoint remain unverified.

Source: https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base
Upstream code and ComfyUI integration: https://github.com/OpenMOSS/MOSS-Image
