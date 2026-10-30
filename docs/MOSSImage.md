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

## Self-Flow

The Self-Flow section of the training tab enables dual-timestep training from Black
Forest Labs' Self-Flow (self-supervised flow matching). It is the same method as on the
FLUX.2 and Mage-Flow branches, adapted to MOSS's single-stream DiT. The
**moss-image Self-Flow LoRA** preset enables it on top of the LoRA preset, with EMA decay
0.999: the paper's 0.9999 suits long pretraining runs, while the teacher of a LoRA run of
a few thousand steps would barely move from the starting weights.

- Each step samples two timesteps per image. `Token Mask Ratio` of the 2x2 patch tokens
  get the second one, the rest the first; text tokens keep the first. The DiT conditions
  every token on its own timestep through the shared block and output-head modulation.
  The velocity target is unchanged.
- An EMA teacher sees all tokens at the cleaner of the two timesteps. Its forward stops
  after `Teacher Block`. The student's output after `Student Block` passes through a
  small trained projector and is aligned to the teacher feature by cosine loss
  (`Representation Weight`). `Structural Alignment` also matches the pairwise token
  similarities of `Structural Tokens` tokens spread over the image. `-1` selects blocks
  14 and 34 of 42.
- The EMA teacher is kept in fp32 in CPU RAM. For the teacher forward, the trained
  parameters point at a GPU copy of their EMA values and back afterwards; the student
  weights are never copied. The EMA update after each optimizer step runs on the GPU.
  With LoRA the EMA covers the LoRA weights and goes to the GPU as a whole (page-locked
  memory, a few hundred MB at rank 32). With full fine-tuning it covers all trained DiT
  weights, about 4 bytes of RAM per parameter. Each block's EMA is then moved to the GPU
  just for that block, which costs one host-to-GPU transfer of the whole fp32 EMA per
  step. Full fine-tuning with Self-Flow does not support layer offloading; activation
  offloading works.
- Expected cost: each step adds one teacher forward up to the teacher block (34 of 42
  by default), without gradients. With gradient checkpointing that is roughly 20% more
  compute than an ordinary step, and with layer offloading roughly 40% more transfers.
  A lower `Teacher Block` makes the teacher cheaper. `Offload Teacher Target` keeps the
  teacher feature in RAM while the student runs.
- Sigma loss weighting weights every token by its own sigma. The identity reward
  reconstructs `x0` with the per-token sigma as well.
- Self-Flow applies to ordinary supervised rows. DPO preference scores (policy,
  reference and streamed replay) and validation keep the native single-timestep
  forward, so preference likelihoods and validation losses stay comparable. Prior
  prediction concepts, masked prior preservation and multi-GPU are not supported
  with Self-Flow.
- The projector and EMA teacher are saved in backups (`self_flow/self_flow.pt`) and
  restored on resume. Exported models contain only the trained weights. The sample
  settings show a `Self-Flow EMA` switch to sample with the EMA teacher during training.
- TensorBoard: `loss/self_flow_rep`, `loss/self_flow_structural`, `loss/generation`,
  `self_flow/cosine_similarity`, the mean timesteps and `performance/self_flow_teacher_ms`.

## T-LoRA

**T-LoRA** in the LoRA tab trains timestep-dependent LoRA ranks, following T-LoRA
(Soboleva et al., AAAI 2026; official code `ControlGenAI/T-LoRA`). High noise levels
overfit most, so the noisier the input, the fewer leading LoRA ranks are active:
`rank(t) = floor((1000 - t) * (rank - min) / 1000) + min` for MOSS's raw time t = 1..1000,
from almost the full rank at the cleanest timestep down to **T-LoRA Min Rank** (1, as
upstream) at pure noise. Each sample of a batch gets its own mask.

- The mask sits between LoRA down and up, so masked ranks get no gradient at that step.
  It applies to training, validation, the RLHF paths and OneTrainer's samples, with
  compiled blocks too. Self-Flow's student uses, per sample, the mean noise level of
  its image tokens; its teacher uses the cleaner timestep.
- An exported LoRA is a normal LoRA without the mask. Other tools apply all ranks at
  every timestep, which differs from OneTrainer's samples. For ComfyUI, the node in
  `comfyui/onetrainer_tlora` applies the mask per sampling step; OneTrainer stores T-LoRA
  Min Rank in the LoRA file for it. The official repository likewise needs its own
  pipeline hook at inference.
- Plain LoRA only: DoRA, LoHa, LoKr and OFT are rejected, as is any model other than
  MOSS Image. On SVDQuant layers only the LoRA ranks are masked, not the SVD residual.
- **T-LoRA Gradient Only** keeps the masked ranks in the forward and cuts only their gradient.
  What a rank learned at other noise levels stays in the output, so the mask can be faded in
  during a run: no mask before **T-LoRA Start**, the full mask (T-LoRA Min Rank) from
  **T-LoRA End** on, linear in between (global steps; both 0 masks from the first step).
  Several concepts in one LoRA can then use every rank at high noise while they are learned,
  before memorization of single layouts is cut off. The exported LoRA is a plain LoRA that
  needs no mask at inference, in ComfyUI's normal loader as in OneTrainer's samples. The upstream SDXL variant's orthogonal initialization
  (Ortho-LoRA) is not included; the official FLUX release uses the timestep masking alone.

## Split Fused Layers (SinkSGD_adv, SINKSGD_ADAPTIVE)

The SinkSGD_adv and SINKSGD_ADAPTIVE option **Split Fused Layers** normalizes layers that hold several
projections in one matrix per part instead of as a whole: FFN up+gate of the 42 main
blocks (2 parts), the text-refiner QKV (3), the shared block modulation (6) and the
video head modulation (2). Muon on DiTs converged much better with this split in CMuon
(Chen, Sun, Yuan, 2026); for SinkSGD it is untested on real training, so compare it in
an A/B run. It applies to the fused weights in full fine-tuning and to their LoRA up
matrices (LoRA, DoRA, LoHa); LoRA down is not fused. In SINKSGD_ADAPTIVE both Sinkhorn
normalizations (gradient and momentum) are split; the agreement mask and the distance
estimate are per entry or over the whole group anyway. In SinkSGD_adv, SNR
preconditioning and geometric weight decay keep their statistics over the whole matrix. Enabling it when no trained
parameter belongs to a fused layer is an error.

## Identity reward

For character LoRAs of real faces, the Identity Reward section of the training tab
adds an ArcFace loss next to the diffusion loss. It follows ID-Aligner: the face in
the one-step clean prediction `x0 = x_t - sigma * v` should have the same ArcFace
embedding as the face in the training image.

- Landmarks: OpenCV's YuNet detector finds the largest face once per training crop,
  after crop, flip and rotation. With image caching, the landmarks are cached with
  the latents; turning the reward on re-caches the images once.
- Loss: the face region of `x0` is decoded by the VAE, aligned to the 112x112 ArcFace
  template and compared with the same crop decoded from the training latent. Each
  face is decoded, embedded and differentiated on its own; only its latent gradient
  is kept for the shared backward pass, so one face crop is in VRAM at a time.
- Which samples: every `Identity Interval` steps, rows with noise level at most
  `Identity Max Sigma` (blurrier predictions say nothing about identity), a detected
  face, a caption that was not dropped, and a caption containing one of the
  comma-separated `Identity Trigger` words. Leave the trigger empty to score every
  face. Prior-prediction concepts, validation and DPO pairs are never scored. In
  mixed DPO training the reward applies only to the ordinary supervised rows.
- Faces whose decoded crop exceeds `Identity Max Crop` pixels are skipped. VRAM grows
  with the square of that size; bf16 training decodes the face in bf16.
- TensorBoard shows `loss/identity` (mean `1 - cosine similarity`) and
  `identity/faces`.

Both model fields take a local file, a Hugging Face file link
(`https://huggingface.co/<repo>/resolve/<revision>/<file>`) or another download URL;
downloads happen once and are cached. The face detector defaults to
`face_detection_yunet_2023mar.onnx` from the OpenCV model zoo, pinned to a commit and
checked by its SHA256. ArcFace defaults to Arc2Face's `arcface.onnx` (an R100 trained
on WebFace42M). Other insightface ONNX recognition models (antelopev2 `glintr100.onnx`,
buffalo_l `w600k_r50.onnx`) and `arcface_torch` `backbone.pth` files work as well; ONNX
models run through a small built-in interpreter, so no ONNX runtime is involved and the
loss stays differentiable. InsightFace's pretrained recognition models are licensed for
non-commercial research only.

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
