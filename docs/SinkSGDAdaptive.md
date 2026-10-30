# SINKSGD_ADAPTIVE (experimental)

SinkSGD_adv with an automatic learning rate and brakes against overfitting. Experimental: tested
on CPU, on a small overfitting toy problem and in MOSS Image LoRA runs.

## How it works

- **Base:** an extension of adv-optm's SinkSGD_adv. All of its options work as there:
  Normalization-then-Momentum, Nesterov (and its coefficient), SNR preconditioning, OrthoGrad,
  orthogonal Sinkhorn, spectral normalization, cautious, geometric and centered weight decay,
  state precisions (bf16/fp16/int8 with stochastic rounding, factored), stochastic rounding of
  bf16 weights, the compiled optimizer (here: the Sinkhorn normalization) and the fused back
  pass. With a fixed step size (no `d` growth) and the extras below off, a step is identical to
  SinkSGD_adv's; the tests check this for every option.
- **Direction (defaults):** the gradient is normalized by square-root Sinkhorn, averaged by
  momentum 0.9 and normalized again (`Normalization-then-Momentum` on, `Renormalize Momentum`
  on), as in the first version of this optimizer. With `Renormalize Momentum` off it behaves like
  SinkSGD_adv's NtM, where entries with a noisy momentum move less. 1D parameters (biases,
  DoRA scales) take the sign, as in SinkSGD_adv.
- **Automatic step size:** the step is `learning rate × d`. `d` estimates the distance to the
  solution like Prodigy, from how well the gradients point back to the start. Unlike Prodigy, `d`
  falls again once progress stalls (`D Shrink`), down to `D Min Ratio` of its maximum: a learning
  rate decay driven by the training itself. Leave the learning rate at 1.0, as with Prodigy.
  `D Coefficient` is 1.0, as in Prodigy: on a MOSS Image LoRA (batch 1), 0.1 kept the step
  near 1e-6 and nothing was learned, while 1.0 with the validation brake off trained well.
  `D Floor` (1e-5) is the lowest step size. `Initial D` (1e-5) is the start: with noisy
  diffusion gradients the estimate rises slowly, often over a few hundred steps. To start with
  large steps that then decay, set `Initial D` to a learning rate that learns fast with plain
  SinkSGD (e.g. 5e-4), keep `D Floor` low (1e-5) and lower `D Shrink` to 0.001–0.002: D then
  approaches the estimate over roughly 1/`D Shrink` steps (0.01 takes only a few hundred).
  TensorBoard's `lr/...` shows the effective rate, `sinksgd_adaptive/d/...` the estimate.
- **Agreement mask:** memorizing single images produces gradients that differ from image to
  image; the features of a concept point the same way across images. Update entries on which
  the recent gradients disagree are damped (`Agreement Threshold`). A layer that only sees
  noise moves less as a whole. Keep the momentum horizon (`1 / (1 - momentum)` steps) well
  below one epoch, otherwise the same image seen again looks like agreement.
- **Validation brake** (off by default; in the MOSS LoRA run above it held the learning rate
  down before anything was learned, so leave it off for now): at every validation, the summed gradient of a few
  validation images is compared with the training direction. `Validation Brake Images` (8)
  are spread evenly over the validation concepts, so one concept cannot decide alone. While
  training no longer helps held-out images, the step is damped in proportion; `d` cannot grow
  while the alignment is zero or negative. It acts
  on the whole learning rate, not per concept. Needs validation concepts and costs one extra
  backward pass per brake image; validate often enough, e.g. every 50–100 steps.
  `Validation Alignment Ref` is the cosine that counts as fully helpful.
- **T-LoRA Aware** (on by default): T-LoRA leaves the LoRA ranks of other noise levels with
  exactly zero gradient. Those ranks are skipped: their momentum, agreement statistics and
  weights stay as they are, and the Sinkhorn normalization covers only the active ranks. With
  the MOSS defaults most ranks are inactive in most steps; without this, their momentum would
  decay towards zero between their rare steps, and the zero entries would distort the
  normalization of the active ones. Recognized from the LoRA up/down tags, so it needs no setting
  in the T-LoRA options. Without T-LoRA every rank has a gradient and nothing changes. Spectral
  normalization still measures the whole matrix.
- **Split Fused Layers** (off by default): normalizes fused weights such as MOSS Image's FFN
  up+gate per part, as for SinkSGD_adv; see the MOSS Image docs.
- **Drift budget** (off by default): the step falls linearly with the RMS distance of the
  trained weights from their start and reaches zero at `Drift Budget`.

## Toy results

Small MLP, 48 training samples with noisy labels, batch 1, 6000 steps, mean of 3 seeds. Lower
is better; the best plain SinkSGD learning rate was found by a sweep.

| Optimizer | best validation loss | final validation loss |
|---|---|---|
| SinkSGD, best of 3 learning rates | 3.02 | 4.11 |
| SINKSGD_ADAPTIVE, no brake | 3.12 | 4.07 |
| SINKSGD_ADAPTIVE, validation brake | 3.03 | 3.07 |

Without the brake, the automatic learning rate reaches the tuned one, and the mask helps a
little, but training still memorizes noise towards the end. With the brake, training ends at
its best validation loss. Whether this carries over to LoRA training needs an A/B run.

## Limits

- Fused back pass: `d` is estimated over all parameters of a step. With the fused back pass the
  parameters are updated during the backward pass, so they use the `d` of the previous step.
- Memory: momentum and agreement statistics in the selected state precision (bf16 with
  stochastic rounding by default; with factored momentum the agreement statistics stay bf16),
  and the distance estimate on every 11th entry only (`slice_p`, as in Prodigy). For MOSS LoRA
  rank 32 on all attention and MLP projections that is 0.33 GiB, against 0.28 GiB for
  SinkSGD_adv with momentum. Full fine-tuning needs about 4 bytes per parameter on top, less
  with the agreement mask off.
- T-LoRA Aware reads one value per LoRA matrix back from the GPU per step.
- Backups of the first version (before it was based on SinkSGD_adv) resume with `d` but start
  momentum anew; the distance estimate restarts from the current weights.
