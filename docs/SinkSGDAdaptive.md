# SINKSGD_ADAPTIVE (experimental)

SinkSGD with an automatic learning rate and brakes against overfitting. A prototype: it is
tested on CPU and on a small overfitting toy problem, not yet on real diffusion training.

## How it works

- **Direction:** like adv-optm's SinkSGD. The gradient is normalized by square-root Sinkhorn
  (alternating row and column L2 normalization), averaged by momentum and normalized again, so
  every entry moves by about the same amount and only the learning rate sets the step size.
- **Automatic step size:** the step is `learning rate × d`. `d` estimates the distance to the
  solution like Prodigy, from how well the gradients point back to the start. Unlike Prodigy, `d`
  falls again once progress stalls (`D Shrink`), down to `D Min Ratio` of its maximum: a learning
  rate decay driven by the training itself. Leave the learning rate at 1.0, as with Prodigy.
  `Initial D` (1e-5) is also the lowest step size. With noisy diffusion gradients the estimate
  rises slowly, often over a few hundred steps; a higher `Initial D` near the lower end of the
  learning rates that work with plain SinkSGD shortens that ramp.
  TensorBoard's `lr/...` shows the effective rate, `sinksgd_adaptive/d/...` the estimate.
- **Agreement mask:** memorizing single images produces gradients that differ from image to
  image; the features of a concept point the same way across images. Update entries on which
  the recent gradients disagree are damped (`Agreement Threshold`). A layer that only sees
  noise moves less as a whole. Keep the momentum horizon (`1 / (1 - momentum)` steps) well
  below one epoch, otherwise the same image seen again looks like agreement.
- **Validation brake** (off by default): at every validation, the summed gradient of a few
  validation images is compared with the training direction. `Validation Brake Images` (8)
  are spread evenly over the validation concepts, so one concept cannot decide alone. While
  training no longer helps held-out images, the step is damped in proportion; `d` cannot grow
  while the alignment is zero or negative. It acts
  on the whole learning rate, not per concept. Needs validation concepts and costs one extra
  backward pass per brake image; validate often enough, e.g. every 50–100 steps.
  `Validation Alignment Ref` is the cosine that counts as fully helpful.
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

- No fused back pass: `d` is estimated over all parameters of a group before any is updated.
  Full fine-tuning with layer offloading, which needs a fused back pass, is therefore not
  possible.
- Memory: momentum and agreement in bf16 with stochastic rounding (`state_precision` bf16_sr,
  or fp32), and the distance estimate on every 11th entry only (`slice_p`, as in Prodigy). For
  MOSS LoRA rank 32 on all attention and MLP projections that is 0.33 GiB, against 0.28 GiB for
  SinkSGD_adv with momentum. Full fine-tuning needs about 4 bytes per parameter on top.
