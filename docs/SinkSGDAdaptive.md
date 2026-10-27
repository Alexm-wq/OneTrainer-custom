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
  TensorBoard's `lr/...` shows the effective rate, `sinksgd_adaptive/d/...` the estimate.
- **Agreement mask:** memorizing single images produces gradients that differ from image to
  image; the features of a concept point the same way across images. Update entries on which
  the recent gradients disagree are damped (`Agreement Threshold`). A layer that only sees
  noise moves less as a whole. Keep the momentum horizon (`1 / (1 - momentum)` steps) well
  below one epoch, otherwise the same image seen again looks like agreement.
- **Validation brake** (off by default): at every validation, the gradient of one validation
  batch is compared with the training direction. While training no longer helps held-out
  images, the step is damped and `d` cannot grow. Needs validation concepts and costs one
  forward and backward pass per validation; validate often enough, e.g. every 50–100 steps.
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
- Per parameter, a copy of the start weights plus three float32 states (gradient sum, momentum,
  agreement); fine for LoRA, large for full fine-tuning.
