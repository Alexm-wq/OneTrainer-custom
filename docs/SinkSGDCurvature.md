# SINKSGD_CURVATURE (experimental)

SinkSGD whose step size is controlled by the measured slope and curvature of the loss, instead of
being set by hand. A prototype: tested on CPU on noisy toy problems and with a miniature MOSS
model, not yet on real diffusion training.

## How it works

- **Direction:** as SINKSGD_ADAPTIVE, so every SinkSGD_adv option works, as do T-LoRA Aware and
  Split Fused Layers. Defaults: Normalization-then-Momentum with Nesterov momentum 0.95 and the
  momentum normalized once more, so every step has the same size and the controller alone
  decides how large it is. The Agreement Threshold is off by default.
- **Slope:** after each step, the next batch's gradient along that step shows whether steps stop
  short (still downhill) or overshoot (uphill again). That batch is independent of the noise in
  the step, so its average is the true slope.
- **Curvature:** every `Curvature Interval` steps (10) the trainer computes the gradient of the
  same batch once more at the new weights, with the same noise and timestep. Its change along
  the step is the curvature, which says how much slope one step removes.
- **Control:** the step size grows while the remaining slope exceeds `Slope Target` (0.25 steps'
  worth) and shrinks below it, by at most `Controller Gain` (1 % in log space) per step. With
  momentum, each step keeps acting for about 20 steps; a direct Newton step on the step size
  therefore oscillated, and a slow controller with fast measurements (slope averaged over about
  20 steps, curvature over 5 measurements) did not. While the measured curvature is negative,
  as at the saddle a LoRA starts in (up = 0), the step size follows the sign of the slope at the
  same rate.

While the loss falls steeply the step size rises; once the remaining slope is gradient noise, it
falls on its own. Leave the learning rate at 1.0; it multiplies the step. TensorBoard shows the
step size as `lr/...` and `sinksgd_curvature/step`, `/slope` and `/curvature`.

## Settings

| Setting | Default | Meaning |
|---|---|---|
| Initial D | 1e-5 | step size until 3 curvature measurements exist |
| D Floor / D Ceiling | 1e-6 / 1e-2 | limits; keep the ceiling well above where it settles, a ceiling it keeps hitting distorts the control |
| Growth Rate / Decay Rate | 1.1 / 0.95 | outer limits per step; the controller itself changes by at most 1 % |
| Controller Gain | 0.01 | how fast the step size follows the slope |
| Slope Target | 0.25 | 0 steps to the bottom of the local parabola |
| Curvature Interval | 10 | one extra forward and backward pass per measurement: about 10 % more time |

## Toy results

Batch 1 to 4 with label noise, 3000 steps, validation loss averaged over seeds, the same SinkSGD
direction with fixed step sizes for comparison:

| Problem | SINKSGD_CURVATURE | best fixed step | worst fixed step tried |
|---|---|---|---|
| MLP | 0.85 | 0.84 (1e-3) | 3.62 (1e-2) |
| Linear regression | 1.36 | 1.03 (3e-4) | 3.99 (1e-2) |
| Deeper MLP | 0.93 | 0.91 | diverged (3e-3) |

Without tuning it lands close to the best fixed step size on each problem, which differs between
them, but does not beat a well tuned one. Whether this carries over to LoRA training needs an
A/B run against your best fixed SinkSGD learning rate.

## Limits

- No fused back pass: the curvature measurement needs the gradients of all parameters at the
  stepped weights. Full fine-tuning with layer offloading is therefore not possible.
- Steps with preference pairs (mixed RLHF DPO) and fp16 training with a gradient scaler skip the
  measurement; a later step measures. LoRA dropout draws a new mask in the repeated pass, which
  adds noise to the curvature.
- Gradient clipping applies to the repeated pass as well.
