# adv-optm 2.5.16 update

The fork previously pinned adv-optm 2.5.12. The stable PyPI release 2.5.16
matches upstream main at
[76fbdc606e8effd8885df32bdc8cfbbe92c5817d](https://github.com/Koratahiu/Advanced_Optimizers/commit/76fbdc606e8effd8885df32bdc8cfbbe92c5817d)
(2026-10-15).

[Upstream comparison](https://github.com/Koratahiu/Advanced_Optimizers/compare/0e5fcea491570327e62373c98acbe7123d41e2b4...76fbdc606e8effd8885df32bdc8cfbbe92c5817d):

- 2.5.13 fixes ADOPT's first step when Kourkoutas beta is disabled.
- 2.5.14 corrects Atan2 plus spectral scaling in AdamW, ADOPT and Prodigy,
  and lets Sinkhorn preserve its input precision.
- 2.5.15 enables scale-invariant epsilon with `eps=None`.
  The existing OneTrainer advanced-optimizer factory already forwards an
  empty epsilon setting as None.
- 2.5.16 fixes the factored Muon/AdaMuon path using an uninitialized shape.

The package's public optimizer classes and constructor arguments remain
compatible. OneTrainer's custom defaults, Nesterov coefficient, PEFT parameter
tags, Muon/AuxAdam grouping, per-group learning rates, fused backward integration
and optimizer-state mapping are retained. Existing user presets are not rewritten.
Upstream did not change SinkSGD's Nesterov mixing formula.

The upgrade also fixes two demonstrated upstream resume failures in the fork:
missing precision metadata in Lion/Muon auxiliary groups, and signed INT8
momentum being reinterpreted as UINT8. A narrow loading helper retains the
upstream loader and restores saved state tensors directly, preserving FP32
moments without first rounding them through BF16 parameter precision.

The focused GitHub Actions workflow generates real checkpoints with 2.5.12,
then upgrades to 2.5.16 and checks factory construction, optimization steps,
custom settings, fused stepping and resume. It uses tiny CPU tensors and does
not download model weights. CUDA compilation/performance and long training
convergence require separate GPU validation.
