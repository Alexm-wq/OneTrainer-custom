# Fast MOSS text-encoder kernels

The Linux CUDA 12 and CUDA 13 environments include `fla-core==0.5.2`.
MOSS uses its Triton Gated DeltaNet, causal convolution, and gated RMSNorm
kernels when Qwen3.5 processes CUDA FP16/BF16 activations. This also applies
to FP8 weight storage: OneTrainer's FP8 projections produce normal
FP16/BF16 compute tensors. The text encoder does not need to be trainable.

The kernel-only FLA package does not change Transformers, require model
downloads, or build a separate `causal-conv1d` C++/CUDA extension. Pixi's
committed lock file includes the dependency. Update the checkout and start
OneTrainer normally; the CUDA environment is synchronized by the launcher.
Restart the Python process after updating.

On the first prompt, look for:

```text
[MOSS TE] FLA/Triton selected: Gated DeltaNet, causal conv1d, gated RMSNorm ...
```

The first invocation may compile/autotune kernels. Measure subsequent
uncached captions, not the first caption or already cached embeddings.
No particular speedup factor is guaranteed.

Transformers 5.5.4 may initially print its generic warning about a missing
native `causal-conv1d` extension while constructing Qwen3.5. OneTrainer binds
FLA's Triton replacement after loading; the later `[MOSS TE]` message
reports the kernels actually selected for encoding.

CPU, FP32, ROCm and missing optional dependencies use the PyTorch reference
path with a diagnostic. Windows does not automatically install FLA.
Unexpected kernel execution/compilation errors remain visible rather than
silently switching to a slow backend.

For an A/B comparison, start a fresh process with
`OT_MOSS_TEXT_KERNELS=0 ./start-ui.sh` to force the reference kernels.
Use a separate empty cache for each comparison. Existing caches and
checkpoint keys remain compatible. Prompt unpadding, position order,
Q/K normalization, decay, scheduler and LoRA training are unchanged.

The targeted test file is `tests/test_moss_text_kernels.py`. It checks
reference-path equivalence, gradients, parameter/checkpoint identity,
layout and gate forwarding, dependency/API compatibility and diagnostics.
The CUDA test compares real FLA outputs against the PyTorch reference at
FP16/BF16 with Qwen-sized heads and uneven prompt lengths. It is explicitly
skipped on runners without an NVIDIA GPU; CPU CI does not establish GPU
correctness or an RTX 5090 throughput improvement.
