# Model integration requirements

These preferences were explicitly requested by Alex on 2026-10-17.

- Deliver a complete model integration from the start. A loader-only implementation or a minimal training path is not a finished integration.
- Compare the official model implementation with the relevant existing OneTrainer model integrations before implementing. Preserve the model's conditioning, latent transforms, scheduler/timestep semantics and inference behavior.
- Cover model selection, model overrides, all applicable training-tab controls, sensible presets, data/text/latent caching, multi-resolution caching, training and adapter variants, mixed RLHF, sampling, validation, saving, export and resume.
- Connect applicable precision and quantization options for both the denoiser and text encoder, including filters and device moves. Connect appropriate attention backends, gradient checkpointing, block compilation, layer offloading and activation offloading.
- Adapt features to the model's architecture. Do not expose unsupported controls or silently ignore settings. State concrete technical exceptions, such as an unsupported checkpoint format or a text encoder that must stay frozen.
- Use focused tests for newly added or modified functionality and directly affected shared paths. Do not run unrelated whole-project test suites. Report GPU-only checks separately when only CPU testing is available.
- For remote-only work requested by Alex, edit the fork and run checks remotely. Do not clone repositories, download weights, install dependencies or place task files on Alex's PC.
