# Training Krea 2 and Kroma LoRAs

Kroma uses the existing **Krea 2** model type. Its native transformer safetensors can
be loaded directly; no manual conversion or separate model type is needed.

1. Select **Krea 2** and the **LoRA** training method.
2. Set **Base Model** to `krea/Krea-2-Raw`, or your local copy of that Diffusers
   model. This supplies the transformer configuration, tokenizer, Qwen3-VL text
   encoder, VAE and scheduler. For a gated repository, accept its license on
   Hugging Face and enter an authorized token in OneTrainer.
3. Set **Override Transformer** to the downloaded Kroma safetensors file. You can
   also use a Hugging Face file URL, for example:
   `https://huggingface.co/lodestones/Kroma/blob/main/kroma-v0.3-base.safetensors`.
   Only the overridden transformer is loaded; the base repository's transformer
   weights are not needed.
4. Start with **Transformer Data Type: bfloat16** and your usual Krea 2 LoRA,
   dataset and memory settings. For lower memory use, select OneTrainer's weight
   quantization on the original checkpoint. Pre-quantized ComfyUI files and GGUF
   are not supported by this Krea 2 import path.
5. For use in ComfyUI, select **Comfy** as the LoRA output format. **Original**
   exports the native Krea 2 names without the `diffusion_model.` prefix.

For Kroma v0.3.1 OPD, the [model author](https://huggingface.co/lodestones/Kroma)
recommends training LoRAs against either `kroma-v0.3-base.safetensors` or
`kroma-sensei-booru-e6-teacher-velocity-v0.3.safetensors`, then applying them to
`kroma-v0.3.1-turbo-opd.safetensors`. The Turbo checkpoint can also be imported,
but it is not the author's recommended training base. Use the checkpoint's own
sampling guidance when generating previews; Turbo inference settings are not
training timestep settings.

The override accepts native Krea/Kroma and Diffusers transformer state dicts,
including common transformer prefixes. Every weight name and shape is checked;
an incomplete or incompatible file fails rather than leaving random weights in
the model. Existing Diffusers base-model loading and internal training backups
continue to use their existing paths.
