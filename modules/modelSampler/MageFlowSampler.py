from collections.abc import Callable
from contextlib import contextmanager

from modules.model.MageFlowModel import MageFlowModel
from modules.modelSampler.BaseModelSampler import BaseModelSampler, ModelSamplerOutput
from modules.util import factory
from modules.util.checkpointing_util import BaseCheckpointLayer
from modules.util.config.SampleConfig import SampleConfig
from modules.util.enum.AudioFormat import AudioFormat
from modules.util.enum.FileType import FileType
from modules.util.enum.ImageFormat import ImageFormat
from modules.util.enum.ModelType import ModelType
from modules.util.enum.VideoFormat import VideoFormat
from modules.util.torch_util import torch_gc

import torch


@contextmanager
def _eager_mage_transformer_blocks(transformer):
    """Temporarily expose the original Mage blocks for inference.

    OneTrainer wraps Mage blocks in compiled CheckpointLayers for training. The
    official Mage sampler calls ``official.transformer`` directly under no_grad,
    bypassing the Self-Flow dispatch that already unwraps those layers. Inductor
    can consequently try to autotune BF16 inference inputs against FP32-backed
    weights and fail before the actual denoising kernel is launched.

    Sampling does not need gradient checkpointing or Inductor. Replace only the
    ModuleList entries with the original eager Mage blocks for the duration of
    inference, then restore the exact same wrapper objects so training state,
    optimizer references and compiled training graphs remain intact.
    """
    blocks = transformer.transformer_blocks
    wrapped = []
    for index, block in enumerate(list(blocks)):
        if not isinstance(block, BaseCheckpointLayer):
            continue
        eager_block = getattr(block, "checkpoint", None)
        if eager_block is None:
            continue
        wrapped.append((index, block))
        blocks[index] = eager_block

    try:
        yield len(wrapped)
    finally:
        for index, block in wrapped:
            blocks[index] = block


@factory.register(BaseModelSampler, ModelType.MAGE_FLOW)
class MageFlowSampler(BaseModelSampler):
    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            model: MageFlowModel,
            model_type: ModelType,
    ):
        super().__init__(train_device, temp_device)
        self.model = model
        self.model_type = model_type

    @torch.no_grad()
    def _sample(self, sample_config: SampleConfig, on_update_progress) -> ModelSamplerOutput:
        try:
            from mage_flow.pipeline import generate_images
        except ImportError as exc:
            raise RuntimeError("Mage sampling requires Microsoft's official mage_flow package") from exc

        official = getattr(self.model, "official_model", None)
        if official is None:
            raise RuntimeError("Mage official model wrapper was not retained by the loader")

        self.model.to(self.train_device)
        seed = -1 if sample_config.random_seed else int(sample_config.seed)
        height = self.quantize_resolution(sample_config.height, 16)
        width = self.quantize_resolution(sample_config.width, 16)
        steps = int(sample_config.diffusion_steps)

        # Microsoft's top-level generate_images() always runs an autoregressive
        # Qwen policy-screening generate() before diffusion. On a quantized
        # Qwen3-VL this can otherwise look like a frozen sampler because the
        # diffusion progress bar has not started yet. Run it explicitly with a
        # bounded response length, report the phase, and feed the cached verdict
        # back to generate_images so the expensive prepass is not repeated.
        original_screen_text = official.txt_enc.screen_text
        cached_verdict = None
        try:
            print("[Mage-Flow sample] Qwen prompt screening...")
            on_update_progress(0, steps)
            # Upstream allows 160 new tokens for a response whose contract is a
            # short one-line JSON object. 64 leaves ample room for the complete
            # verdict while avoiding a long autoregressive prepass before every
            # OneTrainer sample.
            cached_verdict = original_screen_text(sample_config.prompt, max_new_tokens=64)
            print(
                "[Mage-Flow sample] prompt screening complete; "
                f"starting {steps}-step diffusion at {width}x{height}"
            )

            def cached_screen_text(_prompt, max_new_tokens=64):
                return cached_verdict

            official.txt_enc.screen_text = cached_screen_text

            # The official Mage pipeline creates BF16 latent/image tokens but
            # does not establish an autocast context around transformer forwards.
            # OneTrainer training does. Reuse the exact OneTrainer transformer
            # autocast here so FP32-backed/quantized Linear weights receive the
            # same mixed-dtype treatment as they do during training.
            #
            # Also unwrap Mage's compiled CheckpointLayers for this no-grad
            # inference pass. The official sampler walks transformer_blocks
            # directly, so the Self-Flow no-grad dispatch cannot protect this
            # path from Inductor's inference-time BF16/FP32 matmul autotuner.
            with self.model.autocast_context, _eager_mage_transformer_blocks(official.transformer) as eager_blocks:
                if eager_blocks:
                    print(
                        f"[Mage-Flow sample] eager inference blocks={eager_blocks}; "
                        "training compile remains enabled"
                    )
                # TextEncoder.forward is wrapped by the Mage loader to temporarily
                # use packed SDPA; it restores model.mage_attention_backend before
                # the first DiT denoising call, so FA4 remains active for sampling.
                images = generate_images(
                    official,
                    prompts=[sample_config.prompt],
                    neg_prompts=[sample_config.negative_prompt or " "],
                    seeds=[seed],
                    steps=steps,
                    cfg=float(sample_config.cfg_scale),
                    heights=[height],
                    widths=[width],
                    device=str(self.train_device),
                    prompt_template="mage-flow",
                    static_shift=None,
                    batch_cfg=True,
                )
            on_update_progress(steps, steps)
            return ModelSamplerOutput(FileType.IMAGE, images[0])
        finally:
            official.txt_enc.screen_text = original_screen_text
            self.model.to(self.temp_device)
            torch_gc()

    def sample(
            self,
            sample_config: SampleConfig,
            destination: str,
            image_format: ImageFormat | None = None,
            video_format: VideoFormat | None = None,
            audio_format: AudioFormat | None = None,
            on_sample: Callable[[ModelSamplerOutput], None] = lambda _: None,
            on_update_progress: Callable[[int, int], None] = lambda _, __: None,
    ):
        output = self._sample(sample_config, on_update_progress)
        self.save_sampler_output(output, destination, image_format, video_format, audio_format)
        on_sample(output)
