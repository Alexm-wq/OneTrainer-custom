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
    bypassing the Self-Flow dispatch that already unwraps those layers. Sampling
    does not need gradient checkpointing or Inductor, so replace only the
    ModuleList entries with the original eager Mage blocks for the duration of
    inference, then restore the exact same wrapper objects afterward.
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


@contextmanager
def _stable_mage_sampling_attention(model: MageFlowModel):
    """Use non-FA4 SDPA only while Mage is sampling.

    Repeated FA4/CuTe varlen inference on consumer Blackwell has produced a
    delayed device-side illegal-memory-access after many denoise steps. Training
    can still use the configured FA4 backend; inference temporarily switches the
    Mage backend shim and the mutable Qwen restore state to SDPA. PyTorch Flash
    SDPA is disabled in this scope as well, preferring cuDNN or memory-efficient
    SDPA and retaining the math implementation only as a final fallback.
    """
    from mage_flow.models.modules._attn_backend import set_attn_backend

    previous_backend = getattr(model, "mage_attention_backend", "sdpa")
    backend_state = getattr(model, "mage_attention_backend_state", None)
    previous_state_backend = (
        backend_state.get("dit", previous_backend)
        if backend_state is not None
        else previous_backend
    )

    model.mage_attention_backend = "sdpa"
    if backend_state is not None:
        backend_state["dit"] = "sdpa"
    set_attn_backend("sdpa")

    try:
        # The official Mage SDPA shim dispatches each packed segment through
        # torch.nn.functional.scaled_dot_product_attention. Exclude PyTorch's
        # Flash backend too so sampling cannot fall back onto another flash
        # implementation while we are diagnosing/stabilizing Blackwell inference.
        if torch.cuda.is_available():
            with torch.backends.cuda.sdp_kernel(
                    enable_flash=False,
                    enable_math=True,
                    enable_mem_efficient=True,
                    enable_cudnn=True,
            ):
                yield
        else:
            yield
    finally:
        model.mage_attention_backend = previous_backend
        if backend_state is not None:
            backend_state["dit"] = previous_state_backend
        set_attn_backend(previous_state_backend)


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
            cached_verdict = original_screen_text(sample_config.prompt, max_new_tokens=64)
            print(
                "[Mage-Flow sample] prompt screening complete; "
                f"starting {steps}-step diffusion at {width}x{height}"
            )

            def cached_screen_text(_prompt, max_new_tokens=64):
                return cached_verdict

            official.txt_enc.screen_text = cached_screen_text

            # Official Mage creates BF16 diffusion tokens but does not establish
            # the transformer autocast context used by OneTrainer training. Reuse
            # that context, unwrap training-only compiled/checkpoint wrappers, and
            # keep sampling off FA4/CuTe. The original backend and wrappers are
            # restored immediately after inference.
            with (
                self.model.autocast_context,
                _stable_mage_sampling_attention(self.model),
                _eager_mage_transformer_blocks(official.transformer) as eager_blocks,
            ):
                print(
                    "[Mage-Flow sample] attention=SDPA (Flash disabled; "
                    "cuDNN/efficient fallback enabled)"
                )
                if eager_blocks:
                    print(
                        f"[Mage-Flow sample] eager inference blocks={eager_blocks}; "
                        "training compile remains enabled"
                    )
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
