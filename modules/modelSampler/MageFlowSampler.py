import random
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

    OneTrainer wraps Mage blocks in compiled CheckpointLayers for training.
    Sampling does not need gradient checkpointing or Inductor, so replace only
    the ModuleList entries with the original eager Mage blocks for the duration
    of inference, then restore the exact same wrapper objects afterward.
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


def _lens_to_cu(lengths: list[int], device: torch.device) -> torch.Tensor:
    """Convert packed segment lengths to int32 cumulative sequence offsets."""
    tensor = torch.tensor(lengths, device=device, dtype=torch.int32)
    return torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device=device),
            torch.cumsum(tensor, dim=0, dtype=torch.int32),
        ]
    )


def _encode_mage_text(
        model,
        prompts: list[str],
        template: str,
        drop_idx: int,
        device: torch.device,
) -> tuple[torch.Tensor, list[int]]:
    """Encode one or more prompts through Mage's normal Qwen conditioning path."""
    tokenizer = model.txt_enc.tokenizer
    max_len = model.txt_enc.tokenizer_max_length + drop_idx
    ids_list = [
        tokenizer(
            template.format(prompt),
            max_length=max_len,
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)
        for prompt in prompts
    ]
    input_ids = torch.cat(ids_list).to(device)
    input_cu = _lens_to_cu([int(ids.numel()) for ids in ids_list], device)

    encoded = model.txt_enc(
        input_ids,
        input_cu,
        drop_idx_override=drop_idx,
    )
    return encoded["txt"], encoded["txt_seq_lens"].tolist()


def _slice_encoded_text(
        txt_flat: torch.Tensor,
        lengths: list[int],
        index: int,
        device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract one packed text segment and its cumulative sequence offsets."""
    token_start = sum(lengths[:index])
    token_end = token_start + lengths[index]
    txt = txt_flat[token_start:token_end].reshape(1, -1, txt_flat.shape[-1]).to(device)
    return txt, _lens_to_cu([lengths[index]], device)


def _mage_velocity(
        transformer,
        image_tokens: torch.Tensor,
        image_shape: tuple[int, int, int],
        txt: torch.Tensor,
        txt_cu: torch.Tensor,
        neg_txt: torch.Tensor | None,
        neg_cu: torch.Tensor | None,
        sigma: float,
        cfg: float,
) -> torch.Tensor:
    """Compute the Mage velocity, fusing positive/negative CFG when needed."""
    device = image_tokens.device
    image_len = image_tokens.shape[1]

    if neg_txt is None:
        image_cu = _lens_to_cu([image_len], device)
        timesteps = torch.full((1,), sigma, dtype=image_tokens.dtype, device=device)
        return transformer(
            img=image_tokens,
            txt=txt,
            timesteps=timesteps,
            img_shapes=[[image_shape]],
            img_cu_seqlens=image_cu,
            txt_cu_seqlens=txt_cu,
        )

    doubled_images = torch.cat([image_tokens, image_tokens], dim=1)
    doubled_image_cu = _lens_to_cu([image_len, image_len], device)
    doubled_txt = torch.cat([txt, neg_txt], dim=1)
    pos_len = int((txt_cu[1] - txt_cu[0]).item())
    neg_len = int((neg_cu[1] - neg_cu[0]).item())
    doubled_txt_cu = _lens_to_cu([pos_len, neg_len], device)
    timesteps = torch.full((2,), sigma, dtype=image_tokens.dtype, device=device)

    output = transformer(
        img=doubled_images,
        txt=doubled_txt,
        timesteps=timesteps,
        img_shapes=[[image_shape, image_shape]],
        img_cu_seqlens=doubled_image_cu,
        txt_cu_seqlens=doubled_txt_cu,
    )
    conditional = output[:, :image_len, :]
    unconditional = output[:, image_len:, :]
    return unconditional + cfg * (conditional - unconditional)


def _get_mage_scheduler(model, steps: int, device: torch.device):
    """Configure the same static-shift scheduler used by the pinned Mage pipeline."""
    scheduler = getattr(model, "scheduler", None)
    if scheduler is None:
        from diffusers import FlowMatchEulerDiscreteScheduler

        scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000,
            shift=6.0,
            use_dynamic_shifting=False,
        )

    scheduler.set_timesteps(
        sigmas=torch.linspace(1.0, 1.0 / steps, steps).tolist(),
        device=device,
    )
    return scheduler


def _decode_mage_image(
        model,
        image_tokens: torch.Tensor,
        height: int,
        width: int,
        device: torch.device,
):
    """Unpack Mage image tokens, VAE-decode them, and return a PIL image."""
    from einops import rearrange
    from mage_flow.models.utils import unpack
    from PIL import Image

    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        output = model.vae.decode(unpack(image_tokens.float(), height, width))
    output = rearrange(output.clamp(-1, 1), "b c h w -> b h w c")
    output = (127.5 * (output + 1.0)).cpu().byte().numpy()
    return Image.fromarray(output[0])


@torch.no_grad()
def _generate_mage_image(
        model,
        prompt: str,
        negative_prompt: str,
        seed: int,
        steps: int,
        cfg: float,
        height: int,
        width: int,
        device: torch.device,
        on_update_progress: Callable[[int, int], None],
):
    """Run OneTrainer's local Mage text-to-image inference path.

    The denoise path matches the pinned Microsoft Mage sampler for a single
    sample: same prompt template, Qwen conditioning, fused CFG, static-shift
    scheduler and VAE decoding. Deployment policy classification and
    distribution-preserving watermark injection are intentionally not part of
    OneTrainer sampling.
    """
    try:
        from einops import rearrange
        from mage_flow.models.utils import PROMPT_TEMPLATE, get_noise
    except ImportError as exc:
        raise RuntimeError("Mage sampling requires Microsoft's official mage_flow package") from exc

    info = PROMPT_TEMPLATE["mage-flow"]
    template = info.get("template", "{}")
    drop_idx = int(info.get("start_idx", 0))
    dev = torch.device(device)

    height = max(16, 16 * (int(height) // 16))
    width = max(16, 16 * (int(width) // 16))
    if seed == -1:
        seed = random.randint(0, 2**32 - 1)

    # Initialize diffusion from ordinary Gaussian noise. The upstream
    # distribution-preserving watermark replacement is deliberately absent.
    torch.manual_seed(seed)
    noise = get_noise(
        num_samples=1,
        channel=model.vae.latent_channels,
        height=height,
        width=width,
        device=dev,
        dtype=torch.bfloat16,
        seed=seed,
    )
    _, _, grid_h, grid_w = noise.shape
    image_tokens = rearrange(noise, "b c h w -> b (h w) c")
    image_shape = (1, grid_h, grid_w)

    negative_prompt = negative_prompt or " "
    use_negative = cfg > 1.0 and bool(negative_prompt)
    conditioning_prompts = [prompt, negative_prompt] if use_negative else [prompt]
    txt_flat, txt_lengths = _encode_mage_text(
        model,
        conditioning_prompts,
        template,
        drop_idx,
        dev,
    )
    txt, txt_cu = _slice_encoded_text(txt_flat, txt_lengths, 0, dev)
    if use_negative:
        neg_txt, neg_cu = _slice_encoded_text(txt_flat, txt_lengths, 1, dev)
    else:
        neg_txt = neg_cu = None

    scheduler = _get_mage_scheduler(model, steps, dev)
    on_update_progress(0, steps)
    for step_index, timestep in enumerate(scheduler.timesteps):
        prediction = _mage_velocity(
            model.transformer,
            image_tokens,
            image_shape,
            txt,
            txt_cu,
            neg_txt,
            neg_cu,
            scheduler.sigmas[step_index].item(),
            cfg,
        )
        image_tokens = scheduler.step(
            prediction,
            timestep,
            image_tokens,
            return_dict=False,
        )[0]
        on_update_progress(step_index + 1, steps)

    return _decode_mage_image(model, image_tokens, height, width, dev)


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
        official = getattr(self.model, "official_model", None)
        if official is None:
            raise RuntimeError("Mage official model wrapper was not retained by the loader")

        self.model.to(self.train_device)
        seed = -1 if sample_config.random_seed else int(sample_config.seed)
        height = self.quantize_resolution(sample_config.height, 16)
        width = self.quantize_resolution(sample_config.width, 16)
        steps = int(sample_config.diffusion_steps)

        try:
            print(
                "[Mage-Flow sample] local conditioning; policy screening disabled; "
                "Gaussian-Shading watermark disabled"
            )
            print(
                f"[Mage-Flow sample] starting {steps}-step diffusion at "
                f"{width}x{height}"
            )

            # Keep the existing stable inference execution context: OneTrainer's
            # autocast, eager transformer blocks, and SDPA instead of FA4/CuTe.
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
                image = _generate_mage_image(
                    official,
                    prompt=sample_config.prompt,
                    negative_prompt=sample_config.negative_prompt or " ",
                    seed=seed,
                    steps=steps,
                    cfg=float(sample_config.cfg_scale),
                    height=height,
                    width=width,
                    device=self.train_device,
                    on_update_progress=on_update_progress,
                )

            return ModelSamplerOutput(FileType.IMAGE, image)
        finally:
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
