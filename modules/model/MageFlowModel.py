from __future__ import annotations

from contextlib import nullcontext
from random import Random
from typing import Any

from modules.model.BaseModel import BaseModel
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.module.MageFlowSelfFlow import MageFlowSelfFlowEMA, MageFlowSelfFlowProjector
from modules.util.enum.ModelType import ModelType

import torch
from torch import Tensor


MAGE_PROMPT_TEMPLATE = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, "
    "text, spatial relationships of the objects and background:<|im_end|>\n"
    "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
)
MAGE_PROMPT_CROP_START = 34
MAGE_PROMPT_MAX_LENGTH = 2048


class MageFlowModel(BaseModel):
    """OneTrainer adapter around Microsoft's official ``mage_flow`` package."""

    tokenizer: Any
    noise_scheduler: Any
    text_encoder_wrapper: Any
    text_encoder: torch.nn.Module | None
    vae: torch.nn.Module | None
    transformer: torch.nn.Module | None

    transformer_lora: LoRAModuleWrapper | None
    lora_state_dict: dict | None

    self_flow_projector: MageFlowSelfFlowProjector | None
    self_flow_ema: MageFlowSelfFlowEMA | None
    self_flow_state_dict: dict | None
    self_flow_student_layer: int | None
    self_flow_teacher_layer: int | None

    def __init__(self, model_type: ModelType):
        super().__init__(model_type=model_type)
        self.tokenizer = None
        self.noise_scheduler = None
        self.text_encoder_wrapper = None
        self.text_encoder = None
        self.vae = None
        self.transformer = None
        self.base_model_name = None

        self.text_encoder_autocast_context = nullcontext()
        self.transformer_lora = None
        self.lora_state_dict = None

        self.self_flow_projector = None
        self.self_flow_ema = None
        self.self_flow_state_dict = None
        self.self_flow_student_layer = None
        self.self_flow_teacher_layer = None

    def adapters(self) -> list[LoRAModuleWrapper]:
        return [adapter for adapter in [self.transformer_lora] if adapter is not None]

    def vae_to(self, device: torch.device):
        if self.vae is not None:
            self.vae.to(device=device)

    def text_encoder_to(self, device: torch.device):
        if self.text_encoder_wrapper is not None:
            self.text_encoder_wrapper.to(device=device)
        elif self.text_encoder is not None:
            self.text_encoder.to(device=device)

    def transformer_to(self, device: torch.device):
        if self.transformer is not None:
            self.transformer.to(device=device)
        if self.transformer_lora is not None:
            self.transformer_lora.to(device)
        if self.self_flow_projector is not None:
            self.self_flow_projector.to(device=device)

    def to(self, device: torch.device):
        self.vae_to(device)
        self.text_encoder_to(device)
        self.transformer_to(device)
        return self

    def eval(self):
        if self.vae is not None:
            self.vae.eval()
        if self.text_encoder_wrapper is not None:
            self.text_encoder_wrapper.eval()
        elif self.text_encoder is not None:
            self.text_encoder.eval()
        if self.transformer is not None:
            self.transformer.eval()
        if self.self_flow_projector is not None:
            self.self_flow_projector.eval()
        return self

    def self_flow_adapter_modules(self) -> list[torch.nn.Module]:
        if self.transformer_lora is not None:
            return list(self.transformer_lora.lora_modules.values())
        # Full fine-tuning is supported as well, but maintaining/swapping a full
        # 4B CPU EMA is intentionally expensive. LoRA remains the recommended
        # Self-Flow mode on a single GPU.
        return [self.transformer] if self.transformer is not None else []

    def self_flow_parameters(self) -> list[torch.nn.Parameter]:
        if self.transformer_lora is not None:
            return [p for module in self.self_flow_adapter_modules() for p in module.parameters() if p.requires_grad]
        return [p for p in self.transformer.parameters() if p.requires_grad] if self.transformer is not None else []

    def get_self_flow_state_dict(self) -> dict | None:
        if self.self_flow_projector is None or self.self_flow_ema is None:
            return None
        return {
            "version": 1,
            "projector": self.self_flow_projector.state_dict(),
            "ema": self.self_flow_ema.state_dict(),
            "student_layer": self.self_flow_student_layer,
            "teacher_layer": self.self_flow_teacher_layer,
        }

    @staticmethod
    def pack_latents(latents: Tensor) -> Tensor:
        if latents.ndim != 4:
            raise ValueError(f"Mage latents must be [B,C,H,W], got {tuple(latents.shape)}")
        return latents.flatten(2).transpose(1, 2)

    @staticmethod
    def unpack_latents(tokens: Tensor, height: int, width: int) -> Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"Mage latent tokens must be [B,N,C], got {tuple(tokens.shape)}")
        return tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[-1], height, width)

    @staticmethod
    def image_shapes(batch_size: int, height: int, width: int):
        return [[(1, int(height), int(width))] for _ in range(batch_size)]

    @staticmethod
    def _cu_seqlens(lengths: list[int], device: torch.device) -> Tensor:
        lens = torch.tensor(lengths, dtype=torch.int32, device=device)
        return torch.cat([torch.zeros(1, dtype=torch.int32, device=device), lens.cumsum(0)])

    def encode_text(
            self,
            train_device: torch.device,
            batch_size: int = 1,
            rand: Random | None = None,
            text: str | list[str] | None = None,
            tokens: Tensor | None = None,
            tokens_mask: Tensor | None = None,
            text_encoder_output: Tensor | None = None,
            text_encoder_dropout_probability: float | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return padded conditioning + boolean validity mask.

        The official Mage TextEncoder is internally packed/varlen. We preserve
        that exact encoder path and only pad its already-encoded conditioning
        so MGDS can cache/batch it. ``prepare_packed_text`` removes the padding
        again before the Mage transformer forward.
        """
        if text_encoder_dropout_probability not in (None, 0.0):
            raise NotImplementedError("Mage text-encoder dropout is not implemented; use caption dropout instead")

        if text_encoder_output is not None:
            if tokens_mask is None:
                mask = torch.ones(text_encoder_output.shape[:2], dtype=torch.bool, device=text_encoder_output.device)
            else:
                # Cached text embeddings are already cropped to exclude the 34-token system prefix.
                lengths = tokens_mask.sum(dim=1).clamp(max=text_encoder_output.shape[1])
                mask = torch.arange(text_encoder_output.shape[1], device=text_encoder_output.device)[None, :] < lengths[:, None]
            return text_encoder_output, mask

        if self.text_encoder_wrapper is None:
            raise RuntimeError("Mage text encoder has not been loaded")

        tokenizer = self.tokenizer
        if tokens is None:
            if text is None:
                raise ValueError("Mage encode_text requires text, tokens, or a cached text_encoder_output")
            prompts = [text] if isinstance(text, str) else list(text)
            ids = [
                tokenizer(
                    MAGE_PROMPT_TEMPLATE.format(prompt),
                    max_length=MAGE_PROMPT_MAX_LENGTH + MAGE_PROMPT_CROP_START,
                    truncation=True,
                    return_tensors="pt",
                ).input_ids.squeeze(0)
                for prompt in prompts
            ]
        else:
            if tokens.ndim == 1:
                tokens = tokens.unsqueeze(0)
            if tokens_mask is None:
                ids = [row for row in tokens]
            else:
                ids = [row[mask.bool()] for row, mask in zip(tokens, tokens_mask, strict=True)]

        lengths = [int(row.numel()) for row in ids]
        packed_ids = torch.cat(ids, dim=0).to(self.text_encoder.device)
        cu = self._cu_seqlens(lengths, self.text_encoder.device)
        with self.text_encoder_autocast_context:
            result = self.text_encoder_wrapper(
                packed_ids,
                cu,
                drop_idx_override=MAGE_PROMPT_CROP_START,
            )

        flat = result["txt"]
        valid_lengths = [int(x) for x in result["txt_seq_lens"].tolist()]
        split = torch.split(flat, valid_lengths, dim=0)
        padded = torch.nn.utils.rnn.pad_sequence(split, batch_first=True)
        max_len = padded.shape[1]
        mask = torch.arange(max_len, device=padded.device)[None, :] < torch.tensor(valid_lengths, device=padded.device)[:, None]
        return padded, mask

    def prepare_packed_text(self, text: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        if text.ndim != 3 or mask.ndim != 2:
            raise ValueError("Mage text conditioning must be [B,T,D] with mask [B,T]")
        lengths = mask.sum(dim=1).to(dtype=torch.int32)
        pieces = [text[i, :int(length.item())] for i, length in enumerate(lengths)]
        packed = torch.cat(pieces, dim=0).unsqueeze(0)
        return packed, self._cu_seqlens([int(x) for x in lengths.tolist()], text.device)

    def prepare_packed_images(self, tokens: Tensor) -> tuple[Tensor, Tensor]:
        batch, length, channels = tokens.shape
        packed = tokens.reshape(1, batch * length, channels)
        return packed, self._cu_seqlens([length] * batch, tokens.device)

    @staticmethod
    def unprepare_packed_images(tokens: Tensor, batch_size: int) -> Tensor:
        if tokens.shape[0] != 1:
            raise ValueError("Packed Mage output must have batch dimension 1")
        if tokens.shape[1] % batch_size != 0:
            raise ValueError("Packed Mage token count is not divisible by batch size")
        return tokens.reshape(batch_size, tokens.shape[1] // batch_size, tokens.shape[-1])
