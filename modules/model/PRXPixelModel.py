import html
from contextlib import nullcontext
from random import Random

from modules.model.BaseModel import BaseModel
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util.enum.ModelType import ModelType
from modules.util.LayerOffloadConductor import LayerOffloadConductor

import torch
from torch import Tensor

import ftfy
from diffusers import DiffusionPipeline, FlowMatchEulerDiscreteScheduler, PRXPixelPipeline, PRXTransformer2DModel
from transformers import PreTrainedTokenizerBase, Qwen3VLTextModel


PROMPT_MAX_LENGTH = 256


def clean_prompt(text: str) -> str:
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


class PRXPixelModel(BaseModel):
    tokenizer: PreTrainedTokenizerBase | None
    noise_scheduler: FlowMatchEulerDiscreteScheduler | None
    text_encoder: Qwen3VLTextModel | None
    transformer: PRXTransformer2DModel | None

    text_encoder_autocast_context: torch.autocast | nullcontext
    text_encoder_offload_conductor: LayerOffloadConductor | None
    transformer_offload_conductor: LayerOffloadConductor | None

    transformer_lora: LoRAModuleWrapper | None
    lora_state_dict: dict | None

    def __init__(self, model_type: ModelType):
        super().__init__(model_type=model_type)

        self.tokenizer = None
        self.noise_scheduler = None
        self.text_encoder = None
        self.transformer = None

        self.text_encoder_autocast_context = nullcontext()
        self.text_encoder_offload_conductor = None
        self.transformer_offload_conductor = None

        self.transformer_lora = None
        self.lora_state_dict = None

    def adapters(self) -> list[LoRAModuleWrapper]:
        return [adapter for adapter in [self.transformer_lora] if adapter is not None]

    def text_encoder_to(self, device: torch.device):
        if self.text_encoder_offload_conductor is not None:
            self.text_encoder_offload_conductor.to(device)
        else:
            self.text_encoder.to(device=device)

    def transformer_to(self, device: torch.device):
        if self.transformer_offload_conductor is not None:
            self.transformer_offload_conductor.to(device)
        else:
            self.transformer.to(device=device)

        if self.transformer_lora is not None:
            self.transformer_lora.to(device)

    def to(self, device: torch.device):
        self.text_encoder_to(device)
        self.transformer_to(device)

    def eval(self):
        self.text_encoder.eval()
        self.transformer.eval()

    def create_pipeline(self) -> DiffusionPipeline:
        return PRXPixelPipeline(
            transformer=self.transformer,
            scheduler=self.noise_scheduler,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            default_sample_size=1024,
            prompt_max_tokens=PROMPT_MAX_LENGTH,
            noise_scale=2.0,
        )

    def encode_text(
            self,
            train_device: torch.device,
            batch_size: int = 1,
            rand: Random | None = None,
            text: str | list[str] = None,
            tokens: Tensor = None,
            tokens_mask: Tensor = None,
            text_encoder_dropout_probability: float | None = None,
            text_encoder_output: Tensor = None,
    ) -> tuple[Tensor, Tensor]:
        if tokens is None and text is not None:
            if isinstance(text, str):
                text = [text]

            tokenized = self.tokenizer(
                [clean_prompt(value) for value in text],
                truncation=True,
                padding="max_length",
                max_length=PROMPT_MAX_LENGTH,
                return_tensors="pt",
            )
            tokens = tokenized.input_ids.to(self.text_encoder.device)
            tokens_mask = tokenized.attention_mask.bool().to(self.text_encoder.device)

        if text_encoder_output is None:
            with self.text_encoder_autocast_context:
                output = self.text_encoder(
                    input_ids=tokens,
                    attention_mask=tokens_mask,
                    output_hidden_states=True,
                    return_dict=True,
                    use_cache=False,
                )
                text_encoder_output = output.last_hidden_state

        if text_encoder_dropout_probability is not None and text_encoder_dropout_probability > 0.0:
            raise NotImplementedError("Text-encoder dropout is not supported for PRX Pixel")

        return text_encoder_output, tokens_mask.bool()
