from random import Random

from modules.model.Flux2Model import Flux2Model
from modules.module.moss.pipeline_moss_image import MOSSImagePipeline
from modules.util.enum.DataType import DataType

import torch


class MOSSImageModel(Flux2Model):
    """Reuse only FLUX.2's component moves and 2x2 latent/BN transforms."""

    def __init__(self, model_type):
        super().__init__(model_type)
        self.vae_train_dtype = DataType.FLOAT_32
        self.vae_autocast_context = torch.autocast("cpu", enabled=False)
        self.empty_text = {}

    def fusion_groups(self):
        return None

    def diffusers_to_original(self):
        return None

    def create_pipeline(self):
        return MOSSImagePipeline(
            transformer=self.transformer, scheduler=self.noise_scheduler,
            text_encoder=self.text_encoder, tokenizer=self.tokenizer, vae=self.vae,
        )

    def calculate_timestep_shift(self, latent_height, latent_width):
        # MOSS uses a fixed training distribution, not FLUX's resolution-dependent shift.
        return self.noise_scheduler.config.training_shift

    def tokenize(self, text, max_length):
        prompts = [text] if isinstance(text, str) else list(text)
        rendered = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False,
                add_generation_prompt=True, enable_thinking=True,
            )
            for prompt in prompts
        ]
        return self.tokenizer(
            rendered, padding="max_length", truncation=True,
            max_length=max_length or 4096, return_tensors="pt",
        )

    @torch.no_grad()
    def encode_tokens(self, tokens, mask):
        # Qwen3.5's Gated DeltaNet is padding-sensitive. Unpad BEFORE encoding,
        # then scatter back to the original token positions for the disk cache.
        device = self.text_encoder.device
        tokens, mask = tokens.to(device), mask.to(device).bool()
        rows = []
        with self.text_encoder_autocast_context:
            for ids, valid in zip(tokens, mask, strict=True):
                if not valid.any():
                    raise ValueError("MOSS needs at least one valid prompt token.")
                unpadded = ids[valid].unsqueeze(0)
                hidden = self.text_encoder(
                    input_ids=unpadded, attention_mask=torch.ones_like(unpadded),
                    use_cache=False, return_dict=True,
                ).last_hidden_state[0]
                row = hidden.new_zeros((ids.shape[0], hidden.shape[-1]))
                row[valid] = hidden
                rows.append(row)
        return torch.stack(rows)

    @torch.no_grad()
    def prepare_empty_text(self, max_length):
        max_length = max_length or 4096
        if max_length not in self.empty_text:
            tokens = self.tokenize("", max_length)
            hidden = self.encode_tokens(tokens.input_ids, tokens.attention_mask)
            self.empty_text[max_length] = (hidden.cpu(), tokens.attention_mask.cpu())

    def encode_text(
            self, train_device, batch_size=1, rand: Random | None = None,
            text=None, tokens=None, tokens_mask=None, text_encoder_sequence_length=None,
            text_encoder_dropout_probability=None, text_encoder_output=None,
    ):
        if text is not None:
            encoded = self.tokenize(text, text_encoder_sequence_length)
            tokens, tokens_mask = encoded.input_ids, encoded.attention_mask
        if tokens_mask is None:
            raise ValueError("MOSS text conditioning requires its cached token mask.")
        if text_encoder_output is None:
            text_encoder_output = self.encode_tokens(tokens, tokens_mask)
        hidden = text_encoder_output.to(train_device)
        mask = tokens_mask.to(train_device).bool()
        probability = text_encoder_dropout_probability or 0.0
        if probability:
            # Do not mutate cached tensors or replace conditioning by an all-zero mask.
            hidden, mask = hidden.clone(), mask.clone()
            empty = self.empty_text.get(hidden.shape[1])
            if empty is None:
                raise RuntimeError("Empty MOSS prompt must be encoded during text-cache preparation.")
            for index in range(hidden.shape[0]):
                if rand.random() < probability:
                    hidden[index] = empty[0][0].to(hidden)
                    mask[index] = empty[1][0].to(mask)
        return hidden, mask
