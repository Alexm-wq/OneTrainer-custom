from random import Random
from types import SimpleNamespace

from modules.model.Flux2Model import Flux2Model
from modules.module.flux3_window_attention import enable_flux3_window_attention
from modules.util.convert_util import chunk_swap
from modules.util.enum.DataType import DataType

import torch

from diffusers.image_processor import VaeImageProcessor

TEXT_ENCODER_LAYERS = (4, 8, 12, 16, 20, 24, 28, 32)
MAX_TEXT_LENGTH = 4096
TEXT_PAD_MULTIPLE = 80
TRAINING_SHIFT = 6.93
SAMPLING_SHIFT = 16.0


class Flux3Model(Flux2Model):
    """FLUX.3 conditioning; FLUX.2 supplies only component moves and VAE transforms."""

    def __init__(self, model_type):
        super().__init__(model_type)
        self.vae_train_dtype = DataType.FLOAT_32
        self.vae_autocast_context = torch.autocast("cpu", enabled=False)
        self.empty_text = {}
        self.processor = None

    def fusion_groups(self):
        return None

    def diffusers_to_original(self):
        block = [
            ("attn.to_q", "q_proj"), ("attn.to_k", "k_proj"), ("attn.to_v", "v_proj"),
            ("attn.to_out.0", "attn_out"),
            ("attn.norm_q.weight", "norm.query_norm.scale"),
            ("attn.norm_k.weight", "norm.key_norm.scale"),
            ("ff.linear_in", "mlp_in"), ("ff.linear_out", "mlp_out"),
        ]
        return [
            ("x_embedders.{mode}", "emb_in.{mode}"),
            ("context_embedder", "txt_in"),
            ("timestep_embedder.linear_1", "time_in.in_layer"),
            ("timestep_embedder.linear_2", "time_in.out_layer"),
            ("target_timestep_embedder.linear_1", "ref_time_in.in_layer"),
            ("target_timestep_embedder.linear_2", "ref_time_in.out_layer"),
            ("early_stream_modulations.{mode}.linear", "early_stream_modulations.{mode}.lin"),
            ("single_stream_modulations.{mode}.linear", "single_stream_modulations.{mode}.lin"),
            ("txt_mode_blocks.{i}", "txt_mode_blocks.{i}", block),
            ("mode_blocks.{mode}.{i}", "content_mode_blocks.{mode}.{i}", block),
            ("single_transformer_blocks.{i}", "single_blocks.{i}", block),
            ("proj_out", "final_layer.image.linear"),
            *chunk_swap("norm_out.linear", "final_layer.image.adaLN_modulation.1"),
        ]

    def create_pipeline(self):
        # Sampling is owned by Flux3Sampler so training never requires the optional
        # 9B prompt upsampler or a second copy of the model.
        return SimpleNamespace(image_processor=VaeImageProcessor(vae_scale_factor=16))

    @staticmethod
    def text_length(max_length):
        value = MAX_TEXT_LENGTH if max_length is None else int(max_length)
        if not 1 <= value <= MAX_TEXT_LENGTH:
            raise ValueError("FLUX.3 text sequence length must be between 1 and 4096.")
        return value

    @staticmethod
    def window_lengths(mask):
        real = mask.long().sum(dim=-1)
        if (real == 0).any():
            raise ValueError("FLUX.3 needs at least one valid prompt token.")
        return ((real + TEXT_PAD_MULTIPLE - 1) // TEXT_PAD_MULTIPLE * TEXT_PAD_MULTIPLE).clamp(max=mask.shape[-1])

    def tokenize(self, text, max_length=None):
        prompts = [text] if isinstance(text, str) else list(text)
        rendered = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True,
            )
            for prompt in prompts
        ]
        return self.tokenizer(
            rendered, padding="max_length", truncation=True,
            max_length=self.text_length(max_length), return_tensors="pt",
        )

    @torch.no_grad()
    def encode_tokens(self, tokens, mask):
        device = self.text_encoder.device
        tokens, mask = tokens.to(device), mask.to(device).bool()
        rows = []
        with self.text_encoder_autocast_context:
            for ids, valid, length in zip(tokens, mask, self.window_lengths(mask).tolist(), strict=True):
                # Padding inside the 80-token window is learned conditioning.
                # Only the unused cache tail may be zero-padded or discarded.
                ids, valid = ids[:length].unsqueeze(0), valid[:length].unsqueeze(0)
                positions = (valid.long().cumsum(-1) - 1).masked_fill(~valid, 1)
                output = self.text_encoder(
                    input_ids=ids, attention_mask=valid,
                    position_ids=positions[None].expand(4, -1, -1),
                    output_hidden_states=True, use_cache=False, return_dict=True,
                )
                hidden = torch.cat([output.hidden_states[k] for k in TEXT_ENCODER_LAYERS], dim=-1)[0]
                rows.append(torch.nn.functional.pad(hidden, (0, 0, 0, tokens.shape[-1] - length)))
        return torch.stack(rows)

    @torch.no_grad()
    def prepare_empty_text(self, max_length):
        length = self.text_length(max_length)
        if length not in self.empty_text:
            encoded = self.tokenize("", length)
            hidden = self.encode_tokens(encoded.input_ids, encoded.attention_mask)
            self.empty_text[length] = (hidden.cpu(), encoded.attention_mask.cpu())

    def encode_text(
            self, train_device, batch_size=1, rand: Random | None = None,
            text=None, tokens=None, tokens_mask=None, text_encoder_sequence_length=None,
            text_encoder_dropout_probability=None, text_encoder_output=None,
    ):
        if text is not None:
            encoded = self.tokenize(text, text_encoder_sequence_length)
            tokens, tokens_mask = encoded.input_ids, encoded.attention_mask
        if tokens_mask is None:
            raise ValueError("FLUX.3 conditioning requires the original token mask.")
        if text_encoder_output is None:
            text_encoder_output = self.encode_tokens(tokens, tokens_mask)
        hidden, mask = text_encoder_output.to(train_device), tokens_mask.to(train_device).bool()
        probability = text_encoder_dropout_probability or 0.0
        if probability:
            hidden, mask = hidden.clone(), mask.clone()
            empty = self.empty_text.get(hidden.shape[1])
            if empty is None:
                raise RuntimeError("Encode the empty FLUX.3 prompt before moving the text encoder off device.")
            for index in range(hidden.shape[0]):
                if rand.random() < probability:
                    hidden[index], mask[index] = empty[0][0].to(hidden), empty[1][0].to(mask)
        return hidden, mask

    @staticmethod
    def image_ids(height, width, device, reference_index=None):
        h, w = torch.meshgrid(
            torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij",
        )
        ids = torch.zeros(height * width, 4, device=device, dtype=torch.float32)
        ids[:, 1], ids[:, 2] = h.flatten() * 0.5, w.flatten() * 0.5
        if reference_index is not None:
            ids[:, 3] = -(reference_index + 1)
        return ids

    @staticmethod
    def text_ids(length, device):
        ids = torch.zeros(length, 4, device=device, dtype=torch.float32)
        ids[:, 3] = torch.arange(length, device=device)
        return ids

    def predict_velocity(self, latents, timestep, hidden, mask, *, target_timestep=None, reference=None):
        """Preserve trained padding while batching different prompt windows."""
        enable_flux3_window_attention(self.transformer)
        packed = self.pack_latents(latents)
        lengths = self.window_lengths(mask)
        length = int(lengths.max())
        visible = torch.arange(length, device=hidden.device)[None] < lengths[:, None]
        window_mask = visible[:, None, None, :] if not bool(visible.all()) else None
        image_ids = self.image_ids(*latents.shape[-2:], latents.device)
        target = target_timestep.expand(latents.shape[0]) if target_timestep is not None else None
        cond = self.pack_latents(reference).to(packed.dtype) if reference is not None else None
        cond_ids = self.image_ids(*reference.shape[-2:], latents.device, reference_index=0) if reference is not None else None

        def forward(dummy, inputs, context, times, image_ids, text_ids, target, cond, cond_ids, window):
            return self.transformer(
                hidden_states=inputs, timestep=times, encoder_hidden_states=context,
                img_ids=image_ids, txt_ids=text_ids, target_timestep=target,
                image_cond_hidden_states=cond, image_cond_ids=cond_ids,
                text_window_mask=window,
            ).sample

        args = (
            packed.new_zeros((), requires_grad=True), packed, hidden[:, :length].to(packed.dtype),
            timestep.expand(latents.shape[0]), image_ids, self.text_ids(length, latents.device),
            target, cond, cond_ids, window_mask,
        )
        if torch.is_grad_enabled() and self.transformer_offload_conductor is not None:
            # The conductor owns one activation arena. Save activations only in
            # recompute immediately followed by backward, including mixed RLHF.
            output = torch.utils.checkpoint.checkpoint(forward, *args, use_reentrant=True)
        else:
            output = forward(*args)
        return self.unpack_latents(output, *latents.shape[-2:])

    def calculate_timestep_shift(self, latent_height, latent_width):
        return TRAINING_SHIFT
