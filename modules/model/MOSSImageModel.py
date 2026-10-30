import threading
from random import Random

from modules.model.Flux2Model import Flux2Model
from modules.module.moss.pipeline_moss_image import MOSSImagePipeline
from modules.module.SelfFlow import SelfFlowEMA, SelfFlowProjector
from modules.util.enum.DataType import DataType
from modules.util.moss_layout import is_layout

import torch


class MOSSImageModel(Flux2Model):
    """Reuse only FLUX.2's component moves and 2x2 latent/BN transforms."""

    self_flow_projector: SelfFlowProjector | None
    self_flow_ema: SelfFlowEMA | None
    self_flow_state_dict: dict | None
    self_flow_student_layer: int | None
    self_flow_teacher_layer: int | None

    def __init__(self, model_type):
        super().__init__(model_type)
        self.vae_train_dtype = DataType.FLOAT_32
        self.vae_autocast_context = torch.autocast("cpu", enabled=False)
        self.empty_text = {}
        # Self-Flow training state: projector and EMA teacher exist only while training with Self-Flow.
        self.self_flow_projector = None
        self.self_flow_ema = None
        self.self_flow_state_dict = None
        self.self_flow_student_layer = None
        self.self_flow_teacher_layer = None
        self._tlora_hook = None

    def transformer_to(self, device):
        super().transformer_to(device)
        if self.self_flow_projector is not None:
            self.self_flow_projector.to(device=device)

    def eval(self):
        super().eval()
        if self.self_flow_projector is not None:
            self.self_flow_projector.eval()

    def self_flow_adapter_modules(self) -> list[torch.nn.Module]:
        """Modules switched to eval for the EMA teacher, so LoRA dropout stays off."""
        if self.transformer_lora is None:
            return []
        return list(self.transformer_lora.lora_modules.values())

    def get_self_flow_state_dict(self) -> dict | None:
        if self.self_flow_projector is None or self.self_flow_ema is None:
            return None
        return {
            "version": 1,
            "projector": self.self_flow_projector.state_dict(),
            "ema": self.self_flow_ema.state_dict(),
        }

    def fusion_groups(self):
        return None

    def fused_output_chunks(self) -> dict[str, int]:
        """Transformer layers whose output stacks several projections: name pattern -> number of parts."""
        return {
            r"layers\.\d+\.proj\.ffn_up_gate": 2,
            r"text_embedder\.blocks\.\d+\.qkv": 3,
            r"shared_block_modulation": 6,
            r"video_head_modulation": 2,
        }

    def enable_tlora(self):
        """Set the T-LoRA rank mask from the timestep of every transformer call (training, validation, sampling)."""
        if self._tlora_hook is not None:
            self._tlora_hook.remove()
        self._tlora_hook = self.transformer.register_forward_pre_hook(self._tlora_pre_hook, with_kwargs=True)

    def _tlora_pre_hook(self, module, args, kwargs):
        hidden_states = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
        timestep = kwargs["timestep"] if "timestep" in kwargs else args[1]
        self.set_tlora_timestep(timestep, hidden_states.shape[0], hidden_states.device)

    def set_tlora_timestep(self, timestep: torch.Tensor | None, batch_size: int | None = None,
                           device: torch.device | None = None):
        """T-LoRA: the transformer LoRA uses fewer ranks the noisier the input.

        timestep is MOSS's raw time 1..1000 (sigma * 1000), per sample or per image token. As in
        the official T-LoRA, rank(t) = floor((T - t) * (rank - min_rank) / T) + min_rank.
        """
        config = self.train_config
        if self.transformer_lora is None:
            return
        if config is None or not config.tlora:
            # Without T-LoRA every rank is used, also after a run that had it on.
            if self.transformer_lora._rank_mask is not None:
                self.transformer_lora.set_rank_mask(None)
            return
        if timestep is None:
            self.transformer_lora.set_rank_mask(None)
            return
        times = torch.as_tensor(timestep, dtype=torch.float32, device=device)
        times = times.reshape(times.shape[0], -1).mean(1) if times.ndim > 1 else times.reshape(-1)
        if batch_size is not None:
            times = times.expand(batch_size)
        rank, min_rank = config.lora_rank, self.tlora_min_rank()
        total = self.noise_scheduler.config.num_train_timesteps
        ranks = ((total - times) * (rank - min_rank) / total).floor().clamp(0, rank - min_rank) + min_rank
        mask = torch.arange(rank, device=times.device) < ranks[:, None]
        self.transformer_lora.set_rank_mask(mask.float(), gradient_only=config.tlora_gradient_only)

    def tlora_min_rank(self) -> float:
        """Gradient mode fades the mask in: the min rank falls from the full rank at T-LoRA Start to the
        configured value at T-LoRA End. Forward mode uses the configured value from the first step."""
        config = self.train_config
        if not config.tlora_gradient_only:
            return config.tlora_min_rank
        start, end, step = config.tlora_start_step, config.tlora_end_step, self.train_progress.global_step
        if step >= end:
            return config.tlora_min_rank
        if step <= start:
            return config.lora_rank
        return config.lora_rank - (config.lora_rank - config.tlora_min_rank) * (step - start) / (end - start)

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
        limit = max_length or 4096
        if not hasattr(self.tokenizer, "_mgds_tokenizer_lock"):
            self.tokenizer._mgds_tokenizer_lock = threading.Lock()
        with self.tokenizer._mgds_tokenizer_lock:
            if any(is_layout(prompt) for prompt in prompts):
                full = self.tokenizer(rendered, padding=False, truncation=False).input_ids
                for prompt, ids in zip(prompts, full, strict=True):
                    if is_layout(prompt) and len(ids) > limit:
                        raise ValueError(
                            f"MOSS layout needs {len(ids)} tokens including the chat template; "
                            f"Text Encoder Sequence Length is {limit}. Increase it or simplify the layout."
                        )
            return self.tokenizer(
                rendered, padding="max_length", truncation=True,
                max_length=limit, return_tensors="pt",
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
            text_encoder_dropout_probability=None, text_encoder_output=None, return_dropped=False,
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
        dropped = [False] * hidden.shape[0]
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
                    dropped[index] = True
        return (hidden, mask, dropped) if return_dropped else (hidden, mask)
