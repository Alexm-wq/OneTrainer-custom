import os
import csv
import json
import math
from abc import ABCMeta, abstractmethod
from contextlib import contextmanager
from contextvars import ContextVar

from modules.model.BaseModel import BaseModel
from modules.util.config.TrainConfig import TrainConfig, TrainEmbeddingConfig, TrainModelPartConfig
from modules.util.enum.AttentionMechanism import AttentionMechanism
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.ModuleFilter import ModuleFilter
from modules.util.NamedParameterGroup import NamedParameterGroup, NamedParameterGroupCollection
from modules.util.TimedActionMixin import TimedActionMixin
from modules.util.TrainProgress import TrainProgress

import torch
from torch import Tensor
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.tensorboard import SummaryWriter
from modules.util.enum.DPOObjective import DPOObjective
from modules.util.enum.DPORefMode import DPORefMode
import modules.util.multi_gpu_util as multi
import torch.nn.functional as F


class BaseModelSetup(
    TimedActionMixin,
    metaclass=ABCMeta,
):
    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            debug_mode: bool,
    ):
        super().__init__()

        self.train_device = train_device
        self.temp_device = temp_device
        self.debug_mode = debug_mode
        self._dpo_ref_params = None
        self._dpo_ref_params_cpu = None
        self._dpo_ema_ref_params_cpu = None
        self._dpo_ema_policy_cpu_buffers = None
        self._dpo_ema_ref_decay = None
        self._dpo_ema_ref_steps = 0
        self._dpo_concept_ref_params: dict[str, list[list[Tensor]]] = {}
        self._dpo_concept_ref_params_cpu: dict[str, list[list[Tensor]]] = {}
        self._dpo_policy_cpu_buffers = None
        self._last_dpo_metrics = None
        self._last_dpo_pair_losses: list[tuple[str, float, str]] = []
        self._dpo_paired_half = None
        self._dpo_stream_active = ContextVar(
            f"ot_dpo_stream_active_{id(self)}",
            default=False,
        )
        self._dpo_reference_active = ContextVar(
            f"ot_dpo_reference_active_{id(self)}",
            default=False,
        )
        self._dpo_runtime_beta = None
        self._dpo_bad_pair_previous_rewards: dict[str, tuple[float, float, int]] = {}
        self._dpo_curriculum_state: dict[str, dict[str, float | int | str]] = {}
        self._dpo_curriculum_pending: dict[str, dict[str, float | int | str]] = {}
        self.frozen_parameters = {}

    @abstractmethod
    def create_parameters(self, model: BaseModel, config: TrainConfig) -> NamedParameterGroupCollection:
        pass

    @abstractmethod
    def setup_optimizations(self, model: BaseModel, config: TrainConfig):
        pass

    @abstractmethod
    def setup_model(self, model: BaseModel, config: TrainConfig):
        pass

    @abstractmethod
    def setup_train_device(self, model: BaseModel, config: TrainConfig):
        pass

    @abstractmethod
    def predict(self, model: BaseModel, batch: dict, config: TrainConfig, train_progress: TrainProgress, *, deterministic: bool = False) -> dict:
        pass

    @abstractmethod
    def calculate_loss(self, model: BaseModel, batch: dict, data: dict, config: TrainConfig) -> Tensor:
        pass

    @abstractmethod
    def after_optimizer_step(self, model: BaseModel, config: TrainConfig, train_progress: TrainProgress):
        pass

    def after_backward(self, model: BaseModel, config: TrainConfig, train_progress: TrainProgress):
        pass

    def after_streamed_dpo_branch_backward(self, model: BaseModel, config: TrainConfig, train_progress: TrainProgress):
        pass

    def report_to_tensorboard(self, model: BaseModel, config: TrainConfig, scheduler: LRScheduler, tensorboard: SummaryWriter):
        lrs = scheduler.get_last_lr()
        parameters = model.parameters.display_name_mapping
        reported_learning_rates = {}
        if any('optim_type' in g for g in model.optimizer.param_groups):
            for group in model.optimizer.param_groups:
                name = group.get('name')
                if not name or not group['params']:
                    continue
                optim_type = group.get('optim_type', 'unknown')
                unique_name = f"{name}_{optim_type}"
                if unique_name not in reported_learning_rates:
                    reported_learning_rates[unique_name] = group['lr']
        else:
            for lr, parameter in zip(lrs, parameters, strict=True):
                name = parameter.split('/')[0]
                if name not in reported_learning_rates:
                    reported_learning_rates[name] = lr
        reported_learning_rates = config.optimizer.optimizer.maybe_adjust_lrs(reported_learning_rates, model.optimizer)
        for name, lr in reported_learning_rates.items():
            tensorboard.add_scalar(f"lr/{name}", lr, model.train_progress.global_step)
        if hasattr(model.optimizer, 'kourkoutas_helper') and model.optimizer.kourkoutas_helper is not None:
            stats = model.optimizer.kourkoutas_helper.last_beta2_stats
            if stats:
                tensorboard.add_scalar("kourkoutas/beta2_mean", stats['mean'], model.train_progress.global_step)

    @staticmethod
    def _dpo_hard_pair_curriculum_enabled(config: TrainConfig) -> bool:
        return bool(getattr(config, "rlhf_dpo_hard_pair_curriculum", False))

    @staticmethod
    def _dpo_curriculum_settings(config: TrainConfig) -> dict[str, float]:
        ema_decay = float(getattr(config, "rlhf_dpo_hard_pair_curriculum_ema", 0.9))
        minimum_weight = float(getattr(config, "rlhf_dpo_hard_pair_curriculum_min_weight", 0.1))
        full_margin = float(getattr(config, "rlhf_dpo_hard_pair_curriculum_full_margin", 0.05))
        if not 0.0 <= ema_decay < 1.0:
            raise ValueError("Hard-Pair Curriculum EMA must satisfy 0 <= EMA < 1")
        if not 0.0 <= minimum_weight <= 1.0:
            raise ValueError("Hard-Pair Curriculum Minimum Weight must satisfy 0 <= weight <= 1")
        if full_margin <= 0.0:
            raise ValueError("DPO curriculum Full Margin must be > 0")
        return {"ema_decay": ema_decay, "minimum_weight": minimum_weight, "full_margin": full_margin}

    @staticmethod
    def _dpo_normalize_pair_path(path: str) -> str:
        return os.path.normcase(os.path.realpath(os.path.abspath(os.path.expanduser(str(path)))))

    def _dpo_pair_identity(self, batch: dict, index: int) -> str:
        pair_key = str(self._dpo_csv_batch_value(batch, ("dpo_pair_key",), index) or "").strip()
        if pair_key.startswith("dpo-pair-path-v1\nchosen=") and "\nrejected=" in pair_key:
            return pair_key
        chosen_path = str(self._dpo_csv_batch_value(batch, ("image_path", "chosen_image_path", "chosen_source_path", "chosen_image_path_raw"), index) or "").strip()
        rejected_path = str(self._dpo_csv_batch_value(batch, ("image_path_rejected", "rejected_image_path", "rejected_source_path", "rejected_image_path_raw"), index) or "").strip()
        if chosen_path and rejected_path:
            return f"dpo-pair-path-v1\nchosen={self._dpo_normalize_pair_path(chosen_path)}\nrejected={self._dpo_normalize_pair_path(rejected_path)}"
        raise RuntimeError("DPO pair identity requires both chosen and rejected source paths")

    def _dpo_curriculum_pair_key(self, batch: dict, index: int) -> str:
        return self._dpo_pair_identity(batch, index)

    @staticmethod
    def _dpo_curriculum_competence(objective: DPOObjective, config: TrainConfig, margin: Tensor, policy_chosen_score: Tensor, policy_rejected_score: Tensor) -> tuple[Tensor, float | None]:
        objective = DPOObjective(objective)
        if objective == DPOObjective.LINEAR:
            return policy_chosen_score - policy_rejected_score, None
        if objective == DPOObjective.IPO:
            tau = float(config.rlhf_dpo_ipo_tau)
            target_margin = 1.0 / (2.0 * tau)
            configured_full_margin = float(getattr(config, "rlhf_dpo_hard_pair_curriculum_full_margin", 0.05))
            return margin, min(configured_full_margin, target_margin)
        return margin, None

    def _stage_dpo_curriculum_observations(self, batch: dict, config: TrainConfig, competence: Tensor, objective: DPOObjective, full_margin_override: float | None = None) -> tuple[Tensor, Tensor, Tensor]:
        settings = self._dpo_curriculum_settings(config)
        full_margin = settings["full_margin"] if full_margin_override is None else float(full_margin_override)
        weights = torch.ones_like(competence)
        return weights, competence.detach(), torch.ones_like(competence)

    def commit_dpo_curriculum_state(self):
        if self._dpo_curriculum_pending:
            self._dpo_curriculum_state.update(self._dpo_curriculum_pending)
            self._dpo_curriculum_pending.clear()

    def discard_dpo_curriculum_pending(self):
        self._dpo_curriculum_pending.clear()

    def save_dpo_curriculum_state(self, path: str, config: TrainConfig):
        return

    def load_dpo_curriculum_state(self, path: str, config: TrainConfig):
        return

    @staticmethod
    def _is_dpo_rejected_key(key: str) -> bool:
        return key.endswith("_rejected")

    def _create_dpo_batched_batch(self, batch: dict) -> tuple[dict, int]:
        latent_image = batch["latent_image"]
        chosen_b = int(latent_image.shape[0])
        rejected_latent = batch["latent_image_rejected"]
        batched = {}
        rejected_key_map = {"latent_image": "latent_image_rejected", "image": "image_rejected", "image_path": "image_path_rejected", "chosen_image_path": "rejected_image_path", "chosen_source_path": "rejected_source_path"}
        for key, value in batch.items():
            if key.endswith("_rejected") or key.startswith("rejected_"):
                continue
            rejected_key = rejected_key_map.get(key)
            if rejected_key is None and key.startswith("chosen_"):
                candidate = "rejected_" + key[len("chosen_"):]
                if candidate in batch:
                    rejected_key = candidate
            if rejected_key is not None and rejected_key in batch:
                rejected_value = batch[rejected_key]
                if isinstance(value, torch.Tensor):
                    batched[key] = torch.cat([value, rejected_value], dim=0)
                elif isinstance(value, list):
                    batched[key] = value + list(rejected_value)
                elif isinstance(value, tuple):
                    batched[key] = value + tuple(rejected_value)
                else:
                    batched[key] = value
            else:
                if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == chosen_b:
                    batched[key] = torch.cat([value, value], dim=0)
                elif isinstance(value, list) and len(value) == chosen_b:
                    batched[key] = value + value
                elif isinstance(value, tuple) and len(value) == chosen_b:
                    batched[key] = value + value
                else:
                    batched[key] = value
        self._dpo_paired_half = chosen_b
        return batched, chosen_b

    def _create_dpo_stream_batches(self, batch: dict) -> tuple[dict, dict, int]:
        chosen_b = int(batch["latent_image"].shape[0])
        chosen = dict(batch)
        rejected = dict(batch)
        rejected["latent_image"] = batch["latent_image_rejected"]
        return chosen, rejected, chosen_b

    @contextmanager
    def _dpo_stream_predict_context(self):
        token = self._dpo_stream_active.set(True)
        try:
            yield
        finally:
            self._dpo_stream_active.reset(token)

    def _dpo_conditioning_locked(self) -> bool:
        return bool(self._dpo_paired_half is not None or self._dpo_stream_active.get())

    @contextmanager
    def _dpo_reference_predict_context(self):
        token = self._dpo_reference_active.set(True)
        try:
            yield
        finally:
            self._dpo_reference_active.reset(token)

    def _dpo_reference_prediction(self) -> bool:
        return bool(self._dpo_reference_active.get())

    def rlhf_chosen_supervised_requires_separate_forward(self, config: TrainConfig) -> bool:
        return False

    def rlhf_mixed_normal_dpo_requires_sequential_backward(self, config: TrainConfig) -> bool:
        return False

    @staticmethod
    def rlhf_chosen_supervised_weight(config: TrainConfig, objective: DPOObjective) -> float:
        objective = DPOObjective(objective)
        return 1.0 if objective in {DPOObjective.ANCHORED_REJECT, DPOObjective.BALANCED_REJECT} else max(float(config.rlhf_supervised_mix), 0.0)

    def calculate_rlhf_chosen_supervised_loss(self, model: BaseModel, batch: dict, config: TrainConfig, train_progress: TrainProgress) -> Tensor:
        output = self.predict(model, batch, config, train_progress)
        try:
            return self.calculate_loss(model, batch, output, config)
        finally:
            del output

    @staticmethod
    def _split_dpo_batched_output(output: dict, chosen_b: int) -> tuple[dict, dict]:
        chosen_out, rejected_out = {}, {}
        for key, value in output.items():
            if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == 2 * chosen_b:
                chosen_out[key] = value[:chosen_b]
                rejected_out[key] = value[chosen_b:]
            else:
                chosen_out[key] = value
                rejected_out[key] = value
        return chosen_out, rejected_out

    def get_last_dpo_metrics(self) -> dict[str, float]:
        return self._last_dpo_metrics or {}

    def get_last_dpo_pair_losses(self) -> list[tuple[str, float, str]]:
        return list(self._last_dpo_pair_losses)

    def set_dpo_runtime_beta(self, beta: float | None):
        self._dpo_runtime_beta = beta

    def rlhf_logp_per_sample(self, model: BaseModel, batch: dict, data: dict, config: TrainConfig) -> Tensor:
        predicted = data["predicted"]
        target = data["target"]
        error = (predicted - target).pow(2)
        element_loss_weight = data.get("element_loss_weight")
        if element_loss_weight is not None:
            error = error * element_loss_weight.to(device=error.device, dtype=error.dtype)
        return -error.mean(dim=list(range(1, predicted.ndim)), dtype=torch.float32)

    def rlhf_linear_error_per_sample(self, model: BaseModel, batch: dict, data: dict, config: TrainConfig) -> Tensor:
        predicted = data["predicted"]
        target = data["target"]
        error = (predicted - target).pow(2)
        element_loss_weight = data.get("element_loss_weight")
        if element_loss_weight is not None:
            error = error * element_loss_weight.to(device=error.device, dtype=error.dtype)
        return error.mean(dim=list(range(1, predicted.ndim)), dtype=torch.float32)

    def _dpo_score_per_sample(self, model: BaseModel, batch: dict, data: dict, config: TrainConfig, objective: DPOObjective) -> Tensor:
        if objective == DPOObjective.LINEAR:
            return -self.rlhf_linear_error_per_sample(model, batch, data, config)
        return self.rlhf_logp_per_sample(model, batch, data, config)

    @staticmethod
    def _validate_rlhf_logp_per_sample(logp: Tensor, expected_b: int, name: str) -> Tensor:
        if logp.ndim != 1 or int(logp.shape[0]) != int(expected_b):
            raise RuntimeError(f"{name} rlhf_logp_per_sample must return shape [{expected_b}], got {tuple(logp.shape)}")
        return logp.float()

    def rlhf_policy_auxiliary_loss(self, model: BaseModel, batch: dict, data: dict, config: TrainConfig) -> Tensor | None:
        return None

    @staticmethod
    def _linear_dpo_pair_loss(policy_chosen_score: Tensor, policy_rejected_score: Tensor, reference_chosen_score: Tensor, reference_rejected_score: Tensor, beta: float, eta: float):
        chosen_ratio = policy_chosen_score - reference_chosen_score.detach()
        rejected_ratio = policy_rejected_score - reference_rejected_score.detach()
        margin = chosen_ratio - rejected_ratio
        utility = torch.clamp(0.5 - 0.2 * float(beta) * margin.detach(), min=float(eta), max=1.0 - float(eta))
        policy_error_gap = -(policy_chosen_score - policy_rejected_score)
        return utility.detach() * policy_error_gap, utility, policy_error_gap, margin

    @staticmethod
    def _dpo_csv_batch_value(batch: dict, names: tuple[str, ...], index: int | None = None):
        for name in names:
            if name in batch:
                value = batch[name]
                if isinstance(value, (list, tuple)) and index is not None and index < len(value):
                    return value[index]
                return value
        return ""

    def _dpo_localized_metrics(self, batch: dict, batch_size: int) -> dict[str, float]:
        return {"localized_active_fraction": 0.0, "localized_mask_fraction": 0.0, "localized_mean_weight": 1.0}

    def _write_dpo_pair_csv_log(self, **kwargs):
        return

    def _write_dpo_bad_pair_csv_log(self, **kwargs):
        return

    def calculate_dpo_loss(self, model: BaseModel, batch: dict, config: TrainConfig, train_progress: TrainProgress, *, objective: DPOObjective | None = None, reference_mode: DPORefMode | None = None, reference_key: str | None = None, streamed: bool = False, external_chosen_supervised_loss_value: float | None = None) -> Tensor:
        if "latent_image_rejected" not in batch:
            raise RuntimeError("RLHF DPO requires paired chosen/rejected batches")
        objective = DPOObjective(config.rlhf_dpo_objective if objective is None else objective)
        if objective == DPOObjective.LINEAR:
            reference_mode = DPORefMode.EMA_ADAPTER
            reference_key = None
        self._last_dpo_pair_losses = []
        beta = config.rlhf_dpo_beta if self._dpo_runtime_beta is None else self._dpo_runtime_beta
        chosen_supervised_weight = self.rlhf_chosen_supervised_weight(config, objective)
        include_chosen_supervised = chosen_supervised_weight > 0.0
        batched_input, chosen_b = self._create_dpo_batched_batch(batch)
        self._dpo_paired_half = chosen_b
        try:
            with torch.no_grad(), self.reference_model(model, config, reference_mode=reference_mode, reference_key=reference_key), self._dpo_reference_predict_context():
                ref_output = self.predict(model, batched_input, config, train_progress)
                ref_logp = self._validate_rlhf_logp_per_sample(self._dpo_score_per_sample(model, batched_input, ref_output, config, objective), 2 * chosen_b, "reference")
                ref_chosen_logp, ref_rejected_logp = ref_logp[:chosen_b], ref_logp[chosen_b:]
        finally:
            self._dpo_paired_half = None
        supervised_loss = None
        separate_chosen_supervised = include_chosen_supervised and self.rlhf_chosen_supervised_requires_separate_forward(config)
        if separate_chosen_supervised and external_chosen_supervised_loss_value is None:
            supervised_loss = self.calculate_rlhf_chosen_supervised_loss(model, batch, config, train_progress)
        self._dpo_paired_half = chosen_b
        try:
            policy_output = self.predict(model, batched_input, config, train_progress)
        finally:
            self._dpo_paired_half = None
        policy_logp = self._validate_rlhf_logp_per_sample(self._dpo_score_per_sample(model, batched_input, policy_output, config, objective), 2 * chosen_b, "policy")
        policy_chosen_logp, policy_rejected_logp = policy_logp[:chosen_b], policy_logp[chosen_b:]
        policy_auxiliary_loss = self.rlhf_policy_auxiliary_loss(model, batched_input, policy_output, config)
        if include_chosen_supervised and not separate_chosen_supervised:
            chosen_output, _ = self._split_dpo_batched_output(policy_output, chosen_b)
            supervised_loss = self.calculate_loss(model, batch, chosen_output, config)
        chosen_ratio = policy_chosen_logp - ref_chosen_logp.detach()
        rejected_ratio = policy_rejected_logp - ref_rejected_logp.detach()
        margin = chosen_ratio - rejected_ratio
        margin_penalty_loss = torch.zeros_like(margin)
        wrong_order_penalty_loss = torch.zeros_like(margin)
        margin_target_violation = torch.zeros_like(margin)
        wrong_order_violation = torch.zeros_like(margin)
        if objective == DPOObjective.LINEAR:
            raw_pair_total_loss, linear_utility, linear_policy_error_gap, _ = self._linear_dpo_pair_loss(policy_chosen_logp, policy_rejected_logp, ref_chosen_logp, ref_rejected_logp, float(beta), float(config.rlhf_dpo_linear_eta))
        elif objective == DPOObjective.ANCHORED_REJECT:
            rejected_target = float(getattr(config, "rlhf_dpo_anchored_rejected_target", -0.05))
            rejected_weight = max(float(getattr(config, "rlhf_dpo_anchored_rejected_weight", 1.0)), 0.0)
            huber_delta = max(float(getattr(config, "rlhf_dpo_anchored_huber_delta", 0.1)), 1e-8)
            margin_target = max(float(getattr(config, "rlhf_dpo_anchored_margin_target", 0.05)), 0.0)
            margin_weight = max(float(getattr(config, "rlhf_dpo_anchored_margin_weight", 0.5)), 0.0)
            wrong_order_weight = max(float(getattr(config, "rlhf_dpo_anchored_wrong_order_weight", 0.5)), 0.0)
            rejected_violation = F.relu(rejected_ratio - rejected_target)
            rejected_pair_loss = rejected_weight * F.smooth_l1_loss(rejected_violation, torch.zeros_like(rejected_violation), beta=huber_delta, reduction="none")
            chosen_floor_violation = F.relu(-chosen_ratio)
            chosen_floor_pair_loss = F.smooth_l1_loss(chosen_floor_violation, torch.zeros_like(chosen_floor_violation), beta=huber_delta, reduction="none")
            margin_target_violation = F.relu(margin_target - margin)
            margin_penalty_loss = margin_weight * F.smooth_l1_loss(margin_target_violation, torch.zeros_like(margin_target_violation), beta=huber_delta, reduction="none")
            wrong_order_violation = F.relu(-margin)
            wrong_order_penalty_loss = wrong_order_weight * F.smooth_l1_loss(wrong_order_violation, torch.zeros_like(wrong_order_violation), beta=huber_delta, reduction="none")
            raw_pair_total_loss = rejected_pair_loss + chosen_floor_pair_loss + margin_penalty_loss + wrong_order_penalty_loss
            linear_utility = None
            linear_policy_error_gap = None
        elif objective == DPOObjective.BALANCED_REJECT:
            balanced_chosen_budget = F.relu(chosen_ratio.detach())
            balanced_reject_target = -max(float(getattr(config, "rlhf_dpo_balanced_reject_ratio", 1.0)), 0.0) * balanced_chosen_budget
            violation = F.relu(rejected_ratio - balanced_reject_target)
            raw_pair_total_loss = max(float(getattr(config, "rlhf_dpo_balanced_reject_weight", 1.0)), 0.0) * F.smooth_l1_loss(violation, torch.zeros_like(violation), beta=max(float(getattr(config, "rlhf_dpo_balanced_huber_delta", 0.1)), 1e-8), reduction="none")
            linear_utility = None
            linear_policy_error_gap = None
        elif objective == DPOObjective.IPO:
            raw_pair_total_loss = (margin - 1.0 / (2.0 * config.rlhf_dpo_ipo_tau)).pow(2)
            linear_utility = None
            linear_policy_error_gap = None
        else:
            logits = beta * margin
            raw_pair_total_loss = -F.logsigmoid(logits)
            linear_utility = None
            linear_policy_error_gap = None
        pair_total_loss = raw_pair_total_loss
        preference_loss = pair_total_loss.mean()
        loss = preference_loss
        if policy_auxiliary_loss is not None:
            loss = loss + policy_auxiliary_loss
        chosen_supervised_loss_value = 0.0
        if supervised_loss is not None:
            chosen_supervised_loss_value = supervised_loss.detach().item()
            loss = loss + chosen_supervised_weight * supervised_loss
        elif external_chosen_supervised_loss_value is not None:
            chosen_supervised_loss_value = float(external_chosen_supervised_loss_value)
        self._last_dpo_metrics = {
            "objective_loss": preference_loss.detach().item(),
            "chosen_reward": chosen_ratio.detach().mean().item(),
            "rejected_reward": rejected_ratio.detach().mean().item(),
            "reward_margin": margin.detach().mean().item(),
            "accuracy": (margin.detach() > 0).float().mean().item(),
            "chosen_supervised_weight": float(chosen_supervised_weight),
            "chosen_supervised_loss": float(chosen_supervised_loss_value),
            "total_loss": float(loss.detach().item()),
        }
        if objective == DPOObjective.ANCHORED_REJECT:
            self._last_dpo_metrics["anchored_chosen_floor_loss"] = chosen_floor_pair_loss.detach().mean().item()
            self._last_dpo_metrics["anchored_chosen_floor_violation"] = chosen_floor_violation.detach().mean().item()
        if linear_utility is not None:
            self._last_dpo_metrics["linear_utility"] = linear_utility.detach().mean().item()
            self._last_dpo_metrics["linear_policy_error_gap"] = linear_policy_error_gap.detach().mean().item()
        return loss

    def stop_embedding_training_elapsed(self, config: TrainEmbeddingConfig, train_progress: TrainProgress):
        return self.single_action_elapsed("stop_embedding_training_" + str(config.uuid), config.stop_training_after, config.stop_training_after_unit, train_progress)

    def __stop_model_part_training_elapsed(self, unique_name: str, config: TrainModelPartConfig, train_progress: TrainProgress):
        return self.single_action_elapsed("stop_" + unique_name + "_training", config.stop_training_after, config.stop_training_after_unit, train_progress)

    @contextmanager
    def prior_model(self, model: BaseModel, config: TrainConfig):
        if config.training_method is not TrainingMethod.LORA:
            raise NotImplementedError("Prior model is only available with LoRA training")
        for adapter in model.adapters():
            adapter.remove_hook_from_module()
        try:
            yield
        finally:
            for adapter in model.adapters():
                adapter.hook_to_module()

    @contextmanager
    def reference_model(self, model: BaseModel, config: TrainConfig, reference_mode: DPORefMode | None = None, reference_key: str | None = None):
        adapters = model.adapters()
        ref_mode = DPORefMode(config.effective_dpo_ref_mode() if reference_mode is None else reference_mode)
        if ref_mode == DPORefMode.NEW_ADAPTER:
            for adapter in adapters:
                adapter.remove_hook_from_module()
            try:
                yield
            finally:
                for adapter in adapters:
                    adapter.hook_to_module()
        else:
            yield

    def initialize_dpo_reference(self, model: BaseModel, config: TrainConfig, snapshot_path: str | None = None, force_existing_adapter: bool = False, force_cpu_existing_adapter: bool = False):
        return

    def save_dpo_reference(self, snapshot_path: str):
        return

    def update_dpo_ema_reference(self, model: BaseModel, config: TrainConfig):
        return

    def initialize_dpo_concept_references(self, model: BaseModel, gpu_reference_keys: list[str] | tuple[str, ...] = (), cpu_reference_keys: list[str] | tuple[str, ...] = (), snapshot_path: str | None = None):
        return

    def save_dpo_concept_references(self, snapshot_path: str):
        return

    def _create_model_part_parameters(self, parameter_group_collection: NamedParameterGroupCollection, unique_name: str, model: torch.nn.Module, config: TrainModelPartConfig, freeze: list[ModuleFilter] | None = None, debug: bool = False):
        if not config.train:
            return
        parameters = model.parameters()
        parameter_group_collection.add_group(NamedParameterGroup(unique_name=unique_name, parameters=parameters, learning_rate=config.learning_rate))

    def _setup_model_part_requires_grad(self, unique_name: str, model: torch.nn.Module, config: TrainModelPartConfig, train_progress: TrainProgress):
        if model is not None:
            train_model_part = config.train and not self.__stop_model_part_training_elapsed(unique_name, config, train_progress)
            model.requires_grad_(train_model_part)

    @staticmethod
    def _set_attention_backend(component, attn: AttentionMechanism, mask: bool):
        match attn:
            case AttentionMechanism.SDP:
                component.set_attention_backend("native")
            case AttentionMechanism.FLASH:
                component.set_attention_backend("flash")
            case AttentionMechanism.CUDNN:
                component.set_attention_backend("_native_cudnn")
            case _:
                raise NotImplementedError(f"attention mechanism {str(attn)} not implemented")
