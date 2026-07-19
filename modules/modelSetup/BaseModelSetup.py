import os
import csv
import json
from abc import ABCMeta, abstractmethod
from contextlib import contextmanager

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
        self._last_dpo_metrics = None
        self._dpo_paired_half = None
        self._dpo_runtime_beta = None
        # Previous per-pair rewards are kept only for the current process so the
        # separate bad-pair CSV can detect sudden wrong-direction jumps.
        self._dpo_bad_pair_previous_rewards: dict[str, tuple[float, float, int]] = {}
        # Hard-pair curriculum state is committed only after a successful
        # optimizer step. Pending observations belong to the current gradient-
        # accumulation window and are never written into a backup.
        self._dpo_curriculum_state: dict[str, dict[str, float | int]] = {}
        self._dpo_curriculum_pending: dict[str, dict[str, float | int]] = {}
        self.frozen_parameters = {}

    @abstractmethod
    def create_parameters(
            self,
            model: BaseModel,
            config: TrainConfig,
    ) -> NamedParameterGroupCollection:
        pass

    @abstractmethod
    def setup_optimizations(
            self,
            model: BaseModel,
            config: TrainConfig,
    ):
        pass

    @abstractmethod
    def setup_model(
            self,
            model: BaseModel,
            config: TrainConfig,
    ):
        pass

    @abstractmethod
    def setup_train_device(
            self,
            model: BaseModel,
            config: TrainConfig,
    ):
        pass

    @abstractmethod
    def predict(
            self,
            model: BaseModel,
            batch: dict,
            config: TrainConfig,
            train_progress: TrainProgress,
            *,
            deterministic: bool = False,
    ) -> dict:
        pass

    @abstractmethod
    def calculate_loss(
            self,
            model: BaseModel,
            batch: dict,
            data: dict,
            config: TrainConfig,
    ) -> Tensor:
        pass

    @abstractmethod
    def after_optimizer_step(
            self,
            model: BaseModel,
            config: TrainConfig,
            train_progress: TrainProgress,
    ):
        pass

    def report_to_tensorboard(
            self,
            model: BaseModel,
            config: TrainConfig,
            scheduler: LRScheduler,
            tensorboard: SummaryWriter,
    ):
        lrs = scheduler.get_last_lr()
        parameters = model.parameters.display_name_mapping

        reported_learning_rates = {}

        # Handle MuonWithAuxAdam's split parameter groups
        if any('optim_type' in g for g in model.optimizer.param_groups):
            for group in model.optimizer.param_groups:
                name = group.get('name')
                if not name or not group['params']:
                    continue
                # For MuonWithAuxAdam, parameter groups are split for Muon and Adam,
                # but might retain the same base name (e.g., 'unet').
                optim_type = group.get('optim_type', 'unknown')
                unique_name = f"{name}_{optim_type}"
                if unique_name not in reported_learning_rates:
                    reported_learning_rates[unique_name] = group['lr']
        else:
            for lr, parameter in zip(lrs, parameters, strict=True):
                # only use the prefix. this prevents multiple embedding reports. TODO: find a better solution
                name = parameter.split('/')[0]

                if name not in reported_learning_rates:
                    reported_learning_rates[name] = lr

        reported_learning_rates = config.optimizer.optimizer.maybe_adjust_lrs(reported_learning_rates, model.optimizer)

        for name, lr in reported_learning_rates.items():
            tensorboard.add_scalar(
                f"lr/{name}", lr, model.train_progress.global_step
            )

        if hasattr(model.optimizer, 'kourkoutas_helper') and model.optimizer.kourkoutas_helper is not None:
            stats = model.optimizer.kourkoutas_helper.last_beta2_stats
            if stats:
                tensorboard.add_scalar("kourkoutas/beta2_mean", stats['mean'], model.train_progress.global_step)

    @staticmethod
    def _dpo_hard_pair_curriculum_enabled(config: TrainConfig) -> bool:
        return bool(
            getattr(config, "rlhf_dpo_hard_pair_curriculum", False)
            and getattr(config, "rlhf_dpo_objective", None)
            == DPOObjective.ANCHORED_REJECT
        )

    @staticmethod
    def _dpo_curriculum_settings(config: TrainConfig) -> dict[str, float]:
        ema_decay = float(
            getattr(config, "rlhf_dpo_hard_pair_curriculum_ema", 0.9)
        )
        minimum_weight = float(
            getattr(config, "rlhf_dpo_hard_pair_curriculum_min_weight", 0.1)
        )
        full_margin = float(
            getattr(config, "rlhf_dpo_hard_pair_curriculum_full_margin", 0.05)
        )
        margin_target = float(
            getattr(config, "rlhf_dpo_anchored_margin_target", 0.05)
        )
        margin_weight = float(
            getattr(config, "rlhf_dpo_anchored_margin_weight", 0.5)
        )
        wrong_order_weight = float(
            getattr(config, "rlhf_dpo_anchored_wrong_order_weight", 0.5)
        )

        if not 0.0 <= ema_decay < 1.0:
            raise ValueError(
                "Hard-Pair Curriculum EMA must satisfy 0 <= EMA < 1, "
                f"got {ema_decay}"
            )
        if not 0.0 <= minimum_weight <= 1.0:
            raise ValueError(
                "Hard-Pair Curriculum Minimum Weight must satisfy "
                f"0 <= weight <= 1, got {minimum_weight}"
            )
        if full_margin <= 0.0:
            raise ValueError(
                "Hard-Pair Curriculum Full Margin must be > 0, "
                f"got {full_margin}"
            )
        if margin_target < 0.0:
            raise ValueError(
                "Anchored Reject Margin Target must be >= 0, "
                f"got {margin_target}"
            )
        if margin_weight < 0.0:
            raise ValueError(
                "Anchored Reject Margin Weight must be >= 0, "
                f"got {margin_weight}"
            )
        if wrong_order_weight < 0.0:
            raise ValueError(
                "Anchored Reject Wrong-Order Weight must be >= 0, "
                f"got {wrong_order_weight}"
            )

        world_size = int(os.environ.get("WORLD_SIZE", "1") or "1")
        if world_size != 1:
            raise RuntimeError(
                "Hard-Pair Curriculum currently requires single-GPU training. "
                "Per-pair EMA state cannot be resumed exactly across multiple "
                "ranks with OneTrainer's master-only backup path."
            )

        return {
            "ema_decay": ema_decay,
            "minimum_weight": minimum_weight,
            "full_margin": full_margin,
            "margin_target": margin_target,
            "margin_weight": margin_weight,
            "wrong_order_weight": wrong_order_weight,
        }

    def _dpo_curriculum_pair_key(self, batch: dict, index: int) -> str:
        pair_key = str(
            self._dpo_csv_batch_value(batch, ("dpo_pair_key",), index)
            or ""
        ).strip()
        if pair_key:
            return pair_key

        chosen_path = str(
            self._dpo_csv_batch_value(
                batch,
                (
                    "image_path",
                    "chosen_image_path",
                    "chosen_source_path",
                    "chosen_image_path_raw",
                ),
                index,
            )
            or ""
        ).strip()
        rejected_path = str(
            self._dpo_csv_batch_value(
                batch,
                (
                    "image_path_rejected",
                    "rejected_image_path",
                    "rejected_source_path",
                    "rejected_image_path_raw",
                ),
                index,
            )
            or ""
        ).strip()

        if chosen_path or rejected_path:
            return f"chosen={chosen_path}\nrejected={rejected_path}"

        raise RuntimeError(
            "Hard-Pair Curriculum requires a stable dpo_pair_key or chosen/"
            "rejected image paths for every pair. None were present in the batch."
        )

    def _stage_dpo_curriculum_observations(
            self,
            batch: dict,
            config: TrainConfig,
            margin: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        settings = self._dpo_curriculum_settings(config)
        ema_decay = settings["ema_decay"]
        minimum_weight = settings["minimum_weight"]
        full_margin = settings["full_margin"]

        detached_margin = margin.detach().float().reshape(-1)
        weights: list[float] = []
        margin_emas: list[float] = []
        observations: list[float] = []

        for index, current_tensor in enumerate(detached_margin):
            pair_key = self._dpo_curriculum_pair_key(batch, index)
            current_margin = float(current_tensor.cpu().item())

            previous = self._dpo_curriculum_pending.get(pair_key)
            if previous is None:
                previous = self._dpo_curriculum_state.get(pair_key)

            if previous is None:
                margin_ema = current_margin
                count = 1
            else:
                old_ema = float(previous["margin_ema"])
                count = int(previous["observations"]) + 1
                margin_ema = (
                    ema_decay * old_ema
                    + (1.0 - ema_decay) * current_margin
                )

            self._dpo_curriculum_pending[pair_key] = {
                "margin_ema": margin_ema,
                "observations": count,
            }

            progress = max(0.0, min(1.0, margin_ema / full_margin))
            smooth_progress = progress * progress * (3.0 - 2.0 * progress)
            weight = minimum_weight + (1.0 - minimum_weight) * smooth_progress

            weights.append(weight)
            margin_emas.append(margin_ema)
            observations.append(float(count))

        return (
            torch.tensor(weights, device=margin.device, dtype=margin.dtype),
            torch.tensor(
                margin_emas,
                device=margin.device,
                dtype=margin.dtype,
            ),
            torch.tensor(
                observations,
                device=margin.device,
                dtype=margin.dtype,
            ),
        )

    def commit_dpo_curriculum_state(self):
        if not self._dpo_curriculum_pending:
            return
        self._dpo_curriculum_state.update(self._dpo_curriculum_pending)
        self._dpo_curriculum_pending.clear()

    def discard_dpo_curriculum_pending(self):
        self._dpo_curriculum_pending.clear()

    def save_dpo_curriculum_state(self, path: str, config: TrainConfig):
        if not self._dpo_hard_pair_curriculum_enabled(config):
            return
        if self._dpo_curriculum_pending:
            raise RuntimeError(
                "Refusing to save Hard-Pair Curriculum state with uncommitted "
                "gradient-accumulation observations."
            )

        payload = {
            "version": 2,
            "settings": self._dpo_curriculum_settings(config),
            "pairs": {
                key: {
                    "margin_ema": float(value["margin_ema"]),
                    "observations": int(value["observations"]),
                }
                for key, value in sorted(self._dpo_curriculum_state.items())
            },
        }
        temp_path = f"{path}.tmp"
        with open(temp_path, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temp_path, path)

    def load_dpo_curriculum_state(self, path: str, config: TrainConfig):
        self._dpo_curriculum_state.clear()
        self._dpo_curriculum_pending.clear()
        if not self._dpo_hard_pair_curriculum_enabled(config):
            return
        if not os.path.isfile(path):
            raise RuntimeError(
                "Hard-Pair Curriculum is enabled, but the resume backup is "
                f"missing its state file: {path}"
            )

        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        state_version = int(payload.get("version", -1))
        if state_version not in (1, 2):
            raise RuntimeError(
                "Unsupported Hard-Pair Curriculum state version: "
                f"{payload.get('version')}"
            )

        expected = self._dpo_curriculum_settings(config)
        saved = payload.get("settings", {})
        settings_to_check = (
            ("ema_decay", "minimum_weight", "full_margin")
            if state_version == 1
            else tuple(expected.keys())
        )
        for name in settings_to_check:
            expected_value = expected[name]
            if name not in saved or abs(float(saved[name]) - expected_value) > 1e-12:
                raise RuntimeError(
                    "Hard-Pair Curriculum settings changed across resume: "
                    f"{name}: backup={saved.get(name)!r}, current={expected_value!r}. "
                    "Exact resume requires identical curriculum settings."
                )
        if state_version == 1:
            print(
                "[OT-DPO-CURRICULUM] Loading legacy v1 EMA state. Pairwise "
                "margin penalties use the current configuration; backups "
                "created after this point will use strict v2 settings."
            )

        pairs = payload.get("pairs", {})
        if not isinstance(pairs, dict):
            raise RuntimeError("Hard-Pair Curriculum state has an invalid pairs map")

        restored: dict[str, dict[str, float | int]] = {}
        for key, value in pairs.items():
            margin_ema = float(value["margin_ema"])
            observations = int(value["observations"])
            if observations < 1:
                raise RuntimeError(
                    f"Invalid curriculum observation count for pair {key!r}: "
                    f"{observations}"
                )
            if not torch.isfinite(torch.tensor(margin_ema)):
                raise RuntimeError(
                    f"Non-finite curriculum EMA for pair {key!r}: {margin_ema}"
                )
            restored[str(key)] = {
                "margin_ema": margin_ema,
                "observations": observations,
            }

        self._dpo_curriculum_state = restored
        print(
            "[OT-DPO-CURRICULUM] restored "
            f"{len(restored)} per-pair EMA states"
        )

    @staticmethod
    def _is_dpo_rejected_key(key: str) -> bool:
        return key.endswith("_rejected")

    def _create_dpo_batched_batch(self, batch: dict) -> tuple[dict, int]:
        # The chosen latent is the authoritative batch dimension. Inferring B
        # from arbitrary dict order can pick a metadata list with a different
        # length and silently corrupt the chosen/rejected split.
        latent_image = batch.get("latent_image")
        if not isinstance(latent_image, torch.Tensor) or latent_image.ndim == 0:
            raise RuntimeError(
                "DPO batch must contain a batched latent_image tensor"
            )

        chosen_b = int(latent_image.shape[0])
        if chosen_b <= 0:
            raise RuntimeError("DPO batch is empty")

        rejected_latent = batch.get("latent_image_rejected")
        if not isinstance(rejected_latent, torch.Tensor):
            raise RuntimeError(
                "DPO batch must contain latent_image_rejected as a Tensor"
            )
        if rejected_latent.shape != latent_image.shape:
            raise RuntimeError(
                "DPO latent shape mismatch: "
                f"latent_image {tuple(latent_image.shape)} != "
                f"latent_image_rejected {tuple(rejected_latent.shape)}"
            )

        batched = {}

        rejected_key_map = {
            "latent_image": "latent_image_rejected",
            "image": "image_rejected",
            "image_path": "image_path_rejected",
            "chosen_image_path": "rejected_image_path",
            "chosen_source_path": "rejected_source_path",
        }

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
                    if not isinstance(rejected_value, torch.Tensor):
                        raise TypeError(
                            f"DPO batch key '{key}' is Tensor but rejected key '{rejected_key}' "
                            f"is {type(rejected_value).__name__}"
                        )
                    if value.ndim == 0 or rejected_value.ndim == 0:
                        raise RuntimeError(
                            f"DPO paired tensor keys '{key}'/'{rejected_key}' "
                            "must have a batch dimension"
                        )
                    if int(value.shape[0]) != chosen_b or int(rejected_value.shape[0]) != chosen_b:
                        raise RuntimeError(
                            f"DPO paired tensor keys '{key}'/'{rejected_key}' must both "
                            f"have batch size {chosen_b}, got {value.shape[0]} and "
                            f"{rejected_value.shape[0]}"
                        )
                    if key == "latent_image" and value.shape != rejected_value.shape:
                        raise RuntimeError(
                            "DPO latent shape mismatch: "
                            f"latent_image {tuple(value.shape)} != "
                            f"latent_image_rejected {tuple(rejected_value.shape)}"
                        )
                    batched[key] = torch.cat([value, rejected_value], dim=0)

                elif isinstance(value, list):
                    if isinstance(rejected_value, tuple):
                        rejected_value = list(rejected_value)
                    if not isinstance(rejected_value, list):
                        raise TypeError(
                            f"DPO batch key '{key}' is list but rejected key '{rejected_key}' "
                            f"is {type(rejected_value).__name__}"
                        )
                    if len(value) != chosen_b or len(rejected_value) != chosen_b:
                        raise RuntimeError(
                            f"DPO paired list keys '{key}'/'{rejected_key}' must both "
                            f"have length {chosen_b}, got {len(value)} and "
                            f"{len(rejected_value)}"
                        )
                    batched[key] = value + rejected_value

                elif isinstance(value, tuple):
                    if isinstance(rejected_value, list):
                        rejected_value = tuple(rejected_value)
                    if not isinstance(rejected_value, tuple):
                        raise TypeError(
                            f"DPO batch key '{key}' is tuple but rejected key '{rejected_key}' "
                            f"is {type(rejected_value).__name__}"
                        )
                    if len(value) != chosen_b or len(rejected_value) != chosen_b:
                        raise RuntimeError(
                            f"DPO paired tuple keys '{key}'/'{rejected_key}' must both "
                            f"have length {chosen_b}, got {len(value)} and "
                            f"{len(rejected_value)}"
                        )
                    batched[key] = value + rejected_value

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


    @staticmethod
    def _split_dpo_batched_output(output: dict, chosen_b: int) -> tuple[dict, dict]:
        # Splits a model output dict whose batched tensors have leading dim 2B
        # into chosen-only (first B) and rejected-only (last B) dicts.
        chosen_out: dict = {}
        rejected_out: dict = {}
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

    def set_dpo_runtime_beta(self, beta: float | None):
        # Adaptive-beta override from the trainer. The logged reward metrics
        # are computed before beta is applied, so adapting beta from them does
        # not create a feedback loop.
        self._dpo_runtime_beta = beta

    def rlhf_logp_per_sample(
            self,
            model: BaseModel,
            batch: dict,
            data: dict,
            config: TrainConfig,
    ) -> Tensor:
        # Default DPO likelihood proxy: old raw-MSE behavior.
        #
        # Keep the old memory behavior: do not upcast the full [2B,C,H,W]
        # tensors before subtraction. Only the reduction accumulates in fp32.
        #
        # Model families can override this to use their native per-sample
        # training loss math. Krea overrides this to use _flow_matching_losses(),
        # so DPO follows the same MSE/MAE/log-cosh/Huber/loss-weight/sigma math
        # as normal Krea training.
        predicted = data["predicted"]
        target = data["target"]
        return -(
            predicted - target
        ).pow(2).mean(dim=list(range(1, predicted.ndim)), dtype=torch.float32)

    @staticmethod
    def _validate_rlhf_logp_per_sample(logp: Tensor, chosen_b: int, name: str) -> Tensor:
        # DPO requires exactly one scalar logp proxy per chosen/rejected sample.
        # A scalar mean loss or unreduced spatial tensor would silently corrupt
        # the preference objective, so fail hard.
        if not isinstance(logp, torch.Tensor):
            raise TypeError(
                f"{name} rlhf_logp_per_sample must return a Tensor, "
                f"got {type(logp).__name__}"
            )

        expected_b = 2 * int(chosen_b)
        if logp.ndim != 1 or int(logp.shape[0]) != expected_b:
            raise RuntimeError(
                f"{name} rlhf_logp_per_sample must return shape "
                f"[{expected_b}], got {tuple(logp.shape)}"
            )

        # DPO arithmetic is cheap at [2B], so force stable fp32 margins without
        # creating large fp32 activation copies.
        if logp.dtype != torch.float32:
            logp = logp.float()

        return logp

    @staticmethod
    def _dpo_csv_index_value(value, index: int | None = None):
        if value is None:
            return ""

        if isinstance(value, (str, int, float, bool)):
            return value

        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")

        if hasattr(value, "detach"):
            tensor = value.detach()
            if tensor.numel() == 0:
                return ""

            if index is not None and tensor.ndim > 0 and int(tensor.shape[0]) > index:
                tensor = tensor[index]
            else:
                tensor = tensor.flatten()[0]

            if tensor.numel() == 1:
                item = tensor.detach().cpu().item()
                if isinstance(item, float):
                    return float(item)
                if isinstance(item, int):
                    return int(item)
                if isinstance(item, bool):
                    return bool(item)
                return item

            flat = tensor.detach().cpu().flatten().tolist()
            return "x".join(str(x) for x in flat)

        if isinstance(value, (list, tuple)):
            if index is None:
                if len(value) == 0:
                    return ""
                return BaseModelSetup._dpo_csv_index_value(value[0], None)
            if 0 <= index < len(value):
                return BaseModelSetup._dpo_csv_index_value(value[index], None)
            return ""

        return str(value)

    @staticmethod
    def _dpo_csv_float_value(value, index: int | None = None):
        value = BaseModelSetup._dpo_csv_index_value(value, index)
        if value == "":
            return ""
        try:
            return float(value)
        except Exception:
            return value

    @staticmethod
    def _dpo_csv_neg_float_value(value, index: int | None = None):
        value = BaseModelSetup._dpo_csv_float_value(value, index)
        if value == "":
            return ""
        try:
            return -float(value)
        except Exception:
            return ""

    @staticmethod
    def _dpo_csv_batch_value(batch: dict, names: tuple[str, ...], index: int | None = None):
        for name in names:
            if name in batch:
                return BaseModelSetup._dpo_csv_index_value(batch.get(name), index)
        return ""

    @staticmethod
    def _dpo_csv_concept_value(batch: dict, index: int, key: str):
        flat_name = f"concept.{key}"
        if flat_name in batch:
            return BaseModelSetup._dpo_csv_index_value(batch.get(flat_name), index)

        concept = batch.get("concept")
        if isinstance(concept, (list, tuple)) and 0 <= index < len(concept):
            concept = concept[index]

        if isinstance(concept, dict):
            cur = concept
            for part in key.split("."):
                if not isinstance(cur, dict) or part not in cur:
                    return ""
                cur = cur[part]
            return BaseModelSetup._dpo_csv_index_value(cur, None)

        return ""

    @staticmethod
    def _dpo_csv_path(config: TrainConfig) -> str:
        # Default: write into the current OneTrainer working directory.
        # No environment export needed.
        #
        # Optional override still exists if you ever want it:
        #   OT_DPO_PAIR_CSV_PATH=/some/path.csv
        path = os.environ.get("OT_DPO_PAIR_CSV_PATH", "").strip()
        if path:
            return path
        return os.path.join(os.getcwd(), "dpo_pair_log.csv")

    @staticmethod
    def _dpo_csv_scalar(value):
        if value is None:
            return ""
        if hasattr(value, "detach"):
            if value.detach().numel() == 0:
                return ""
            return float(value.detach().float().mean().cpu().item())
        try:
            return float(value)
        except Exception:
            return str(value)

    def _write_dpo_pair_csv_log(
            self,
            batch: dict,
            config: TrainConfig,
            train_progress: TrainProgress,
            chosen_b: int,
            policy_timestep,
            pair_total_loss,
            chosen_ratio,
            rejected_ratio,
            margin,
            raw_pair_total_loss=None,
            curriculum_weight=None,
            curriculum_margin_ema=None,
            curriculum_observations=None,
            margin_penalty_loss=None,
            wrong_order_penalty_loss=None,
            margin_target_violation=None,
            wrong_order_violation=None,
    ):
        if not multi.is_master():
            return

        chosen_b = int(chosen_b)
        if chosen_b <= 0:
            return

        fieldnames = [
            "global_step",
            "epoch",
            "pair_index",
            "objective",
            "chosen_image_path",
            "rejected_image_path",
            "dpo_pair_key",
            "timestep",
            "chosen_reward",
            "rejected_reward",
            "reward_margin",
            "accuracy",
            "raw_pair_loss",
            "curriculum_weight",
            "curriculum_margin_ema",
            "curriculum_observations",
            "margin_penalty_loss",
            "wrong_order_penalty_loss",
            "margin_target_violation",
            "wrong_order_violation",
            "pair_loss",
        ]

        path = self._dpo_csv_path(config)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        # Preserve an existing old-schema CSV instead of appending rows with a
        # different column order beneath its header.
        if os.path.exists(path) and os.path.getsize(path) > 0:
            with open(path, "r", newline="", encoding="utf-8") as f:
                current_header = next(csv.reader(f), [])

            if current_header != fieldnames:
                legacy_index = 1
                legacy_path = f"{path}.legacy-{legacy_index}"
                while os.path.exists(legacy_path):
                    legacy_index += 1
                    legacy_path = f"{path}.legacy-{legacy_index}"
                os.replace(path, legacy_path)
                print(
                    f"[OT-DPO-PAIR-CSV] moved old-schema log to {legacy_path}"
                )

        write_header = not os.path.exists(path) or os.path.getsize(path) == 0

        rows = []
        for i in range(chosen_b):
            rows.append({
                "global_step": getattr(train_progress, "global_step", ""),
                "epoch": getattr(train_progress, "epoch", ""),
                "pair_index": i,
                "objective": str(
                    getattr(config, "rlhf_dpo_objective", "")
                ),
                "chosen_image_path": self._dpo_csv_batch_value(
                    batch,
                    (
                        "image_path",
                        "chosen_image_path",
                        "chosen_source_path",
                        "chosen_image_path_raw",
                    ),
                    i,
                ),
                "rejected_image_path": self._dpo_csv_batch_value(
                    batch,
                    (
                        "image_path_rejected",
                        "rejected_image_path",
                        "rejected_source_path",
                        "rejected_image_path_raw",
                    ),
                    i,
                ),
                "dpo_pair_key": self._dpo_csv_batch_value(
                    batch,
                    ("dpo_pair_key",),
                    i,
                ),
                "timestep": self._dpo_csv_index_value(
                    policy_timestep,
                    i,
                ),
                "chosen_reward": self._dpo_csv_float_value(
                    chosen_ratio,
                    i,
                ),
                "rejected_reward": self._dpo_csv_float_value(
                    rejected_ratio,
                    i,
                ),
                "reward_margin": self._dpo_csv_float_value(
                    margin,
                    i,
                ),
                "accuracy": float(
                    margin.detach()[i].item() > 0.0
                ),
                "raw_pair_loss": self._dpo_csv_float_value(
                    raw_pair_total_loss,
                    i,
                ),
                "curriculum_weight": self._dpo_csv_float_value(
                    curriculum_weight,
                    i,
                ),
                "curriculum_margin_ema": self._dpo_csv_float_value(
                    curriculum_margin_ema,
                    i,
                ),
                "curriculum_observations": self._dpo_csv_float_value(
                    curriculum_observations,
                    i,
                ),
                "margin_penalty_loss": self._dpo_csv_float_value(
                    margin_penalty_loss,
                    i,
                ),
                "wrong_order_penalty_loss": self._dpo_csv_float_value(
                    wrong_order_penalty_loss,
                    i,
                ),
                "margin_target_violation": self._dpo_csv_float_value(
                    margin_target_violation,
                    i,
                ),
                "wrong_order_violation": self._dpo_csv_float_value(
                    wrong_order_violation,
                    i,
                ),
                "pair_loss": self._dpo_csv_float_value(
                    pair_total_loss,
                    i,
                ),
            })

        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
                extrasaction="ignore",
            )
            if write_header:
                writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _dpo_bad_pair_csv_path() -> str:
        path = os.environ.get("OT_DPO_BAD_PAIR_CSV_PATH", "").strip()
        if path:
            return path
        return os.path.join(os.getcwd(), "dpo_bad_pairs.csv")

    def _write_dpo_bad_pair_csv_log(
            self,
            batch: dict,
            config: TrainConfig,
            train_progress: TrainProgress,
            chosen_b: int,
            policy_timestep,
            pair_total_loss,
            chosen_ratio: Tensor,
            rejected_ratio: Tensor,
            margin: Tensor,
    ):
        """Write only severe DPO outliers to a separate CSV.

        This logger is intentionally independent of TensorBoard and of the
        all-pairs dpo_pair_log.csv. A row is emitted when at least one of these
        occurs:
          * non-finite reward/loss
          * chosen target violation exceeds the configured threshold
          * rejected target violation exceeds the configured threshold
          * per-pair objective loss exceeds the configured threshold
          * the same pair suddenly moves in the wrong direction compared with
            its previous observation in this process
        """
        if not multi.is_master():
            return
        if not bool(getattr(config, "rlhf_dpo_bad_pair_logging", True)):
            return

        chosen_b = int(chosen_b)
        if chosen_b <= 0:
            return

        chosen_target = float(
            getattr(config, "rlhf_dpo_anchored_chosen_target", 0.0)
        )
        rejected_target = float(
            getattr(config, "rlhf_dpo_anchored_rejected_target", -0.05)
        )
        violation_threshold = max(
            float(
                getattr(
                    config,
                    "rlhf_dpo_bad_pair_reward_violation_threshold",
                    2.0,
                )
            ),
            0.0,
        )
        change_threshold = max(
            float(
                getattr(
                    config,
                    "rlhf_dpo_bad_pair_reward_change_threshold",
                    2.0,
                )
            ),
            0.0,
        )
        loss_threshold = max(
            float(
                getattr(config, "rlhf_dpo_bad_pair_loss_threshold", 2.0)
            ),
            0.0,
        )

        chosen_values = chosen_ratio.detach().float().cpu().tolist()
        rejected_values = rejected_ratio.detach().float().cpu().tolist()
        margin_values = margin.detach().float().cpu().tolist()
        if pair_total_loss is None:
            loss_values = [float("nan")] * chosen_b
        else:
            loss_values = (
                pair_total_loss.detach().float().reshape(-1).cpu().tolist()
            )
            if len(loss_values) != chosen_b:
                loss_values = [float("nan")] * chosen_b

        fieldnames = [
            "global_step",
            "epoch",
            "pair_index",
            "objective",
            "reason",
            "chosen_image_path",
            "rejected_image_path",
            "dpo_pair_key",
            "timestep",
            "chosen_reward",
            "rejected_reward",
            "reward_margin",
            "previous_chosen_reward",
            "previous_rejected_reward",
            "chosen_reward_change",
            "rejected_reward_change",
            "chosen_target",
            "rejected_target",
            "chosen_target_violation",
            "rejected_target_violation",
            "pair_loss",
        ]

        rows = []
        global_step = int(getattr(train_progress, "global_step", 0))
        epoch = getattr(train_progress, "epoch", "")

        for i in range(chosen_b):
            chosen_path = str(
                self._dpo_csv_batch_value(
                    batch,
                    (
                        "image_path",
                        "chosen_image_path",
                        "chosen_source_path",
                        "chosen_image_path_raw",
                    ),
                    i,
                )
            )
            rejected_path = str(
                self._dpo_csv_batch_value(
                    batch,
                    (
                        "image_path_rejected",
                        "rejected_image_path",
                        "rejected_source_path",
                        "rejected_image_path_raw",
                    ),
                    i,
                )
            )
            pair_key = str(
                self._dpo_csv_batch_value(batch, ("dpo_pair_key",), i)
            ).strip()
            history_key = pair_key or f"{chosen_path}\n{rejected_path}"

            chosen_value = float(chosen_values[i])
            rejected_value = float(rejected_values[i])
            margin_value = float(margin_values[i])
            pair_loss_value = float(loss_values[i])

            chosen_violation = max(chosen_target - chosen_value, 0.0)
            rejected_violation = max(rejected_value - rejected_target, 0.0)

            previous = self._dpo_bad_pair_previous_rewards.get(history_key)
            previous_chosen = previous[0] if previous is not None else None
            previous_rejected = previous[1] if previous is not None else None
            chosen_change = (
                chosen_value - previous_chosen
                if previous_chosen is not None
                else None
            )
            rejected_change = (
                rejected_value - previous_rejected
                if previous_rejected is not None
                else None
            )

            reasons = []
            finite_values = (
                torch.isfinite(torch.tensor(chosen_value)).item()
                and torch.isfinite(torch.tensor(rejected_value)).item()
                and torch.isfinite(torch.tensor(margin_value)).item()
                and torch.isfinite(torch.tensor(pair_loss_value)).item()
            )
            if not finite_values:
                reasons.append("non_finite")
            if chosen_violation >= violation_threshold and violation_threshold > 0:
                reasons.append("chosen_target_violation")
            if rejected_violation >= violation_threshold and violation_threshold > 0:
                reasons.append("rejected_target_violation")
            if pair_loss_value >= loss_threshold and loss_threshold > 0:
                reasons.append("pair_loss")
            if (
                chosen_change is not None
                and chosen_change <= -change_threshold
                and change_threshold > 0
            ):
                reasons.append("chosen_reward_drop")
            if (
                rejected_change is not None
                and rejected_change >= change_threshold
                and change_threshold > 0
            ):
                reasons.append("rejected_reward_rise")

            # Always update history, including normal observations, so a later
            # catastrophic jump is measured against the immediately preceding
            # observation of this exact pair.
            self._dpo_bad_pair_previous_rewards[history_key] = (
                chosen_value,
                rejected_value,
                global_step,
            )

            if not reasons:
                continue

            rows.append({
                "global_step": global_step,
                "epoch": epoch,
                "pair_index": i,
                "objective": str(
                    getattr(config, "rlhf_dpo_objective", "")
                ),
                "reason": "|".join(reasons),
                "chosen_image_path": chosen_path,
                "rejected_image_path": rejected_path,
                "dpo_pair_key": pair_key,
                "timestep": self._dpo_csv_index_value(policy_timestep, i),
                "chosen_reward": chosen_value,
                "rejected_reward": rejected_value,
                "reward_margin": margin_value,
                "previous_chosen_reward": (
                    "" if previous_chosen is None else previous_chosen
                ),
                "previous_rejected_reward": (
                    "" if previous_rejected is None else previous_rejected
                ),
                "chosen_reward_change": (
                    "" if chosen_change is None else chosen_change
                ),
                "rejected_reward_change": (
                    "" if rejected_change is None else rejected_change
                ),
                "chosen_target": chosen_target,
                "rejected_target": rejected_target,
                "chosen_target_violation": chosen_violation,
                "rejected_target_violation": rejected_violation,
                "pair_loss": pair_loss_value,
            })

        if not rows:
            return

        path = self._dpo_bad_pair_csv_path()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        if os.path.exists(path) and os.path.getsize(path) > 0:
            with open(path, "r", newline="", encoding="utf-8") as f:
                current_header = next(csv.reader(f), [])
            if current_header != fieldnames:
                legacy_index = 1
                legacy_path = f"{path}.legacy-{legacy_index}"
                while os.path.exists(legacy_path):
                    legacy_index += 1
                    legacy_path = f"{path}.legacy-{legacy_index}"
                os.replace(path, legacy_path)
                print(
                    f"[OT-DPO-BAD-PAIR-CSV] moved old-schema log to "
                    f"{legacy_path}"
                )

        write_header = not os.path.exists(path) or os.path.getsize(path) == 0
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
                extrasaction="ignore",
            )
            if write_header:
                writer.writeheader()
            writer.writerows(rows)


    def calculate_dpo_loss(
        self,
        model: BaseModel,
        batch: dict,
        config: TrainConfig,
        train_progress: TrainProgress,
    ) -> Tensor:
        if "latent_image_rejected" not in batch:
            raise RuntimeError(
                "RLHF DPO requires paired chosen/rejected batches, but the dataloader did not provide rejected samples."
            )

        beta = config.rlhf_dpo_beta if self._dpo_runtime_beta is None else self._dpo_runtime_beta
        supervised_loss = None

        # 2 forwards: 1 batched ref (no_grad) + 1 batched policy, each over the
        # [chosen; rejected] batch. Both halves share per-pair timestep+noise via
        # _dpo_paired_half, and ref/policy share them too because predict()
        # seeds its generator from global_step. Note for torch.compile users:
        # supervised/validation batches are B-sized while DPO batches are
        # 2B-sized, so mixing them in one session compiles two graphs.
        batched_input, chosen_b = self._create_dpo_batched_batch(batch)

        self._dpo_paired_half = chosen_b
        try:
            with torch.no_grad(), self.reference_model(model, config):
                ref_output = self.predict(model, batched_input, config, train_progress)
                ref_logp = self.rlhf_logp_per_sample(model, batched_input, ref_output, config)
                ref_logp = self._validate_rlhf_logp_per_sample(ref_logp, chosen_b, "reference")
                ref_chosen_logp = ref_logp[:chosen_b]
                ref_rejected_logp = ref_logp[chosen_b:]
                del ref_output, ref_logp

            policy_output = self.predict(model, batched_input, config, train_progress)
        finally:
            self._dpo_paired_half = None
        policy_timestep = policy_output.get("timestep")
        policy_logp = self.rlhf_logp_per_sample(model, batched_input, policy_output, config)
        policy_logp = self._validate_rlhf_logp_per_sample(policy_logp, chosen_b, "policy")
        policy_chosen_logp = policy_logp[:chosen_b]
        policy_rejected_logp = policy_logp[chosen_b:]
        if config.rlhf_supervised_mix > 0:
            chosen_output, _ = self._split_dpo_batched_output(policy_output, chosen_b)
            supervised_loss = self.calculate_loss(model, batch, chosen_output, config)
            del chosen_output
        del policy_output, policy_logp

        chosen_ratio = policy_chosen_logp - ref_chosen_logp.detach()
        rejected_ratio = policy_rejected_logp - ref_rejected_logp.detach()
        margin = chosen_ratio - rejected_ratio

        # Default for logging and non-decoupled / IPO paths.
        dpo_beta_scale = 1.0
        chosen_reward_push_loss = None
        chosen_reward_floor_loss = None
        chosen_reward_aux_loss = None
        chosen_anchor_weight = 0.0
        chosen_reward_floor_value = float(getattr(config, "rlhf_dpo_chosen_reward_floor", 0.0))
        pair_total_loss = None
        raw_pair_total_loss = None
        curriculum_weight = torch.ones_like(margin)
        curriculum_margin_ema = margin.detach()
        curriculum_observations = torch.zeros_like(margin)
        margin_penalty_loss = torch.zeros_like(margin)
        wrong_order_penalty_loss = torch.zeros_like(margin)
        margin_target_violation = torch.zeros_like(margin)
        wrong_order_violation = torch.zeros_like(margin)

        if config.rlhf_dpo_objective == DPOObjective.ANCHORED_REJECT:
            # Independent chosen/rejected anchors remain the primary safety
            # constraints. Two bounded pairwise terms additionally require a
            # positive target margin and add extra rescue pressure while the
            # rejected sample is ranked above the chosen sample.
            chosen_target = float(
                getattr(config, "rlhf_dpo_anchored_chosen_target", 0.0)
            )
            rejected_target = float(
                getattr(config, "rlhf_dpo_anchored_rejected_target", -0.05)
            )
            chosen_weight = max(
                float(getattr(config, "rlhf_dpo_anchored_chosen_weight", 1.0)),
                0.0,
            )
            rejected_weight = max(
                float(getattr(config, "rlhf_dpo_anchored_rejected_weight", 1.0)),
                0.0,
            )
            huber_delta = max(
                float(getattr(config, "rlhf_dpo_anchored_huber_delta", 0.1)),
                1e-8,
            )
            margin_target = max(
                float(getattr(config, "rlhf_dpo_anchored_margin_target", 0.05)),
                0.0,
            )
            margin_weight = max(
                float(getattr(config, "rlhf_dpo_anchored_margin_weight", 0.5)),
                0.0,
            )
            wrong_order_weight = max(
                float(getattr(config, "rlhf_dpo_anchored_wrong_order_weight", 0.5)),
                0.0,
            )

            chosen_violation = F.relu(chosen_target - chosen_ratio)
            rejected_violation = F.relu(rejected_ratio - rejected_target)

            chosen_pair_loss = chosen_weight * F.smooth_l1_loss(
                chosen_violation,
                torch.zeros_like(chosen_violation),
                beta=huber_delta,
                reduction="none",
            )
            rejected_pair_loss = rejected_weight * F.smooth_l1_loss(
                rejected_violation,
                torch.zeros_like(rejected_violation),
                beta=huber_delta,
                reduction="none",
            )
            # Positive-margin term: active whenever the current policy
            # margin is below the configured target, including small positive
            # margins that are correctly ordered but not yet decisive.
            margin_target_violation = F.relu(margin_target - margin)
            margin_penalty_loss = margin_weight * F.smooth_l1_loss(
                margin_target_violation,
                torch.zeros_like(margin_target_violation),
                beta=huber_delta,
                reduction="none",
            )

            # Wrong-order rescue: an additional bounded penalty only while the
            # rejected image is preferred over the chosen image. Negative
            # margins therefore receive both the target-margin term and this
            # extra correction.
            wrong_order_violation = F.relu(-margin)
            wrong_order_penalty_loss = wrong_order_weight * F.smooth_l1_loss(
                wrong_order_violation,
                torch.zeros_like(wrong_order_violation),
                beta=huber_delta,
                reduction="none",
            )

            raw_pair_total_loss = (
                chosen_pair_loss
                + rejected_pair_loss
                + margin_penalty_loss
                + wrong_order_penalty_loss
            )

            if self._dpo_hard_pair_curriculum_enabled(config):
                (
                    curriculum_weight,
                    curriculum_margin_ema,
                    curriculum_observations,
                ) = self._stage_dpo_curriculum_observations(
                    batch,
                    config,
                    margin,
                )
                # The curriculum weight is detached state derived from
                # a per-pair EMA. There is no gradient path through confidence,
                # so the model cannot reduce its loss by intentionally keeping
                # confidence low.
                pair_total_loss = (
                    raw_pair_total_loss * curriculum_weight.detach()
                )
            else:
                pair_total_loss = raw_pair_total_loss

            dpo_loss = pair_total_loss.mean()
            loss = dpo_loss

        elif config.rlhf_dpo_objective == DPOObjective.IPO:
            dpo_loss = (margin - 1.0 / (2.0 * config.rlhf_dpo_ipo_tau)).pow(2).mean()
            loss = dpo_loss
        else:
            logits = beta * margin
            dpo_loss = -F.logsigmoid(logits).mean()
            preference_loss = dpo_loss

            if config.rlhf_dpo_label_smoothing > 0:
                label_smoothing = config.rlhf_dpo_label_smoothing
                preference_loss = (
                    (1.0 - label_smoothing) * preference_loss
                    + label_smoothing * (-F.logsigmoid(-logits).mean())
                )

            if getattr(config, "rlhf_dpo_beta_gradient_decouple", False):
                beta_for_scale = float(beta.detach().item()) if isinstance(beta, torch.Tensor) else float(beta)
                beta_ref = getattr(config, "rlhf_dpo_beta_gradient_reference", None)
                if beta_ref is None or float(beta_ref) <= 0:
                    beta_ref = float(config.rlhf_dpo_beta)
                dpo_beta_scale = float(beta_ref) / max(beta_for_scale, 1e-12)

                # Value-preserving gradient scaling:
                # forward/logged loss stays equal to preference_loss,
                # backward gradient is scaled by dpo_beta_scale.
                preference_loss = (
                    preference_loss.detach()
                    + dpo_beta_scale * (preference_loss - preference_loss.detach())
                )

            loss = preference_loss

        # The legacy chosen anchor remains available for SIGMOID/IPO configs,
        # but is never stacked on top of the independent Anchored Reject loss.
        if (
            config.rlhf_dpo_objective != DPOObjective.ANCHORED_REJECT
            and getattr(config, "rlhf_dpo_chosen_reward_anchor", False)
        ):
            chosen_anchor_weight = float(getattr(config, "rlhf_dpo_chosen_reward_anchor_weight", 0.0))
            if chosen_anchor_weight > 0:
                chosen_reward_target = float(getattr(config, "rlhf_dpo_chosen_reward_target", 0.05))
                chosen_reward_floor_value = float(getattr(config, "rlhf_dpo_chosen_reward_floor", 0.0))
                chosen_reward_floor_multiplier = float(
                    getattr(config, "rlhf_dpo_chosen_reward_floor_multiplier", 4.0)
                )
                chosen_reward_sharpness = max(
                    float(getattr(config, "rlhf_dpo_chosen_reward_sharpness", 20.0)),
                    1e-6,
                )

                chosen_reward_push_loss = F.softplus(
                    (chosen_reward_target - chosen_ratio) * chosen_reward_sharpness
                ).mean() / chosen_reward_sharpness

                chosen_reward_floor_violation = F.relu(chosen_reward_floor_value - chosen_ratio)
                chosen_reward_floor_loss = (
                    chosen_reward_floor_violation.mean()
                    + chosen_reward_sharpness * chosen_reward_floor_violation.pow(2).mean()
                )

                chosen_reward_aux_loss = chosen_anchor_weight * (
                    chosen_reward_push_loss
                    + chosen_reward_floor_multiplier * chosen_reward_floor_loss
                )
                loss = loss + chosen_reward_aux_loss

        # DPO-side logged loss includes beta-grad scaling and chosen-anchor aux loss,
        # but does not include supervised_mix.
        dpo_logged_loss = loss

        if supervised_loss is not None:
            loss = loss + config.rlhf_supervised_mix * supervised_loss
            del supervised_loss

        try:
            if config.rlhf_dpo_objective == DPOObjective.ANCHORED_REJECT:
                # Already computed above as the exact per-pair objective.
                pass
            elif config.rlhf_dpo_objective == DPOObjective.IPO:
                pair_total_loss = (
                    margin
                    - 1.0 / (2.0 * config.rlhf_dpo_ipo_tau)
                ).pow(2)
            else:
                pair_logits = beta * margin
                pair_total_loss = -F.logsigmoid(pair_logits)

                if config.rlhf_dpo_label_smoothing > 0:
                    label_smoothing = config.rlhf_dpo_label_smoothing
                    pair_total_loss = (
                        (1.0 - label_smoothing) * pair_total_loss
                        + label_smoothing
                        * (-F.logsigmoid(-pair_logits))
                    )
        except Exception as e:
            print(
                "[OT-DPO-PAIR-CSV] failed to compute per-pair loss: "
                f"{type(e).__name__}: {e}"
            )

        if raw_pair_total_loss is None:
            raw_pair_total_loss = pair_total_loss

        try:
            self._write_dpo_pair_csv_log(
                batch=batch,
                config=config,
                train_progress=train_progress,
                chosen_b=chosen_b,
                policy_timestep=policy_timestep,
                pair_total_loss=pair_total_loss,
                chosen_ratio=chosen_ratio,
                rejected_ratio=rejected_ratio,
                margin=margin,
                raw_pair_total_loss=raw_pair_total_loss,
                curriculum_weight=curriculum_weight,
                curriculum_margin_ema=curriculum_margin_ema,
                curriculum_observations=curriculum_observations,
                margin_penalty_loss=margin_penalty_loss,
                wrong_order_penalty_loss=wrong_order_penalty_loss,
                margin_target_violation=margin_target_violation,
                wrong_order_violation=wrong_order_violation,
            )
        except Exception as e:
            print(
                "[OT-DPO-PAIR-CSV] failed to write row: "
                f"{type(e).__name__}: {e}"
            )

        try:
            self._write_dpo_bad_pair_csv_log(
                batch=batch,
                config=config,
                train_progress=train_progress,
                chosen_b=chosen_b,
                policy_timestep=policy_timestep,
                pair_total_loss=pair_total_loss,
                chosen_ratio=chosen_ratio,
                rejected_ratio=rejected_ratio,
                margin=margin,
            )
        except Exception as e:
            print(
                "[OT-DPO-BAD-PAIR-CSV] failed to write row: "
                f"{type(e).__name__}: {e}"
            )

        self._last_dpo_metrics = {
            "objective_loss": dpo_logged_loss.detach().item(),
            "chosen_reward": chosen_ratio.detach().mean().item(),
            "rejected_reward": rejected_ratio.detach().mean().item(),
            "reward_margin": margin.detach().mean().item(),
            "accuracy": (margin.detach() > 0).float().mean().item(),
            "hard_pair_curriculum_weight": curriculum_weight.detach().mean().item(),
            "hard_pair_margin_ema": curriculum_margin_ema.detach().mean().item(),
            "hard_pair_observations": curriculum_observations.detach().mean().item(),
            "margin_penalty_loss": margin_penalty_loss.detach().mean().item(),
            "wrong_order_penalty_loss": wrong_order_penalty_loss.detach().mean().item(),
            "margin_target_violation": margin_target_violation.detach().mean().item(),
            "wrong_order_violation": wrong_order_violation.detach().mean().item(),
            "chosen_anchor_active": float(
                chosen_reward_aux_loss is not None
            ),
            "chosen_anchor_weight": float(chosen_anchor_weight),
            "chosen_anchor_floor": float(chosen_reward_floor_value),
            "chosen_anchor_push_loss": (
                chosen_reward_push_loss.detach().item()
                if chosen_reward_push_loss is not None
                else 0.0
            ),
            "chosen_anchor_floor_loss": (
                chosen_reward_floor_loss.detach().item()
                if chosen_reward_floor_loss is not None
                else 0.0
            ),
            "chosen_anchor_aux_loss": (
                chosen_reward_aux_loss.detach().item()
                if chosen_reward_aux_loss is not None
                else 0.0
            ),
        }

        return loss

    def stop_embedding_training_elapsed(
            self,
            config: TrainEmbeddingConfig,
            train_progress: TrainProgress,
    ):
        return self.single_action_elapsed(
            "stop_embedding_training_" + str(config.uuid),
            config.stop_training_after,
            config.stop_training_after_unit,
            train_progress,
        )

    def __stop_model_part_training_elapsed(
            self,
            unique_name: str,
            config: TrainModelPartConfig,
            train_progress: TrainProgress,
    ):
        return self.single_action_elapsed(
            "stop_" + unique_name + "_training",
            config.stop_training_after,
            config.stop_training_after_unit,
            train_progress,
        )

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

    def initialize_dpo_reference(
            self,
            model: BaseModel,
            config: TrainConfig,
            snapshot_path: str | None = None,
    ):
        """Initialize a stable existing-adapter reference before training.

        The old implementation captured the reference lazily on the first DPO
        batch, so ordinary training steps before that batch changed the anchor.
        This method is called after the model is on the training device and can
        restore the original snapshot from an OT backup.
        """
        if not getattr(config, "rlhf_enabled", False):
            return
        if config.effective_dpo_ref_mode() != DPORefMode.EXISTING_ADAPTER:
            return
        if self._dpo_ref_params is not None:
            return

        adapters = list(model.adapters())
        if len(adapters) == 0:
            raise RuntimeError(
                "RLHF DPO existing-adapter reference requires active adapters"
            )

        loaded_groups = None

        if snapshot_path and not os.path.isfile(snapshot_path):
            raise RuntimeError(
                "[DPO] resume backup is missing its fixed reference: "
                f"{snapshot_path}. Refusing to replace it with the resumed "
                "policy because that changes the DPO objective."
            )
        if snapshot_path and os.path.isfile(snapshot_path):
            try:
                payload = torch.load(
                    snapshot_path,
                    map_location="cpu",
                    weights_only=True,
                )
            except TypeError:
                payload = torch.load(snapshot_path, map_location="cpu")

            if isinstance(payload, dict):
                loaded_groups = payload.get("adapter_parameters")
            else:
                loaded_groups = payload

            if not isinstance(loaded_groups, (list, tuple)):
                raise RuntimeError(
                    f"Invalid DPO reference snapshot: {snapshot_path}"
                )
            if len(loaded_groups) != len(adapters):
                raise RuntimeError(
                    "DPO reference snapshot adapter count mismatch: "
                    f"snapshot={len(loaded_groups)}, model={len(adapters)}"
                )

        snapshot_groups = []
        for adapter_index, adapter in enumerate(adapters):
            parameters = list(adapter.parameters())
            loaded_parameters = (
                loaded_groups[adapter_index]
                if loaded_groups is not None
                else None
            )

            if loaded_parameters is not None and len(loaded_parameters) != len(parameters):
                raise RuntimeError(
                    "DPO reference snapshot parameter count mismatch for "
                    f"adapter {adapter_index}: snapshot={len(loaded_parameters)}, "
                    f"model={len(parameters)}"
                )

            group = []
            for parameter_index, parameter in enumerate(parameters):
                if loaded_parameters is None:
                    reference = parameter.detach().clone()
                else:
                    source = loaded_parameters[parameter_index]
                    if not isinstance(source, torch.Tensor):
                        raise RuntimeError(
                            "DPO reference snapshot contains a non-tensor at "
                            f"adapter {adapter_index}, parameter {parameter_index}"
                        )
                    if tuple(source.shape) != tuple(parameter.shape):
                        raise RuntimeError(
                            "DPO reference snapshot shape mismatch at adapter "
                            f"{adapter_index}, parameter {parameter_index}: "
                            f"snapshot={tuple(source.shape)}, "
                            f"model={tuple(parameter.shape)}"
                        )
                    reference = source.to(
                        device=parameter.device,
                        dtype=parameter.dtype,
                    ).clone()

                group.append(reference)
            snapshot_groups.append(group)

        self._dpo_ref_params = snapshot_groups

        if snapshot_path and loaded_groups is None:
            raise RuntimeError(
                "[DPO] failed to restore the saved fixed reference from "
                f"{snapshot_path}. Refusing unsafe DPO resume."
            )

        if loaded_groups is not None:
            print(f"[OT-RLHF] restored fixed DPO reference from {snapshot_path}")
        else:
            print("[OT-RLHF] captured fixed existing-adapter DPO reference")

    def save_dpo_reference(self, snapshot_path: str):
        if self._dpo_ref_params is None:
            return

        os.makedirs(os.path.dirname(snapshot_path) or ".", exist_ok=True)
        payload = {
            "version": 1,
            "adapter_parameters": [
                [parameter.detach().cpu().clone() for parameter in group]
                for group in self._dpo_ref_params
            ],
        }
        torch.save(payload, snapshot_path)

    @contextmanager
    def reference_model(self, model: BaseModel, config: TrainConfig):
        adapters = model.adapters()

        if config.training_method is not TrainingMethod.LORA:
            raise NotImplementedError(
                "RLHF DPO reference modes are currently only implemented for adapter training in the LoRA tab."
            )
        if len(adapters) == 0:
            raise RuntimeError(
                "RLHF DPO requires active adapters, but no trainable adapters are attached to the current model."
            )

        ref_mode = config.effective_dpo_ref_mode()

        if ref_mode == DPORefMode.NEW_ADAPTER:
            for adapter in adapters:
                adapter.remove_hook_from_module()
            try:
                yield
            finally:
                for adapter in adapters:
                    adapter.hook_to_module()

        elif ref_mode == DPORefMode.EXISTING_ADAPTER:
            # Fallback for callers outside GenericTrainer. GenericTrainer
            # initializes this before the first optimizer step and restores it
            # from backups when available.
            if self._dpo_ref_params is None:
                self.initialize_dpo_reference(model, config)

            if self._dpo_ref_params is None:
                raise RuntimeError(
                    "Existing-adapter DPO reference was not initialized"
                )
            if len(self._dpo_ref_params) != len(adapters):
                raise RuntimeError(
                    "Existing-adapter DPO reference adapter count changed"
                )

            # Preserve Parameter storage so optimizer/DDP hooks do not see
            # data-pointer replacement. Adapter tensors are small enough that a
            # temporary policy clone is safer than swapping .data references.
            policy_values = [
                [parameter.detach().clone() for parameter in adapter.parameters()]
                for adapter in adapters
            ]
            try:
                with torch.no_grad():
                    for adapter_index, (adapter, ref_params) in enumerate(
                            zip(adapters, self._dpo_ref_params, strict=True)
                    ):
                        parameters = list(adapter.parameters())
                        if len(parameters) != len(ref_params):
                            raise RuntimeError(
                                "Existing-adapter DPO reference parameter count "
                                f"changed for adapter {adapter_index}"
                            )

                        for parameter_index, (parameter, ref_data) in enumerate(
                                zip(parameters, ref_params, strict=True)
                        ):
                            if tuple(parameter.shape) != tuple(ref_data.shape):
                                raise RuntimeError(
                                    "Existing-adapter DPO reference shape changed at "
                                    f"adapter {adapter_index}, parameter {parameter_index}"
                                )
                            if (
                                ref_data.device != parameter.device
                                or ref_data.dtype != parameter.dtype
                            ):
                                ref_data = ref_data.to(
                                    device=parameter.device,
                                    dtype=parameter.dtype,
                                )
                                self._dpo_ref_params[adapter_index][parameter_index] = ref_data
                            parameter.copy_(ref_data)
                yield
            finally:
                with torch.no_grad():
                    for adapter, saved_values in zip(
                            adapters, policy_values, strict=True
                    ):
                        for parameter, saved_value in zip(
                                adapter.parameters(), saved_values, strict=True
                        ):
                            parameter.copy_(saved_value)
        else:
            raise ValueError(f"Unsupported DPO reference mode: {ref_mode}")

    def _create_model_part_parameters(
        self,
        parameter_group_collection: NamedParameterGroupCollection,
        unique_name: str,
        model: torch.nn.Module,
        config: TrainModelPartConfig,
        freeze: list[ModuleFilter] | None = None,
        debug: bool = False,
    ):
        if not config.train:
            return

        if freeze is not None and len(freeze) > 0:
            selected = []
            deselected = []
            parameters = []
            self.frozen_parameters[unique_name] = []
            for name, param in model.named_parameters():
                if any(f.matches(name) for f in freeze):
                    parameters.append(param)
                    selected.append(name)
                else:
                    self.frozen_parameters[unique_name].append(param)
                    deselected.append(name)

            if debug:
                print(f"Selected layers: {selected}")
                print(f"Deselected layers: {deselected}")
            else:
                print(f"Selected layers: {len(selected)}")
                print(f"Deselected layers: {len(deselected)}")
                print("Note: Enable Debug mode to see the full list of layer names")
        else:
            parameters = model.parameters()

        parameter_group_collection.add_group(NamedParameterGroup(
            unique_name=unique_name,
            parameters=parameters,
            learning_rate=config.learning_rate,
        ))

    def _setup_model_part_requires_grad(
        self,
        unique_name: str,
        model: torch.nn.Module,
        config: TrainModelPartConfig,
        train_progress: TrainProgress,
    ):
        if model is not None:
            train_model_part = config.train and \
                               not self.__stop_model_part_training_elapsed(unique_name, config, train_progress)
            model.requires_grad_(train_model_part)

            #even if frozen parameters are not passed to the optimizer, required_grad has to be False.
            #otherwise, gradients accumulate in param.grad and waste vram
            if unique_name in self.frozen_parameters:
                for param in self.frozen_parameters[unique_name]:
                    param.requires_grad_(False)

    @staticmethod
    def _set_attention_backend(component, attn: AttentionMechanism, mask: bool):
        match attn:
            case AttentionMechanism.SDP:
                component.set_attention_backend("native")

            case AttentionMechanism.FLASH:
                backend = "flash" if mask else "flash"
                print(f"Attention backend: {backend}")
                component.set_attention_backend(backend)

            case AttentionMechanism.CUDNN:
                component.set_attention_backend("_native_cudnn")

            case _:
                raise NotImplementedError(
                    f"attention mechanism {str(attn)} not implemented"
                )

