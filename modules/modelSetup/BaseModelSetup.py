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
        # Linear-DPO owns a moving CPU-fp32 adapter reference. It is separate
        # from both fixed DPO snapshots and any model-family EMA (for example,
        # Self-Flow's representation teacher).
        self._dpo_ema_ref_params_cpu = None
        self._dpo_ema_policy_cpu_buffers = None
        self._dpo_ema_ref_decay = None
        self._dpo_ema_ref_steps = 0
        # Explicit per-concept snapshots are intentionally separate from the
        # RLHF-tab-wide existing-adapter reference.  Otherwise a Sigmoid
        # concept added after an Anchored Reject phase silently compares
        # against the old Anchored snapshot.
        self._dpo_concept_ref_params: dict[str, list[list[Tensor]]] = {}
        self._dpo_concept_ref_params_cpu: dict[str, list[list[Tensor]]] = {}
        self._dpo_policy_cpu_buffers = None
        self._last_dpo_metrics = None
        # Detached per-pair objective losses from the most recent DPO dispatch.
        # GenericTrainer consumes these only after a successful optimizer step
        # to drive the optional Adaptive DPO Dataset sampler.
        self._last_dpo_pair_losses: list[tuple[str, float, str]] = []
        self._dpo_paired_half = None
        # Context-local because checkpoint recomputation may be scheduled by
        # autograd independently for the two policy branches.
        self._dpo_stream_active = ContextVar(
            f"ot_dpo_stream_active_{id(self)}",
            default=False,
        )
        # Reference prediction must be distinguishable from the policy pass.
        # Self-Flow uses this to avoid nesting its EMA-teacher adapter swap
        # inside the fixed DPO-reference adapter swap.
        self._dpo_reference_active = ContextVar(
            f"ot_dpo_reference_active_{id(self)}",
            default=False,
        )
        self._dpo_runtime_beta = None
        # Previous per-pair rewards are kept only for the current process so the
        # separate bad-pair CSV can detect sudden wrong-direction jumps.
        self._dpo_bad_pair_previous_rewards: dict[str, tuple[float, float, int]] = {}
        # Hard-pair curriculum state is committed only after a successful
        # optimizer step. Pending observations belong to the current gradient-
        # accumulation window and are never written into a backup.
        self._dpo_curriculum_state: dict[
            str,
            dict[str, float | int | str],
        ] = {}
        self._dpo_curriculum_pending: dict[
            str,
            dict[str, float | int | str],
        ] = {}
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

    def after_backward(
            self,
            model: BaseModel,
            config: TrainConfig,
            train_progress: TrainProgress,
    ):
        """Optional post-backward lifecycle hook for timing/instrumentation."""
        pass

    def after_streamed_dpo_branch_backward(
            self,
            model: BaseModel,
            config: TrainConfig,
            train_progress: TrainProgress,
    ):
        """Optional hook after one streamed DPO branch replay is differentiated."""
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
        # Curriculum is a detached per-pair confidence gate, not part of any
        # particular preference objective. Keeping this objective-agnostic also
        # makes it work for per-concept objective dispatches in mixed batches.
        return bool(getattr(config, "rlhf_dpo_hard_pair_curriculum", False))

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
        }

    @staticmethod
    def _dpo_normalize_pair_path(path: str) -> str:
        return os.path.normcase(
            os.path.realpath(
                os.path.abspath(os.path.expanduser(str(path)))
            )
        )

    def _dpo_pair_identity(self, batch: dict, index: int) -> str:
        # Prefer an already path-qualified key. This is important in cache-only
        # mode, where legacy diagnostic source-path fields may be reconstructed
        # approximately while the cached pair key remains exact.
        pair_key = str(
            self._dpo_csv_batch_value(
                batch,
                ("dpo_pair_key",),
                index,
            )
            or ""
        ).strip()
        if (
            pair_key.startswith("dpo-pair-path-v1\nchosen=")
            and "\nrejected=" in pair_key
        ):
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

        if chosen_path and rejected_path:
            chosen_path = self._dpo_normalize_pair_path(chosen_path)
            rejected_path = self._dpo_normalize_pair_path(rejected_path)
            return (
                "dpo-pair-path-v1\n"
                f"chosen={chosen_path}\n"
                f"rejected={rejected_path}"
            )

        if pair_key:
            raise RuntimeError(
                "DPO received a legacy basename-only dpo_pair_key "
                f"{pair_key!r} without both source paths. Rebuild the DPO "
                "cache so pair identities are path-qualified."
            )

        raise RuntimeError(
            "DPO pair identity requires both chosen and rejected source paths, "
            "or a path-qualified dpo_pair_key."
        )

    def _dpo_curriculum_pair_key(self, batch: dict, index: int) -> str:
        # Compatibility wrapper for the existing hard-pair curriculum.
        return self._dpo_pair_identity(batch, index)

    @staticmethod
    def _dpo_curriculum_competence(
            objective: DPOObjective,
            config: TrainConfig,
            margin: Tensor,
            policy_chosen_score: Tensor,
            policy_rejected_score: Tensor,
    ) -> tuple[Tensor, float | None]:
        """Return the objective-appropriate curriculum signal and threshold.

        A moving-reference margin is not a universal competence measure:
        Linear-DPO's reference catches up by design, and IPO has a finite
        optimum.  The shared curriculum therefore keeps one gating mechanism
        while choosing a mathematically meaningful signal for each objective.
        """
        objective = DPOObjective(objective)
        if objective == DPOObjective.LINEAR:
            # Linear-DPO's policy-vs-EMA reward margin converges back toward
            # zero as the EMA reference catches up.  The policy's direct
            # chosen-vs-rejected score gap remains positive when it ranks the
            # pair correctly and is independent of EMA lag.
            return policy_chosen_score - policy_rejected_score, None

        if objective == DPOObjective.IPO:
            tau = float(config.rlhf_dpo_ipo_tau)
            if not math.isfinite(tau) or tau <= 0.0:
                raise ValueError(f"IPO Tau must be finite and > 0, got {tau}")
            target_margin = 1.0 / (2.0 * tau)
            configured_full_margin = float(
                getattr(
                    config,
                    "rlhf_dpo_hard_pair_curriculum_full_margin",
                    0.05,
                )
            )
            # Never require IPO to overshoot its own optimum merely to reach
            # full curriculum weight. A smaller user threshold remains valid.
            return margin, min(configured_full_margin, target_margin)

        # Sigmoid DPO and Anchored Reject both improve monotonically with the
        # policy-vs-reference chosen/rejected reward margin.
        return margin, None

    def _stage_dpo_curriculum_observations(
            self,
            batch: dict,
            config: TrainConfig,
            competence: Tensor,
            objective: DPOObjective,
            full_margin_override: float | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        settings = self._dpo_curriculum_settings(config)
        ema_decay = settings["ema_decay"]
        minimum_weight = settings["minimum_weight"]
        full_margin = (
            settings["full_margin"]
            if full_margin_override is None
            else float(full_margin_override)
        )
        if full_margin <= 0.0:
            raise ValueError(
                "DPO curriculum effective Full Margin must be > 0, "
                f"got {full_margin} for {objective}"
            )
        objective_name = str(DPOObjective(objective))

        detached_margin = competence.detach().float().reshape(-1)
        weights: list[float] = []
        margin_emas: list[float] = []
        observations: list[float] = []

        for index, current_tensor in enumerate(detached_margin):
            pair_key = self._dpo_curriculum_pair_key(batch, index)
            current_margin = float(current_tensor.cpu().item())
            if not math.isfinite(current_margin):
                raise RuntimeError(
                    "DPO curriculum received a non-finite competence value "
                    f"for pair {pair_key!r}: {current_margin}"
                )

            previous = self._dpo_curriculum_pending.get(pair_key)
            if previous is None:
                previous = self._dpo_curriculum_state.get(pair_key)
            if (
                previous is not None
                and str(previous.get("objective", "")) != objective_name
            ):
                # A changed global/per-concept objective changes the meaning
                # and scale of the competence signal. Never reuse its EMA as
                # though it belonged to the new objective.
                previous = None

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
                "objective": objective_name,
            }

            progress = max(0.0, min(1.0, margin_ema / full_margin))
            smooth_progress = progress * progress * (3.0 - 2.0 * progress)
            weight = minimum_weight + (1.0 - minimum_weight) * smooth_progress
            weights.append(weight)
            margin_emas.append(margin_ema)
            observations.append(float(count))

        return (
            torch.tensor(
                weights,
                device=competence.device,
                dtype=competence.dtype,
            ),
            torch.tensor(
                margin_emas,
                device=competence.device,
                dtype=competence.dtype,
            ),
            torch.tensor(
                observations,
                device=competence.device,
                dtype=competence.dtype,
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
            "version": 4,
            "settings": self._dpo_curriculum_settings(config),
            "pairs": {
                key: {
                    "margin_ema": float(value["margin_ema"]),
                    "observations": int(value["observations"]),
                    "objective": str(value["objective"]),
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
        if state_version not in (1, 2, 3, 4):
            raise RuntimeError(
                "Unsupported Hard-Pair Curriculum state version: "
                f"{payload.get('version')}"
            )

        expected = self._dpo_curriculum_settings(config)
        saved = payload.get("settings", {})
        # v2/v3 stored Anchored-Reject-only settings that never affected the
        # curriculum weight itself. Ignore those extra keys when restoring the
        # now objective-agnostic v4 state.
        settings_to_check = tuple(expected.keys())
        mismatches: list[str] = []
        for name in settings_to_check:
            expected_value = expected[name]
            saved_value = saved.get(name)

            try:
                matches = (
                    saved_value is not None
                    and abs(float(saved_value) - expected_value) <= 1e-12
                )
            except (TypeError, ValueError):
                matches = False

            if not matches:
                mismatches.append(
                    f"{name}: backup={saved_value!r}, current={expected_value!r}"
                )

        if mismatches:
            print(
                "[OT-DPO-CURRICULUM] WARNING: curriculum settings changed "
                "across resume: " + "; ".join(mismatches) + ". Applying the "
                "current settings without aborting the model resume."
            )
        if state_version == 1:
            print(
                "[OT-DPO-CURRICULUM] Loading legacy v1 EMA state. Pairwise "
                "margin penalties use the current configuration; backups "
                "created after this point will use objective-agnostic v4 "
                "settings."
            )

        pairs = payload.get("pairs", {})
        if not isinstance(pairs, dict):
            raise RuntimeError("Hard-Pair Curriculum state has an invalid pairs map")
        if state_version < 3:
            legacy_pair_count = len(pairs)
            pairs = {}
            print(
                "[OT-DPO-CURRICULUM] WARNING: discarded "
                f"{legacy_pair_count} legacy basename-keyed EMA states. "
                "They cannot be safely split across duplicate filenames; "
                "only curriculum EMA history is being reset."
            )

        restored: dict[str, dict[str, float | int | str]] = {}
        for key, value in pairs.items():
            margin_ema = float(value["margin_ema"])
            observations = int(value["observations"])
            objective_name = (
                str(DPOObjective.ANCHORED_REJECT)
                if state_version <= 3
                else str(value.get("objective", ""))
            )
            try:
                objective_name = str(DPOObjective(objective_name))
            except ValueError as exc:
                raise RuntimeError(
                    f"Invalid curriculum objective for pair {key!r}: "
                    f"{objective_name!r}"
                ) from exc
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
                "objective": objective_name,
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

    def _create_dpo_stream_batches(self, batch: dict) -> tuple[dict, dict, int]:
        """Build B-sized chosen/rejected batches without allocating a 2B tensor.

        The key mapping intentionally mirrors _create_dpo_batched_batch. Shared
        conditioning and metadata are reused; paired rejected values are exposed
        under the ordinary key names expected by predict().
        """
        latent_image = batch.get("latent_image")
        rejected_latent = batch.get("latent_image_rejected")
        if not isinstance(latent_image, torch.Tensor) or latent_image.ndim == 0:
            raise RuntimeError(
                "Streamed DPO batch must contain a batched latent_image tensor"
            )
        if not isinstance(rejected_latent, torch.Tensor):
            raise RuntimeError(
                "Streamed DPO batch must contain latent_image_rejected as a Tensor"
            )
        if rejected_latent.shape != latent_image.shape:
            raise RuntimeError(
                "Streamed DPO latent shape mismatch: "
                f"latent_image {tuple(latent_image.shape)} != "
                f"latent_image_rejected {tuple(rejected_latent.shape)}"
            )

        chosen_b = int(latent_image.shape[0])
        if chosen_b <= 0:
            raise RuntimeError("Streamed DPO batch is empty")

        rejected_key_map = {
            "latent_image": "latent_image_rejected",
            "image": "image_rejected",
            "image_path": "image_path_rejected",
            "chosen_image_path": "rejected_image_path",
            "chosen_source_path": "rejected_source_path",
        }
        chosen: dict = {}
        rejected: dict = {}

        for key, value in batch.items():
            if key.endswith("_rejected") or key.startswith("rejected_"):
                continue

            rejected_key = rejected_key_map.get(key)
            if rejected_key is None and key.startswith("chosen_"):
                candidate = "rejected_" + key[len("chosen_"):]
                if candidate in batch:
                    rejected_key = candidate

            rejected_value = (
                batch[rejected_key]
                if rejected_key is not None and rejected_key in batch
                else value
            )

            if rejected_key is not None and rejected_key in batch:
                if isinstance(value, torch.Tensor):
                    if not isinstance(rejected_value, torch.Tensor):
                        raise TypeError(
                            f"DPO batch key '{key}' is Tensor but rejected key "
                            f"'{rejected_key}' is {type(rejected_value).__name__}"
                        )
                    if (
                        value.ndim == 0
                        or rejected_value.ndim == 0
                        or int(value.shape[0]) != chosen_b
                        or int(rejected_value.shape[0]) != chosen_b
                    ):
                        raise RuntimeError(
                            f"DPO paired tensor keys '{key}'/'{rejected_key}' "
                            f"must both have batch size {chosen_b}"
                        )
                elif isinstance(value, (list, tuple)):
                    if not isinstance(rejected_value, (list, tuple)):
                        raise TypeError(
                            f"DPO batch key '{key}' is a sequence but rejected "
                            f"key '{rejected_key}' is "
                            f"{type(rejected_value).__name__}"
                        )
                    if len(value) != chosen_b or len(rejected_value) != chosen_b:
                        raise RuntimeError(
                            f"DPO paired sequence keys '{key}'/'{rejected_key}' "
                            f"must both have length {chosen_b}"
                        )

            chosen[key] = value
            rejected[key] = rejected_value

        return chosen, rejected, chosen_b

    @contextmanager
    def _dpo_stream_predict_context(self):
        """Mark a B-sized streamed forward as paired DPO.

        The context lives inside the checkpointed callable so backward
        recomputation gets exactly the same conditioning behavior as its
        original forward. ContextVar keeps independent autograd worker contexts
        from racing on shared setup state.
        """
        token = self._dpo_stream_active.set(True)
        try:
            yield
        finally:
            self._dpo_stream_active.reset(token)

    def _dpo_conditioning_locked(self) -> bool:
        return bool(
            self._dpo_paired_half is not None
            or self._dpo_stream_active.get()
        )

    @contextmanager
    def _dpo_reference_predict_context(self):
        token = self._dpo_reference_active.set(True)
        try:
            yield
        finally:
            self._dpo_reference_active.reset(token)

    def _dpo_reference_prediction(self) -> bool:
        return bool(self._dpo_reference_active.get())

    def rlhf_chosen_supervised_requires_separate_forward(
            self,
            config: TrainConfig,
    ) -> bool:
        """Whether DPO's chosen supervised term needs its own forward."""
        return False

    def rlhf_mixed_normal_dpo_requires_sequential_backward(
            self,
            config: TrainConfig,
    ) -> bool:
        """Whether normal and DPO subbatches must backward sequentially.

        This is independent of DPO chosen supervision. Model families whose
        normal training path temporarily swaps parameters (for example Flux2
        Self-Flow's EMA teacher) can request:
            normal forward/backward -> free graph -> DPO forward/backward
        while preserving one optimizer step and the existing item weighting.
        """
        return False

    @staticmethod
    def rlhf_chosen_supervised_weight(
            config: TrainConfig,
            objective: DPOObjective,
    ) -> float:
        """Return the positive-data training weight for a DPO objective.

        Anchored Reject and Balanced Reject carry a full chosen reconstruction term.
        Other objectives use the RLHF Supervised Mix control. Keeping this in
        one helper lets GenericTrainer externalize the expensive chosen
        Self-Flow graph without changing the objective semantics.
        """
        objective = DPOObjective(objective)
        return (
            1.0
            if objective in {
                DPOObjective.ANCHORED_REJECT,
                DPOObjective.BALANCED_REJECT,
            }
            else max(float(config.rlhf_supervised_mix), 0.0)
        )

    def calculate_rlhf_chosen_supervised_loss(
            self,
            model: BaseModel,
            batch: dict,
            config: TrainConfig,
            train_progress: TrainProgress,
    ) -> Tensor:
        """Run one DPO chosen batch through the ordinary training path.

        This intentionally runs outside all DPO conditioning/reference
        contexts. For Flux2 Self-Flow it therefore executes the same teacher,
        student, generation, representation and optional structural objective
        as an ordinary positive training item. ``latent_image_rejected`` may
        remain present in ``batch``; normal predictors consume ``latent_image``
        and ignore the paired rejected cache field.
        """
        output = self.predict(model, batch, config, train_progress)
        try:
            return self.calculate_loss(model, batch, output, config)
        finally:
            del output


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

    def get_last_dpo_pair_losses(self) -> list[tuple[str, float, str]]:
        return list(self._last_dpo_pair_losses)

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
        error = (predicted - target).pow(2)
        element_loss_weight = data.get("element_loss_weight")
        if element_loss_weight is not None:
            error = error * element_loss_weight.to(
                device=error.device,
                dtype=error.dtype,
            )
        return -error.mean(
            dim=list(range(1, predicted.ndim)),
            dtype=torch.float32,
        )

    def rlhf_linear_error_per_sample(
            self,
            model: BaseModel,
            batch: dict,
            data: dict,
            config: TrainConfig,
    ) -> Tensor:
        """Paper-faithful squared flow/denoising error for Linear-DPO.

        This intentionally does not reuse a model family's configurable native
        training loss: Linear-DPO derives its correction and policy term from
        squared prediction errors. Timestep/noise matching remains provided by
        the existing paired DPO forward path.
        """
        predicted = data["predicted"]
        target = data["target"]
        error = (predicted - target).pow(2)
        element_loss_weight = data.get("element_loss_weight")
        if element_loss_weight is not None:
            error = error * element_loss_weight.to(
                device=error.device,
                dtype=error.dtype,
            )
        return error.mean(
            dim=list(range(1, predicted.ndim)),
            dtype=torch.float32,
        )

    @staticmethod
    def _dpo_per_sample_tensor(
            value,
            batch_size: int,
            device: torch.device,
            dtype: torch.dtype,
            default: float,
    ) -> Tensor:
        if value is None:
            return torch.full(
                (batch_size,),
                float(default),
                device=device,
                dtype=dtype,
            )
        if isinstance(value, torch.Tensor):
            result = value.detach().to(device=device, dtype=dtype).reshape(-1)
        elif isinstance(value, (list, tuple)):
            result = torch.tensor(value, device=device, dtype=dtype).reshape(-1)
        else:
            result = torch.full(
                (batch_size,),
                float(value),
                device=device,
                dtype=dtype,
            )

        if int(result.numel()) == 1 and batch_size != 1:
            result = result.expand(batch_size)
        if int(result.numel()) != batch_size:
            raise RuntimeError(
                "Localized DPO metadata must have one value per sample: "
                f"expected {batch_size}, got {int(result.numel())}"
            )
        return result

    @staticmethod
    def _resize_dpo_mask_like(mask: Tensor, predicted: Tensor) -> Tensor:
        batch_size = int(predicted.shape[0])
        if mask.ndim == 0 or int(mask.shape[0]) != batch_size:
            raise RuntimeError(
                "Localized DPO mask must have the same leading batch size as "
                f"the prediction: mask={tuple(mask.shape)}, "
                f"prediction={tuple(predicted.shape)}"
            )

        mask = mask.to(device=predicted.device, dtype=torch.float32)

        if predicted.ndim in (3, 4, 5):
            # Sequence-shaped denoisers are supported when the cached spatial
            # mask flattens exactly to their token axis.
            if predicted.ndim == 3 and mask.ndim >= 3:
                flat = mask.reshape(batch_size, -1)
                if int(predicted.shape[1]) == int(flat.shape[1]):
                    return flat.unsqueeze(-1)
                if int(predicted.shape[2]) == int(flat.shape[1]):
                    return flat.unsqueeze(1)

            while mask.ndim < predicted.ndim:
                mask = mask.unsqueeze(1)
            while mask.ndim > predicted.ndim and int(mask.shape[1]) == 1:
                mask = mask.squeeze(1)

            if mask.ndim != predicted.ndim:
                raise RuntimeError(
                    "Localized DPO mask cannot be aligned with prediction: "
                    f"mask={tuple(mask.shape)}, "
                    f"prediction={tuple(predicted.shape)}"
                )

            if predicted.ndim >= 4 and tuple(mask.shape[2:]) != tuple(predicted.shape[2:]):
                spatial_dims = predicted.ndim - 2
                mode = {1: "linear", 2: "bilinear", 3: "trilinear"}.get(
                    spatial_dims
                )
                if mode is None:
                    raise RuntimeError(
                        "Localized DPO supports one-, two-, or three-"
                        f"dimensional prediction grids, got {spatial_dims}"
                    )
                mask = F.interpolate(
                    mask,
                    size=tuple(int(x) for x in predicted.shape[2:]),
                    mode=mode,
                    align_corners=False,
                )

            # A single mask channel broadcasts across prediction channels.
            if int(mask.shape[1]) not in (1, int(predicted.shape[1])):
                raise RuntimeError(
                    "Localized DPO mask channel count must be 1 or match the "
                    f"prediction: mask={tuple(mask.shape)}, "
                    f"prediction={tuple(predicted.shape)}"
                )
            return mask.clamp(0.0, 1.0)

        raise RuntimeError(
            "Localized DPO requires a spatial or sequence prediction tensor, "
            f"got shape {tuple(predicted.shape)}"
        )

    def _with_dpo_localized_weight(
            self,
            batch: dict,
            data: dict,
    ) -> dict:
        predicted = data["predicted"]
        batch_size = int(predicted.shape[0])
        flags = self._dpo_per_sample_tensor(
            batch.get("dpo_masked"),
            batch_size,
            predicted.device,
            torch.float32,
            0.0,
        ).clamp(0.0, 1.0)

        if not bool(torch.any(flags > 0.0).item()):
            return data

        raw_mask = batch.get("dpo_mask")
        if not isinstance(raw_mask, torch.Tensor):
            raise RuntimeError(
                "A concept enables Localized DPO, but the dataloader did not "
                "provide dpo_mask. Rebuild the image cache after installing "
                "the complete localized-DPO patch."
            )

        multipliers = self._dpo_per_sample_tensor(
            batch.get("dpo_mask_weight"),
            batch_size,
            predicted.device,
            torch.float32,
            10.0,
        )
        active_multipliers = multipliers[flags > 0.0]
        if not bool(torch.isfinite(active_multipliers).all().item()):
            raise ValueError("Localized DPO Mask Weight must be finite")
        if bool(torch.any(active_multipliers < 1.0).item()):
            raise ValueError("Localized DPO Mask Weight must be >= 1")
        # Inactive concepts may carry arbitrary legacy metadata. Make it
        # neutral before arithmetic so an unused NaN/Inf cannot contaminate a
        # mixed batch through IEEE ``0 * Inf`` behavior.
        multipliers = torch.where(
            flags > 0.0,
            multipliers,
            torch.ones_like(multipliers),
        )

        mask = self._resize_dpo_mask_like(raw_mask, predicted)
        broadcast_shape = (batch_size,) + (1,) * (predicted.ndim - 1)
        flags = flags.reshape(broadcast_shape)
        multipliers = multipliers.reshape(broadcast_shape)
        localized_weight = 1.0 + flags * (multipliers - 1.0) * mask

        existing_weight = data.get("element_loss_weight")
        if existing_weight is not None:
            localized_weight = localized_weight * existing_weight.to(
                device=localized_weight.device,
                dtype=localized_weight.dtype,
            )

        weighted_data = dict(data)
        weighted_data["element_loss_weight"] = localized_weight
        return weighted_data

    def _dpo_localized_metrics(self, batch: dict, batch_size: int) -> dict[str, float]:
        mask = batch.get("dpo_mask")
        if not isinstance(mask, torch.Tensor):
            return {
                "localized_active_fraction": 0.0,
                "localized_mask_fraction": 0.0,
                "localized_mean_weight": 1.0,
            }

        device = mask.device
        flags = self._dpo_per_sample_tensor(
            batch.get("dpo_masked"),
            batch_size,
            device,
            torch.float32,
            0.0,
        ).clamp(0.0, 1.0)
        multipliers = self._dpo_per_sample_tensor(
            batch.get("dpo_mask_weight"),
            batch_size,
            device,
            torch.float32,
            10.0,
        )
        multipliers = torch.where(
            flags > 0.0,
            multipliers,
            torch.ones_like(multipliers),
        )
        per_sample_mask = mask.detach().float().reshape(batch_size, -1).mean(1)
        active_count = flags.sum()
        if float(active_count.item()) > 0.0:
            mask_fraction = (per_sample_mask * flags).sum() / active_count
            mean_weight = (
                (1.0 + (multipliers - 1.0) * per_sample_mask) * flags
            ).sum() / active_count
        else:
            mask_fraction = mask.new_tensor(0.0, dtype=torch.float32)
            mean_weight = mask.new_tensor(1.0, dtype=torch.float32)
        return {
            "localized_active_fraction": flags.mean().item(),
            "localized_mask_fraction": mask_fraction.item(),
            "localized_mean_weight": mean_weight.item(),
        }

    def _dpo_score_per_sample(
            self,
            model: BaseModel,
            batch: dict,
            data: dict,
            config: TrainConfig,
            objective: DPOObjective,
    ) -> Tensor:
        data = self._with_dpo_localized_weight(batch, data)
        if objective == DPOObjective.LINEAR:
            return -self.rlhf_linear_error_per_sample(
                model,
                batch,
                data,
                config,
            )
        return self.rlhf_logp_per_sample(model, batch, data, config)

    @staticmethod
    def _linear_dpo_pair_loss(
            policy_chosen_score: Tensor,
            policy_rejected_score: Tensor,
            reference_chosen_score: Tensor,
            reference_rejected_score: Tensor,
            beta: float,
            eta: float,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return pair loss, utility, error gap, and preference margin."""
        chosen_ratio = policy_chosen_score - reference_chosen_score.detach()
        rejected_ratio = policy_rejected_score - reference_rejected_score.detach()
        margin = chosen_ratio - rejected_ratio
        utility = torch.clamp(
            0.5 - 0.2 * float(beta) * margin.detach(),
            min=float(eta),
            max=1.0 - float(eta),
        )
        policy_error_gap = -(
            policy_chosen_score - policy_rejected_score
        )
        pair_loss = utility.detach() * policy_error_gap
        return pair_loss, utility, policy_error_gap, margin

    @staticmethod
    def _linear_dpo_adaptive_difficulty(
            policy_chosen_score: Tensor,
            policy_rejected_score: Tensor,
            competence_scale: float,
    ) -> Tensor:
        """Stable non-negative pair difficulty for adaptive Linear-DPO.

        The signed Linear-DPO objective is not a difficulty measure: correctly
        ordered pairs have negative loss, and its detached utility depends on
        policy-vs-EMA lag.  Sampling instead uses a smooth ranking error from
        the current policy's direct chosen/rejected score gap.  Zero gap maps
        to log(2), wrong ordering maps above it, and increasingly confident
        correct ordering approaches zero.
        """
        scale = float(competence_scale)
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError(
                "Linear-DPO adaptive difficulty scale must be finite and > 0, "
                f"got {scale}"
            )
        direct_gap = policy_chosen_score - policy_rejected_score
        return F.softplus(-direct_gap / scale)

    def rlhf_policy_auxiliary_loss(
            self,
            model: BaseModel,
            batch: dict,
            data: dict,
            config: TrainConfig,
    ) -> Tensor | None:
        """Optional policy-only loss excluded from the DPO logp proxy."""
        return None

    @staticmethod
    def _validate_rlhf_logp_per_sample(logp: Tensor, expected_b: int, name: str) -> Tensor:
        # DPO requires exactly one scalar logp proxy per chosen/rejected sample.
        # A scalar mean loss or unreduced spatial tensor would silently corrupt
        # the preference objective, so fail hard.
        if not isinstance(logp, torch.Tensor):
            raise TypeError(
                f"{name} rlhf_logp_per_sample must return a Tensor, "
                f"got {type(logp).__name__}"
            )

        expected_b = int(expected_b)
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
    def _dpo_csv_timestep_value(value, index: int | None = None):
        """Serialize scalar or per-token timesteps without exploding CSV rows."""
        if value is None or not hasattr(value, "detach"):
            return BaseModelSetup._dpo_csv_index_value(value, index)

        tensor = value.detach()
        if tensor.numel() == 0:
            return ""
        if (
            index is not None
            and tensor.ndim > 0
            and int(tensor.shape[0]) > index
        ):
            tensor = tensor[index]
        elif index is not None:
            tensor = tensor.flatten()[0]

        if tensor.numel() == 1:
            return BaseModelSetup._dpo_csv_index_value(tensor, None)

        flat = tensor.detach().cpu().flatten()
        unique, counts = torch.unique(
            flat,
            sorted=True,
            return_counts=True,
        )
        if unique.numel() <= 8:
            histogram = "|".join(
                f"{item.item()}:{int(count.item())}"
                for item, count in zip(unique, counts, strict=True)
            )
            return f"n={flat.numel()};values={histogram}"

        # Keep unusual continuous/per-token schedulers compact too.
        flat_float = flat.float()
        return (
            f"n={flat.numel()};unique={unique.numel()};"
            f"min={flat_float.min().item():.8g};"
            f"max={flat_float.max().item():.8g};"
            f"mean={flat_float.mean().item():.8g}"
        )

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
            objective: DPOObjective,
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
                    objective
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
                "timestep": self._dpo_csv_timestep_value(
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
            objective: DPOObjective,
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
          * chosen target violation exceeds the configured threshold for an
            objective that actually uses a chosen reward target
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

        # Only enabled auxiliaries/objectives have absolute reward targets.
        # Linear-DPO and plain Sigmoid/IPO are relative objectives, so applying
        # Anchored-Reject thresholds to them would create false bad-pair rows.
        chosen_target = (
            float(getattr(config, "rlhf_dpo_chosen_reward_target", 0.05))
            if (
                objective in {DPOObjective.SIGMOID, DPOObjective.IPO}
                and bool(getattr(config, "rlhf_dpo_chosen_reward_anchor", False))
            )
            else None
        )
        rejected_target = (
            float(getattr(config, "rlhf_dpo_anchored_rejected_target", -0.05))
            if objective == DPOObjective.ANCHORED_REJECT
            else None
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

            chosen_violation = (
                0.0
                if chosen_target is None
                else max(chosen_target - chosen_value, 0.0)
            )
            rejected_violation = (
                0.0
                if rejected_target is None
                else max(rejected_value - rejected_target, 0.0)
            )

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
            if (
                chosen_target is not None
                and chosen_violation >= violation_threshold
                and violation_threshold > 0
            ):
                reasons.append("chosen_target_violation")
            if (
                rejected_target is not None
                and rejected_violation >= violation_threshold
                and violation_threshold > 0
            ):
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
                    objective
                ),
                "reason": "|".join(reasons),
                "chosen_image_path": chosen_path,
                "rejected_image_path": rejected_path,
                "dpo_pair_key": pair_key,
                "timestep": self._dpo_csv_timestep_value(policy_timestep, i),
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
                "chosen_target": "" if chosen_target is None else chosen_target,
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
        *,
        objective: DPOObjective | None = None,
        reference_mode: DPORefMode | None = None,
        reference_key: str | None = None,
        streamed: bool = False,
        external_chosen_supervised_loss_value: float | None = None,
    ) -> Tensor:
        if "latent_image_rejected" not in batch:
            raise RuntimeError(
                "RLHF DPO requires paired chosen/rejected batches, but the dataloader did not provide rejected samples."
            )

        objective = DPOObjective(
            config.rlhf_dpo_objective
            if objective is None
            else objective
        )
        if objective == DPOObjective.LINEAR:
            # Per-concept fixed/base overrides cannot replace the moving
            # reference required by the Linear-DPO derivation.
            reference_mode = DPORefMode.EMA_ADAPTER
            reference_key = None
        self._last_dpo_pair_losses = []
        beta = config.rlhf_dpo_beta if self._dpo_runtime_beta is None else self._dpo_runtime_beta
        supervised_loss = None
        policy_auxiliary_loss = None
        # Anchored Reject always models its chosen image exactly like one
        # ordinary training item. Other objectives use rlhf_supervised_mix as
        # the optional chosen-data training weight; Anchored Reject must not
        # stack a second copy of that chosen loss.
        chosen_supervised_weight = self.rlhf_chosen_supervised_weight(
            config,
            objective,
        )
        include_chosen_supervised = chosen_supervised_weight > 0.0
        external_chosen_supervised = bool(
            external_chosen_supervised_loss_value is not None
            and include_chosen_supervised
            and self.rlhf_chosen_supervised_requires_separate_forward(config)
        )

        if streamed:
            # Low-VRAM DPO keeps every expensive model call B-sized. Reference
            # branches are sequential and no-grad. Policy branches use a custom
            # autograd bridge: their value forwards retain no graph, then each
            # branch is recomputed and differentiated to completion from its
            # backward callback. This is deliberately not an outer
            # torch.utils.checkpoint wrapper. OT transformer layers already own
            # checkpoint/offload conductors whose one-forward/one-backward
            # lifecycle must not have two nested replay graphs outstanding.
            chosen_input, rejected_input, chosen_b = (
                self._create_dpo_stream_batches(batch)
            )

            def reference_logp(branch_input: dict, branch_name: str) -> Tensor:
                with self._dpo_stream_predict_context():
                    output = self.predict(
                        model,
                        branch_input,
                        config,
                        train_progress,
                    )
                    logp = self._dpo_score_per_sample(
                        model,
                        branch_input,
                        output,
                        config,
                        objective,
                    )
                    logp = self._validate_rlhf_logp_per_sample(
                        logp,
                        chosen_b,
                        branch_name,
                    )
                    del output
                    return logp

            with (
                torch.no_grad(),
                self.reference_model(
                    model,
                    config,
                    reference_mode=reference_mode,
                    reference_key=reference_key,
                ),
                self._dpo_reference_predict_context(),
            ):
                ref_chosen_logp = reference_logp(
                    chosen_input,
                    "streamed reference chosen",
                )
                ref_rejected_logp = reference_logp(
                    rejected_input,
                    "streamed reference rejected",
                )

            optimizer = getattr(model, "optimizer", None)
            parameter_groups = getattr(optimizer, "param_groups", None)
            if not isinstance(parameter_groups, list):
                raise RuntimeError(
                    "Streamed DPO requires an initialized optimizer with "
                    "parameter groups"
                )

            trainable_parameters: list[Tensor] = []
            seen_parameter_ids: set[int] = set()
            for parameter_group in parameter_groups:
                for parameter in parameter_group.get("params", []):
                    if (
                        isinstance(parameter, torch.Tensor)
                        and parameter.requires_grad
                        and id(parameter) not in seen_parameter_ids
                    ):
                        seen_parameter_ids.add(id(parameter))
                        trainable_parameters.append(parameter)
            trainable_parameters_tuple = tuple(trainable_parameters)
            if not trainable_parameters_tuple:
                raise RuntimeError(
                    "Streamed DPO could not find any trainable optimizer "
                    "parameters"
                )

            # On one GPU, accumulate each replay directly into the existing
            # .grad buffers. Returning branch VJPs through the outer autograd
            # graph temporarily materializes a complete new gradient tuple while
            # previous accumulation gradients remain resident, and can keep both
            # chosen/rejected tuples pending before AccumulateGrad runs. That
            # defeats the low-VRAM purpose. Multi-GPU still returns VJPs so DDP
            # sees one ordinary outer-autograd reduction.
            active_offload_conductor = False
            for model_value in vars(model).values():
                offload_activated = getattr(
                    model_value,
                    "offload_activated",
                    None,
                )
                if callable(offload_activated) and offload_activated():
                    active_offload_conductor = True
                    break

            single_gpu_stream = multi.world_size() <= 1
            fused_back_pass = bool(
                config.optimizer.optimizer.supports_fused_back_pass()
                and config.optimizer.fused_back_pass
            )
            streamed_direct_backward = single_gpu_stream

            if active_offload_conductor and not single_gpu_stream:
                raise RuntimeError(
                    "Streamed DPO with layer/activation offloading currently "
                    "supports one GPU. Disable offloading for multi-GPU "
                    "streamed DPO."
                )
            if streamed_direct_backward and fused_back_pass:
                raise RuntimeError(
                    "Single-GPU Streamed DPO requires Fused Back Pass to be "
                    "disabled. It accumulates chosen/rejected replay gradients "
                    "directly to avoid duplicate branch gradient tensors; "
                    "ordinary gradient accumulation remains supported."
                )

            def stream_policy_branch(
                    branch_input: dict,
                    branch_name: str,
                    include_supervised: bool,
                    include_policy_auxiliary: bool,
            ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
                latent = branch_input["latent_image"]
                timestep_holder: dict[str, Tensor] = {}

                def evaluate_branch(latent_input: Tensor):
                    # Never mutate the closure-owned batch. The same function
                    # runs once for its value and once for its VJP recompute.
                    replay_input = dict(branch_input)
                    replay_input["latent_image"] = latent_input
                    separate_supervised_forward = bool(
                        include_supervised
                        and self.rlhf_chosen_supervised_requires_separate_forward(
                            config,
                        )
                    )

                    branch_supervised_loss = None
                    if separate_supervised_forward:
                        # IMPORTANT: normal Flux2 Self-Flow temporarily swaps
                        # EMA teacher adapter weights into the live transformer
                        # and restores them in-place. That swap must happen
                        # BEFORE any trainable DPO graph is constructed, or
                        # autograd's saved parameter version counters become
                        # invalid. During streamed backward replay this ordering
                        # is just as important as it is on the fast DPO path.
                        supervised_output = self.predict(
                            model,
                            replay_input,
                            config,
                            train_progress,
                        )
                        branch_supervised_loss = self.calculate_loss(
                            model,
                            replay_input,
                            supervised_output,
                            config,
                        )
                        del supervised_output

                    # Preference scoring stays inside the DPO-conditioning lock.
                    # In Flux2 this deliberately bypasses Self-Flow so the
                    # chosen/rejected preference gradient remains pure DPO.
                    # No parameter-swapping operation may occur after this graph
                    # is constructed and before it is differentiated.
                    with self._dpo_stream_predict_context():
                        output = self.predict(
                            model,
                            replay_input,
                            config,
                            train_progress,
                        )
                        logp = self._dpo_score_per_sample(
                            model,
                            replay_input,
                            output,
                            config,
                            objective,
                        )
                        logp = self._validate_rlhf_logp_per_sample(
                            logp,
                            chosen_b,
                            branch_name,
                        )

                        if include_policy_auxiliary:
                            branch_auxiliary_loss = self.rlhf_policy_auxiliary_loss(
                                model,
                                replay_input,
                                output,
                                config,
                            )
                            if branch_auxiliary_loss is None:
                                branch_auxiliary_loss = logp.new_empty((0,))
                        else:
                            branch_auxiliary_loss = logp.new_empty((0,))

                        timestep = output.get("timestep")
                        if isinstance(timestep, torch.Tensor):
                            timestep = timestep.detach()
                        else:
                            timestep = logp.new_empty((0,))

                        if include_supervised and not separate_supervised_forward:
                            branch_supervised_loss = self.calculate_loss(
                                model,
                                replay_input,
                                output,
                                config,
                            )
                        elif branch_supervised_loss is None:
                            branch_supervised_loss = logp.new_empty((0,))

                        del output

                    return (
                        logp,
                        branch_supervised_loss,
                        branch_auxiliary_loss,
                        timestep,
                    )

                class StreamedDPOBranch(torch.autograd.Function):
                    @staticmethod
                    def forward(ctx, latent_input: Tensor, *parameters):
                        # torch.autograd.Function.forward executes with gradient
                        # recording disabled. Capture RNG state so uncommon
                        # stochastic layers also replay exactly during backward;
                        # diffusion noise/timesteps additionally use OT's local
                        # per-step generator.
                        ctx.save_for_backward(latent_input)
                        ctx.cpu_rng_state = torch.get_rng_state()
                        ctx.cuda_device_index = None
                        ctx.cuda_rng_state = None
                        if latent_input.device.type == "cuda":
                            device_index = latent_input.device.index
                            if device_index is None:
                                device_index = torch.cuda.current_device()
                            ctx.cuda_device_index = int(device_index)
                            ctx.cuda_rng_state = torch.cuda.get_rng_state(
                                device_index,
                            )

                        (
                            logp,
                            branch_supervised_loss,
                            branch_auxiliary_loss,
                            timestep,
                        ) = (
                            evaluate_branch(latent_input)
                        )
                        timestep_holder["value"] = timestep
                        return logp, branch_supervised_loss, branch_auxiliary_loss

                    @staticmethod
                    def backward(ctx, grad_logp, grad_supervised, grad_auxiliary):
                        (latent_input,) = ctx.saved_tensors
                        cuda_devices = (
                            []
                            if ctx.cuda_device_index is None
                            else [ctx.cuda_device_index]
                        )

                        # fork_rng restores the caller's current states after
                        # replay, so custom backward does not consume randomness
                        # that subsequent training code expects.
                        with torch.random.fork_rng(
                                devices=cuda_devices,
                                enabled=True,
                        ):
                            torch.set_rng_state(ctx.cpu_rng_state)
                            if ctx.cuda_rng_state is not None:
                                torch.cuda.set_rng_state(
                                    ctx.cuda_rng_state,
                                    ctx.cuda_device_index,
                                )

                            with torch.enable_grad():
                                (
                                    recomputed_logp,
                                    recomputed_supervised,
                                    recomputed_auxiliary,
                                    _,
                                ) = evaluate_branch(latent_input)

                                outputs = []
                                output_gradients = []
                                if grad_logp is not None:
                                    outputs.append(recomputed_logp)
                                    output_gradients.append(grad_logp)
                                if include_supervised and grad_supervised is not None:
                                    outputs.append(recomputed_supervised)
                                    output_gradients.append(grad_supervised)
                                if grad_auxiliary is not None and recomputed_auxiliary.numel() > 0:
                                    outputs.append(recomputed_auxiliary)
                                    output_gradients.append(grad_auxiliary)

                                if outputs and streamed_direct_backward:
                                    # Required for OT's use_reentrant=True
                                    # offload checkpoints. The incoming gradient
                                    # already contains AMP scaling and gradient-
                                    # accumulation weighting, so direct nested
                                    # backward produces the same accumulated
                                    # parameter gradients as the ordinary path.
                                    torch.autograd.backward(
                                        tensors=tuple(outputs),
                                        grad_tensors=tuple(output_gradients),
                                    )
                                    parameter_gradients = (
                                        None,
                                    ) * len(trainable_parameters_tuple)
                                elif outputs:
                                    parameter_gradients = torch.autograd.grad(
                                        outputs=tuple(outputs),
                                        inputs=trainable_parameters_tuple,
                                        grad_outputs=tuple(output_gradients),
                                        allow_unused=True,
                                    )
                                else:
                                    parameter_gradients = (
                                        None,
                                    ) * len(trainable_parameters_tuple)

                                self.after_streamed_dpo_branch_backward(
                                    model,
                                    config,
                                    train_progress,
                                )

                        # Latent cache tensors are targets/conditioning, not
                        # trainable inputs. Remaining returns correspond exactly
                        # to the optimizer parameters passed to apply().
                        return (None, *parameter_gradients)

                policy_logp, branch_supervised_loss, branch_auxiliary_loss = (
                    StreamedDPOBranch.apply(
                        latent,
                        *trainable_parameters_tuple,
                    )
                )
                policy_timestep = timestep_holder.get(
                    "value",
                    policy_logp.new_empty((0,)),
                )
                return (
                    policy_logp,
                    branch_supervised_loss,
                    branch_auxiliary_loss,
                    policy_timestep,
                )

            (
                policy_chosen_logp,
                chosen_supervised_loss,
                chosen_auxiliary_loss,
                policy_timestep,
            ) = stream_policy_branch(
                chosen_input,
                "streamed policy chosen",
                include_chosen_supervised and not external_chosen_supervised,
                True,
            )
            (
                policy_rejected_logp,
                _,
                rejected_auxiliary_loss,
                _,
            ) = stream_policy_branch(
                rejected_input,
                "streamed policy rejected",
                False,
                True,
            )
            if include_chosen_supervised and not external_chosen_supervised:
                supervised_loss = chosen_supervised_loss
            branch_auxiliary_losses = [
                auxiliary
                for auxiliary in (
                    chosen_auxiliary_loss,
                    rejected_auxiliary_loss,
                )
                if auxiliary.numel() > 0
            ]
            if branch_auxiliary_losses:
                # Match the fast 2B path: each B-sized branch auxiliary is a
                # per-branch mean, so averaging chosen/rejected gives the mean
                # across all 2B policy samples.
                policy_auxiliary_loss = torch.stack(
                    branch_auxiliary_losses
                ).mean()

        else:
            # Fast path: one batched reference and one batched policy forward,
            # each over [chosen; rejected]. Both halves share per-pair timestep
            # and noise via _dpo_paired_half. Reference and policy share them too
            # because predict() seeds its generator from global_step.
            batched_input, chosen_b = self._create_dpo_batched_batch(batch)

            # Complete the DPO reference swap first. reference_model() may
            # temporarily replace live adapter parameters and restore them
            # in-place, so no trainable graph may exist across this context.
            self._dpo_paired_half = chosen_b
            try:
                with (
                    torch.no_grad(),
                    self.reference_model(
                        model,
                        config,
                        reference_mode=reference_mode,
                        reference_key=reference_key,
                    ),
                    self._dpo_reference_predict_context(),
                ):
                    ref_output = self.predict(
                        model,
                        batched_input,
                        config,
                        train_progress,
                    )
                    ref_logp = self._dpo_score_per_sample(
                        model,
                        batched_input,
                        ref_output,
                        config,
                        objective,
                    )
                    ref_logp = self._validate_rlhf_logp_per_sample(
                        ref_logp,
                        2 * chosen_b,
                        "reference",
                    )
                    ref_chosen_logp = ref_logp[:chosen_b]
                    ref_rejected_logp = ref_logp[chosen_b:]
                    del ref_output, ref_logp
            finally:
                self._dpo_paired_half = None

            separate_chosen_supervised = bool(
                include_chosen_supervised
                and self.rlhf_chosen_supervised_requires_separate_forward(config)
            )
            if separate_chosen_supervised and not external_chosen_supervised:
                # Normal Flux2 Self-Flow performs its EMA teacher parameter
                # swap before constructing the student graph. Run it AFTER all
                # DPO reference swapping, but BEFORE the DPO policy graph.
                supervised_output = self.predict(
                    model,
                    batch,
                    config,
                    train_progress,
                )
                supervised_loss = self.calculate_loss(
                    model,
                    batch,
                    supervised_output,
                    config,
                )
                del supervised_output

            # The pure DPO policy graph is constructed last. From here until
            # backward there must be no EMA/reference parameter swap.
            self._dpo_paired_half = chosen_b
            try:
                policy_output = self.predict(
                    model,
                    batched_input,
                    config,
                    train_progress,
                )
            finally:
                self._dpo_paired_half = None
            policy_timestep = policy_output.get("timestep")
            policy_logp = self._dpo_score_per_sample(
                model,
                batched_input,
                policy_output,
                config,
                objective,
            )
            policy_logp = self._validate_rlhf_logp_per_sample(
                policy_logp,
                2 * chosen_b,
                "policy",
            )
            policy_chosen_logp = policy_logp[:chosen_b]
            policy_rejected_logp = policy_logp[chosen_b:]
            # Full Self-Flow DPO: representation/structural learning sees both
            # chosen and rejected policy samples. Preference direction is still
            # supplied by the DPO objective; this auxiliary does not alter the
            # policy/reference reward definition.
            policy_auxiliary_loss = self.rlhf_policy_auxiliary_loss(
                model,
                batched_input,
                policy_output,
                config,
            )
            if include_chosen_supervised and not separate_chosen_supervised:
                chosen_output, _ = self._split_dpo_batched_output(
                    policy_output,
                    chosen_b,
                )
                supervised_loss = self.calculate_loss(
                    model,
                    batch,
                    chosen_output,
                    config,
                )
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
        chosen_reward_aux_pair_loss = None
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
        sigmoid_objective_loss = None
        linear_utility = None
        linear_policy_error_gap = None
        balanced_reject_target = None
        balanced_reject_violation = None
        balanced_reject_pair_loss = None
        balanced_chosen_budget = None

        if objective == DPOObjective.LINEAR:
            # Linear-DPO notation uses DeltaD = (E_policy_chosen -
            # E_ref_chosen) - (E_policy_rejected - E_ref_rejected).
            # Our score is -E, therefore margin == -DeltaD and the paper's
            # detached utility becomes
            # clip(0.5 - 0.2*beta*margin, eta, 1-eta).
            eta = float(config.rlhf_dpo_linear_eta)
            beta_value = (
                float(beta.detach().item())
                if isinstance(beta, torch.Tensor)
                else float(beta)
            )
            (
                raw_pair_total_loss,
                linear_utility,
                linear_policy_error_gap,
                _,
            ) = self._linear_dpo_pair_loss(
                policy_chosen_logp,
                policy_rejected_logp,
                ref_chosen_logp,
                ref_rejected_logp,
                beta_value,
                eta,
            )

        elif objective == DPOObjective.ANCHORED_REJECT:
            # The chosen branch receives one full ordinary flow-matching loss
            # below. The preference objective therefore only needs an absolute
            # rejected anchor plus bounded pairwise terms. Those pairwise terms
            # provide extra chosen pressure precisely for close or misordered
            # negatives, without imposing an arbitrary chosen reward target.
            rejected_target = float(
                getattr(config, "rlhf_dpo_anchored_rejected_target", -0.05)
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

            rejected_violation = F.relu(rejected_ratio - rejected_target)

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
                rejected_pair_loss
                + margin_penalty_loss
                + wrong_order_penalty_loss
            )

        elif objective == DPOObjective.BALANCED_REJECT:
            # This preference branch is strictly rejected-only. Model-family
            # policy may separately add chosen supervision; the Flux2 full
            # Self-Flow DPO experiment disables that extra chosen-only pass.
            #
            # Positive chosen improvement vs. the fixed reference determines
            # how far rejected should be suppressed:
            #
            #   rejected_target = -ratio * max(stopgrad(chosen_reward), 0)
            #
            # If chosen reward falls to/below zero, the target returns to zero.
            # The preference loss therefore cannot manufacture separation by
            # dragging chosen down, and cannot chase a deteriorating chosen.
            balance_ratio = max(
                float(getattr(
                    config, "rlhf_dpo_balanced_reject_ratio", 1.0
                )),
                0.0,
            )
            balanced_weight = max(
                float(getattr(
                    config, "rlhf_dpo_balanced_reject_weight", 1.0
                )),
                0.0,
            )
            balanced_huber_delta = max(
                float(getattr(
                    config, "rlhf_dpo_balanced_huber_delta", 0.1
                )),
                1e-8,
            )

            balanced_chosen_budget = F.relu(chosen_ratio.detach())
            balanced_reject_target = (
                -balance_ratio * balanced_chosen_budget
            )
            balanced_reject_violation = F.relu(
                rejected_ratio - balanced_reject_target
            )
            balanced_reject_pair_loss = (
                balanced_weight
                * F.smooth_l1_loss(
                    balanced_reject_violation,
                    torch.zeros_like(balanced_reject_violation),
                    beta=balanced_huber_delta,
                    reduction="none",
                )
            )
            raw_pair_total_loss = balanced_reject_pair_loss

        elif objective == DPOObjective.IPO:
            raw_pair_total_loss = (
                margin - 1.0 / (2.0 * config.rlhf_dpo_ipo_tau)
            ).pow(2)
        else:
            logits = beta * margin
            raw_pair_total_loss = -F.logsigmoid(logits)

            if config.rlhf_dpo_label_smoothing > 0:
                label_smoothing = config.rlhf_dpo_label_smoothing
                raw_pair_total_loss = (
                    (1.0 - label_smoothing) * raw_pair_total_loss
                    + label_smoothing * (-F.logsigmoid(-logits))
                )

            if getattr(config, "rlhf_dpo_beta_gradient_decouple", False):
                beta_for_scale = float(beta.detach().item()) if isinstance(beta, torch.Tensor) else float(beta)
                beta_ref = getattr(config, "rlhf_dpo_beta_gradient_reference", None)
                if beta_ref is None or float(beta_ref) <= 0:
                    beta_ref = float(config.rlhf_dpo_beta)
                dpo_beta_scale = float(beta_ref) / max(beta_for_scale, 1e-12)

        if raw_pair_total_loss is None:
            raise RuntimeError(
                f"DPO objective {objective} did not produce per-pair losses"
            )

        (
            curriculum_competence,
            curriculum_full_margin,
        ) = self._dpo_curriculum_competence(
            objective,
            config,
            margin,
            policy_chosen_logp,
            policy_rejected_logp,
        )
        curriculum_margin_ema = curriculum_competence.detach()

        if self._dpo_hard_pair_curriculum_enabled(config):
            (
                curriculum_weight,
                curriculum_margin_ema,
                curriculum_observations,
            ) = self._stage_dpo_curriculum_observations(
                batch,
                config,
                curriculum_competence,
                objective,
                curriculum_full_margin,
            )
            # The weight is detached EMA state. Every objective is gated at
            # the same unreduced pair-loss layer, while the competence signal
            # above respects that objective's own optimum/reference behavior.
            # This also composes with per-concept objective dispatch without
            # changing any score definition.
            pair_total_loss = (
                raw_pair_total_loss * curriculum_weight.detach()
            )
        else:
            pair_total_loss = raw_pair_total_loss

        preference_loss = pair_total_loss.mean()
        if (
            objective == DPOObjective.SIGMOID
            and getattr(config, "rlhf_dpo_beta_gradient_decouple", False)
        ):
            # Value-preserving gradient scaling: forward/logged loss stays
            # equal to the curriculum-weighted preference loss.
            preference_loss = (
                preference_loss.detach()
                + dpo_beta_scale
                * (preference_loss - preference_loss.detach())
            )

        dpo_loss = preference_loss
        loss = preference_loss
        if objective == DPOObjective.SIGMOID:
            # Exact sigmoid objective after label smoothing, curriculum, and
            # the value-preserving beta-gradient wrapper.
            sigmoid_objective_loss = preference_loss.detach().item()

        # The legacy chosen anchor remains available for SIGMOID/IPO configs,
        # but is never stacked on top of the independent Anchored Reject loss.
        if (
            objective not in {
                DPOObjective.ANCHORED_REJECT,
                DPOObjective.BALANCED_REJECT,
                DPOObjective.LINEAR,
            }
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

                chosen_reward_push_pair_loss = F.softplus(
                    (chosen_reward_target - chosen_ratio) * chosen_reward_sharpness
                ) / chosen_reward_sharpness

                chosen_reward_floor_violation = F.relu(
                    chosen_reward_floor_value - chosen_ratio
                )
                chosen_reward_floor_pair_loss = (
                    chosen_reward_floor_violation
                    + chosen_reward_sharpness
                    * chosen_reward_floor_violation.pow(2)
                )

                chosen_reward_aux_pair_loss = chosen_anchor_weight * (
                    chosen_reward_push_pair_loss
                    + chosen_reward_floor_multiplier
                    * chosen_reward_floor_pair_loss
                )
                chosen_reward_push_loss = chosen_reward_push_pair_loss.mean()
                chosen_reward_floor_loss = chosen_reward_floor_pair_loss.mean()
                chosen_reward_aux_loss = chosen_reward_aux_pair_loss.mean()
                loss = loss + chosen_reward_aux_loss

        # DPO-side logged loss includes beta-grad scaling and chosen-anchor aux loss,
        # but not the independent chosen supervised_mix training term.
        dpo_logged_loss = loss

        if policy_auxiliary_loss is not None:
            if policy_auxiliary_loss.ndim != 0:
                raise RuntimeError(
                    "rlhf_policy_auxiliary_loss must return a scalar Tensor, "
                    f"got shape {tuple(policy_auxiliary_loss.shape)}"
                )
            loss = loss + policy_auxiliary_loss

        chosen_supervised_loss_value = (
            float(external_chosen_supervised_loss_value)
            if external_chosen_supervised
            else 0.0
        )
        if supervised_loss is not None:
            chosen_supervised_loss_value = supervised_loss.detach().item()
            loss = loss + chosen_supervised_weight * supervised_loss
            del supervised_loss

        # When GenericTrainer externalizes chosen Self-Flow, its gradient was
        # already accumulated and the returned tensor intentionally contains
        # only the DPO-side graph. Preserve the historical full-objective
        # scalar for diagnostics without attaching the supervised graph again.
        reported_total_loss_value = loss.detach().item()
        if external_chosen_supervised:
            reported_total_loss_value += (
                chosen_supervised_weight * chosen_supervised_loss_value
            )

        # Adaptive DPO Dataset uses a detached, non-negative per-pair
        # difficulty before curriculum or supervised_mix; otherwise curriculum
        # could make a hard pair look artificially easy. Ordinary objectives
        # already provide a non-negative loss. Linear-DPO is signed and its
        # utility follows EMA-reference lag, so use a separate smooth ranking
        # difficulty derived only from the current policy's direct score gap.
        if objective == DPOObjective.LINEAR:
            adaptive_pair_loss = self._linear_dpo_adaptive_difficulty(
                policy_chosen_logp,
                policy_rejected_logp,
                float(getattr(
                    config,
                    "rlhf_dpo_hard_pair_curriculum_full_margin",
                    0.05,
                )),
            )
        else:
            adaptive_pair_loss = raw_pair_total_loss
        if (
            adaptive_pair_loss is not None
            and chosen_reward_aux_pair_loss is not None
        ):
            adaptive_pair_loss = (
                adaptive_pair_loss + chosen_reward_aux_pair_loss
            )

        if isinstance(adaptive_pair_loss, torch.Tensor):
            detached_adaptive_loss = (
                adaptive_pair_loss.detach().float().reshape(-1).cpu()
            )
            if int(detached_adaptive_loss.numel()) == int(chosen_b):
                self._last_dpo_pair_losses = [
                    (
                        self._dpo_pair_identity(batch, index),
                        max(float(detached_adaptive_loss[index].item()), 0.0),
                        str(objective),
                    )
                    for index in range(int(chosen_b))
                ]

        try:
            self._write_dpo_pair_csv_log(
                batch=batch,
                config=config,
                objective=objective,
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
                objective=objective,
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

        localized_metrics = self._dpo_localized_metrics(batch, int(chosen_b))
        self._last_dpo_metrics = {
            "objective_loss": dpo_logged_loss.detach().item(),
            "chosen_reward": chosen_ratio.detach().mean().item(),
            "rejected_reward": rejected_ratio.detach().mean().item(),
            "reward_margin": margin.detach().mean().item(),
            "accuracy": (margin.detach() > 0).float().mean().item(),
            "hard_pair_curriculum_weight": curriculum_weight.detach().mean().item(),
            # Keep the old margin-named metric for dashboard compatibility,
            # but expose the accurate generic name for Linear-DPO and IPO.
            "hard_pair_competence_ema": curriculum_margin_ema.detach().mean().item(),
            "hard_pair_margin_ema": curriculum_margin_ema.detach().mean().item(),
            "hard_pair_observations": curriculum_observations.detach().mean().item(),
            "adaptive_pair_difficulty": (
                adaptive_pair_loss.detach().float().mean().item()
                if isinstance(adaptive_pair_loss, torch.Tensor)
                else 0.0
            ),
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
            "policy_auxiliary_loss": (
                policy_auxiliary_loss.detach().item()
                if policy_auxiliary_loss is not None
                else 0.0
            ),
            "chosen_supervised_weight": float(chosen_supervised_weight),
            "chosen_supervised_loss": float(chosen_supervised_loss_value),
            "total_loss": float(reported_total_loss_value),
            **localized_metrics,
        }
        if balanced_reject_pair_loss is not None:
            self._last_dpo_metrics.update({
                "balanced_reject_loss": (
                    balanced_reject_pair_loss.detach().mean().item()
                ),
                "balanced_chosen_budget": (
                    balanced_chosen_budget.detach().mean().item()
                ),
                "balanced_reject_target": (
                    balanced_reject_target.detach().mean().item()
                ),
                "balanced_reject_violation": (
                    balanced_reject_violation.detach().mean().item()
                ),
                "balanced_target_satisfied": (
                    (rejected_ratio.detach() <= balanced_reject_target.detach())
                    .float()
                    .mean()
                    .item()
                ),
            })
        if linear_utility is not None:
            self._last_dpo_metrics.update({
                "linear_objective_loss": dpo_loss.detach().item(),
                "linear_utility": linear_utility.detach().mean().item(),
                "linear_policy_error_gap": (
                    linear_policy_error_gap.detach().mean().item()
                ),
                "linear_direct_accuracy": (
                    (policy_chosen_logp.detach() > policy_rejected_logp.detach())
                    .float()
                    .mean()
                    .item()
                ),
                "linear_effective_pair_weight": (
                    linear_utility.detach()
                    * curriculum_weight.detach()
                ).mean().item(),
            })
        if sigmoid_objective_loss is not None:
            self._last_dpo_metrics["sigmoid_objective_loss"] = (
                sigmoid_objective_loss
            )
            # These objective-scoped diagnostics must not be confused with
            # the legacy aggregate reward curves when a batch mixes Anchored
            # Reject and Sigmoid concepts.
            self._last_dpo_metrics["sigmoid_chosen_reward"] = (
                chosen_ratio.detach().mean().item()
            )
            self._last_dpo_metrics["sigmoid_rejected_reward"] = (
                rejected_ratio.detach().mean().item()
            )
            self._last_dpo_metrics["sigmoid_reward_margin"] = (
                margin.detach().mean().item()
            )

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

    @staticmethod
    def _dpo_cpu_buffer_like(tensor: Tensor) -> Tensor:
        """Allocate an exact CPU buffer, pinned when CUDA supports it."""
        use_pinned_memory = bool(
            tensor.device.type == "cuda"
            and torch.cuda.is_available()
        )
        try:
            return torch.empty_like(
                tensor,
                device="cpu",
                pin_memory=use_pinned_memory,
                memory_format=torch.preserve_format,
            )
        except RuntimeError:
            # Pinned allocations can fail on constrained hosts or non-CUDA
            # compatibility layers. Pageable CPU memory remains exact, only
            # the transfers become synchronous.
            return torch.empty_like(
                tensor,
                device="cpu",
                memory_format=torch.preserve_format,
            )

    @classmethod
    def _dpo_clone_to_cpu(cls, tensor: Tensor) -> Tensor:
        buffer = cls._dpo_cpu_buffer_like(tensor)
        non_blocking = bool(
            tensor.device.type == "cuda"
            and buffer.is_pinned()
        )
        buffer.copy_(tensor.detach(), non_blocking=non_blocking)
        return buffer

    @staticmethod
    def _dpo_sync_cuda_devices(devices: set[torch.device]):
        for device in devices:
            torch.cuda.current_stream(device=device).synchronize()

    def initialize_dpo_reference(
            self,
            model: BaseModel,
            config: TrainConfig,
            snapshot_path: str | None = None,
            force_existing_adapter: bool = False,
            force_cpu_existing_adapter: bool = False,
    ):
        """Initialize a stable existing-adapter reference before training.

        The old implementation captured the reference lazily on the first DPO
        batch, so ordinary training steps before that batch changed the anchor.
        This method is called after the model is on the training device and can
        restore the original snapshot from an OT backup.
        """
        if not getattr(config, "rlhf_enabled", False):
            return

        configured_mode = DPORefMode(config.effective_dpo_ref_mode())
        require_gpu_reference = bool(
            force_existing_adapter
            or configured_mode == DPORefMode.EXISTING_ADAPTER
        )
        require_cpu_reference = bool(
            force_cpu_existing_adapter
            or configured_mode == DPORefMode.EXISTING_ADAPTER_CPU
        )
        require_ema_reference = configured_mode == DPORefMode.EMA_ADAPTER
        create_gpu_reference = bool(
            require_gpu_reference and self._dpo_ref_params is None
        )
        create_cpu_reference = bool(
            require_cpu_reference and self._dpo_ref_params_cpu is None
        )
        create_ema_reference = bool(
            require_ema_reference and self._dpo_ema_ref_params_cpu is None
        )
        if (
            not create_gpu_reference
            and not create_cpu_reference
            and not create_ema_reference
        ):
            return

        adapters = list(model.adapters())
        if len(adapters) == 0:
            raise RuntimeError(
                "RLHF DPO existing-adapter reference requires active adapters"
            )

        loaded_groups = None
        loaded_ema_groups = None
        loaded_ema_decay = None
        loaded_ema_steps = 0

        if snapshot_path and not os.path.isfile(snapshot_path):
            raise RuntimeError(
                "[DPO] resume backup is missing its saved reference: "
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
                loaded_ema_groups = payload.get("ema_adapter_parameters")
                loaded_ema_decay = payload.get("ema_decay")
                loaded_ema_steps = int(payload.get("ema_optimization_steps", 0))
            else:
                loaded_groups = payload

            required_loaded_groups = (
                loaded_ema_groups
                if create_ema_reference
                else loaded_groups
            )
            if not isinstance(required_loaded_groups, (list, tuple)):
                raise RuntimeError(
                    f"Invalid DPO reference snapshot: {snapshot_path}"
                )
            if len(required_loaded_groups) != len(adapters):
                raise RuntimeError(
                    "DPO reference snapshot adapter count mismatch: "
                    f"snapshot={len(required_loaded_groups)}, model={len(adapters)}"
                )

            if create_ema_reference:
                configured_decay = float(config.rlhf_dpo_linear_ema_decay)
                if loaded_ema_decay is None:
                    raise RuntimeError(
                        "Linear-DPO resume snapshot is missing its EMA decay."
                    )
                if abs(float(loaded_ema_decay) - configured_decay) > 1e-12:
                    raise RuntimeError(
                        "Linear-DPO EMA decay changed across resume: "
                        f"snapshot={float(loaded_ema_decay)}, "
                        f"config={configured_decay}."
                    )

        gpu_snapshot_groups = [] if create_gpu_reference else None
        cpu_snapshot_groups = [] if create_cpu_reference else None
        ema_snapshot_groups = [] if create_ema_reference else None
        asynchronous_cpu_devices: set[torch.device] = set()

        for adapter_index, adapter in enumerate(adapters):
            parameters = list(adapter.parameters())
            loaded_parameters = (
                loaded_groups[adapter_index]
                if loaded_groups is not None
                else None
            )
            loaded_ema_parameters = (
                loaded_ema_groups[adapter_index]
                if loaded_ema_groups is not None
                else None
            )

            if loaded_parameters is not None and len(loaded_parameters) != len(parameters):
                raise RuntimeError(
                    "DPO reference snapshot parameter count mismatch for "
                    f"adapter {adapter_index}: snapshot={len(loaded_parameters)}, "
                    f"model={len(parameters)}"
                )
            if (
                loaded_ema_parameters is not None
                and len(loaded_ema_parameters) != len(parameters)
            ):
                raise RuntimeError(
                    "Linear-DPO EMA snapshot parameter count mismatch for "
                    f"adapter {adapter_index}: "
                    f"snapshot={len(loaded_ema_parameters)}, "
                    f"model={len(parameters)}"
                )

            gpu_group = [] if create_gpu_reference else None
            cpu_group = [] if create_cpu_reference else None
            ema_group = [] if create_ema_reference else None
            for parameter_index, parameter in enumerate(parameters):
                if loaded_parameters is not None:
                    source = loaded_parameters[parameter_index]
                elif self._dpo_ref_params is not None:
                    source = self._dpo_ref_params[adapter_index][parameter_index]
                elif self._dpo_ref_params_cpu is not None:
                    source = self._dpo_ref_params_cpu[adapter_index][parameter_index]
                else:
                    source = parameter.detach()

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

                if gpu_group is not None:
                    gpu_group.append(source.to(
                        device=parameter.device,
                        dtype=parameter.dtype,
                    ).clone())

                if cpu_group is not None:
                    cpu_reference = self._dpo_clone_to_cpu(source)
                    if (
                        source.device.type == "cuda"
                        and cpu_reference.is_pinned()
                    ):
                        asynchronous_cpu_devices.add(source.device)
                    cpu_group.append(cpu_reference)

                if ema_group is not None:
                    ema_source = (
                        loaded_ema_parameters[parameter_index]
                        if loaded_ema_parameters is not None
                        else parameter.detach()
                    )
                    if not isinstance(ema_source, torch.Tensor):
                        raise RuntimeError(
                            "Linear-DPO EMA snapshot contains a non-tensor at "
                            f"adapter {adapter_index}, parameter "
                            f"{parameter_index}"
                        )
                    if tuple(ema_source.shape) != tuple(parameter.shape):
                        raise RuntimeError(
                            "Linear-DPO EMA snapshot shape mismatch at adapter "
                            f"{adapter_index}, parameter {parameter_index}: "
                            f"snapshot={tuple(ema_source.shape)}, "
                            f"model={tuple(parameter.shape)}"
                        )
                    ema_group.append(
                        ema_source.detach().to(
                            device="cpu",
                            dtype=torch.float32,
                        ).clone()
                    )

            if gpu_snapshot_groups is not None:
                gpu_snapshot_groups.append(gpu_group)
            if cpu_snapshot_groups is not None:
                cpu_snapshot_groups.append(cpu_group)
            if ema_snapshot_groups is not None:
                ema_snapshot_groups.append(ema_group)

        # Finish asynchronous D2H snapshot capture once. Training can then use
        # the immutable CPU tensors without any synchronization ambiguity.
        self._dpo_sync_cuda_devices(asynchronous_cpu_devices)

        if gpu_snapshot_groups is not None:
            self._dpo_ref_params = gpu_snapshot_groups
        if cpu_snapshot_groups is not None:
            self._dpo_ref_params_cpu = cpu_snapshot_groups
            self._dpo_policy_cpu_buffers = None
        if ema_snapshot_groups is not None:
            self._dpo_ema_ref_params_cpu = ema_snapshot_groups
            self._dpo_ema_policy_cpu_buffers = None
            self._dpo_ema_ref_decay = float(
                config.rlhf_dpo_linear_ema_decay
            )
            self._dpo_ema_ref_steps = (
                loaded_ema_steps if loaded_ema_groups is not None else 0
            )

        if (
            snapshot_path
            and loaded_groups is None
            and loaded_ema_groups is None
        ):
            raise RuntimeError(
                "[DPO] failed to restore the saved reference from "
                f"{snapshot_path}. Refusing unsafe DPO resume."
            )

        storage_modes = []
        if create_gpu_reference:
            storage_modes.append("GPU")
        if create_cpu_reference:
            storage_modes.append("CPU offload")
        if create_ema_reference:
            storage_modes.append("CPU fp32 EMA")
        storage_text = " + ".join(storage_modes)
        if loaded_groups is not None or loaded_ema_groups is not None:
            print(
                f"[OT-RLHF] restored DPO reference from "
                f"{snapshot_path} ({storage_text})"
            )
        elif create_ema_reference:
            print(
                "[OT-RLHF] initialized Linear-DPO EMA reference from the "
                f"loaded adapter ({storage_text})"
            )
        else:
            print(
                "[OT-RLHF] captured fixed existing-adapter DPO reference "
                f"({storage_text})"
            )

    def save_dpo_reference(self, snapshot_path: str):
        reference_groups = (
            self._dpo_ref_params_cpu
            if self._dpo_ref_params_cpu is not None
            else self._dpo_ref_params
        )
        ema_reference_groups = self._dpo_ema_ref_params_cpu
        if reference_groups is None and ema_reference_groups is None:
            return

        os.makedirs(os.path.dirname(snapshot_path) or ".", exist_ok=True)
        payload = {
            "version": 2,
        }
        # Preserve the version-1 field name and structure so checkpoints from
        # the previous implementation remain a strict subset of this payload.
        if reference_groups is not None:
            payload["adapter_parameters"] = [
                [parameter.detach().cpu().clone() for parameter in group]
                for group in reference_groups
            ]
        if ema_reference_groups is not None:
            payload.update({
                "ema_adapter_parameters": [
                    [
                        parameter.detach().to(
                            device="cpu",
                            dtype=torch.float32,
                        ).clone()
                        for parameter in group
                    ]
                    for group in ema_reference_groups
                ],
                "ema_decay": float(self._dpo_ema_ref_decay),
                "ema_optimization_steps": int(self._dpo_ema_ref_steps),
            })
        torch.save(payload, snapshot_path)

    @torch.no_grad()
    def update_dpo_ema_reference(
            self,
            model: BaseModel,
            config: TrainConfig,
    ):
        """Update Linear-DPO's adapter-only EMA after a successful step."""
        if self._dpo_ema_ref_params_cpu is None:
            return
        if DPORefMode(config.effective_dpo_ref_mode()) != DPORefMode.EMA_ADAPTER:
            return

        decay = float(config.rlhf_dpo_linear_ema_decay)
        if self._dpo_ema_ref_decay is None:
            self._dpo_ema_ref_decay = decay
        if abs(float(self._dpo_ema_ref_decay) - decay) > 1e-12:
            raise RuntimeError(
                "Linear-DPO EMA decay changed after initialization."
            )

        adapters = list(model.adapters())
        if len(adapters) != len(self._dpo_ema_ref_params_cpu):
            raise RuntimeError(
                "Linear-DPO EMA reference adapter count changed."
            )

        one_minus_decay = 1.0 - decay
        for adapter_index, (adapter, ema_group) in enumerate(zip(
                adapters,
                self._dpo_ema_ref_params_cpu,
                strict=True,
        )):
            parameters = list(adapter.parameters())
            if len(parameters) != len(ema_group):
                raise RuntimeError(
                    "Linear-DPO EMA reference parameter count changed for "
                    f"adapter {adapter_index}."
                )
            for parameter_index, (parameter, ema_parameter) in enumerate(zip(
                    parameters,
                    ema_group,
                    strict=True,
            )):
                if tuple(parameter.shape) != tuple(ema_parameter.shape):
                    raise RuntimeError(
                        "Linear-DPO EMA reference shape changed at adapter "
                        f"{adapter_index}, parameter {parameter_index}."
                    )
                current = parameter.detach().to(
                    device="cpu",
                    dtype=torch.float32,
                )
                ema_parameter.mul_(decay).add_(
                    current,
                    alpha=one_minus_decay,
                )
        self._dpo_ema_ref_steps += 1

    @staticmethod
    def _load_dpo_concept_reference_payload(
            snapshot_path: str | None,
    ) -> dict[str, list[list[Tensor]]]:
        if not snapshot_path:
            return {}
        if not os.path.isfile(snapshot_path):
            raise RuntimeError(
                "[DPO] resume backup is missing its per-concept reference "
                f"file: {snapshot_path}"
            )
        try:
            payload = torch.load(
                snapshot_path,
                map_location="cpu",
                weights_only=True,
            )
        except TypeError:
            payload = torch.load(snapshot_path, map_location="cpu")

        references = (
            payload.get("concept_references")
            if isinstance(payload, dict)
            else None
        )
        if not isinstance(references, dict):
            raise RuntimeError(
                f"Invalid per-concept DPO reference snapshot: {snapshot_path}"
            )
        return {str(key): value for key, value in references.items()}

    def initialize_dpo_concept_references(
            self,
            model: BaseModel,
            gpu_reference_keys: list[str] | tuple[str, ...] = (),
            cpu_reference_keys: list[str] | tuple[str, ...] = (),
            snapshot_path: str | None = None,
    ):
        """Restore or capture fixed snapshots for explicit concept overrides.

        Keys absent from a restored v5 snapshot intentionally start a new
        reference phase from the adapter currently loaded in the model.  This
        is what makes adding a Sigmoid concept mid-training compare against
        that current adapter instead of an older global Anchored snapshot.
        """
        gpu_keys = list(dict.fromkeys(str(key) for key in gpu_reference_keys))
        cpu_keys = list(dict.fromkeys(str(key) for key in cpu_reference_keys))
        overlap = set(gpu_keys).intersection(cpu_keys)
        if overlap:
            raise RuntimeError(
                "The same DPO concept reference key requested both GPU and "
                f"CPU storage: {sorted(overlap)}"
            )

        missing_gpu = [
            key for key in gpu_keys
            if key not in self._dpo_concept_ref_params
        ]
        missing_cpu = [
            key for key in cpu_keys
            if key not in self._dpo_concept_ref_params_cpu
        ]
        if not missing_gpu and not missing_cpu:
            return

        adapters = list(model.adapters())
        if not adapters:
            raise RuntimeError(
                "Per-concept DPO snapshots require active adapters"
            )
        loaded_references = self._load_dpo_concept_reference_payload(
            snapshot_path
        )
        asynchronous_cpu_devices: set[torch.device] = set()

        def capture(key: str, to_cpu: bool) -> list[list[Tensor]]:
            source_groups = loaded_references.get(key)
            if source_groups is not None and not isinstance(
                    source_groups, (list, tuple)
            ):
                raise RuntimeError(
                    "Invalid per-concept DPO reference entry for key "
                    f"{key!r}"
                )
            if source_groups is not None and len(source_groups) != len(adapters):
                raise RuntimeError(
                    "Per-concept DPO reference adapter count mismatch for "
                    f"key {key!r}: snapshot={len(source_groups)}, "
                    f"model={len(adapters)}"
                )

            result: list[list[Tensor]] = []
            for adapter_index, adapter in enumerate(adapters):
                parameters = list(adapter.parameters())
                sources = (
                    source_groups[adapter_index]
                    if source_groups is not None
                    else parameters
                )
                if len(sources) != len(parameters):
                    raise RuntimeError(
                        "Per-concept DPO reference parameter count mismatch "
                        f"for key {key!r}, adapter {adapter_index}"
                    )

                group: list[Tensor] = []
                for parameter_index, (source, parameter) in enumerate(zip(
                        sources,
                        parameters,
                        strict=True,
                )):
                    if not isinstance(source, Tensor):
                        raise RuntimeError(
                            "Per-concept DPO reference contains a non-tensor "
                            f"for key {key!r}, adapter {adapter_index}, "
                            f"parameter {parameter_index}"
                        )
                    if tuple(source.shape) != tuple(parameter.shape):
                        raise RuntimeError(
                            "Per-concept DPO reference shape mismatch for key "
                            f"{key!r}, adapter {adapter_index}, parameter "
                            f"{parameter_index}: snapshot={tuple(source.shape)}, "
                            f"model={tuple(parameter.shape)}"
                        )
                    if to_cpu:
                        copied = self._dpo_clone_to_cpu(source)
                        if (
                            source.device.type == "cuda"
                            and copied.is_pinned()
                        ):
                            asynchronous_cpu_devices.add(source.device)
                    else:
                        copied = source.detach().to(
                            device=parameter.device,
                            dtype=parameter.dtype,
                        ).clone()
                    group.append(copied)
                result.append(group)
            return result

        for key in missing_gpu:
            self._dpo_concept_ref_params[key] = capture(key, False)
        for key in missing_cpu:
            self._dpo_concept_ref_params_cpu[key] = capture(key, True)
        self._dpo_sync_cuda_devices(asynchronous_cpu_devices)
        if missing_cpu:
            self._dpo_policy_cpu_buffers = None

        restored = sorted(
            key for key in missing_gpu + missing_cpu
            if key in loaded_references
        )
        captured = sorted(
            key for key in missing_gpu + missing_cpu
            if key not in loaded_references
        )
        if restored:
            print(
                "[OT-RLHF] restored per-concept DPO references: "
                + ", ".join(restored)
            )
        if captured:
            print(
                "[OT-RLHF] captured new per-concept DPO references from the "
                "currently loaded adapter: " + ", ".join(captured)
            )

    def save_dpo_concept_references(self, snapshot_path: str):
        keys = sorted(
            set(self._dpo_concept_ref_params)
            | set(self._dpo_concept_ref_params_cpu)
        )
        if not keys:
            return

        references = {}
        for key in keys:
            groups = self._dpo_concept_ref_params_cpu.get(key)
            if groups is None:
                groups = self._dpo_concept_ref_params[key]
            references[key] = [
                [parameter.detach().cpu().clone() for parameter in group]
                for group in groups
            ]

        os.makedirs(os.path.dirname(snapshot_path) or ".", exist_ok=True)
        torch.save(
            {
                "version": 1,
                "concept_references": references,
            },
            snapshot_path,
        )

    @contextmanager
    def reference_model(
            self,
            model: BaseModel,
            config: TrainConfig,
            reference_mode: DPORefMode | None = None,
            reference_key: str | None = None,
    ):
        adapters = model.adapters()

        if config.training_method is not TrainingMethod.LORA:
            raise NotImplementedError(
                "RLHF DPO reference modes are currently only implemented for adapter training in the LoRA tab."
            )
        if len(adapters) == 0:
            raise RuntimeError(
                "RLHF DPO requires active adapters, but no trainable adapters are attached to the current model."
            )

        ref_mode = DPORefMode(
            config.effective_dpo_ref_mode()
            if reference_mode is None
            else reference_mode
        )

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
            if reference_key is not None:
                reference_key = str(reference_key)
                if reference_key not in self._dpo_concept_ref_params:
                    self.initialize_dpo_concept_references(
                        model,
                        gpu_reference_keys=[reference_key],
                    )
                reference_params = self._dpo_concept_ref_params.get(
                    reference_key
                )
            else:
                if self._dpo_ref_params is None:
                    self.initialize_dpo_reference(
                        model,
                        config,
                        force_existing_adapter=True,
                    )
                reference_params = self._dpo_ref_params

            if reference_params is None:
                raise RuntimeError(
                    "Existing-adapter DPO reference was not initialized for "
                    f"key {reference_key!r}"
                )
            if len(reference_params) != len(adapters):
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
                            zip(adapters, reference_params, strict=True)
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
                                reference_params[adapter_index][parameter_index] = ref_data
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

        elif ref_mode in {
            DPORefMode.EXISTING_ADAPTER_CPU,
            DPORefMode.EMA_ADAPTER,
        }:
            if ref_mode == DPORefMode.EMA_ADAPTER:
                if reference_key is not None:
                    raise RuntimeError(
                        "Linear-DPO EMA reference cannot use a per-concept "
                        "fixed-reference key."
                    )
                if self._dpo_ema_ref_params_cpu is None:
                    self.initialize_dpo_reference(model, config)
                reference_params_cpu = self._dpo_ema_ref_params_cpu
                policy_cpu_buffers = self._dpo_ema_policy_cpu_buffers
            elif reference_key is not None:
                reference_key = str(reference_key)
                if reference_key not in self._dpo_concept_ref_params_cpu:
                    self.initialize_dpo_concept_references(
                        model,
                        cpu_reference_keys=[reference_key],
                    )
                reference_params_cpu = self._dpo_concept_ref_params_cpu.get(
                    reference_key
                )
                policy_cpu_buffers = self._dpo_policy_cpu_buffers
            else:
                if self._dpo_ref_params_cpu is None:
                    self.initialize_dpo_reference(
                        model,
                        config,
                        force_cpu_existing_adapter=True,
                    )
                reference_params_cpu = self._dpo_ref_params_cpu
                policy_cpu_buffers = self._dpo_policy_cpu_buffers

            if reference_params_cpu is None:
                raise RuntimeError(
                    "CPU-offloaded DPO reference was not "
                    f"initialized for key {reference_key!r}"
                )
            if len(reference_params_cpu) != len(adapters):
                raise RuntimeError(
                    "CPU-offloaded DPO reference adapter count changed"
                )

            if policy_cpu_buffers is None:
                policy_cpu_buffers = [
                    [
                        self._dpo_cpu_buffer_like(parameter)
                        for parameter in adapter.parameters()
                    ]
                    for adapter in adapters
                ]
                if ref_mode == DPORefMode.EMA_ADAPTER:
                    self._dpo_ema_policy_cpu_buffers = policy_cpu_buffers
                else:
                    self._dpo_policy_cpu_buffers = policy_cpu_buffers

            parameter_groups = [
                list(adapter.parameters())
                for adapter in adapters
            ]
            for adapter_index, (
                    parameters,
                    ref_params,
                    policy_buffers,
            ) in enumerate(zip(
                parameter_groups,
                reference_params_cpu,
                policy_cpu_buffers,
                strict=True,
            )):
                if (
                    len(parameters) != len(ref_params)
                    or len(parameters) != len(policy_buffers)
                ):
                    raise RuntimeError(
                        "CPU-offloaded DPO reference parameter count changed "
                        f"for adapter {adapter_index}"
                    )
                for parameter_index, (
                        parameter,
                        ref_data,
                        policy_buffer,
                ) in enumerate(zip(
                    parameters,
                    ref_params,
                    policy_buffers,
                    strict=True,
                )):
                    if (
                        tuple(parameter.shape) != tuple(ref_data.shape)
                        or tuple(parameter.shape) != tuple(policy_buffer.shape)
                    ):
                        raise RuntimeError(
                            "CPU-offloaded DPO reference shape changed at "
                            f"adapter {adapter_index}, parameter "
                            f"{parameter_index}"
                        )

            policy_stashed = False
            try:
                with torch.no_grad():
                    # Queue every D2H policy copy before overwriting any live
                    # adapter tensor. On CUDA the pinned copies and following
                    # H2D copies are ordered on the current stream.
                    for parameters, policy_buffers in zip(
                            parameter_groups,
                            policy_cpu_buffers,
                            strict=True,
                    ):
                        for parameter, policy_buffer in zip(
                                parameters,
                                policy_buffers,
                                strict=True,
                        ):
                            policy_buffer.copy_(
                                parameter.detach(),
                                non_blocking=bool(
                                    parameter.device.type == "cuda"
                                    and policy_buffer.is_pinned()
                                ),
                            )
                    policy_stashed = True

                    for parameters, ref_params in zip(
                            parameter_groups,
                            reference_params_cpu,
                            strict=True,
                    ):
                        for parameter, ref_data in zip(
                                parameters,
                                ref_params,
                                strict=True,
                        ):
                            parameter.copy_(
                                ref_data,
                                non_blocking=bool(
                                    parameter.device.type == "cuda"
                                    and ref_data.is_pinned()
                                ),
                            )
                yield
            finally:
                # Restore into the same Parameter storage. Optimizer, DDP and
                # compiled-module references therefore remain valid, while
                # the only extra adapter-sized allocations live on the CPU.
                if policy_stashed:
                    with torch.no_grad():
                        for parameters, policy_buffers in zip(
                            parameter_groups,
                            policy_cpu_buffers,
                            strict=True,
                        ):
                            for parameter, policy_buffer in zip(
                                parameters,
                                policy_buffers,
                                strict=True,
                            ):
                                parameter.copy_(
                                    policy_buffer,
                                    non_blocking=bool(
                                        parameter.device.type == "cuda"
                                        and policy_buffer.is_pinned()
                                    ),
                                )
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
