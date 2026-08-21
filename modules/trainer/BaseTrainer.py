import csv
import math
import os
import shutil
import subprocess
from abc import ABCMeta, abstractmethod
from pathlib import Path

from modules.model.BaseModel import BaseModel
from modules.modelLoader.BaseModelLoader import BaseModelLoader
from modules.modelSampler.BaseModelSampler import BaseModelSampler
from modules.modelSaver.BaseModelSaver import BaseModelSaver
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.util import create
from modules.util.callbacks.TrainCallbacks import TrainCallbacks
from modules.util.commands.TrainCommands import TrainCommands
from modules.util.config.TrainConfig import TrainConfig
from modules.util.encryption_util import configure_data_encryption
from modules.util.TimedActionMixin import TimedActionMixin
from modules.util.TrainProgress import TrainProgress

import torch


class BaseTrainer(
    TimedActionMixin,
    metaclass=ABCMeta,
):

    tensorboard_subprocess: subprocess.Popen

    def __init__(
            self,
            config: TrainConfig,
            callbacks: TrainCallbacks,
            commands: TrainCommands,
            *,
            require_encryption_key: bool = True,
    ):
        super().__init__()
        self.config = config
        configure_data_encryption(
            self.config,
            require_key=require_encryption_key,
        )
        self.callbacks = callbacks
        self.commands = commands
        self.train_device = torch.device(self.config.train_device)
        self.temp_device = torch.device(self.config.temp_device)
        # GenericTrainer accumulates a raw DPO item fraction into
        # _dpo_bypass_update_weight. SinkSGD normalizes the captured DPO
        # gradient before applying it, so scalar curriculum attenuation would
        # otherwise disappear. The property below converts that raw fraction
        # into the effective post-normalization step fraction.
        self._dpo_bypass_effective_update_weight = 0.0
        self._dpo_curriculum_metric_sum_seen = 0.0
        self._dpo_curriculum_metric_weight_seen = 0.0

    @property
    def _dpo_bypass_update_weight(self) -> float:
        return self._dpo_bypass_effective_update_weight

    @_dpo_bypass_update_weight.setter
    def _dpo_bypass_update_weight(self, value: float):
        """Preserve DPO curriculum magnitude through SinkSGD normalization.

        GenericTrainer updates this field with ``+= dpo_item_fraction / GA``
        after each RLHF microbatch. The DPO gradient already contains each
        pair's detached curriculum weight, which correctly changes the weighted
        gradient direction, but SinkSGD subsequently Sinkhorn/sign-normalizes
        that gradient and removes its global scalar attenuation. For SinkSGD's
        momentum-bypass step only, reapply the pair-count-weighted curriculum
        mean to the step size.

        The DPO metric accumulator is updated once per dispatch group before
        GenericTrainer increments this field. Taking deltas from that accumulator
        therefore gives the exact mean curriculum weight for the just-finished
        microbatch, including mixed per-concept objective dispatch groups. When
        curriculum is disabled the metric is 1.0, preserving the old behavior.
        """
        value = float(value)
        if not math.isfinite(value):
            raise RuntimeError(
                f"Non-finite DPO momentum-bypass update weight: {value!r}"
            )

        # GenericTrainer resets the accumulator to zero after applying the
        # isolated DPO update. Keep metric cursors intact: TensorBoard metric
        # accumulators may continue across optimizer windows.
        if value == 0.0:
            self._dpo_bypass_effective_update_weight = 0.0
            return

        current = float(self._dpo_bypass_effective_update_weight)
        raw_increment = value - current
        if raw_increment < -1e-12:
            raise RuntimeError(
                "DPO momentum-bypass update weight moved backwards without "
                "being reset to zero."
            )
        if raw_increment <= 0.0:
            return

        optimizer = getattr(getattr(self, "model", None), "optimizer", None)
        is_sinksgd = (
            optimizer is not None
            and optimizer.__class__.__name__ == "SinkSGD_adv"
        )
        if not is_sinksgd:
            self._dpo_bypass_effective_update_weight = value
            return

        metric_name = "hard_pair_curriculum_weight"
        metric_sums = getattr(self, "_dpo_metric_sums", {})
        metric_weights = getattr(self, "_dpo_metric_weights", {})
        current_sum = float(metric_sums.get(metric_name, 0.0))
        current_weight = float(metric_weights.get(metric_name, 0.0))

        previous_sum = float(self._dpo_curriculum_metric_sum_seen)
        previous_weight = float(self._dpo_curriculum_metric_weight_seen)

        # The trainer periodically flushes its metric accumulators. Detect that
        # reset and treat the current totals as the new delta.
        if current_weight + 1e-12 < previous_weight:
            delta_sum = current_sum
            delta_weight = current_weight
        else:
            delta_sum = current_sum - previous_sum
            delta_weight = current_weight - previous_weight

        self._dpo_curriculum_metric_sum_seen = current_sum
        self._dpo_curriculum_metric_weight_seen = current_weight

        if delta_weight > 0.0:
            curriculum_scale = delta_sum / delta_weight
            if (
                not math.isfinite(curriculum_scale)
                or curriculum_scale < -1e-6
                or curriculum_scale > 1.0 + 1e-6
            ):
                raise RuntimeError(
                    "Invalid DPO hard-pair curriculum scale for SinkSGD "
                    f"momentum bypass: {curriculum_scale!r}"
                )
            curriculum_scale = min(1.0, max(0.0, curriculum_scale))
        else:
            # Defensive compatibility fallback. Older/custom model setups that
            # do not report the curriculum metric retain the historical scale.
            curriculum_scale = 1.0

        self._dpo_bypass_effective_update_weight = (
            current + raw_increment * curriculum_scale
        )

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def end(self):
        pass

    def create_model_loader(self) -> BaseModelLoader:
        return create.create_model_loader(self.config.model_type, self.config.training_method)

    def create_model_setup(self) -> BaseModelSetup:
        return create.create_model_setup(
            self.config.model_type,
            self.train_device,
            self.temp_device,
            self.config.training_method,
            self.config.debug_mode,
        )

    def create_data_loader(self, model: BaseModel, model_setup: BaseModelSetup, train_progress: TrainProgress, is_validation=False):
        return create.create_data_loader(
            self.train_device,
            self.temp_device,
            model,
            self.config.model_type,
            model_setup,
            self.config.training_method,
            self.config,
            train_progress,
            is_validation,
        )

    def create_model_saver(self) -> BaseModelSaver:
        return create.create_model_saver(self.config.model_type, self.config.training_method)

    def create_model_sampler(self, model: BaseModel) -> BaseModelSampler:
        return create.create_model_sampler(
            self.train_device,
            self.temp_device,
            model,
            self.config.model_type,
            self.config.training_method
        )

    def _gradient_l2_from_parameter_grads(self) -> float:
        total_sq = 0.0
        for parameter in self.parameters:
            grad = parameter.grad
            if grad is None:
                continue
            detached_f32 = grad.detach().float()
            total_sq += float(
                torch.sum(detached_f32 * detached_f32).detach().cpu().item()
            )
        return math.sqrt(max(total_sq, 0.0))

    @staticmethod
    def _format_gradient_value(value: float | None) -> str:
        return "" if value is None else f"{value:.12g}"

    def _write_gradient_magnitude_row(
            self,
            global_step: int,
            self_flow_gradient_magnitude: float,
            dpo_gradient_magnitude: float,
    ):
        csv_path = Path(__file__).resolve().parents[2] / "gradient_magnitude.csv"
        fieldnames = [
            "global_step",
            "self_flow_gradient_magnitude",
            "dpo_gradient_magnitude",
            "dpo_to_self_flow_ratio",
        ]

        # Transparently upgrade the original two-column CSV while preserving
        # rows already collected before Self-Flow probing was added.
        if csv_path.exists() and csv_path.stat().st_size > 0:
            with csv_path.open("r", newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                old_fieldnames = reader.fieldnames or []
                old_rows = list(reader)
            if old_fieldnames != fieldnames:
                with csv_path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fieldnames)
                    writer.writeheader()
                    for row in old_rows:
                        writer.writerow({
                            "global_step": row.get("global_step", ""),
                            "self_flow_gradient_magnitude": row.get(
                                "self_flow_gradient_magnitude", ""
                            ),
                            "dpo_gradient_magnitude": row.get(
                                "dpo_gradient_magnitude", ""
                            ),
                            "dpo_to_self_flow_ratio": row.get(
                                "dpo_to_self_flow_ratio", ""
                            ),
                        })

        ratio = (
            dpo_gradient_magnitude / self_flow_gradient_magnitude
            if self_flow_gradient_magnitude > 0.0
            else None
        )
        write_header = not csv_path.exists() or csv_path.stat().st_size == 0
        with csv_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow({
                "global_step": global_step,
                "self_flow_gradient_magnitude": self._format_gradient_value(
                    self_flow_gradient_magnitude
                ),
                "dpo_gradient_magnitude": self._format_gradient_value(
                    dpo_gradient_magnitude
                ),
                "dpo_to_self_flow_ratio": self._format_gradient_value(ratio),
            })

    def _GenericTrainer__backward_dpo_with_gradient_probe(self, loss: torch.Tensor):
        """Backward DPO while recording DPO and active Self-Flow magnitudes.

        Historically this probe labeled the full pre-existing ``parameter.grad``
        norm as Self-Flow. That is only meaningful when Self-Flow is enabled;
        otherwise those gradients can be ordinary supervised gradients or
        carry-over from gradient accumulation. Never report those as Self-Flow.
        """
        self_flow_enabled = bool(
            getattr(self.config, "self_flow_enabled", False)
        )
        self_flow_gradient_magnitude = (
            self._gradient_l2_from_parameter_grads()
            if self_flow_enabled
            else 0.0
        )
        if not math.isfinite(self_flow_gradient_magnitude):
            raise RuntimeError("Self-Flow gradient magnitude became NaN or Inf.")

        grad_sq_by_device: dict[torch.device, torch.Tensor] = {}
        handles = []

        def capture(grad: torch.Tensor | None):
            if grad is None:
                return None
            detached_f32 = grad.detach().float()
            sq = torch.sum(detached_f32 * detached_f32)
            existing = grad_sq_by_device.get(grad.device)
            grad_sq_by_device[grad.device] = sq if existing is None else existing + sq
            return grad

        try:
            for parameter in self.parameters:
                if parameter.requires_grad:
                    handles.append(parameter.register_hook(capture))

            # Some local/custom GenericTrainer revisions route DPO backward
            # through this probe. Preserve the actual momentum-bypass semantics
            # in that case instead of turning the logger into a training change.
            bypass_enabled_fn = getattr(
                self,
                "_GenericTrainer__dpo_momentum_bypass_enabled",
                None,
            )
            bypass_backward_fn = getattr(
                self,
                "_GenericTrainer__backward_dpo_without_momentum",
                None,
            )
            if (
                callable(bypass_enabled_fn)
                and callable(bypass_backward_fn)
                and bool(bypass_enabled_fn())
            ):
                # Probe hooks were registered first, so they observe the
                # original incoming gradient before the bypass hook zeroes the
                # transient tensor and stores its FP32 CPU copy.
                bypass_backward_fn(loss)
            else:
                loss.backward()
        finally:
            for handle in handles:
                handle.remove()

        if not hasattr(self, "tensorboard"):
            return

        total_sq = sum(
            float(value.detach().cpu().item())
            for value in grad_sq_by_device.values()
        )
        dpo_gradient_magnitude = math.sqrt(max(total_sq, 0.0))
        if not math.isfinite(dpo_gradient_magnitude):
            raise RuntimeError("DPO gradient magnitude became NaN or Inf.")

        train_progress = getattr(getattr(self, "model", None), "train_progress", None)
        global_step = int(getattr(train_progress, "global_step", -1))

        self.tensorboard.add_scalar(
            "rlhf/self_flow_gradient_magnitude",
            self_flow_gradient_magnitude,
            global_step,
        )
        self.tensorboard.add_scalar(
            "rlhf/dpo_gradient_magnitude",
            dpo_gradient_magnitude,
            global_step,
        )
        if self_flow_gradient_magnitude > 0.0:
            self.tensorboard.add_scalar(
                "rlhf/dpo_to_self_flow_gradient_ratio",
                dpo_gradient_magnitude / self_flow_gradient_magnitude,
                global_step,
            )

        self._write_gradient_magnitude_row(
            global_step,
            self_flow_gradient_magnitude,
            dpo_gradient_magnitude,
        )

    def _start_tensorboard(self):
        tensorboard_executable = shutil.which("tensorboard")
        tensorboard_log_dir = os.path.join(self.config.workspace_dir, "tensorboard")

        tensorboard_args = [
            tensorboard_executable,
            "--logdir",
            tensorboard_log_dir,
            "--port",
            str(self.config.tensorboard_port),
            "--samples_per_plugin=images=100,scalars=10000",
        ]

        if self.config.tensorboard_expose:
            tensorboard_args.append("--bind_all")

        self.tensorboard_subprocess = subprocess.Popen(tensorboard_args)

    def _stop_tensorboard(self):
        self.tensorboard_subprocess.kill()
