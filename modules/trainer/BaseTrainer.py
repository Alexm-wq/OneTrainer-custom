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
