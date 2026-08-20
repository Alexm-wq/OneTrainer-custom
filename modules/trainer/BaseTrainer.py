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

    def _GenericTrainer__backward_dpo_with_gradient_probe(self, loss: torch.Tensor):
        """Backward a DPO component while recording its incoming gradient L2 norm."""
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
        gradient_magnitude = math.sqrt(max(total_sq, 0.0))
        if not math.isfinite(gradient_magnitude):
            raise RuntimeError("DPO gradient magnitude became NaN or Inf.")

        train_progress = getattr(getattr(self, "model", None), "train_progress", None)
        global_step = int(getattr(train_progress, "global_step", -1))

        self.tensorboard.add_scalar(
            "rlhf/dpo_gradient_magnitude",
            gradient_magnitude,
            global_step,
        )

        csv_path = Path(__file__).resolve().parents[2] / "gradient_magnitude.csv"
        write_header = not csv_path.exists() or csv_path.stat().st_size == 0
        with csv_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            if write_header:
                writer.writerow(["global_step", "dpo_gradient_magnitude"])
            writer.writerow([global_step, f"{gradient_magnitude:.12g}"])

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
