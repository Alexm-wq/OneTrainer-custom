from __future__ import annotations

import io
import os
from collections.abc import Iterable
from contextlib import contextmanager, redirect_stdout

from modules.module.MageFlowSelfFlow import MageFlowSelfFlowEMA
from modules.util.enum.DPORefMode import DPORefMode

import torch
from torch import Tensor, nn


_TRUE_VALUES = {"1", "true", "yes", "on", "gpu", "cuda"}
_FALSE_VALUES = {"0", "false", "no", "off", "cpu", "host"}


def _parse_gpu_choice(value: str, source: str) -> bool:
    normalized = value.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise ValueError(
        f"{source} must be one of cpu, gpu, false, true, 0, or 1"
    )


def _read_gpu_choice(config, config_attr: str, env_name: str) -> bool:
    """Resolve EMA residency, with explicit environment overrides taking priority."""
    raw = os.environ.get(env_name)
    source = env_name
    if raw is None:
        raw = os.environ.get("OT_MAGE_EMA_DEVICE")
        source = "OT_MAGE_EMA_DEVICE"
    if raw is not None:
        return _parse_gpu_choice(raw, source)

    configured = getattr(config, config_attr, None)
    if configured is None:
        return False
    if isinstance(configured, str):
        return _parse_gpu_choice(configured, config_attr)
    return bool(configured)


def _module_device(module) -> str:
    if module is None:
        return "absent"
    try:
        parameters = module.parameters()
        parameter = next(iter(parameters))
    except (StopIteration, AttributeError, TypeError):
        return "no-parameters"
    return f"{parameter.device}/{str(parameter.dtype).removeprefix('torch.')}"


def _first_nested_tensor_device(groups) -> str:
    if groups is None:
        return "not-initialized"
    for group in groups:
        for tensor in group:
            return f"{tensor.device}/{str(tensor.dtype).removeprefix('torch.')}"
    return "empty"


def log_mage_runtime_devices(setup, model, config, *, phase: str) -> None:
    """Print actual live residency after setup/reference relocation has completed."""
    self_flow_ema = getattr(model, "self_flow_ema", None)
    if self_flow_ema is None:
        self_flow_ema_device = "disabled"
        self_flow_shadow_device = "disabled"
    else:
        ema_parameters = getattr(self_flow_ema, "ema_parameters", [])
        student_parameters = getattr(self_flow_ema, "student_parameters", [])
        self_flow_ema_device = (
            f"{ema_parameters[0].device}/{str(ema_parameters[0].dtype).removeprefix('torch.')}"
            if ema_parameters else "empty"
        )
        self_flow_shadow_device = (
            f"{student_parameters[0].device}/{str(student_parameters[0].dtype).removeprefix('torch.')}"
            if student_parameters else "empty"
        )

    dpo_ema = _first_nested_tensor_device(
        getattr(setup, "_dpo_ema_ref_params_cpu", None)
    )
    dpo_stash = _first_nested_tensor_device(
        getattr(setup, "_dpo_ema_policy_cpu_buffers", None)
    )

    print(f"[Mage-Flow devices] {phase}")
    print(f"  transformer:       {_module_device(getattr(model, 'transformer', None))}")
    print(f"  LoRA adapter:      {_module_device(getattr(model, 'transformer_lora', None))}")
    print(f"  text encoder:      {_module_device(getattr(model, 'text_encoder', None))}")
    print(f"  VAE:               {_module_device(getattr(model, 'vae', None))}")
    print(f"  Self-Flow EMA:     {self_flow_ema_device}")
    print(f"  Self-Flow shadow:  {self_flow_shadow_device}")
    print(f"  Linear-DPO EMA:    {dpo_ema}")
    print(f"  Linear-DPO stash:  {dpo_stash}")
    print(
        "  requested:         "
        f"self-flow={'GPU' if _read_gpu_choice(config, 'self_flow_ema_on_gpu', 'OT_MAGE_SELF_FLOW_EMA_DEVICE') else 'CPU'}, "
        f"linear-dpo={'GPU' if _read_gpu_choice(config, 'rlhf_dpo_linear_ema_on_gpu', 'OT_MAGE_LINEAR_DPO_EMA_DEVICE') else 'CPU'}"
    )


def mage_self_flow_ema_device(config, parameter: Tensor) -> torch.device:
    use_gpu = _read_gpu_choice(
        config,
        "self_flow_ema_on_gpu",
        "OT_MAGE_SELF_FLOW_EMA_DEVICE",
    )
    if not use_gpu:
        return torch.device("cpu")
    if parameter.device.type != "cuda":
        raise RuntimeError(
            "Mage Self-Flow GPU EMA was requested, but the trainable policy "
            f"parameter is on {parameter.device} instead of CUDA."
        )
    return parameter.device


class MageFlowSelfFlowDeviceEMA(MageFlowSelfFlowEMA):
    """Mage Self-Flow EMA whose float32 storage can remain on the training GPU."""

    def __init__(
            self,
            modules: Iterable[nn.Module],
            decay: float = 0.9999,
            state_dict: dict | None = None,
            *,
            storage_device: torch.device,
    ):
        self.storage_device = torch.device(storage_device)
        if self.storage_device.type != "cuda":
            raise ValueError(
                "MageFlowSelfFlowDeviceEMA is only needed for CUDA storage"
            )
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Mage Self-Flow GPU EMA was requested but CUDA is unavailable"
            )
        super().__init__(modules, decay=decay, state_dict=state_dict)

    def _cpu_copy(self, parameter: Tensor) -> Tensor:
        # Base MageFlowSelfFlowEMA routes persistent storage through this helper.
        # Override the storage primitive only; swap/update semantics stay intact.
        return parameter.detach().to(
            device=self.storage_device,
            dtype=torch.float32,
        ).clone()

    def state_dict(self) -> dict:
        # Backups remain device-agnostic even with a live CUDA EMA.
        return {
            "decay": self.decay,
            "optimization_steps": self.optimization_steps,
            "ema_parameters": [
                parameter.detach().to(
                    device="cpu",
                    dtype=torch.float32,
                ).clone()
                for parameter in self.ema_parameters
            ],
        }

    def load_state_dict(self, state: dict):
        stored = state.get("ema_parameters")
        if stored is None or len(stored) != len(self.parameters):
            raise RuntimeError(
                "Mage Self-Flow checkpoint EMA parameter count mismatch"
            )

        loaded = []
        for index, (source, parameter) in enumerate(
                zip(stored, self.parameters, strict=True)
        ):
            if source.shape != parameter.shape:
                raise RuntimeError(
                    "Mage Self-Flow checkpoint EMA shape mismatch at "
                    f"parameter {index}"
                )
            loaded.append(
                source.detach().to(
                    device=self.storage_device,
                    dtype=torch.float32,
                ).clone()
            )

        self.decay = float(state.get("decay", self.decay))
        self.optimization_steps = int(state.get("optimization_steps", 0))
        self.ema_parameters = loaded
        self.student_parameters = [
            self._cpu_copy(parameter)
            for parameter in self.parameters
        ]


def create_mage_self_flow_ema(
        modules: Iterable[nn.Module],
        config,
        parameter: Tensor,
        *,
        state_dict: dict | None = None,
) -> MageFlowSelfFlowEMA:
    storage_device = mage_self_flow_ema_device(config, parameter)
    if storage_device.type == "cuda":
        ema = MageFlowSelfFlowDeviceEMA(
            modules,
            decay=config.self_flow_ema_decay,
            state_dict=state_dict,
            storage_device=storage_device,
        )
        print(
            "[Mage Self-Flow] EMA storage=GPU FP32 "
            f"({storage_device}); teacher swaps remain on-device"
        )
        return ema

    print("[Mage Self-Flow] EMA storage=CPU FP32")
    return MageFlowSelfFlowEMA(
        modules,
        decay=config.self_flow_ema_decay,
        state_dict=state_dict,
    )


class MageFlowLinearDPOGPUReferenceMixin:
    """Optional CUDA-resident Linear-DPO EMA for Mage LoRA training."""

    @staticmethod
    def _mage_linear_dpo_gpu_requested(config) -> bool:
        return _read_gpu_choice(
            config,
            "rlhf_dpo_linear_ema_on_gpu",
            "OT_MAGE_LINEAR_DPO_EMA_DEVICE",
        )

    def initialize_dpo_reference(
            self,
            model,
            config,
            snapshot_path: str | None = None,
            force_existing_adapter: bool = False,
            force_cpu_existing_adapter: bool = False,
    ):
        gpu_ema = (
            DPORefMode(config.effective_dpo_ref_mode()) == DPORefMode.EMA_ADAPTER
            and self._mage_linear_dpo_gpu_requested(config)
        )

        # The generic loader intentionally deserializes reference checkpoints on
        # CPU first. In GPU mode that is only staging, not the final runtime
        # residency. Rewrite that one diagnostic so it cannot falsely claim the
        # active EMA remains on CPU.
        if gpu_ema:
            captured = io.StringIO()
            with redirect_stdout(captured):
                result = super().initialize_dpo_reference(
                    model,
                    config,
                    snapshot_path,
                    force_existing_adapter=force_existing_adapter,
                    force_cpu_existing_adapter=force_cpu_existing_adapter,
                )
            for line in captured.getvalue().splitlines():
                if "restored DPO reference from" in line and "CPU fp32 EMA" in line:
                    line = line.replace(
                        "CPU fp32 EMA",
                        "CPU fp32 checkpoint staging for EMA",
                    )
                print(line)
        else:
            result = super().initialize_dpo_reference(
                model,
                config,
                snapshot_path,
                force_existing_adapter=force_existing_adapter,
                force_cpu_existing_adapter=force_cpu_existing_adapter,
            )

        if not gpu_ema:
            log_mage_runtime_devices(
                self,
                model,
                config,
                phase="DPO reference initialized (final runtime residency)",
            )
            return result

        references = self._dpo_ema_ref_params_cpu
        if references is None:
            return result

        adapters = list(model.adapters())
        if len(adapters) != len(references):
            raise RuntimeError(
                "Linear-DPO GPU EMA reference adapter count mismatch"
            )

        gpu_references = []
        gpu_policy_buffers = []
        for adapter_index, (adapter, ref_group) in enumerate(
                zip(adapters, references, strict=True)
        ):
            parameters = list(adapter.parameters())
            if len(parameters) != len(ref_group):
                raise RuntimeError(
                    "Linear-DPO GPU EMA parameter count mismatch for "
                    f"adapter {adapter_index}"
                )

            gpu_ref_group = []
            gpu_buffer_group = []
            for parameter_index, (parameter, reference) in enumerate(
                    zip(parameters, ref_group, strict=True)
            ):
                if parameter.device.type != "cuda":
                    raise RuntimeError(
                        "Linear-DPO GPU EMA was requested, but adapter "
                        f"{adapter_index} parameter {parameter_index} is on "
                        f"{parameter.device} instead of CUDA."
                    )
                if tuple(parameter.shape) != tuple(reference.shape):
                    raise RuntimeError(
                        "Linear-DPO GPU EMA shape mismatch at adapter "
                        f"{adapter_index}, parameter {parameter_index}"
                    )

                gpu_ref_group.append(
                    reference.detach().to(
                        device=parameter.device,
                        dtype=torch.float32,
                    ).clone()
                )
                gpu_buffer_group.append(
                    torch.empty_like(
                        parameter,
                        device=parameter.device,
                        memory_format=torch.preserve_format,
                    )
                )

            gpu_references.append(gpu_ref_group)
            gpu_policy_buffers.append(gpu_buffer_group)

        # Retain legacy attribute names for generic backup compatibility; their
        # tensors are CUDA-resident in this mode despite the historical suffix.
        self._dpo_ema_ref_params_cpu = gpu_references
        self._dpo_ema_policy_cpu_buffers = gpu_policy_buffers

        first_device = gpu_references[0][0].device if gpu_references and gpu_references[0] else "cuda"
        print(
            "[OT-RLHF] Linear-DPO EMA active runtime storage=GPU FP32 "
            f"({first_device}); policy stash remains on-device"
        )
        log_mage_runtime_devices(
            self,
            model,
            config,
            phase="DPO reference initialized (final runtime residency)",
        )
        return result

    @torch.no_grad()
    def update_dpo_ema_reference(self, model, config):
        if not self._mage_linear_dpo_gpu_requested(config):
            return super().update_dpo_ema_reference(model, config)

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
                "Linear-DPO GPU EMA reference adapter count changed."
            )

        one_minus_decay = 1.0 - decay
        for adapter_index, (adapter, ema_group) in enumerate(
                zip(adapters, self._dpo_ema_ref_params_cpu, strict=True)
        ):
            parameters = list(adapter.parameters())
            if len(parameters) != len(ema_group):
                raise RuntimeError(
                    "Linear-DPO GPU EMA parameter count changed for "
                    f"adapter {adapter_index}."
                )

            for parameter_index, (parameter, ema_parameter) in enumerate(
                    zip(parameters, ema_group, strict=True)
            ):
                if tuple(parameter.shape) != tuple(ema_parameter.shape):
                    raise RuntimeError(
                        "Linear-DPO GPU EMA shape changed at adapter "
                        f"{adapter_index}, parameter {parameter_index}."
                    )
                if ema_parameter.device != parameter.device:
                    raise RuntimeError(
                        "Linear-DPO GPU EMA device changed at adapter "
                        f"{adapter_index}, parameter {parameter_index}: "
                        f"EMA={ema_parameter.device}, policy={parameter.device}."
                    )

                ema_parameter.mul_(decay).add_(
                    parameter.detach(),
                    alpha=one_minus_decay,
                )

        self._dpo_ema_ref_steps += 1

    @contextmanager
    def reference_model(
            self,
            model,
            config,
            reference_mode: DPORefMode | None = None,
            reference_key: str | None = None,
    ):
        ref_mode = DPORefMode(
            config.effective_dpo_ref_mode()
            if reference_mode is None
            else reference_mode
        )
        if (
            ref_mode != DPORefMode.EMA_ADAPTER
            or not self._mage_linear_dpo_gpu_requested(config)
        ):
            with super().reference_model(
                model,
                config,
                reference_mode=reference_mode,
                reference_key=reference_key,
            ) as value:
                yield value
            return

        if reference_key is not None:
            raise RuntimeError(
                "Linear-DPO EMA reference cannot use a per-concept "
                "fixed-reference key."
            )

        adapters = list(model.adapters())
        if not adapters:
            raise RuntimeError(
                "RLHF DPO requires active adapters, but no trainable adapters "
                "are attached to the current model."
            )
        if self._dpo_ema_ref_params_cpu is None:
            self.initialize_dpo_reference(model, config)

        references = self._dpo_ema_ref_params_cpu
        if references is None or len(references) != len(adapters):
            raise RuntimeError(
                "Linear-DPO GPU EMA reference was not initialized correctly."
            )

        buffers = self._dpo_ema_policy_cpu_buffers
        if buffers is None:
            buffers = [
                [
                    torch.empty_like(
                        parameter,
                        device=parameter.device,
                        memory_format=torch.preserve_format,
                    )
                    for parameter in adapter.parameters()
                ]
                for adapter in adapters
            ]
            self._dpo_ema_policy_cpu_buffers = buffers

        parameter_groups = [
            list(adapter.parameters())
            for adapter in adapters
        ]
        for adapter_index, (parameters, ref_group, buffer_group) in enumerate(
                zip(parameter_groups, references, buffers, strict=True)
        ):
            if (
                len(parameters) != len(ref_group)
                or len(parameters) != len(buffer_group)
            ):
                raise RuntimeError(
                    "Linear-DPO GPU EMA parameter count changed for "
                    f"adapter {adapter_index}."
                )
            for parameter_index, (parameter, reference, buffer) in enumerate(
                    zip(parameters, ref_group, buffer_group, strict=True)
            ):
                if (
                    tuple(parameter.shape) != tuple(reference.shape)
                    or tuple(parameter.shape) != tuple(buffer.shape)
                ):
                    raise RuntimeError(
                        "Linear-DPO GPU EMA shape changed at adapter "
                        f"{adapter_index}, parameter {parameter_index}."
                    )
                if (
                    parameter.device.type != "cuda"
                    or reference.device != parameter.device
                    or buffer.device != parameter.device
                ):
                    raise RuntimeError(
                        "Linear-DPO GPU EMA tensors must remain on the same "
                        f"CUDA device at adapter {adapter_index}, parameter "
                        f"{parameter_index}: policy={parameter.device}, "
                        f"EMA={reference.device}, stash={buffer.device}."
                    )

        policy_stashed = False
        try:
            with torch.no_grad():
                for parameters, buffer_group in zip(
                        parameter_groups, buffers, strict=True
                ):
                    for parameter, buffer in zip(
                            parameters, buffer_group, strict=True
                    ):
                        buffer.copy_(parameter.detach())
                policy_stashed = True

                for parameters, ref_group in zip(
                        parameter_groups, references, strict=True
                ):
                    for parameter, reference in zip(
                            parameters, ref_group, strict=True
                    ):
                        parameter.copy_(reference)
            yield
        finally:
            if policy_stashed:
                with torch.no_grad():
                    for parameters, buffer_group in zip(
                            parameter_groups, buffers, strict=True
                    ):
                        for parameter, buffer in zip(
                                parameters, buffer_group, strict=True
                        ):
                            parameter.copy_(buffer)
