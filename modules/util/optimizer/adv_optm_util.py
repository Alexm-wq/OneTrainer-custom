"""Lossless loading for adv-optm's mixed-precision optimizer states."""
import torch


# These dense states follow parameter precision only in the automatic mode.
_AUTO_MOMENTS = {"exp_avg", "exp_avg_sq", "momentum_buffer", "second_momentum_buffer"}
_PARAMETER_STATES = {"p0", "s", "last_grad"}


def load_adv_optimizer_state_dict(optimizer: torch.optim.Optimizer, state_dict: dict) -> None:
    """Retain signed quantized states and FP32 moments when resuming.

    adv-optm 2.5.16 assumes every group has actual_state_precision, although
    Lion and auxiliary Adam groups do not write that key. Its post-load dtype
    repair also converts signed INT8 moments to UINT8. PyTorch has already
    rounded FP32 state through the parameter dtype before that repair runs.

    Keep the upstream loader (including its validation and initialization), then
    restore tensor values directly from the checkpoint with their storage dtype.
    The parameter-group mapping remains owned by create_optimizer.
    """
    prepared = dict(state_dict)
    prepared["param_groups"] = []
    for saved_group in state_dict["param_groups"]:
        group = dict(saved_group)
        auxiliary_adam = group.get("optim_type") == "adam" or group.get("use_muon") is False
        prefix = "adam_" if auxiliary_adam else ""
        precision = group.get(prefix + "actual_state_precision", group.get(prefix + "state_precision", "auto"))
        group.setdefault("actual_state_precision", "auto" if precision == "factored" else precision)
        prepared["param_groups"].append(group)

    optimizer.load_state_dict(prepared)

    for saved_group, group in zip(prepared["param_groups"], optimizer.param_groups, strict=True):
        auxiliary_adam = group.get("optim_type") == "adam" or group.get("use_muon") is False
        prefix = "adam_" if auxiliary_adam else ""
        precision = group.get(prefix + "actual_state_precision", group.get(prefix + "state_precision", "auto"))
        for state_id, parameter in zip(saved_group["params"], group["params"], strict=True):
            saved_state = state_dict["state"].get(state_id, {})
            restored_state = optimizer.state[parameter]
            for key, value in saved_state.items():
                if not isinstance(value, torch.Tensor):
                    continue
                dtype = value.dtype
                if (
                    precision == "auto" and key in _AUTO_MOMENTS and not saved_state.get("factored", False)
                    or key in _PARAMETER_STATES
                    or key == "anchor_data" and group.get("centered_wd_mode", "full") == "full"
                ):
                    dtype = parameter.dtype
                restored_state[key] = value.to(device=parameter.device, dtype=dtype)
