import math
from collections.abc import Callable


def lr_lambda_warmup(warmup_steps: int, lr_lambda: Callable[[int], float]):
    def warmup(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(warmup_steps)
        else:
            return lr_lambda(current_step - warmup_steps)

    return warmup


def lr_lambda_constant():
    def lr_lambda(current_step: int):
        return 1

    return lr_lambda


def lr_lambda_linear(
        scheduler_steps: int,
        min_factor: float = 1.0,
):
    def lr_lambda(current_step: int):
        lin_val = max(0.0, float(scheduler_steps - current_step) / float(scheduler_steps))
        factor = apply_min_factor(lin_val, min_factor)
        return factor

    return lr_lambda



def lr_lambda_cosine(
        scheduler_steps: int,
        min_factor: float = 1.0,
):
    def lr_lambda(current_step: int):
        progress = float(current_step) / float(scheduler_steps)
        cos_val = 0.5 * (1.0 + math.cos(progress * math.pi))
        factor = max(0.0, cos_val)
        factor = apply_min_factor(factor, min_factor)
        return factor

    return lr_lambda


def lr_lambda_cosine_with_restarts(
        scheduler_steps: int,
        num_cycles: float,
        min_factor: float = 1.0,
):
    def lr_lambda(current_step: int):
        progress = float(min(current_step, scheduler_steps - 1)) / float(scheduler_steps)
        cos_val = 0.5 * (1.0 + math.cos(progress * 2.0 * math.pi * num_cycles))
        factor = max(0.0, cos_val)
        factor = apply_min_factor(factor, min_factor)
        return factor

    return lr_lambda



def lr_lambda_cosine_with_hard_restarts(
        scheduler_steps: int,
        num_cycles: float,
        min_factor: float = 1.0,
):
    def lr_lambda(current_step: int):
        progress = float(min(current_step, scheduler_steps - 1)) / float(scheduler_steps)
        cos_val = 0.5 * (1.0 + math.cos(((progress * num_cycles) % 1.0) * math.pi))
        factor = max(0.0, cos_val)
        factor = apply_min_factor(factor, min_factor)
        return factor

    return lr_lambda



def lr_lambda_rex(
        scheduler_steps: int,
        min_factor: float = 1.0,
):
    def lr_lambda(current_step: int):
        # https://arxiv.org/abs/2107.04197
        max_lr = 1
        min_lr = 0
        d = 0.9
        if current_step < scheduler_steps:
            progress = (current_step / scheduler_steps)
            div = (1 - d) + (d * (1 - progress))
            val = min_lr + (max_lr - min_lr) * ((1 - progress) / div)
        else:
            val = min_lr

        factor = apply_min_factor(val, min_factor)
        return factor

    return lr_lambda


def lr_lambda_drop_then_cosine(
        drop_steps: int,
        drop_factor: float,
        scheduler_steps: int,
        min_factor: float = 0.0,
):
    """A fast first phase, then the usual cosine: linear from the full rate to drop_factor of it
    over drop_steps, then a cosine from there down to min_factor of it at the end."""
    if not 0.0 < drop_factor <= 1.0:
        raise ValueError(f"Drop Factor must be in (0, 1], got {drop_factor}.")
    if min_factor > drop_factor:
        raise ValueError(f"Min Factor ({min_factor}) must not exceed Drop Factor ({drop_factor}).")

    def lr_lambda(current_step: int):
        if current_step < drop_steps:
            return 1.0 - (1.0 - drop_factor) * current_step / drop_steps
        progress = min(1.0, (current_step - drop_steps) / max(1, scheduler_steps - drop_steps))
        cos_val = 0.5 * (1.0 + math.cos(progress * math.pi))
        return min_factor + (drop_factor - min_factor) * cos_val

    return lr_lambda


def apply_min_factor(value: float, min_factor: float) -> float:
    return min_factor + (1.0 - min_factor) * value
