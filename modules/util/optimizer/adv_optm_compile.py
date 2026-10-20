"""Stop adv_optm's compiled parameter steps from recompiling on every optimizer step.

adv_optm compiles each per-parameter step with dynamic=False, so Python scalars that change on
every step become value guards: AdamW_adv's bias correction and Adopt_adv's step-dependent update
clipping. OneTrainer raises the recompile limit (init_compile), so instead of falling back to eager
after a few steps, every parameter shape was recompiled on every step, which looks like a hang in
the (fused) backward pass. Passing those values as 0-d tensors keeps a single graph per shape.
"""
import torch


class _TensorScalarArgs(dict):
    """Compiled-step cache whose functions receive the given positional scalars as tensors."""

    def __init__(self, indices: set[int]):
        super().__init__()
        self.indices = indices

    def __setitem__(self, key, compiled_fn):
        indices = self.indices

        def call(p, *args):
            args = [
                torch.as_tensor(arg, dtype=torch.float32, device=p.device) if i in indices else arg
                for i, arg in enumerate(args, start=1)
            ]
            return compiled_fn(p, *args)

        super().__setitem__(key, call)


class _TensorClip:
    """Clip schedule evaluated eagerly; the compiled step only reads the resulting tensor."""

    def __init__(self, clip_lambda):
        self.clip_lambda = clip_lambda
        self.value = None

    def __call__(self, step):
        return self.value

    def prepare(self, step, device):
        self.value = torch.as_tensor(self.clip_lambda(step), dtype=torch.float32, device=device)


def fix_compiled_step_guards(optimizer):
    from adv_optm import AdamW_adv, Adopt_adv

    if not any(group.get('compiled_optimizer', False) for group in optimizer.param_groups):
        return optimizer

    if isinstance(optimizer, AdamW_adv):
        # _step_parameter(p, grad, state, group, step_size, beta1, beta2, sqrt_bias_correction2, ...)
        optimizer._compiled_step_fns = _TensorScalarArgs({7})
    elif isinstance(optimizer, Adopt_adv) and optimizer.clip_lambda is not None:
        clip = _TensorClip(optimizer.clip_lambda)
        optimizer.clip_lambda = clip
        step_parameter = optimizer.step_parameter

        def step_parameter_with_clip(p, group, i=None):
            if p.grad is not None:
                clip.prepare(optimizer.state[p].get('step', 0), p.device)
            return step_parameter(p, group, i)

        optimizer.step_parameter = step_parameter_with_clip

    return optimizer
