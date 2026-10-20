"""Compiled adv-optm steps must not recompile on every step; tiny CPU tensors, no model weights.

AdamW_adv passes its bias correction and Adopt_adv its update clipping as Python scalars that
change on every step. With OneTrainer's raised recompile limit this recompiled every parameter
shape on every step, which looked like a hang in the fused backward pass.
"""
import unittest
from types import SimpleNamespace

from modules.util import create
from modules.util.compile_util import init_compile
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType
from modules.util.enum.Optimizer import Optimizer
from modules.util.NamedParameterGroup import NamedParameterGroup, NamedParameterGroupCollection
from modules.util.optimizer_util import OPTIMIZER_DEFAULT_PARAMETERS

import torch
from torch import nn
from torch._dynamo.utils import counters


def make_optimizer(kind, compile, **overrides):
    torch.manual_seed(3)
    config = TrainConfig.default_values()
    config.train_device = config.temp_device = "cpu"
    config.model_type = ModelType.KREA_2
    config.learning_rate = 1e-3
    config.optimizer.from_dict(OPTIMIZER_DEFAULT_PARAMETERS[kind])
    config.optimizer.optimizer = kind
    config.optimizer.stochastic_rounding = False
    config.optimizer.compile = compile
    for key, value in overrides.items():
        setattr(config.optimizer, key, value)
    transformer = nn.Module()
    transformer.transformer_blocks = nn.ModuleList([nn.Linear(8, 16, bias=False)])
    transformer.proj = nn.Linear(16, 8, bias=False)
    model = SimpleNamespace(model_type=config.model_type, transformer=transformer)
    groups = NamedParameterGroupCollection()
    groups.add_group(NamedParameterGroup("transformer", transformer.parameters(), config.learning_rate))
    return transformer, create.create_optimizer(groups, None, config, model=model)


def train(kind, compile, fused, steps=5, **overrides):
    transformer, optimizer = make_optimizer(kind, compile, **overrides)
    group = optimizer.param_groups[0]
    graphs = []
    for step in range(steps):
        group["lr"] = 1e-3 * (step + 1) / steps  # warmup changes the LR on every step
        before = counters["stats"]["unique_graphs"]
        for index, parameter in enumerate(group["params"]):
            values = torch.arange(parameter.numel(), dtype=torch.float32).reshape(parameter.shape)
            parameter.grad = values.add(step + 1).sin().mul(0.15)
            if fused:
                optimizer.step_parameter(parameter, group, index)
        if not fused:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        graphs.append(counters["stats"]["unique_graphs"] - before)
    return [parameter.detach().clone() for parameter in transformer.parameters()], graphs


class CompiledAdvOptimizerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_compile()

    def test_compiled_steps_compile_once_and_match_eager(self):
        cases = [
            (Optimizer.ADAMW_ADV, {}),
            (Optimizer.ADOPT_ADV, {"use_atan2": False}),  # step-dependent clipping
            (Optimizer.ADOPT_ADV, {"use_atan2": True}),
        ]
        for kind, overrides in cases:
            for fused in (False, True):
                with self.subTest(kind=kind, fused=fused, **overrides):
                    torch._dynamo.reset()
                    expected, _ = train(kind, False, fused, **overrides)
                    actual, graphs = train(kind, True, fused, **overrides)
                    # Adopt's first step only initializes its second moment.
                    first_update = 1 if kind == Optimizer.ADOPT_ADV and not overrides["use_atan2"] else 0
                    self.assertGreater(graphs[first_update], 0)
                    self.assertEqual(graphs[first_update + 1:], [0] * (len(graphs) - first_update - 1))
                    for a, b in zip(actual, expected, strict=True):
                        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
