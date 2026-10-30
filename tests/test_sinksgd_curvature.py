"""SINKSGD_CURVATURE (experimental): step size from measured slope and curvature. CPU only."""
import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from modules.trainer.GenericTrainer import GenericTrainer
from modules.util import create
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType
from modules.util.enum.Optimizer import Optimizer
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.NamedParameterGroup import NamedParameterGroup, NamedParameterGroupCollection
from modules.util.optimizer.adv_optm_util import load_adv_optimizer_state_dict
from modules.util.optimizer.SinkSGDAdaptive import SinkSGDAdaptive
from modules.util.optimizer.SinkSGDCurvature import SinkSGDCurvature
from modules.util.optimizer_util import change_optimizer
from modules.util.TrainProgress import TrainProgress

import torch
from torch import nn

from test_moss_image import tiny_model, train_config


def _noisy_regression(seed):
    torch.manual_seed(seed)
    teacher = nn.Sequential(nn.Linear(16, 64), nn.Tanh(), nn.Linear(64, 16))
    for layer in (teacher[0], teacher[2]):
        nn.init.normal_(layer.weight, std=0.5)
    model = nn.Sequential(nn.Linear(16, 64), nn.Tanh(), nn.Linear(64, 16))

    def sample(generator):
        x = torch.randn(4, 16, generator=generator)
        with torch.no_grad():
            return x, teacher(x) + 0.5 * torch.randn(4, 16, generator=generator)

    validation = [sample(torch.Generator().manual_seed(seed + 1)) for _ in range(32)]
    return model, sample, validation


def _train(optimizer_factory, steps, seed=0):
    model, sample, validation = _noisy_regression(seed)
    optimizer = optimizer_factory(list(model.parameters()))
    generator = torch.Generator().manual_seed(seed + 2)
    steps_taken = []
    for _ in range(steps):
        x, y = sample(generator)
        optimizer.zero_grad()
        (model(x) - y).square().mean().backward()
        optimizer.step()
        if getattr(optimizer, "wants_curvature_probe", lambda: False)():
            optimizer.zero_grad()
            (model(x) - y).square().mean().backward()
            optimizer.finish_curvature_probe()
        steps_taken.append(optimizer.param_groups[0]["d"])
    with torch.no_grad():
        loss = sum((model(x) - y).square().mean().item() for x, y in validation) / len(validation)
    return loss, steps_taken


class SinkSGDCurvatureTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        torch.set_num_threads(1)

    def test_probe_measures_the_curvature_along_the_step(self):
        # Quadratic loss 0.5 x'Ax: the probe must return u'Au for the step direction u.
        a = torch.diag(torch.linspace(0.5, 3.0, 12))
        parameter = nn.Parameter(torch.randn(12, 1))
        optimizer = SinkSGDCurvature([parameter], d0=1e-2, d_ceiling=1e-2, state_precision="fp32", slice_p=1)

        def gradient():
            optimizer.zero_grad()
            (0.5 * parameter.T @ a @ parameter).sum().backward()

        gradient()
        before = parameter.detach().clone()
        optimizer.step()
        self.assertTrue(optimizer.wants_curvature_probe())
        u = (before - parameter.detach()) / optimizer.param_groups[0]["last_step"]
        gradient()
        optimizer.finish_curvature_probe()
        group = optimizer.param_groups[0]
        measured = group["curvature"] / group["curvature_weight"]
        torch.testing.assert_close(torch.tensor(measured), (u.T @ a @ u).squeeze(), rtol=1e-4, atol=1e-6)
        self.assertFalse(optimizer.wants_curvature_probe())

    def test_finds_a_step_size_close_to_the_best_fixed_one(self):
        loss, steps = _train(lambda p: SinkSGDCurvature(p, state_precision="fp32"), 3000)
        self.assertGreater(max(steps), 30 * 1e-5)
        self.assertLess(max(steps), 1e-2)
        # The same direction with fixed step sizes: within 40 % of the best, far better than the worst.
        fixed_losses = [_train(lambda p, fixed=fixed: SinkSGDAdaptive(
            p, d0=fixed, d_floor=fixed, growth_rate=1.0, d_shrink=0.0, agreement_threshold=0.0,
            momentum=0.95, nesterov=True, normed_momentum=True, momentum_renorm=True, state_precision="fp32",
        ), 3000)[0] for fixed in (1e-5, 3e-4, 1e-3, 1e-2)]
        self.assertLess(loss, 1.4 * min(fixed_losses))
        self.assertLess(loss, 0.5 * max(fixed_losses))

    def test_stays_at_initial_step_until_curvature_is_known(self):
        parameter = nn.Parameter(torch.randn(8, 8))
        optimizer = SinkSGDCurvature([parameter], d0=1e-4, min_probes=3, curvature_interval=10, state_precision="fp32")
        for _ in range(25):
            parameter.grad = torch.randn(8, 8)
            optimizer.step()
            optimizer.cancel_curvature_probe()  # no probe ever finishes
        self.assertEqual(optimizer.param_groups[0]["d"], 1e-4)
        self.assertEqual(optimizer.param_groups[0]["probes"], 0)

    def test_rejects_fused_back_pass_and_invalid_limits(self):
        parameter = nn.Parameter(torch.zeros(4, 4))
        optimizer = SinkSGDCurvature([parameter])
        self.assertFalse(optimizer.supports_fused_back_pass)
        parameter.grad = torch.ones(4, 4)
        with self.assertRaisesRegex(RuntimeError, "fused back pass"):
            optimizer.step_parameter(parameter, optimizer.param_groups[0], 0)
        with self.assertRaisesRegex(ValueError, "D Ceiling"):
            SinkSGDCurvature([parameter], d0=1e-2, d_ceiling=1e-3)

    def test_state_dict_roundtrip(self):
        model, _, _ = _noisy_regression(0)
        optimizer = SinkSGDCurvature(list(model.parameters()))
        for _ in range(30):
            for p in model.parameters():
                p.grad = torch.randn_like(p)
            optimizer.step()
            for p in model.parameters():
                p.grad = torch.randn_like(p)
            optimizer.finish_curvature_probe()
        restored_model = copy.deepcopy(model)
        restored = SinkSGDCurvature(list(restored_model.parameters()))
        load_adv_optimizer_state_dict(restored, optimizer.state_dict())
        for key in ("d", "slope", "slope_weight", "curvature", "curvature_weight", "probes", "steps", "last_step"):
            self.assertEqual(restored.param_groups[0][key], optimizer.param_groups[0][key], key)
        saved, state = optimizer.state[next(model.parameters())], restored.state[next(restored_model.parameters())]
        self.assertEqual(state["c_x"].dtype, torch.float32)
        torch.testing.assert_close(state["c_x"], saved["c_x"])

    def test_created_from_config(self):
        config = TrainConfig.default_values()
        config.optimizer.optimizer = Optimizer.SINKSGD_CURVATURE
        config.optimizer = change_optimizer(config)
        config.learning_rate = 1.0
        parameter = nn.Parameter(torch.zeros(4, 4))
        collection = NamedParameterGroupCollection()
        collection.add_group(NamedParameterGroup("transformer", [parameter], learning_rate=1.0))
        optimizer = create.create_optimizer(collection, None, config, SimpleNamespace())
        self.assertIsInstance(optimizer, SinkSGDCurvature)
        group = optimizer.param_groups[0]
        self.assertEqual((group["d0"], group["d_floor"], group["d_ceiling"]), (1e-5, 1e-6, 1e-2))
        self.assertEqual((group["gain"], group["slope_target"], group["curvature_interval"]), (0.01, 0.25, 10))
        self.assertEqual((group["momentum"], group["nesterov"], group["momentum_renorm"]), (0.95, True, True))
        self.assertTrue(Optimizer.SINKSGD_CURVATURE.is_adaptive and Optimizer.SINKSGD_CURVATURE.is_adv)
        self.assertFalse(Optimizer.SINKSGD_CURVATURE.supports_fused_back_pass())
        self.assertIn("sinksgd_curvature/step/0", optimizer.tensorboard_stats())
        config.optimizer.fused_back_pass = True
        with self.assertRaisesRegex(RuntimeError, "fused back pass"):
            create.create_optimizer(collection, None, config, SimpleNamespace())

    def test_trainer_probe_repeats_the_same_batch_at_the_new_weights(self):
        model, config = tiny_model(), train_config()
        model.train_config = config
        setup = create.create_model_setup(
            ModelType.MOSS_IMAGE, torch.device("cpu"), torch.device("cpu"), TrainingMethod.LORA,
        )
        setup.setup_optimizations(model, config)
        setup.setup_model(model, config)
        setup.setup_train_device(model, config)
        parameters = list(model.transformer_lora.parameters())
        for module in model.transformer_lora.lora_modules.values():
            nn.init.normal_(module.lora_up.weight, std=0.1)
        optimizer = SinkSGDCurvature(parameters, d0=1e-3, state_precision="fp32")
        model.optimizer = optimizer
        trainer = object.__new__(GenericTrainer)
        trainer.config, trainer.model, trainer.model_setup, trainer.parameters = config, model, setup, parameters
        batch = {
            "latent_image": torch.randn(1, 32, 4, 4), "tokens": torch.ones(1, 4, dtype=torch.long),
            "tokens_mask": torch.ones(1, 4), "text_encoder_hidden_state": torch.randn(1, 4, 8),
            "loss_weight": torch.ones(1), "concept_type": ["STANDARD"],
        }
        progress = TrainProgress(global_step=7)
        setup.calculate_loss(model, batch, setup.predict(model, batch, config, progress), config).backward()
        optimizer.step()
        # Expected: the same batch (same noise and timestep from the progress) at the new weights.
        optimizer_copy = [p.detach().clone() for p in parameters]
        for p in parameters:
            p.grad = None
        setup.calculate_loss(model, batch, setup.predict(model, batch, config, progress), config).backward()
        expected = [p.grad.clone() for p in parameters]
        captured = []
        with patch.object(optimizer, "finish_curvature_probe", lambda: captured.extend(p.grad.clone() for p in parameters)):
            trainer._GenericTrainer__run_curvature_probe([(batch, progress)], None)
        for got, want, weight, p in zip(captured, expected, optimizer_copy, parameters, strict=True):
            torch.testing.assert_close(got, want)
            torch.testing.assert_close(p.detach(), weight)


if __name__ == "__main__":
    unittest.main()
