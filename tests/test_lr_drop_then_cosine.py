"""DROP_THEN_COSINE learning rate scheduler. CPU only."""
import math
import unittest
from types import SimpleNamespace

from modules.ui.BaseTrainingTabView import BaseTrainingTabView
from modules.util import create
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.LearningRateScheduler import LearningRateScheduler
from modules.util.lr_scheduler_util import lr_lambda_drop_then_cosine

import torch
from torch import nn


class DropThenCosineTest(unittest.TestCase):
    def test_linear_drop_then_cosine_to_the_min_factor(self):
        factor = lr_lambda_drop_then_cosine(drop_steps=100, drop_factor=1 / 3, scheduler_steps=1100, min_factor=0.05)
        self.assertAlmostEqual(factor(0), 1.0)
        self.assertAlmostEqual(factor(50), 1 - (2 / 3) / 2)  # halfway down the linear drop
        self.assertAlmostEqual(factor(100), 1 / 3)  # the knee: cosine starts here
        self.assertAlmostEqual(factor(600), 0.05 + (1 / 3 - 0.05) * 0.5)  # cosine halfway
        self.assertAlmostEqual(factor(1100), 0.05)
        self.assertAlmostEqual(factor(5000), 0.05)
        values = [factor(step) for step in range(1101)]
        self.assertTrue(all(a >= b for a, b in zip(values, values[1:], strict=False)))

    def test_rejects_inconsistent_factors(self):
        with self.assertRaisesRegex(ValueError, "Drop Factor"):
            lr_lambda_drop_then_cosine(10, 0.0, 100)
        with self.assertRaisesRegex(ValueError, "must not exceed Drop Factor"):
            lr_lambda_drop_then_cosine(10, 0.3, 100, min_factor=0.5)

    def test_created_from_config_in_epochs_with_accumulation_and_warmup(self):
        config = TrainConfig.default_values()
        config.learning_rate = 3e-3
        config.learning_rate_drop_factor, config.learning_rate_drop_epochs = 1 / 3, 5
        config.learning_rate_min_factor = 0.0
        parameter = nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.SGD([parameter], lr=3e-3)
        # 400 batches per epoch, accumulation 2: 200 optimizer steps per epoch, 4000 in 20 epochs.
        scheduler = create.create_lr_scheduler(
            config, optimizer, LearningRateScheduler.DROP_THEN_COSINE, warmup_steps=0, num_cycles=1,
            min_factor=0.0, num_epochs=20, batch_size=1, approximate_epoch_length=400, gradient_accumulation_steps=2,
        )
        lrs = []
        for _ in range(4000):
            lrs.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()
        self.assertAlmostEqual(lrs[0], 3e-3)
        self.assertAlmostEqual(lrs[1000], 1e-3)  # epoch 5
        self.assertAlmostEqual(lrs[2500], 1e-3 * 0.5 * (1 + math.cos(math.pi * 0.5)))  # cosine halfway
        self.assertLess(lrs[-1], 1e-5)

    def test_training_tab_shows_both_fields_on_their_own_rows(self):
        rows = {}

        def record(kind):
            def widget(frame, row, column, *args, **kwargs):
                key = args[0] if kind == "label" else args[-1] if kind == "entry" else kind
                rows.setdefault((row, column), []).append(key)
                if kind == "options_adv":
                    return None, {"component": None, "button_component": None}
                return None
            return widget

        components = SimpleNamespace(
            section_frame=lambda master, row: object(), label=record("label"), entry=record("entry"),
            options=record("options"), options_adv=record("options_adv"),
        )
        view = SimpleNamespace(components=components, restore_scheduler=lambda value: None,
                               restore_optimizer_config=None, open_optimizer_params=None, open_scheduler_params=None)
        ui_state = SimpleNamespace(get_var=lambda name: SimpleNamespace(get=lambda: "COSINE"))
        BaseTrainingTabView._BaseTrainingTabView__create_base_frame(view, None, 0, None, ui_state)
        self.assertTrue(all(len(widgets) == 1 for widgets in rows.values()), rows)
        labels = {key[0] for (row, column), key in rows.items() if column == 0}
        self.assertTrue({"Learning Rate Drop Factor", "Learning Rate Drop Epochs", "Clip Grad Norm"} <= labels)
        self.assertEqual(rows[(6, 1)], ["learning_rate_drop_factor"])
        self.assertEqual(rows[(7, 1)], ["learning_rate_drop_epochs"])


if __name__ == "__main__":
    unittest.main()
