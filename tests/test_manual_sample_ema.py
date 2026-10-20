"""Manual samples can bypass the EMA weights while EMA training is enabled.

Run with QT_QPA_PLATFORM=offscreen xvfb-run -a python -m unittest discover
-s tests -p test_manual_sample_ema.py -v.
"""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from modules.trainer.GenericTrainer import GenericTrainer
from modules.ui.PySide6SampleWindowView import PySide6SampleWindowView
from modules.ui.SampleWindowController import SampleWindowController
from modules.util.callbacks.TrainCallbacks import TrainCallbacks
from modules.util.commands.TrainCommands import TrainCommands
from modules.util.config.SampleConfig import SampleConfig
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.EMAMode import EMAMode
from modules.util.enum.ModelType import ModelType
from modules.util.TrainProgress import TrainProgress

import torch

from PySide6.QtWidgets import QApplication, QCheckBox, QPushButton

try:
    from modules.ui.CtkSampleWindowView import CtkSampleWindowView

    import customtkinter as ctk
except ImportError:  # tkinter is not installed
    ctk = None

SWITCH_TEXT = "sample with EMA weights"


def sample(prompt, use_ema=True):
    config = SampleConfig.default_values(ModelType.MOSS_IMAGE)
    config.prompt, config.use_ema = prompt, use_ema
    return config


def train_config(ema):
    config = TrainConfig.default_values()
    config.model_type = ModelType.MOSS_IMAGE
    config.ema = ema
    return config


class ManualSampleTrainerTest(unittest.TestCase):
    def run_samples(self, samples, ema=EMAMode.GPU):
        events = []
        trainer = object.__new__(GenericTrainer)
        trainer.config = train_config(ema)
        trainer.config.non_ema_sampling = False
        trainer.parameters = []
        trainer.callbacks = TrainCallbacks()
        trainer.model_setup = Mock()
        trainer.model = SimpleNamespace(ema=None if ema == EMAMode.OFF else SimpleNamespace(
            copy_ema_to=lambda parameters, store_temp: events.append("ema weights"),
            copy_temp_to=lambda parameters: events.append("training weights"),
        ))
        trainer._GenericTrainer__sample_loop = lambda **kwargs: events.append((
            [config.prompt for config in kwargs["sample_config_list"]],
            kwargs["ema_applied"], kwargs["is_custom_sample"],
        ))
        trainer._GenericTrainer__sample_during_training(TrainProgress(), torch.device("cpu"), samples)
        return events

    def test_non_ema_manual_samples_use_restored_training_weights(self):
        self.assertEqual(
            self.run_samples([sample("a"), sample("b", use_ema=False)]),
            ["ema weights", (["a"], True, True), "training weights", (["b"], False, True)],
        )

    def test_default_manual_samples_keep_ema_weights(self):
        self.assertEqual(
            self.run_samples([sample("a"), sample("b")]),
            ["ema weights", (["a", "b"], True, True), "training weights"],
        )

    def test_choice_is_irrelevant_without_ema(self):
        self.assertEqual(
            self.run_samples([sample("a", use_ema=False)], ema=EMAMode.OFF),
            [(["a"], False, True)],
        )


def controller(ema):
    commands = TrainCommands()
    return SampleWindowController(
        train_config(ema), use_external_model=True, callbacks=TrainCallbacks(), commands=commands,
    ), commands


class PySide6ManualSampleWindowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def switches(self, window):
        return [box for box in window.findChildren(QCheckBox) if box.text() == SWITCH_TEXT]

    def test_switch_sends_non_ema_sample_command(self):
        sample_controller, commands = controller(EMAMode.GPU)
        window = PySide6SampleWindowView(None, sample_controller)
        switches = self.switches(window)
        self.assertEqual(len(switches), 1)
        self.assertTrue(switches[0].isChecked())
        switches[0].click()
        next(button for button in window.findChildren(QPushButton) if button.text() == "sample").click()
        self.assertFalse(commands.get_and_reset_sample_custom_commands()[0].use_ema)
        window.deleteLater()

    def test_switch_hidden_without_ema(self):
        window = PySide6SampleWindowView(None, controller(EMAMode.OFF)[0])
        self.assertEqual(self.switches(window), [])
        window.deleteLater()


@unittest.skipIf(ctk is None, "tkinter is not installed")
class CtkManualSampleWindowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = ctk.CTk()
        cls.root.withdraw()

    @classmethod
    def tearDownClass(cls):
        cls.root.destroy()

    def widgets(self, parent, cls):
        for child in parent.winfo_children():
            if isinstance(child, cls):
                yield child
            yield from self.widgets(child, cls)

    def switches(self, window):
        return [switch for switch in self.widgets(window, ctk.CTkSwitch) if switch.cget("text") == SWITCH_TEXT]

    def test_switch_sends_non_ema_sample_command(self):
        sample_controller, commands = controller(EMAMode.GPU)
        window = CtkSampleWindowView(self.root, sample_controller)
        switches = self.switches(window)
        self.assertEqual(len(switches), 1)
        self.assertEqual(switches[0].get(), 1)
        switches[0].toggle()
        next(button for button in self.widgets(window, ctk.CTkButton) if button.cget("text") == "sample").invoke()
        self.assertFalse(commands.get_and_reset_sample_custom_commands()[0].use_ema)
        window.destroy()

    def test_switch_hidden_without_ema(self):
        window = CtkSampleWindowView(self.root, controller(EMAMode.OFF)[0])
        self.assertEqual(self.switches(window), [])
        window.destroy()


if __name__ == "__main__":
    unittest.main()
