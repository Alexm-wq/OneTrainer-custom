"""Build both real training UIs without loading model weights.

Run with QT_QPA_PLATFORM=offscreen xvfb-run -a python -m unittest discover
-s tests -p test_moss_training_ui.py -v.
"""
import json
import unittest
from pathlib import Path

# Initialize the model factories before optimizer dialogs, as the full UI does.
from modules.util import create  # noqa: F401

# isort: split
from modules.ui.CtkModelTabView import CtkModelTabView
from modules.ui.PySide6ModelTabView import PySide6ModelTabView
from modules.ui.ModelTabController import ModelTabController
from modules.util.enum.DataType import DataType
from modules.ui.CtkTrainingTabView import CtkTrainingTabView
from modules.ui.PySide6TrainingTabView import PySide6TrainingTabView
from modules.ui.TrainingTabController import TrainingTabController
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.ui.CtkUIState import CtkUIState
from modules.util.ui.PySide6UIState import PySide6UIState

import customtkinter as ctk
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication, QComboBox, QLabel, QLineEdit, QWidget

PRESET_DIR = Path(__file__).resolve().parents[1] / "training_presets" / "MOSS Image"


def preset_config(filename="#moss-image LoRA.json"):
    return TrainConfig.default_values().from_dict(json.loads((PRESET_DIR / filename).read_text()))


class TrainingUIAssertions:
    def assert_moss_controls(self):
        labels = set(self.labels())
        required = {
            "Optimizer", "Learning Rate Scheduler", "Learning Rate", "Epochs",
            "Local Batch Size", "Accumulation Steps", "Resolution", "Train Transformer",
            "Transformer Learning Rate", "Gradient Checkpointing", "Text Encoder",
            "Text Encoder Sequence Length", "Caption Dropout Probability", "EMA",
            "Timestep Distribution", "Timestep Shift", "Masked Training", "MSE Strength",
            "Loss Weight Function", "Layer Filter", "Layer Offload Fraction", "Offload Activations",
        }
        self.assertTrue(required <= labels, f"Missing controls: {required - labels}")
        unsupported = {
            "Train Text Encoder", "Text Encoder Learning Rate", "Clip Skip",
            "Force Attention Mask",
            "Guidance Scale", "Dynamic Timestep Shifting",
        }
        self.assertFalse(unsupported & labels, f"Unsupported controls: {unsupported & labels}")
        self.assertEqual(self.options("Train Data Type"), ["float32", "float16", "bfloat16", "tfloat32"])
        self.assertEqual(self.options("Attention"), ["torch SDPA", "flash-attn", "torch cuDNN"])
        self.assertEqual(self.options("Layer Filter"), ["attn-mlp", "attention", "mlp", "text-refiner", "full", "custom"])
        self.assertNotIn("Gamma", labels)  # Flow-matching loss controls.

    def test_new_moss_quantization_and_offload_controls(self):
        for path in sorted(PRESET_DIR.glob("*.json")):
            self.config.from_dict(json.loads(path.read_text()))
            self.state.update(self.config)
            self.view.refresh_ui()
            self.assert_moss_controls()
            self.assertEqual(self.labels().count("Layer Offload Fraction"), 2)
            old = self.config.transformer.activation_offloading
            self.toggle("Offload Activations")
            self.assertEqual(self.config.transformer.activation_offloading, not old)
            self.assertEqual(self.config.vae.weight_dtype, DataType.FLOAT_32)
            self.assertFalse(self.config.text_encoder.train)
        view_cls = PySide6ModelTabView if isinstance(self.state, PySide6UIState) else CtkModelTabView
        self.view = view_cls(self.parent, ModelTabController(self.config), self.state)
        for label in ("Transformer Data Type", "Text Encoder Data Type"):
            options = self.options(label)
            for dtype in ("int8", "nfloat4", "float8 (W8)"):
                self.assertIn(dtype, options)
            self.assertNotIn("GGUF", options)
        self.assertIn("int W8A8", self.options("Transformer Data Type"))
        self.assertEqual(self.options("VAE Data Type"), ["float32"])
        self.assertIn("Override Transformer", self.labels())
        self.assertIn("Compile transformer blocks", self.labels())

    def test_moss_presets_build_controls_and_edit_real_config(self):
        for filename in ("#moss-image LoRA.json", "#moss-image Mixed DPO LoRA.json"):
            with self.subTest(preset=filename):
                self.config.from_dict(json.loads((PRESET_DIR / filename).read_text()))
                self.state.update(self.config)
                self.view.refresh_ui()
                self.assert_moss_controls()
                self.edit("Learning Rate", "0.0002")
                self.edit("Transformer Learning Rate", "0.0001")
                self.edit("Text Encoder Sequence Length", "384")
                self.edit("Resolution", "768,1024")
                self.assertEqual(self.config.learning_rate, 0.0002)
                self.assertEqual(self.config.transformer.learning_rate, 0.0001)
                self.assertEqual(self.config.text_encoder_sequence_length, 384)
                self.assertEqual(self.config.resolution, "768,1024")
                old_checkpointing = self.config.transformer.gradient_checkpointing
                self.toggle("Gradient Checkpointing")
                self.assertEqual(self.config.transformer.gradient_checkpointing, not old_checkpointing)
                self.assertFalse(self.config.text_encoder.train)

    def test_switching_existing_models_and_back_preserves_their_controls(self):
        for model_type in (ModelType.KREA_2, ModelType.FLUX_2):
            with self.subTest(model=model_type):
                self.state.get_var("model_type").set(str(model_type))
                self.view.refresh_ui()
                labels = set(self.labels())
                self.assertIn("Optimizer", labels)
                self.assertIn("Layer Offload Fraction", labels)
                self.assertIn("Offload Activations", labels)
                self.assertIn("Dynamic Timestep Shifting", labels)
                self.assertEqual(
                    self.options("Train Data Type"),
                    ["float32", "float16", "bfloat16", "tfloat32"],
                )
                self.assertEqual(self.options("Attention"), ["torch SDPA", "flash-attn", "torch cuDNN"])
                self.state.get_var("model_type").set(str(ModelType.MOSS_IMAGE))
                self.view.refresh_ui()
                self.assert_moss_controls()

    def test_finetune_also_builds_training_controls(self):
        self.state.get_var("training_method").set(str(TrainingMethod.FINE_TUNE))
        self.view.refresh_ui()
        self.assert_moss_controls()


class PySide6TrainingUITest(TrainingUIAssertions, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.config = preset_config()
        self.state = PySide6UIState(self.config)
        self.parent = QWidget()
        self.view = PySide6TrainingTabView(self.parent, TrainingTabController(self.config), self.state)

    def tearDown(self):
        self.parent.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.app.processEvents()

    def labels(self):
        return [label.text() for label in self.view.scroll_frame.findChildren(QLabel)]

    def control(self, text):
        labels = [label for label in self.view.scroll_frame.findChildren(QLabel) if label.text() == text]
        self.assertEqual(len(labels), 1, text)
        label = labels[0]
        layout = label.parentWidget().layout()
        row, column, _, _ = layout.getItemPosition(layout.indexOf(label))
        return layout.itemAtPosition(row, column + 1).widget()

    def options(self, text):
        combo = self.control(text)
        self.assertIsInstance(combo, QComboBox)
        return [combo.itemText(i) for i in range(combo.count())]

    def edit(self, text, value):
        entry = self.control(text)
        self.assertIsInstance(entry, QLineEdit)
        entry.setText(value)
        entry.editingFinished.emit()

    def toggle(self, text):
        self.control(text).click()


class CtkTrainingUITest(TrainingUIAssertions, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = ctk.CTk()
        cls.root.withdraw()

    @classmethod
    def tearDownClass(cls):
        cls.root.destroy()

    def setUp(self):
        self.config = preset_config()
        self.state = CtkUIState(self.root, self.config)
        self.parent = ctk.CTkFrame(self.root)
        self.view = CtkTrainingTabView(self.parent, TrainingTabController(self.config), self.state)

    def tearDown(self):
        self.parent.destroy()
        self.root.update_idletasks()

    def widgets(self, parent):
        for child in parent.winfo_children():
            yield child
            yield from self.widgets(child)

    def labels(self):
        return [
            widget.cget("text") for widget in self.widgets(self.view.scroll_frame)
            if isinstance(widget, ctk.CTkLabel)
        ]

    def control(self, text):
        labels = [
            widget for widget in self.widgets(self.view.scroll_frame)
            if isinstance(widget, ctk.CTkLabel) and widget.cget("text") == text
        ]
        self.assertEqual(len(labels), 1, text)
        label = labels[0]
        position = label.grid_info()
        return label.master.grid_slaves(row=position["row"], column=position["column"] + 1)[0]

    def options(self, text):
        return self.control(text).cget("values")

    def edit(self, text, value):
        entry = self.control(text)
        self.assertIsInstance(entry, ctk.CTkEntry)
        entry.delete(0, "end")
        entry.insert(0, value)
        self.assertIsNone(entry._validator.flush())

    def toggle(self, text):
        self.control(text).toggle()


if __name__ == "__main__":
    unittest.main()
