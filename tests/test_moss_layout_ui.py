"""Real Qt and CTk layout editor tests, including phrase selection and export."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from modules.util import create  # noqa: F401

# isort: split
from modules.ui.BaseConceptWindowView import BaseConceptWindowView
from modules.ui.ConceptWindowController import ConceptWindowController
from modules.ui.CtkSampleFrameView import CtkSampleFrameView
from modules.ui.MOSSLayoutEditor import MOSSLayoutEditor, open_sample_layout
from modules.ui.PySide6SampleFrameView import PySide6SampleFrameView
from modules.ui.SampleFrameController import SampleFrameController
from modules.util.config.ConceptConfig import ConceptConfig
from modules.util.config.SampleConfig import SampleConfig
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType
from modules.util.ui import ctk_components, pyside6_components
from modules.util.ui.CtkUIState import CtkUIState
from modules.util.ui.PySide6UIState import PySide6UIState

import customtkinter as ctk
from PIL import Image
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import QApplication, QPushButton, QWidget


class EditorAssertions:
    def make_editor(self, **kwargs):
        editor = MOSSLayoutEditor(self.parent, qt=self.qt, **kwargs)
        self.editors.append(editor)
        return editor

    def event(self, editor, x, y):
        return SimpleNamespace(inaxes=editor.axes, xdata=x, ydata=y, button=1)

    def test_selected_caption_phrase_becomes_drawn_box_description(self):
        editor = self.make_editor(width=200, height=100)
        editor._set("summary", "a red car on the street")
        if self.qt:
            cursor = editor.caption_widget.textCursor()
            cursor.setPosition(2)
            cursor.setPosition(9, QTextCursor.MoveMode.KeepAnchor)
            editor.caption_widget.setTextCursor(cursor)
        else:
            editor.caption_widget.tag_add("sel", "1.2", "1.9")
        editor.press(self.event(editor, 20, 10))
        editor.motion(self.event(editor, 80, 60))
        self.assertEqual(editor.preview_box, (20, 10, 80, 60))
        editor.release(self.event(editor, 80, 60))
        item = editor.document["items"][0]
        self.assertEqual(item["caption"], "red car")
        self.assertEqual(item["bbox"], [-40, -80, 10, -20])
        self.assertEqual(editor._get("caption"), "red car")

    def test_move_resize_description_and_delete(self):
        editor = self.make_editor(width=200, height=100)
        editor.press(self.event(editor, 20, 10))
        editor.release(self.event(editor, 80, 60))
        if self.qt:
            editor.mode.setCurrentText("Select / move")
        else:
            editor.mode.set("Select / move")
        editor.press(self.event(editor, 50, 35))
        editor.release(self.event(editor, 60, 45))
        self.assertEqual(editor.document["items"][0]["bbox"], [-30, -70, 20, -10])
        editor._set("caption", "changed description")
        editor._set("label", "text")
        editor._set("text", "HELLO")
        editor.apply_item()
        self.assertEqual(editor.document["items"][0]["text"], "HELLO")
        editor.press(self.event(editor, 30, 20))
        editor.release(self.event(editor, 20, 10))
        self.assertEqual(editor.document["items"][0]["bbox"], [-40, -80, 20, -10])
        editor.delete_item()
        self.assertEqual(editor.document["items"], [])

    def test_training_bucket_export_reloads_as_matching_annotation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "input.png"
            Image.new("RGB", (201, 127), "red").save(image_path)
            image_path.with_suffix(".txt").write_text("red car on a road\nsoft daylight", encoding="utf-8")
            editor = self.make_editor()
            editor.load_image(image_path)
            self.assertEqual(editor._get("summary"), "red car on a road\nsoft daylight")
            editor._set("resolution", "128")
            editor.apply_bucket()
            editor.pending_phrase = "red car"
            editor.press(self.event(editor, 10, 10))
            editor.release(self.event(editor, 50, 50))
            image, caption, layout = editor.write_bundle(root / "export")
            self.assertEqual(caption.read_text(), "red car on a road\nsoft daylight")
            self.assertEqual(json.loads(layout.read_text())["items"][0]["caption"], "red car")
            restored = self.make_editor()
            restored.load_image(image)
            self.assertEqual(restored.image.size, editor.image.size)
            self.assertEqual(restored.document["items"][0]["bbox"], editor.document["items"][0]["bbox"])
            with Image.open(image_path) as original:
                self.assertEqual(original.size, (201, 127))

    def test_sample_editor_updates_config_and_both_ui_entry_points_exist(self):
        sample = SampleConfig.default_values(ModelType.MOSS_IMAGE)
        sample.prompt = "a scene"
        sample_state = self.state(sample)
        editor = open_sample_layout(self.parent, self.components, sample, sample_state)
        self.editors.append(editor)
        editor.pending_phrase = "a subject"
        editor.press(self.event(editor, 100, 100))
        editor.release(self.event(editor, 400, 600))
        editor.save()
        self.assertTrue(sample.moss_layout_enabled)
        self.assertEqual(json.loads(sample.moss_layout_json)["items"][0]["caption"], "a subject")
        frame_cls = PySide6SampleFrameView if self.qt else CtkSampleFrameView
        frame = frame_cls(self.parent, SampleFrameController(sample, ModelType.MOSS_IMAGE), sample_state)
        self.assertIn("Edit MOSS layout", self.buttons(frame))
        config = TrainConfig.default_values()
        config.model_type = ModelType.MOSS_IMAGE
        concept = ConceptConfig.default_values()
        parent = QWidget(self.parent) if self.qt else ctk.CTkFrame(self.parent)
        BaseConceptWindowView(self.components).build_general_tab(
            parent, ConceptWindowController(config, concept), self.state(concept), self.state(concept.text),
        )
        self.assertIn("Annotate MOSS layouts", self.buttons(parent))
        self.assertEqual(concept.text.moss_layout_mode, "auto")


class QtLayoutEditorTests(EditorAssertions, unittest.TestCase):
    qt = True
    components = pyside6_components

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.parent = QWidget()
        self.editors = []

    def state(self, config):
        return PySide6UIState(config)

    def buttons(self, parent):
        return [widget.text() for widget in parent.findChildren(QPushButton)]

    def tearDown(self):
        for editor in self.editors:
            editor.window.close()
        self.parent.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.app.processEvents()


class CtkLayoutEditorTests(EditorAssertions, unittest.TestCase):
    qt = False
    components = ctk_components

    @classmethod
    def setUpClass(cls):
        cls.app = ctk.CTk()
        cls.app.withdraw()

    @classmethod
    def tearDownClass(cls):
        cls.app.destroy()

    def setUp(self):
        self.parent = ctk.CTkFrame(self.app)
        self.editors = []

    def state(self, config):
        return CtkUIState(self.app, config)

    def buttons(self, parent):
        result = []
        for child in parent.winfo_children():
            if isinstance(child, ctk.CTkButton):
                result.append(child.cget("text"))
            result.extend(self.buttons(child))
        return result

    def tearDown(self):
        for editor in self.editors:
            editor.window.destroy()
        self.parent.destroy()
        self.app.update_idletasks()


if __name__ == "__main__":
    unittest.main()
