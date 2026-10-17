"""One native layout editor shared by the Qt and CustomTkinter frontends."""
import copy
import json
from pathlib import Path

from modules.util.moss_layout import (
    empty_layout,
    is_layout,
    layout_document,
    parse_layout,
    read_layout_source,
    transform_layout,
)

from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from PIL import Image, ImageOps


class MOSSLayoutEditor:
    def __init__(self, parent, *, qt, layout=None, width=1024, height=1024, on_save=None, concept_path=None):
        self.qt, self.on_save, self.concept_path = qt, on_save, concept_path
        self.document = parse_layout(layout or empty_layout(width, height), width, height)
        self.document = transform_layout(
            self.document, width, height,
            scale=(height / self.document["canvas_height"], width / self.document["canvas_width"]),
        )
        self.image = None
        self.image_path = None
        self.bucketed = False
        self.pending_phrase = ""
        self.selected = None
        self.drag = None
        self.preview_box = None
        self.fields = {}
        self._selecting = False
        if qt:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from PySide6.QtWidgets import QDialog, QGridLayout, QWidget
            self.window = QDialog(parent)
            self.window.setWindowTitle("MOSS layout")
            self.window.resize(1160, 760)
            self.window.setModal(True)
            outer = QGridLayout(self.window)
            from PySide6.QtWidgets import QScrollArea
            self.form = QWidget(self.window)
            self.grid = QGridLayout(self.form)
            scroll = QScrollArea(self.window)
            scroll.setWidgetResizable(True)
            scroll.setWidget(self.form)
            scroll.setMinimumWidth(420)
            outer.addWidget(scroll, 0, 0)
            self.figure = Figure(figsize=(7, 7))
            self.canvas = FigureCanvasQTAgg(self.figure)
            # Matplotlib may have queued an idle draw when the parent is deleted.
            self.canvas.destroyed.connect(lambda: setattr(self.canvas, "_draw_pending", False))
            outer.addWidget(self.canvas, 0, 1)
            outer.setColumnStretch(1, 1)
        else:
            import customtkinter as ctk
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            self.window = ctk.CTkToplevel(parent)
            self.window.title("MOSS layout")
            self.window.geometry("1160x760")
            self.window.grid_columnconfigure(1, weight=1)
            self.window.grid_rowconfigure(0, weight=1)
            self.form = ctk.CTkScrollableFrame(self.window, width=370)
            self.form.grid(row=0, column=0, sticky="nsew")
            self.figure = Figure(figsize=(7, 7))
            self.canvas = FigureCanvasTkAgg(self.figure, master=self.window)
            self.canvas.get_tk_widget().grid(row=0, column=1, sticky="nsew")
        self.axes = self.figure.add_subplot()
        row = 0
        if on_save is None:
            self._button("Open training image", row, self.open_image)
            row += 1
        self._button("Import JSON", row, self.import_json)
        row += 1
        self._button("Export JSON", row, self.export_json)
        row += 1
        self._field("summary", "Caption / scene description", row, multiline=True)
        row += 1
        self._button("Use selected phrase for next box", row, self.use_selected_phrase)
        row += 1
        if on_save is None:
            self._button("Load caption file", row, self.load_caption)
            row += 1
            self._field("resolution", "MGDS resolution", row)
            self._set("resolution", "1024")
            row += 1
            self._field("variant", "Resolution variant (0-based)", row)
            self._set("variant", "0")
            row += 1
            self._button("Apply MGDS bucket (center crop)", row, self.apply_bucket)
            row += 1
            self._button("Export image + caption + layout", row, self.export_bundle)
            row += 1
        self.mode = self._choice("Mode", row, ["Draw", "Select / move"], lambda _: None)
        row += 1
        self.object_choice = self._choice("Object", row, ["None"], self.choose_object)
        row += 1
        for key, label in (("label", "Label (object / text)"), ("caption", "Object description"),
                           ("text", "Rendered text"), ("x0", "Left"), ("y0", "Top"),
                           ("x1", "Right"), ("y1", "Bottom")):
            self._field(key, label, row, multiline=key in ("caption", "text"))
            row += 1
        self._button("Apply object changes", row, self.apply_item)
        self._button("Delete selected object", row + 1, self.delete_item)
        self._button("Apply layout" if on_save else "Save .layout.json", row + 2, self.save)
        self._label("Draw a box, or select to move it.\nDrag a corner to resize. Coordinates are source pixels.", row + 3)
        self.status = self._label("", row + 4)
        for event, handler in (("button_press_event", self.press), ("motion_notify_event", self.motion),
                               ("button_release_event", self.release)):
            self.canvas.mpl_connect(event, handler)
        self._set("summary", self.document["summary"]["caption"])
        self._set("label", "object")
        self.refresh()
        if qt:
            self.window.show()
        else:
            self.window.after_idle(self.window.lift)

    def _place(self, widget, row, column, colspan=1):
        if self.qt:
            self.grid.addWidget(widget, row, column, 1, colspan)
        else:
            widget.grid(row=row, column=column, columnspan=colspan, padx=5, pady=3, sticky="ew")

    def _label(self, text, row):
        if self.qt:
            from PySide6.QtWidgets import QLabel
            widget = QLabel(text, self.form)
            widget.setWordWrap(True)
        else:
            import customtkinter as ctk
            widget = ctk.CTkLabel(self.form, text=text, wraplength=340, justify="left")
        self._place(widget, row, 0, 2)
        return widget

    def _button(self, text, row, command):
        if self.qt:
            from PySide6.QtWidgets import QPushButton
            widget = QPushButton(text, self.form)
            widget.clicked.connect(lambda: self.guard(command))
        else:
            import customtkinter as ctk
            widget = ctk.CTkButton(self.form, text=text, command=lambda: self.guard(command))
        self._place(widget, row, 0, 2)

    def _field(self, key, label, row, multiline=False):
        if self.qt:
            from PySide6.QtWidgets import QLabel, QLineEdit, QPlainTextEdit
            title = QLabel(label, self.form)
            widget = QPlainTextEdit(self.form) if multiline else QLineEdit(self.form)
            if multiline:
                widget.setMaximumHeight(65)
            get = widget.toPlainText if multiline else widget.text
            put = widget.setPlainText if multiline else widget.setText
        else:
            import customtkinter as ctk
            title = ctk.CTkLabel(self.form, text=label)
            widget = ctk.CTkTextbox(self.form, width=210, height=65) if multiline else ctk.CTkEntry(self.form, width=210)
            get = (lambda: widget.get("1.0", "end-1c")) if multiline else widget.get
            def put(value):
                widget.delete("1.0" if multiline else 0, "end")
                widget.insert("1.0" if multiline else 0, value)
        self._place(title, row, 0)
        self._place(widget, row, 1)
        self.fields[key] = (get, put)
        if key == "summary":
            self.caption_widget = widget
            if not self.qt:
                widget._textbox.configure(exportselection=False)

    def _choice(self, title, row, choices, callback):
        if self.qt:
            from PySide6.QtWidgets import QComboBox, QLabel
            label = QLabel(title, self.form)
            widget = QComboBox(self.form)
            widget.addItems(choices)
            widget.currentTextChanged.connect(lambda value: self.guard(lambda: callback(value)))
        else:
            import customtkinter as ctk
            label = ctk.CTkLabel(self.form, text=title)
            widget = ctk.CTkOptionMenu(self.form, values=choices, command=lambda value: self.guard(lambda: callback(value)))
        self._place(label, row, 0)
        self._place(widget, row, 1)
        return widget

    def _get(self, name):
        return self.fields[name][0]()

    def _set(self, name, value):
        self.fields[name][1](str(value))

    def _mode(self):
        return self.mode.currentText() if self.qt else self.mode.get()

    def guard(self, action):
        try:
            action()
        except (ValueError, OSError, TypeError) as error:
            self.message(str(error), error=True)

    def message(self, message, error=False):
        if self.qt:
            self.status.setText(message)
            if error:
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.warning(self.window, "MOSS layout", message)
        else:
            self.status.configure(text=message)
            if error:
                from tkinter import messagebox
                messagebox.showerror("MOSS layout", message, parent=self.window)

    def file_dialog(self, *, save=False, image=False, caption=False):
        if self.qt:
            from PySide6.QtWidgets import QFileDialog
            method = QFileDialog.getSaveFileName if save else QFileDialog.getOpenFileName
            filters = "Images (*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff)" if image else "Captions (*.txt)" if caption else "JSON (*.json)"
            return method(self.window, "MOSS layout", str(self.concept_path or ""), filters)[0]
        from tkinter import filedialog
        method = filedialog.asksaveasfilename if save else filedialog.askopenfilename
        types = [("Images", "*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff")] if image else [("Captions", "*.txt")] if caption else [("JSON", "*.json")]
        return method(parent=self.window, initialdir=self.concept_path, filetypes=types,
                      **({"defaultextension": ".json"} if save else {}))

    def load_image(self, path):
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
        raw, _ = read_layout_source(path)
        width, height = image.size
        caption_path = Path(path).with_suffix(".txt")
        caption = caption_path.read_text(encoding="utf-8-sig").strip() if caption_path.is_file() else ""
        document = parse_layout(raw or empty_layout(width, height, caption), width, height, match_image=True)
        self.image, self.image_path, self.document = image, Path(path), document
        self.bucketed = False
        self.selected = None
        self._set("summary", document["summary"]["caption"])
        self.refresh()

    def open_image(self):
        path = self.file_dialog(image=True)
        if path:
            self.load_image(path)

    def import_json(self):
        path = self.file_dialog()
        if path:
            width, height = self.document["canvas_width"], self.document["canvas_height"]
            document = parse_layout(Path(path).read_text(encoding="utf-8-sig"), width, height)
            if self.image is not None and (document["canvas_width"], document["canvas_height"]) != self.image.size:
                raise ValueError("Imported layout dimensions do not match the open training image.")
            self.document, self.selected = document, None
            self._set("summary", document["summary"]["caption"])
            self.refresh()

    def export_json(self):
        self.apply_item()
        self.document["summary"]["caption"] = self._get("summary")
        path = self.file_dialog(save=True)
        if path:
            Path(path).write_text(json.dumps(layout_document(self.document), ensure_ascii=False, indent=2), encoding="utf-8")
            self.message(f"Exported {path}")

    def save(self):
        self.apply_item()
        self.document["summary"]["caption"] = self._get("summary")
        raw = json.dumps(layout_document(self.document), ensure_ascii=False, indent=2)
        parse_layout(raw)
        if self.on_save:
            self.on_save(raw)
            self.message("Layout applied to the sample.")
        else:
            if self.bucketed:
                self.export_bundle()
                return
            if self.image_path is None:
                raise ValueError("Open a training image before saving its annotation.")
            path = self.image_path.with_suffix(".layout.json")
            path.write_text(raw, encoding="utf-8")
            self.message(f"Saved {path.name}")

    def selected_phrase(self):
        if self.qt:
            return self.caption_widget.textCursor().selectedText().replace("\u2029", "\n")
        from tkinter import TclError
        try:
            return self.caption_widget.get("sel.first", "sel.last")
        except TclError:
            return ""

    def use_selected_phrase(self):
        phrase = self.selected_phrase()
        if not phrase:
            raise ValueError("Select a word or phrase in the caption first.")
        self.pending_phrase = phrase
        self.message(f"Next box: {phrase}")

    def load_caption(self):
        path = self.file_dialog(caption=True)
        if path:
            self._set("summary", Path(path).read_text(encoding="utf-8-sig").strip())

    def apply_bucket(self):
        if self.image is None:
            raise ValueError("Open a training image first.")
        self.apply_item()
        self.document["summary"]["caption"] = self._get("summary")
        from modules.util.moss_bucketing import bucket_image
        result = bucket_image(self.image, layout_document(self.document), self._get("resolution"),
                              variant=int(self._get("variant")))
        self.image, self.document = result["image"], result["layout"]
        self.bucketed, self.selected = True, None
        self.refresh()
        self.message(f"MGDS bucket {self.image.width}x{self.image.height}; "
                     f"scale={result['scale_resolution']}; crop offset={result['crop_offset']}. "
                     "Export the image together with its layout.")

    def write_bundle(self, directory):
        if self.image is None or self.image_path is None:
            raise ValueError("Open a training image first.")
        self.apply_item()
        self.document["summary"]["caption"] = self._get("summary")
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        stem = self.image_path.stem + f"-{self.image.width}x{self.image.height}"
        image_path = destination / (stem + ".png")
        caption_path = destination / (stem + ".txt")
        layout_path = destination / (stem + ".layout.json")
        if image_path.resolve() == self.image_path.resolve():
            raise ValueError("Choose a different export folder to preserve the source image.")
        self.image.save(image_path)
        caption_path.write_text(self._get("summary"), encoding="utf-8")
        layout_path.write_text(json.dumps(layout_document(self.document), ensure_ascii=False, indent=2), encoding="utf-8")
        self.message(f"Exported {image_path.name}, caption and layout.")
        return image_path, caption_path, layout_path

    def export_bundle(self):
        if self.qt:
            from PySide6.QtWidgets import QFileDialog
            directory = QFileDialog.getExistingDirectory(self.window, "Export annotated image")
        else:
            from tkinter import filedialog
            directory = filedialog.askdirectory(parent=self.window, title="Export annotated image")
        if directory:
            self.write_bundle(directory)

    def choose_object(self, value):
        if self._selecting:
            return
        self.selected = None if value == "None" else int(value.split(":", 1)[0])
        self.fill_selection()
        self.redraw()

    def fill_selection(self):
        if self.selected is None:
            return
        item = self.document["items"][self.selected]
        for key in ("label", "caption", "text"):
            self._set(key, item.get(key, ""))
        y0, x0, y1, x1 = item["bbox"]
        w, h = self.document["canvas_width"], self.document["canvas_height"]
        for key, value in zip(("x0", "y0", "x1", "y1"), (x0 + w / 2, y0 + h / 2, x1 + w / 2, y1 + h / 2), strict=True):
            self._set(key, round(value, 4))

    def apply_item(self):
        if self.selected is None:
            return
        w, h = self.document["canvas_width"], self.document["canvas_height"]
        x0, y0, x1, y1 = (float(self._get(key)) for key in ("x0", "y0", "x1", "y1"))
        if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
            raise ValueError("Box edges must be finite, ordered and inside the canvas.")
        item = copy.deepcopy(self.document["items"][self.selected])
        item.update(label=self._get("label"), caption=self._get("caption"), bbox=[y0 - h / 2, x0 - w / 2, y1 - h / 2, x1 - w / 2])
        text = self._get("text")
        if text or item["label"] == "text":
            item["text"] = text
        else:
            item.pop("text", None)
        self.document["items"][self.selected] = item
        self.document = parse_layout(layout_document(self.document))
        self.refresh()

    def delete_item(self):
        if self.selected is not None:
            del self.document["items"][self.selected]
            self.selected = None
            self.refresh()

    def refresh(self):
        self._selecting = True
        choices = ["None"] + [f"{i}: {item['label']} {item['caption'][:28]}" for i, item in enumerate(self.document["items"])]
        if self.qt:
            self.object_choice.clear()
            self.object_choice.addItems(choices)
            self.object_choice.setCurrentIndex(self.selected + 1 if self.selected is not None else 0)
        else:
            self.object_choice.configure(values=choices)
            self.object_choice.set(choices[self.selected + 1 if self.selected is not None else 0])
        self._selecting = False
        self.fill_selection()
        self.redraw()

    def redraw(self):
        self.axes.clear()
        w, h = self.document["canvas_width"], self.document["canvas_height"]
        if self.image is not None:
            self.axes.imshow(self.image, extent=(0, w, h, 0))
        self.axes.set(xlim=(0, w), ylim=(h, 0), aspect="equal", title=f"{w} x {h} — draw / move / resize")
        for index, item in enumerate(self.document["items"]):
            y0, x0, y1, x1 = item["bbox"]
            x0, x1, y0, y1 = x0 + w / 2, x1 + w / 2, y0 + h / 2, y1 + h / 2
            color = "#ff9e32" if index == self.selected else "#00a6dd"
            self.axes.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=color, linewidth=2))
            self.axes.text(x0, y0, f"{index}: {item['label']}", color=color, va="bottom")
            if index == self.selected:
                self.axes.plot([x0, x1, x1, x0], [y0, y0, y1, y1], "s", color=color, markersize=5)
        if self.preview_box:
            x0, y0, x1, y1 = self.preview_box
            self.axes.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                                          fill=False, edgecolor="#ff9e32", linestyle="--"))
        self.canvas.draw_idle()

    def _point(self, event):
        if event.inaxes is not self.axes or event.xdata is None or event.ydata is None:
            return None
        w, h = self.document["canvas_width"], self.document["canvas_height"]
        return max(0, min(w, event.xdata)), max(0, min(h, event.ydata))

    def press(self, event):
        point = self._point(event)
        if event.button != 1 or point is None:
            return
        if self._mode() == "Draw":
            self.pending_phrase = self.selected_phrase() or self.pending_phrase
            self.drag = ("new", point, None)
            return
        x, y = point
        w, h = self.document["canvas_width"], self.document["canvas_height"]
        self.selected = None
        for index in reversed(range(len(self.document["items"]))):
            item = self.document["items"][index]
            y0, x0, y1, x1 = item["bbox"]
            x0, x1, y0, y1 = x0 + w / 2, x1 + w / 2, y0 + h / 2, y1 + h / 2
            corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
            near = next((i for i, (cx, cy) in enumerate(corners)
                         if abs(cx - x) <= w * 0.015 and abs(cy - y) <= h * 0.015), None)
            if near is not None or (x0 <= x <= x1 and y0 <= y <= y1):
                self.selected = index
                self.drag = ("resize" if near is not None else "move", point,
                             (copy.deepcopy(item), near))
                break
        self.refresh()

    def motion(self, event):
        if self.drag is None:
            return
        point = self._point(event)
        if point is None:
            return
        self._drag_to(point)
        self.redraw()

    def _drag_to(self, point):
        kind, start, original = self.drag
        w, h = self.document["canvas_width"], self.document["canvas_height"]
        if kind == "new":
            x0, x1 = sorted([start[0], point[0]])
            y0, y1 = sorted([start[1], point[1]])
            self.preview_box = (x0, y0, x1, y1)
            return
        item, corner = original
        y0, x0, y1, x1 = item["bbox"]
        x0, x1, y0, y1 = x0 + w / 2, x1 + w / 2, y0 + h / 2, y1 + h / 2
        if kind == "move":
            dx = max(-x0, min(w - x1, point[0] - start[0]))
            dy = max(-y0, min(h - y1, point[1] - start[1]))
            x0, x1, y0, y1 = x0 + dx, x1 + dx, y0 + dy, y1 + dy
        else:
            opposite = [(x1, y1), (x0, y1), (x0, y0), (x1, y0)][corner]
            x0, x1 = sorted([point[0], opposite[0]])
            y0, y1 = sorted([point[1], opposite[1]])
        if x1 > x0 and y1 > y0:
            self.document["items"][self.selected]["bbox"] = [y0 - h / 2, x0 - w / 2, y1 - h / 2, x1 - w / 2]

    def release(self, event):
        if self.drag is None:
            return
        point = self._point(event)
        kind, start, _ = self.drag
        if point is not None:
            if kind == "new":
                x0, x1 = sorted([start[0], point[0]])
                y0, y1 = sorted([start[1], point[1]])
                w, h = self.document["canvas_width"], self.document["canvas_height"]
                if x1 > x0 and y1 > y0:
                    self.document["items"].append({
                        "label": self._get("label") or "object", "caption": self.pending_phrase, "bbox": [y0 - h / 2, x0 - w / 2, y1 - h / 2, x1 - w / 2],
                    })
                    self.selected = len(self.document["items"]) - 1
            else:
                self._drag_to(point)
        self.drag = None
        self.preview_box = None
        self.document = parse_layout(layout_document(self.document))
        self.refresh()


def open_sample_layout(parent, components, sample, ui_state):
    def save(raw):
        ui_state.get_var("moss_layout_json").set(raw)
        ui_state.get_var("moss_layout_enabled").set(True)
    editor = MOSSLayoutEditor(
        parent, qt="pyside6" in components.__name__,
        layout=sample.moss_layout_json or (sample.prompt if is_layout(sample.prompt) else empty_layout(sample.width, sample.height, sample.prompt)),
        width=sample.width, height=sample.height, on_save=save,
    )
    # Retain the Python controller while the native dialog is open.
    editor.window._moss_editor = editor
    return editor


def open_concept_layout(parent, components, controller):
    editor = MOSSLayoutEditor(
        parent, qt="pyside6" in components.__name__,
        concept_path=controller.get_concept_path(controller.concept.path),
    )
    editor.window._moss_editor = editor
    return editor
