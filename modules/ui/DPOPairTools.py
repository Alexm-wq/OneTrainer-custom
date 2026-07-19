from __future__ import annotations

import os
import threading
from tkinter import filedialog, messagebox

from PIL import Image

from modules.util.dpo_bucket_analysis_util import (
    analyze_concept,
    parse_target_resolutions,
    quantization_for_model,
)
from modules.util.dpo_curation_util import remove_finalized_pair, scan_finalized_pairs


def _is_qt_parent(parent) -> bool:
    try:
        from PySide6.QtWidgets import QWidget
        return isinstance(parent, QWidget)
    except Exception:
        return False


def open_pair_review(parent, concept_pairs):
    if _is_qt_parent(parent):
        window = QtPairReviewDialog(parent, concept_pairs)
        window.show()
        return window
    window = CtkPairReviewWindow(
        parent.winfo_toplevel() if parent is not None else None,
        concept_pairs,
    )
    return window


def open_bucket_analysis(parent, train_config):
    if _is_qt_parent(parent):
        window = QtBucketAnalysisDialog(parent, train_config)
        window.show()
        return window
    window = CtkBucketAnalysisWindow(
        parent.winfo_toplevel() if parent is not None else None,
        train_config,
    )
    return window


class PairReviewState:
    def __init__(self, concept_pairs):
        self.pairs = scan_finalized_pairs(concept_pairs)
        self.index = 0
        self.removed = 0

    def current(self):
        if not self.pairs:
            return None
        self.index = max(0, min(self.index, len(self.pairs) - 1))
        return self.pairs[self.index]

    def move(self, delta):
        if self.pairs:
            self.index = max(0, min(self.index + delta, len(self.pairs) - 1))

    def remove(self):
        entry = self.current()
        if entry is None:
            return
        remove_finalized_pair(entry.get("chosen_path"), entry.get("rejected_path"))
        self.pairs.pop(self.index)
        self.removed += 1
        if self.index >= len(self.pairs):
            self.index = max(0, len(self.pairs) - 1)


try:
    import customtkinter as ctk

    class CtkPairReviewWindow(ctk.CTkToplevel):
        def __init__(self, parent, concept_pairs):
            super().__init__(parent)
            self.title("Review DPO Pairs")
            self.geometry("1400x900")
            self.transient(parent)
            self.state_data = PairReviewState(concept_pairs)
            self.protocol("WM_DELETE_WINDOW", self._close)
            self.wait_visibility()
            self.grab_set()
            self._render()

        def _close(self):
            try:
                self.grab_release()
            except Exception:
                pass
            self.destroy()

        def _clear(self):
            for child in self.winfo_children():
                child.destroy()

        def _render(self):
            self._clear()
            entry = self.state_data.current()
            if entry is None:
                ctk.CTkLabel(
                    self,
                    text=f"No pairs remain. Removed: {self.state_data.removed}",
                    font=("", 22, "bold"),
                ).pack(expand=True)
                ctk.CTkButton(self, text="Close", command=self._close).pack(pady=20)
                return

            ctk.CTkLabel(
                self,
                text=(
                    f"Pair {self.state_data.index + 1}/{len(self.state_data.pairs)} | "
                    f"Key: {entry.get('key')} | Removed: {self.state_data.removed}"
                ),
                font=("", 14, "bold"),
            ).pack(fill="x", padx=15, pady=8)

            grid = ctk.CTkFrame(self, fg_color="transparent")
            grid.pack(expand=True, fill="both", padx=10, pady=5)
            grid.grid_columnconfigure(0, weight=1)
            grid.grid_columnconfigure(1, weight=1)
            grid.grid_rowconfigure(1, weight=1)
            ctk.CTkLabel(grid, text="Chosen", text_color="green", font=("", 14, "bold")).grid(row=0, column=0)
            ctk.CTkLabel(grid, text="Rejected", text_color="red", font=("", 14, "bold")).grid(row=0, column=1)
            self._image(grid, entry.get("chosen_path"), 1, 0)
            self._image(grid, entry.get("rejected_path"), 1, 1)

            buttons = ctk.CTkFrame(self, fg_color="transparent")
            buttons.pack(fill="x", padx=10, pady=10)
            ctk.CTkButton(buttons, text="← Back", command=lambda: self._move(-1)).pack(side="left")
            ctk.CTkButton(
                buttons,
                text="Remove Pair",
                fg_color="#B22222",
                command=self._remove,
            ).pack(side="left", expand=True, padx=20)
            ctk.CTkButton(buttons, text="Keep →", command=lambda: self._move(1)).pack(side="right")

        def _image(self, master, path, row, col):
            if not path or not os.path.isfile(path):
                ctk.CTkLabel(master, text="(missing)", text_color="gray").grid(row=row, column=col)
                return
            try:
                max_w = max(400, (self.winfo_width() or 1400) // 2 - 50)
                max_h = max(400, (self.winfo_height() or 900) - 210)
                with Image.open(path) as source:
                    image = source.convert("RGB")
                    scale = min(max_w / image.width, max_h / image.height)
                    image = image.resize(
                        (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
                        Image.Resampling.LANCZOS,
                    ).copy()
                ctk_image = ctk.CTkImage(light_image=image, dark_image=image, size=image.size)
                label = ctk.CTkLabel(master, text="", image=ctk_image)
                label.image = ctk_image
                label.pil_image = image
                label.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")
            except Exception as ex:
                ctk.CTkLabel(master, text=f"{os.path.basename(path)}\n{ex}").grid(row=row, column=col)

        def _move(self, delta):
            self.state_data.move(delta)
            self._render()

        def _remove(self):
            self.state_data.remove()
            self._render()


    class CtkBucketAnalysisWindow(ctk.CTkToplevel):
        def __init__(self, parent, train_config):
            super().__init__(parent)
            self.title("DPO Bucket / Batch-Size Analyzer")
            self.geometry("1050x760")
            self.transient(parent)
            self.path_var = ctk.StringVar(value="")
            self.batch_var = ctk.StringVar(value=str(max(1, int(getattr(train_config, "batch_size", 1) or 1))))
            resolutions = parse_target_resolutions(str(getattr(train_config, "resolution", "") or "")) or [512]
            self.target_var = ctk.StringVar(value=str(min(resolutions)))
            self.quant_var = ctk.StringVar(value=str(quantization_for_model(getattr(train_config, "model_type", ""))))
            self._thread = None
            self._result = {}
            self._build()
            self.wait_visibility()
            self.grab_set()

        def _build(self):
            row = ctk.CTkFrame(self, fg_color="transparent")
            row.pack(fill="x", padx=15, pady=10)
            ctk.CTkButton(row, text="Select Chosen Folder", command=self._browse).pack(side="left")
            ctk.CTkEntry(row, textvariable=self.path_var).pack(side="left", fill="x", expand=True, padx=10)

            settings = ctk.CTkFrame(self, fg_color="transparent")
            settings.pack(fill="x", padx=15)
            for label, variable, width in [
                ("Batch", self.batch_var, 70),
                ("Target", self.target_var, 90),
                ("Quantization", self.quant_var, 80),
            ]:
                ctk.CTkLabel(settings, text=label).pack(side="left", padx=(0, 4))
                ctk.CTkEntry(settings, textvariable=variable, width=width).pack(side="left", padx=(0, 15))
            ctk.CTkButton(settings, text="Analyze", command=self._run).pack(side="right")

            self.output = ctk.CTkTextbox(self, wrap="none")
            self.output.pack(expand=True, fill="both", padx=15, pady=12)
            self.output.insert("1.0", "Select a chosen-side folder and click Analyze.")

        def _browse(self):
            try:
                self.grab_release()
            except Exception:
                pass
            path = filedialog.askdirectory(title="Select chosen-side DPO folder")
            try:
                self.grab_set()
            except Exception:
                pass
            if path:
                self.path_var.set(path)

        def _run(self):
            try:
                path = self.path_var.get().strip()
                batch = int(self.batch_var.get())
                target = int(self.target_var.get())
                quant = int(self.quant_var.get())
                if not path or batch <= 0 or target <= 0 or quant <= 0:
                    raise ValueError
            except ValueError:
                messagebox.showwarning("Invalid Settings", "Choose a folder and enter positive integers.", parent=self)
                return
            self.output.delete("1.0", "end")
            self.output.insert("1.0", "Scanning images...")
            self._result = {}

            def worker():
                try:
                    self._result["value"] = analyze_concept(path, batch, [target], quant)
                except Exception as ex:
                    self._result["error"] = str(ex)
            self._thread = threading.Thread(target=worker, daemon=True)
            self._thread.start()
            self.after(100, self._poll)

        def _poll(self):
            if self._thread and self._thread.is_alive():
                self.after(100, self._poll)
                return
            if "error" in self._result:
                self._set_output(self._result["error"])
                return
            self._set_output(format_bucket_result(self._result["value"]))

        def _set_output(self, text):
            self.output.delete("1.0", "end")
            self.output.insert("1.0", text)

except Exception:
    CtkPairReviewWindow = CtkBucketAnalysisWindow = None


try:
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtGui import QPixmap
    from PySide6.QtWidgets import (
        QDialog,
        QFileDialog,
        QFormLayout,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMessageBox,
        QPlainTextEdit,
        QPushButton,
        QVBoxLayout,
    )

    class QtPairReviewDialog(QDialog):
        def __init__(self, parent, concept_pairs):
            super().__init__(parent)
            self.setWindowTitle("Review DPO Pairs")
            self.resize(1400, 900)
            self.state_data = PairReviewState(concept_pairs)
            root = QVBoxLayout(self)
            self.header = QLabel()
            root.addWidget(self.header)
            images = QHBoxLayout()
            root.addLayout(images, 1)
            self.chosen = QLabel()
            self.rejected = QLabel()
            self.chosen.setAlignment(Qt.AlignCenter)
            self.rejected.setAlignment(Qt.AlignCenter)
            images.addWidget(self.chosen, 1)
            images.addWidget(self.rejected, 1)
            buttons = QHBoxLayout()
            root.addLayout(buttons)
            back = QPushButton("← Back")
            remove = QPushButton("Remove Pair")
            keep = QPushButton("Keep →")
            buttons.addWidget(back)
            buttons.addStretch(1)
            buttons.addWidget(remove)
            buttons.addStretch(1)
            buttons.addWidget(keep)
            back.clicked.connect(lambda: self._move(-1))
            keep.clicked.connect(lambda: self._move(1))
            remove.clicked.connect(self._remove)
            self._render()

        def _set_image(self, label, path):
            if not path or not os.path.isfile(path):
                label.setPixmap(QPixmap())
                label.setText("(missing)")
                return
            pixmap = QPixmap(path)
            if pixmap.isNull():
                label.setText(os.path.basename(path))
                return
            label.setText("")
            label.setPixmap(pixmap.scaled(640, 720, Qt.KeepAspectRatio, Qt.SmoothTransformation))

        def _render(self):
            entry = self.state_data.current()
            if entry is None:
                self.header.setText(f"No pairs remain. Removed: {self.state_data.removed}")
                self.chosen.clear()
                self.rejected.clear()
                return
            self.header.setText(
                f"Pair {self.state_data.index + 1}/{len(self.state_data.pairs)} | "
                f"Key: {entry.get('key')} | Removed: {self.state_data.removed}"
            )
            self._set_image(self.chosen, entry.get("chosen_path"))
            self._set_image(self.rejected, entry.get("rejected_path"))

        def _move(self, delta):
            self.state_data.move(delta)
            self._render()

        def _remove(self):
            self.state_data.remove()
            self._render()


    class QtBucketAnalysisDialog(QDialog):
        def __init__(self, parent, train_config):
            super().__init__(parent)
            self.setWindowTitle("DPO Bucket / Batch-Size Analyzer")
            self.resize(1050, 760)
            root = QVBoxLayout(self)
            form = QFormLayout()
            root.addLayout(form)
            path_row = QHBoxLayout()
            self.path = QLineEdit()
            browse = QPushButton("Browse")
            path_row.addWidget(self.path, 1)
            path_row.addWidget(browse)
            form.addRow("Chosen-side folder", path_row)
            self.batch = QLineEdit(str(max(1, int(getattr(train_config, "batch_size", 1) or 1))))
            resolutions = parse_target_resolutions(str(getattr(train_config, "resolution", "") or "")) or [512]
            self.target = QLineEdit(str(min(resolutions)))
            self.quant = QLineEdit(str(quantization_for_model(getattr(train_config, "model_type", ""))))
            form.addRow("Batch size", self.batch)
            form.addRow("Target resolution", self.target)
            form.addRow("Quantization", self.quant)
            analyze = QPushButton("Analyze")
            root.addWidget(analyze)
            self.output = QPlainTextEdit()
            self.output.setReadOnly(True)
            root.addWidget(self.output, 1)
            self._thread = None
            self._result = {}
            browse.clicked.connect(self._browse)
            analyze.clicked.connect(self._run)

        def _browse(self):
            path = QFileDialog.getExistingDirectory(self, "Select chosen-side DPO folder")
            if path:
                self.path.setText(path)

        def _run(self):
            try:
                path = self.path.text().strip()
                batch = int(self.batch.text())
                target = int(self.target.text())
                quant = int(self.quant.text())
                if not path or min(batch, target, quant) <= 0:
                    raise ValueError
            except ValueError:
                QMessageBox.warning(self, "Invalid Settings", "Choose a folder and enter positive integers.")
                return
            self.output.setPlainText("Scanning images...")
            self._result = {}
            def worker():
                try:
                    self._result["value"] = analyze_concept(path, batch, [target], quant)
                except Exception as ex:
                    self._result["error"] = str(ex)
            self._thread = threading.Thread(target=worker, daemon=True)
            self._thread.start()
            QTimer.singleShot(100, self._poll)

        def _poll(self):
            if self._thread and self._thread.is_alive():
                QTimer.singleShot(100, self._poll)
                return
            if "error" in self._result:
                self.output.setPlainText(self._result["error"])
            else:
                self.output.setPlainText(format_bucket_result(self._result["value"]))

except Exception:
    QtPairReviewDialog = QtBucketAnalysisDialog = None


def format_bucket_result(result):
    lines = [
        f"Path: {result['concept_path']}",
        f"Scanned: {result['scanned']}",
        f"Unreadable: {result['unreadable']}",
        f"Batch size: {result['batch_size']}",
        f"Quantization: {result['quantization']}",
        "",
    ]
    for target in result.get("targets", []):
        lines.extend([
            f"Target resolution: {target['target']}",
            (
                f"Pairs={target['total_pairs']} Drops={target['total_drops']} "
                f"Add={target['total_add']} Remove={target['total_remove']}"
            ),
            "",
            f"{'Aspect':28} {'Bucket':14} {'Count':>8} {'Drops':>8} {'Add':>8} {'Remove':>8}",
            "-" * 84,
        ])
        for row in target.get("buckets", []):
            lines.append(
                f"{row['aspect_label'][:28]:28} "
                f"{row['h']}x{row['w']:<8} "
                f"{row['count']:>8} {row['drops']:>8} "
                f"{row['add']:>8} {row['remove']:>8}"
            )
        lines.append("")
    return "\n".join(lines)
