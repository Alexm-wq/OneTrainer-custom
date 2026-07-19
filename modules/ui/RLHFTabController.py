from __future__ import annotations

import json
import os
import threading
from tkinter import messagebox as tk_messagebox

from modules.util.config.ConceptConfig import ConceptConfig
from modules.util.config.TrainConfig import TrainConfig
from modules.util.dpo_curation_util import (
    check_dpo_pairs,
    correct_all_captions_to_chosen,
    dpo_pair_key,
    find_caption_mismatches,
    fix_multiline_captions,
    remove_finalized_pair,
    repair_rejected_pairs,
)
from modules.util.dpo_pattern_util import dpo_concept_pattern_dirs
from modules.util.path_util import supported_image_extensions

try:
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication, QMessageBox, QProgressDialog, QWidget
except Exception:
    QApplication = QMessageBox = QProgressDialog = QTimer = QWidget = None


class RLHFTabController:
    def __init__(self, train_config: TrainConfig):
        self.train_config = train_config
        self.parent_widget = None
        self._tool_window = None
        self._worker_thread = None

    def set_parent(self, parent_widget):
        self.parent_widget = parent_widget

    def _parent(self):
        if self.parent_widget is not None:
            return self.parent_widget
        if QApplication is not None:
            return QApplication.activeWindow()
        return None

    def _is_qt(self) -> bool:
        return QWidget is not None and isinstance(self._parent(), QWidget)

    def _info(self, title: str, text: str, details: str | None = None):
        if self._is_qt():
            box = QMessageBox(self._parent())
            box.setIcon(QMessageBox.Information)
            box.setWindowTitle(title)
            box.setText(text)
            if details:
                box.setDetailedText(details)
            box.exec()
        else:
            body = text if not details else f"{text}\n\n{details}"
            tk_messagebox.showinfo(title, body, parent=self._parent())

    def _error(self, title: str, text: str):
        if self._is_qt():
            QMessageBox.critical(self._parent(), title, text)
        else:
            tk_messagebox.showerror(title, text, parent=self._parent())

    def _ask(self, title: str, text: str) -> bool:
        if self._is_qt():
            return QMessageBox.question(
                self._parent(),
                title,
                text,
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            ) == QMessageBox.Yes
        return bool(tk_messagebox.askyesno(title, text, parent=self._parent()))

    def _load_concepts(self):
        concepts = self.train_config.concepts
        if concepts is None:
            filename = self.train_config.concept_file_name
            if not filename or not os.path.isfile(filename):
                raise RuntimeError("No concepts configured.\nSet up concepts in the Concepts tab first.")
            with open(filename, "r", encoding="utf-8") as file:
                concepts = [
                    ConceptConfig.default_values().from_dict(item)
                    for item in json.load(file)
                ]
        return concepts

    def _load_concept_pairs(self) -> list[tuple[str, str]]:
        pairs = dpo_concept_pattern_dirs(self._load_concepts())
        if not pairs:
            raise RuntimeError(
                "No DPO concepts found.\n"
                "Set chosen/rejected patterns on a concept in the Concepts tab."
            )
        return pairs

    @staticmethod
    def _index_images(root: str) -> dict[str, str]:
        indexed = {}
        extensions = supported_image_extensions()
        if not root or not os.path.isdir(root):
            return indexed
        for current, dirs, files in os.walk(root):
            dirs[:] = [name for name in dirs if not name.startswith(".")]
            for filename in files:
                if os.path.splitext(filename)[1].lower() not in extensions:
                    continue
                path = os.path.join(current, filename)
                indexed[dpo_pair_key(path, root)] = path
        return indexed

    def _remove_strays(self, concept_pairs, result) -> int:
        removed = 0
        infos = result.get("pairs", [])
        for index, (chosen_root, rejected_root) in enumerate(concept_pairs):
            info = infos[index] if index < len(infos) else {}
            if not info.get("chosen_stray", 0) and not info.get("rejected_stray", 0):
                continue
            chosen = self._index_images(chosen_root)
            rejected = self._index_images(rejected_root)
            matched = set(chosen) & set(rejected)
            for key, path in chosen.items():
                if key not in matched:
                    remove_finalized_pair(path, None)
                    removed += 1
            for key, path in rejected.items():
                if key not in matched:
                    remove_finalized_pair(None, path)
                    removed += 1
        return removed

    def check_pairs(self):
        try:
            concept_pairs = self._load_concept_pairs()
            result = check_dpo_pairs(concept_pairs)
        except Exception as ex:
            self._error("Check Pairs Error", str(ex))
            return

        matched = int(result.get("total_matched", 0))
        chosen_strays = int(result.get("total_chosen_stray", 0))
        rejected_strays = int(result.get("total_rejected_stray", 0))
        multiline = int(result.get("multiline_captions", 0))
        total_strays = chosen_strays + rejected_strays

        lines = [
            f"Matched pairs: {matched}",
            f"Chosen strays: {chosen_strays}",
            f"Rejected strays: {rejected_strays}",
            f"Multiline captions: {multiline}",
        ]
        formats = result.get("format_stats", {})
        if formats:
            lines.append("Formats: " + ", ".join(
                f"{ext}: {count}" for ext, count in sorted(formats.items())
            ))

        details = []
        for info in result.get("pairs", []):
            details.extend([
                "--- Concept pair ---",
                f"Chosen: {info.get('chosen_path')}",
                f"Rejected: {info.get('rejected_path')}",
                (
                    f"Matched: {info.get('matched', 0)}, "
                    f"chosen stray: {info.get('chosen_stray', 0)}, "
                    f"rejected stray: {info.get('rejected_stray', 0)}"
                ),
                "",
            ])
        self._info("Check Pairs Results", "\n".join(lines), "\n".join(details))

        if total_strays and self._ask(
            "Remove Strays?",
            f"Remove {total_strays} stray image(s) and their sidecar captions?",
        ):
            try:
                removed = self._remove_strays(concept_pairs, result)
                self._info("Strays Removed", f"Removed {removed} stray image(s).")
            except Exception as ex:
                self._error("Remove Strays Error", str(ex))
                return

        if multiline and self._ask(
            "Fix Multiline Captions?",
            f"Flatten {multiline} multiline caption file(s) to one line?",
        ):
            try:
                fixed = fix_multiline_captions(concept_pairs)
                self._info("Captions Fixed", f"Flattened {fixed} caption file(s).")
            except Exception as ex:
                self._error("Fix Captions Error", str(ex))
                return

        try:
            mismatches = find_caption_mismatches(concept_pairs)
        except Exception as ex:
            self._error("Caption Mismatch Error", str(ex))
            return
        if mismatches and self._ask(
            "Caption Mismatches",
            (
                f"Found {len(mismatches)} chosen/rejected caption mismatch(es).\n\n"
                "Overwrite the rejected captions with the matched chosen captions?"
            ),
        ):
            try:
                corrected = correct_all_captions_to_chosen(mismatches)
                self._info("Captions Corrected", f"Corrected {corrected} rejected captions.")
            except Exception as ex:
                self._error("Caption Correction Error", str(ex))

    def review_pairs(self):
        try:
            pairs = self._load_concept_pairs()
            from modules.ui.DPOPairTools import open_pair_review
            self._tool_window = open_pair_review(self._parent(), pairs)
        except Exception as ex:
            self._error("Review Pairs Error", str(ex))

    def bucket_analysis(self):
        try:
            from modules.ui.DPOPairTools import open_bucket_analysis
            self._tool_window = open_bucket_analysis(self._parent(), self.train_config)
        except Exception as ex:
            self._error("DPO Bucket Analysis Error", str(ex))

    def repair_rejected(self):
        try:
            concept_pairs = self._load_concept_pairs()
        except Exception as ex:
            self._error("Re-pair Error", str(ex))
            return

        if not self._ask(
            "Re-pair Rejected Images?",
            (
                "This renames rejected files in place so each is paired with the "
                "most structurally similar chosen image inside its caption group.\n\n"
                "DINOv2 weights may download on the first run. Back up the dataset first.\n\n"
                "Proceed?"
            ),
        ):
            return

        result = {}
        def worker():
            try:
                result["summary"] = repair_rejected_pairs(concept_pairs)
            except Exception as ex:
                result["error"] = str(ex)

        self._worker_thread = threading.Thread(target=worker, daemon=True)
        self._worker_thread.start()

        if self._is_qt():
            progress = QProgressDialog(
                "Computing DINOv2 embeddings and re-pairing...",
                "",
                0,
                0,
                self._parent(),
            )
            progress.setWindowTitle("Re-pairing")
            progress.setCancelButton(None)
            progress.setMinimumDuration(0)
            progress.show()

            def poll():
                if self._worker_thread.is_alive():
                    QTimer.singleShot(150, poll)
                    return
                progress.close()
                self._finish_repair(result)
            QTimer.singleShot(150, poll)
            return

        import customtkinter as ctk
        parent = self._parent().winfo_toplevel() if self._parent() is not None else None
        dialog = ctk.CTkToplevel(parent)
        dialog.title("Re-pairing")
        dialog.transient(parent)
        ctk.CTkLabel(
            dialog,
            text="Computing embeddings and re-pairing rejected images...",
        ).pack(padx=20, pady=(20, 10))
        bar = ctk.CTkProgressBar(dialog, width=340, mode="indeterminate")
        bar.pack(padx=20, pady=(0, 20))
        bar.start()
        dialog.wait_visibility()
        dialog.grab_set()

        def poll():
            if self._worker_thread.is_alive():
                dialog.after(150, poll)
                return
            bar.stop()
            try:
                dialog.grab_release()
            except Exception:
                pass
            dialog.destroy()
            self._finish_repair(result)
        dialog.after(150, poll)

    def _finish_repair(self, result):
        if "error" in result:
            self._error("Re-pair Error", str(result["error"]))
            return
        summary = result.get("summary", {})
        self._info(
            "Re-pair Complete",
            (
                f"Re-paired: {summary.get('pairs_repaired', 0)}\n"
                f"Groups processed: {summary.get('groups_processed', 0)}\n"
                f"Single-pair groups skipped: {summary.get('groups_skipped_single', 0)}\n"
                f"Pairs considered: {summary.get('pairs_total', 0)}"
            ),
        )
