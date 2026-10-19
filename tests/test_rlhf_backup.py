"""Focused RLHF backup/resume tests with real CPU LoRA gradients and optimizer state."""
import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

from modules.modelLoader.mixin.InternalModelLoaderMixin import InternalModelLoaderMixin
from modules.modelSaver.mixin.InternalModelSaverMixin import InternalModelSaverMixin
from modules.trainer.GenericTrainer import GenericTrainer
from modules.util.dpo_beta_controller import DPOBetaController
from modules.util.enum.DPOObjective import DPOObjective
from modules.util.enum.DPORefMode import DPORefMode

import torch

import test_mixed_rlhf as fixtures


class TinyBackupSaver(InternalModelSaverMixin):
    """Use OT's internal optimizer/progress serialization with a tiny CPU model."""

    def save(self, model, model_type, output_format, destination, dtype):
        Path(destination).mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), Path(destination) / "toy_model.pt")
        self._save_internal_data(model, destination)


class RLHFBackupTest(unittest.TestCase):
    def setUp(self):
        fixtures.MixedRLHFTest.setUp(self)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.config.workspace_dir = self.directory.name
        self.config.gradient_accumulation_steps = 1
        self.config.rolling_backup = False
        self.config.concept_file_name = ""
        self.config.sample_definition_file_name = ""
        self.config.rlhf_dpo_adaptive_beta = False
        self.trainer = self.make_trainer(self.model, self.setup)
        self.model.train_progress = self.progress

    pair_batch = fixtures.MixedRLHFTest.pair_batch

    def make_trainer(self, model, setup):
        model.train_config = self.config
        model.ema = None
        model.param_group_mapping = ["adapter"]
        params = [p for adapter in model.adapters() for p in adapter.parameters() if p.requires_grad]
        model.optimizer = torch.optim.SGD(params, lr=0.01, momentum=0.9)
        trainer = object.__new__(GenericTrainer)
        trainer.config = self.config
        trainer.model = model
        trainer.model_setup = setup
        trainer.model_saver = TinyBackupSaver()
        trainer.callbacks = Mock()
        trainer._gradient_accumulation_dirty = False
        trainer._resume_integrity_payload = None
        trainer._resume_saved_gradient_accumulation_steps = None
        trainer._dpo_beta_controller = DPOBetaController(self.config.rlhf_dpo_beta) if (
            self.config.rlhf_dpo_adaptive_beta and self.config.rlhf_dpo_objective == DPOObjective.SIGMOID
        ) else None
        trainer._dpo_best_validation = None
        trainer._dpo_validation_bad_steps = 0
        return trainer

    def gradient(self, trainer, streamed=False):
        trainer.model.optimizer.zero_grad(set_to_none=True)
        with trainer.model_setup.preference_dropout_disabled(trainer.model):
            loss = trainer.model_setup.calculate_dpo_loss(
                trainer.model, self.pair_batch(), self.config,
                trainer.model.train_progress, streamed=streamed,
            )
            loss.backward()
        params = [p for a in trainer.model.adapters() for p in a.parameters() if p.requires_grad]
        return loss.detach(), torch.cat([p.grad.flatten() for p in params])

    def backup(self):
        with patch("modules.trainer.GenericTrainer.torch_gc"):
            self.trainer._GenericTrainer__backup(self.model.train_progress, print_msg=False)
        paths = list((Path(self.directory.name) / "backup").iterdir())
        self.assertEqual(len(paths), 1)
        path = paths[0]
        self.assertTrue((path / "onetrainer_resume_integrity.json").is_file())
        self.assertFalse(list(path.glob("*.tmp")))
        return path

    def restore(self, path):
        model = fixtures.TinyModel()
        model.load_state_dict(torch.load(path / "toy_model.pt", weights_only=True))
        setup = fixtures.TinySetup()
        trainer = self.make_trainer(model, setup)
        trainer._GenericTrainer__validate_resume_backup_files(str(path))
        InternalModelLoaderMixin()._load_internal_data(model, str(path))
        model.optimizer.load_state_dict(model.optimizer_state_dict)
        trainer._GenericTrainer__validate_loaded_resume_progress(str(path))
        trainer._GenericTrainer__load_rlhf_state(str(path))
        setup.load_dpo_curriculum_state(str(path / "onetrainer_dpo_hard_pair_curriculum.json"), self.config)
        setup.initialize_dpo_reference(model, self.config, str(path / "onetrainer_dpo_reference.pt"))
        return trainer

    def test_full_backup_restores_next_loss_gradients_optimizer_and_rlhf_state(self):
        for objective in (DPOObjective.SIGMOID, DPOObjective.ANCHORED_REJECT):
            with self.subTest(objective=objective):
                self.setUp()
                self.config.rlhf_dpo_objective = objective
                self.config.rlhf_dpo_adaptive_beta = objective == DPOObjective.SIGMOID
                self.config.rlhf_dpo_hard_pair_curriculum = objective == DPOObjective.ANCHORED_REJECT
                self.trainer = self.make_trainer(self.model, self.setup)
                if self.trainer._dpo_beta_controller:
                    for i in range(30):
                        beta = self.trainer._dpo_beta_controller.update(0.01 * i)
                    self.setup.set_dpo_runtime_beta(beta)
                self.gradient(self.trainer)
                self.model.optimizer.step()
                self.setup.commit_dpo_curriculum_state()
                self.model.train_progress.global_step = 8
                self.trainer._dpo_best_validation = 0.375
                self.trainer._dpo_validation_bad_steps = 3
                saved_curriculum = copy.deepcopy(self.setup._dpo_curriculum_state)
                path = self.backup()
                restored = self.restore(path)

                self.assertEqual(restored.model.train_progress.global_step, 8)
                self.assertEqual(restored._dpo_best_validation, 0.375)
                self.assertEqual(restored._dpo_validation_bad_steps, 3)
                self.assertEqual(restored.model_setup._dpo_runtime_beta, self.setup._dpo_runtime_beta)
                self.assertEqual(restored.model_setup._dpo_curriculum_state, saved_curriculum)
                if self.trainer._dpo_beta_controller:
                    self.assertEqual(
                        restored._dpo_beta_controller.state_dict(), self.trainer._dpo_beta_controller.state_dict(),
                    )
                    self.assertEqual(
                        restored._dpo_beta_controller.update(0.8), self.trainer._dpo_beta_controller.update(0.8),
                    )
                old_state = self.model.optimizer.state_dict()
                new_state = restored.model.optimizer.state_dict()
                self.assertTrue(old_state["state"])
                self.assertEqual(old_state["param_groups"], new_state["param_groups"])
                for key in old_state["state"]:
                    torch.testing.assert_close(
                        old_state["state"][key]["momentum_buffer"], new_state["state"][key]["momentum_buffer"],
                    )
                for a, b in zip(self.setup._dpo_ref_params, restored.model_setup._dpo_ref_params, strict=True):
                    for old, new in zip(a, b, strict=True):
                        torch.testing.assert_close(old, new)
                for streamed in (False, True):
                    expected = self.gradient(self.trainer, streamed)
                    actual = self.gradient(restored, streamed)
                    torch.testing.assert_close(actual[0], expected[0])
                    torch.testing.assert_close(actual[1], expected[1])
                    self.setup.discard_dpo_curriculum_pending()
                    restored.model_setup.discard_dpo_curriculum_pending()

    def test_missing_trainer_json_keeps_saved_reference_and_resumes_training(self):
        self.config.rlhf_dpo_adaptive_beta = True
        self.trainer = self.make_trainer(self.model, self.setup)
        with torch.no_grad():
            self.params[0].add_(0.15)
        path = self.backup()
        (path / "onetrainer_rlhf_state.json").unlink()
        output = io.StringIO()
        with redirect_stdout(output):
            restored = self.restore(path)
        self.assertIn(str(path / "onetrainer_rlhf_state.json"), output.getvalue())
        self.assertIn("fresh adaptive-beta", output.getvalue())
        self.assertNotIn("new fixed reference", output.getvalue())
        self.assertIsNone(restored.model_setup._dpo_runtime_beta)
        self.assertIsNone(restored._dpo_best_validation)
        self.assertEqual(restored._dpo_beta_controller.state_dict()["_steps"], 0)
        torch.testing.assert_close(
            restored.model_setup._dpo_ref_params[0][0], self.setup._dpo_ref_params[0][0],
        )
        self.assertFalse(torch.equal(restored.model_setup._dpo_ref_params[0][0], self.params[0]))
        expected = self.gradient(self.trainer, streamed=True)
        actual = self.gradient(restored, streamed=True)
        torch.testing.assert_close(actual[0], expected[0])
        torch.testing.assert_close(actual[1], expected[1])
        restored._GenericTrainer__save_rlhf_state(str(path))
        self.assertTrue((path / "onetrainer_rlhf_state.json").is_file())

    def test_supervised_backup_can_start_rlhf_with_new_fixed_reference(self):
        self.config.rlhf_dpo_objective = DPOObjective.ANCHORED_REJECT
        self.config.rlhf_dpo_hard_pair_curriculum = False
        self.config.rlhf_enabled = False
        self.setup._dpo_ref_params = None
        path = self.backup()
        self.assertFalse((path / "onetrainer_dpo_hard_pair_curriculum.json").exists())
        self.assertFalse((path / "onetrainer_rlhf_state.json").exists())
        self.assertFalse((path / "onetrainer_dpo_reference.pt").exists())
        self.config.rlhf_enabled = True
        self.config.rlhf_dpo_hard_pair_curriculum = True
        output = io.StringIO()
        with redirect_stdout(output):
            restored = self.restore(path)
        self.assertIn("not an exact continuation", output.getvalue())
        self.assertIn("fresh per-pair", output.getvalue())
        self.assertEqual(restored.model_setup._dpo_curriculum_state, {})
        before = [p.detach().clone() for a in restored.model.adapters() for p in a.parameters()]
        refs = [p for group in restored.model_setup._dpo_ref_params for p in group]
        for a, b in zip(before, refs, strict=True):
            torch.testing.assert_close(a, b)
        loss, grads = self.gradient(restored, streamed=True)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(grads).all())
        restored.model.optimizer.step()
        for a, b in zip(before, refs, strict=True):
            torch.testing.assert_close(a, b)
        restored.model_setup.commit_dpo_curriculum_state()
        restored.model_setup.save_dpo_reference(str(path / "onetrainer_dpo_reference.pt"))
        restored.model_setup.save_dpo_curriculum_state(
            str(path / "onetrainer_dpo_hard_pair_curriculum.json"), self.config,
        )
        restored._GenericTrainer__save_rlhf_state(str(path))
        again = self.restore(path)
        self.assertEqual(again.model_setup._dpo_curriculum_state, restored.model_setup._dpo_curriculum_state)

    def test_base_reference_needs_no_snapshot_and_rlhf_disabled_needs_no_sidecar(self):
        self.config.rlhf_dpo_ref_mode = DPORefMode.NEW_ADAPTER
        self.setup._dpo_ref_params = None
        path = self.backup()
        self.assertFalse((path / "onetrainer_dpo_reference.pt").exists())
        restored = self.restore(path)
        self.assertIsNone(restored.model_setup._dpo_ref_params)
        self.assertTrue(torch.isfinite(self.gradient(restored)[0]))
        self.config.rlhf_enabled = False
        (path / "onetrainer_rlhf_state.json").write_text("invalid JSON", encoding="utf-8")
        restored._GenericTrainer__load_rlhf_state(str(path))

    def test_legacy_partial_trainer_state_does_not_require_runtime_beta(self):
        self.config.rlhf_dpo_adaptive_beta = True
        self.trainer = self.make_trainer(self.model, self.setup)
        state = self.trainer._dpo_beta_controller.state_dict()
        state["_steps"] = 4
        path = Path(self.directory.name) / "onetrainer_rlhf_state.json"
        path.write_text(json.dumps({"beta_controller": state}), encoding="utf-8")
        self.trainer._GenericTrainer__load_rlhf_state(self.directory.name)
        self.assertEqual(self.trainer._dpo_beta_controller.state_dict()["_steps"], 4)
        self.assertIsNone(self.setup._dpo_runtime_beta)
        self.assertIsNone(self.trainer._dpo_best_validation)
        self.assertEqual(self.trainer._dpo_validation_bad_steps, 0)

    def test_failed_json_write_preserves_previous_state(self):
        save = self.trainer._GenericTrainer__save_rlhf_state
        save(self.directory.name)
        path = Path(self.directory.name) / "onetrainer_rlhf_state.json"
        original = path.read_bytes()
        self.trainer._dpo_best_validation = float("nan")
        with self.assertRaises(ValueError):
            save(self.directory.name)
        self.assertEqual(path.read_bytes(), original)
        self.assertFalse(Path(str(path) + ".tmp").exists())

    def test_present_corrupt_files_and_missing_optimizer_remain_visible(self):
        path = self.backup()
        (path / "onetrainer_rlhf_state.json").write_text("{broken", encoding="utf-8")
        with self.assertRaises(json.JSONDecodeError):
            self.trainer._GenericTrainer__load_rlhf_state(str(path))
        torch.save({"adapter_parameters": []}, path / "onetrainer_dpo_reference.pt")
        with self.assertRaisesRegex(RuntimeError, "adapter count mismatch"):
            fixtures.TinySetup().initialize_dpo_reference(
                self.model, self.config, str(path / "onetrainer_dpo_reference.pt"),
            )
        (path / "optimizer" / "optimizer.pt").unlink()
        with self.assertRaisesRegex(RuntimeError, "missing optimizer state"):
            self.trainer._GenericTrainer__validate_resume_backup_files(str(path))


if __name__ == "__main__":
    unittest.main()
