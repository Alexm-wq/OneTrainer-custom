"""Focused checks for streamed DPO and actionable path diagnostics."""
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import test_mixed_rlhf as fixtures
from test_mixed_rlhf import TinySetup

from modules.dataLoader.moss.MOSSLayout import MOSSRestoreConcept
from modules.ui.BaseConceptWindowView import BaseConceptWindowView
from modules.util.config.ConceptConfig import ConceptConfig
from modules.util.enum.DPOObjective import DPOObjective
from modules.util.enum.DPORefMode import DPORefMode
from modules.util.ui import validation
from modules.util.ui.validation import BaseFieldValidator, validate_path

import torch
from torch.utils.checkpoint import checkpoint


class ReplaySetup(TinySetup):
    def __init__(self):
        super().__init__()
        self.calls = []
        self.reentrant = False

    def predict(self, model, batch, config, progress, *, deterministic=False):
        output = super().predict(model, batch, config, progress, deterministic=deterministic)
        if self.reentrant:
            latent = batch["latent_image"]
            noise = output["target"] + latent
            inputs = (0.5 * (latent + noise)).detach().requires_grad_(torch.is_grad_enabled())
            output["predicted"] = checkpoint(model.net, inputs, use_reentrant=True)
        self.calls.append({
            "size": len(batch["latent_image"]), "grad": torch.is_grad_enabled(),
            "locked": self._dpo_conditioning_locked(),
            "target": output["target"].detach().clone(),
            "timestep": output["timestep"].detach().clone(),
        })
        return output


class StreamedDPOTest(unittest.TestCase):
    def setUp(self):
        fixtures.MixedRLHFTest.setUp(self)
        self.setup = ReplaySetup()
        self.setup.initialize_dpo_reference(self.model, self.config)
        self.trainer.model_setup = self.setup
        self.model.optimizer = torch.optim.SGD(self.params, lr=0.01)
        with torch.no_grad():
            self.params[0].add_(0.013)

    pair_batch = fixtures.MixedRLHFTest.pair_batch

    def backward(self, loss):
        (loss / self.config.gradient_accumulation_steps).backward()

    def run_pair(self, streamed, scale=1):
        self.model.optimizer.zero_grad(set_to_none=True)
        self.setup.calls.clear()
        loss = self.setup.calculate_dpo_loss(
            self.model, self.pair_batch(), self.config, self.progress, streamed=streamed,
        )
        (loss * scale).backward()
        gradients = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten() for p in self.params])
        return loss.detach(), gradients, copy.deepcopy(self.setup.get_last_dpo_metrics())

    def test_objectives_references_and_loss_weights_match(self):
        for objective in DPOObjective:
            for reference in (DPORefMode.NEW_ADAPTER, DPORefMode.EXISTING_ADAPTER):
                for weights in ((1, 0.1), (0, 1), (1, 0)):
                    with self.subTest(objective=objective, reference=reference, weights=weights):
                        self.config.rlhf_dpo_objective = objective
                        self.config.rlhf_dpo_ref_mode = reference
                        self.config.rlhf_dpo_weight, self.config.rlhf_chosen_weight = weights
                        expected = self.run_pair(False, scale=8 / 3)
                        actual = self.run_pair(True, scale=8 / 3)
                        torch.testing.assert_close(actual[0], expected[0])
                        torch.testing.assert_close(actual[1], expected[1], atol=2e-5, rtol=2e-4)
                        for key in ("objective_loss", "chosen_reward", "rejected_reward", "reward_margin"):
                            self.assertAlmostEqual(actual[2][key], expected[2][key], places=5)

    def test_b_sized_forwards_and_checkpointed_backward_replay(self):
        for checkpointed, reentrant in ((False, False), (True, False), (False, True)):
            with self.subTest(checkpointed=checkpointed, reentrant=reentrant):
                self.setup.checkpointed = checkpointed
                self.setup.reentrant = reentrant
                for adapter in self.model.adapters():
                    adapter.dropout.p = 0.7
                self.run_pair(True)
                calls = self.setup.calls
                self.assertEqual(len(calls), 6)
                self.assertEqual([x["grad"] for x in calls], [False] * 4 + [True] * 2)
                self.assertTrue(all(x["size"] == 1 and x["locked"] for x in calls))
                for replay in calls[4:]:
                    matching = [x for x in calls[2:4] if torch.equal(x["target"], replay["target"])]
                    self.assertEqual(len(matching), 1)
                    torch.testing.assert_close(matching[0]["timestep"], replay["timestep"])
                self.assertTrue(all(adapter.dropout.p == 0.7 for adapter in self.model.adapters()))
                self.assertFalse(self.setup._dpo_conditioning_locked())

    def test_mixed_groups_and_existing_accumulation_buffers(self):
        self.config.gradient_accumulation_steps = 3
        self.config.rlhf_normal_weight = 2.3
        self.config.rlhf_dpo_weight = 0.7
        self.config.rlhf_chosen_weight = 0.4
        batch = self.trainer._GenericTrainer__subbatch(self.batch, [0, 1, 1])
        batch["dpo_streamed"] = [False, False, False]
        self.model.optimizer.zero_grad(set_to_none=True)
        with self.setup.preference_dropout_disabled(self.model):
            expected = self.trainer._GenericTrainer__calculate_mixed_rlhf_training_loss(batch, self.progress)
            self.backward(expected)
        gradients = [p.grad.clone() for p in self.params]
        batch["dpo_streamed"][-1] = True
        for p in self.params:
            p.grad = torch.ones_like(p)
        with self.setup.preference_dropout_disabled(self.model):
            actual = self.trainer._GenericTrainer__calculate_mixed_rlhf_training_loss(
                batch, self.progress, backward=self.backward,
            )
        torch.testing.assert_close(actual, expected.detach())
        for p, expected_grad in zip(self.params, gradients, strict=True):
            torch.testing.assert_close(p.grad, expected_grad + 1, atol=2e-5, rtol=2e-4)
        self.assertTrue(self.trainer._rlhf_backward_done)

    def test_validation_is_deterministic_and_has_no_gradients(self):
        self.setup.calls.clear()
        first = self.setup.calculate_dpo_loss(
            self.model, self.pair_batch(), self.config, self.progress,
            streamed=True, validation=True, validation_seed=1234,
        )
        self.progress.global_step += 5
        second = self.setup.calculate_dpo_loss(
            self.model, self.pair_batch(), self.config, self.progress,
            streamed=True, validation=True, validation_seed=1234,
        )
        torch.testing.assert_close(first, second)
        self.assertFalse(first.requires_grad)
        self.assertEqual(len(self.setup.calls), 8)
        self.assertTrue(all(x["size"] == 1 and not x["grad"] for x in self.setup.calls))
        self.assertTrue(all(p.grad is None for p in self.params))

    def test_multi_gpu_vjp_path_matches_batched_gradients(self):
        expected = self.run_pair(False)
        with patch("modules.util.multi_gpu_util.world_size", return_value=2):
            actual = self.run_pair(True)
        torch.testing.assert_close(actual[1], expected[1], atol=2e-5, rtol=2e-4)

    def test_replay_preserves_rng_state(self):
        loss = self.setup.calculate_dpo_loss(
            self.model, self.pair_batch(), self.config, self.progress, streamed=True,
        )
        state = torch.get_rng_state().clone()
        loss.backward()
        torch.testing.assert_close(torch.get_rng_state(), state)

    def test_stream_flag_roundtrip_live_cached_concept_and_ui(self):
        concept = ConceptConfig.default_values()
        self.assertFalse(concept.dpo_streamed)
        concept.dpo_streamed = True
        restored = ConceptConfig.default_values().from_dict(concept.to_dict())
        self.assertTrue(restored.dpo_streamed)
        inventory = SimpleNamespace(by_source={"image": {"concept": restored.to_dict()}})
        module = MOSSRestoreConcept(inventory)
        module._get_previous_item = lambda *_args: "image"
        self.assertTrue(module.get_item(0, 0)["concept"]["dpo_streamed"])
        inventory.by_source["image"]["concept"]["dpo_streamed"] = False
        self.assertFalse(module.get_item(0, 0)["concept"]["dpo_streamed"])
        components = Mock()
        view = BaseConceptWindowView(components)
        controller = SimpleNamespace(concept=concept, train_config=self.config, download_dataset_threaded=lambda: None)
        view.build_general_tab(None, controller, Mock(), Mock())
        self.assertTrue(any(call.args[-1] == "dpo_streamed" for call in components.switch.call_args_list))

    def test_shared_conditioning_and_latents_are_not_copied(self):
        batch = self.pair_batch()
        batch["tokens"] = torch.ones(1, 4)
        batch["prompt"] = ['{"summary":{"caption":"Mira"},"items":[]}']
        chosen, rejected, count = self.setup._create_dpo_stream_batches(batch)
        self.assertEqual(count, 1)
        self.assertIs(chosen["latent_image"], batch["latent_image"])
        self.assertIs(rejected["latent_image"], batch["latent_image_rejected"])
        for key in ("tokens", "prompt"):
            self.assertIs(chosen[key], rejected[key])


class DiagnosticValidator(BaseFieldValidator):
    def __init__(self, obj, value):
        state = SimpleNamespace(obj=obj, get_field_metadata=lambda _: SimpleNamespace(type=str, nullable=False, default=""))
        super().__init__(state, "path", extra_validate=validate_path)
        self.value = value

    def _apply_error(self):
        pass

    def _clear_error(self):
        pass

    def flush(self):
        return self.validate(self.value)


class PathDiagnosticTest(unittest.TestCase):
    def test_error_identifies_concept_and_actual_uncommitted_path(self):
        concept = ConceptConfig.default_values()
        concept.name = "Character pairs"
        concept.path = "/stale/value"
        with tempfile.TemporaryDirectory() as root:
            missing = str(Path(root) / "missing")
            validator = DiagnosticValidator(concept, missing)
            with patch.object(validation, "_active_validators", {validator}):
                errors = validation.flush_and_validate_all()
            self.assertEqual(len(errors), 1)
            self.assertIn("Character pairs", errors[0])
            self.assertIn(missing, errors[0])
            self.assertNotIn("/stale/value", errors[0])

    def test_existing_path_passes_and_relative_error_shows_resolution(self):
        with tempfile.TemporaryDirectory() as root:
            validator = DiagnosticValidator(ConceptConfig.default_values(), root)
            self.assertIsNone(validator.flush())
        self.assertIn(str(Path("missing-concept-92fd65").resolve()), validate_path("missing-concept-92fd65"))


if __name__ == "__main__":
    unittest.main()
