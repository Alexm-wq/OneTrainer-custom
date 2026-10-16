"""CPU regression tests for actual mixed trainer, loss, adapter and holdout paths."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from modules.dataLoader.dpo.FilterDPOChosenPaths import FilterDPOChosenPaths
from modules.dataLoader.mixin.DataLoaderMgdsMixin import DataLoaderMgdsMixin
from modules.modelSetup.BaseFlux2Setup import BaseFlux2Setup
from modules.modelSetup.BaseFluxSetup import BaseFluxSetup
from modules.modelSetup.BaseKrea2Setup import BaseKrea2Setup
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.modelSetup.mixin.ModelSetupDiffusionLossMixin import ModelSetupDiffusionLossMixin
from modules.module.LoRAModule import LoRAModule
from modules.trainer.GenericTrainer import GenericTrainer
from modules.ui.BaseRLHFTabView import BaseRLHFTabView
from modules.util.commands.TrainCommands import TrainCommands
from modules.util.config.ConceptConfig import ConceptConfig
from modules.util.config.TrainConfig import TrainConfig
from modules.util.dpo_beta_controller import DPOBetaController
from modules.util.enum.ConceptType import ConceptType
from modules.util.enum.DPOObjective import DPOObjective
from modules.util.enum.DPORefMode import DPORefMode
from modules.util.enum.LossScaler import LossScaler
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.rlhf_util import is_validation_pair, preference_pair_key, validate_rlhf_config
from modules.util.TrainProgress import TrainProgress

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, 4), nn.Tanh(), nn.Linear(4, 4))
        self.net.requires_grad_(False)
        self.loras = nn.ModuleList()
        for index in (0, 2):
            adapter = LoRAModule(str(index), self.net[index], rank=2, alpha=2)
            nn.init.normal_(adapter.lora_up.weight, std=0.1)
            adapter.hook_to_module()
            self.loras.append(adapter)
        self.noise_scheduler = SimpleNamespace(
            config={"num_train_timesteps": 1000}, sigmas=torch.linspace(1, 0, 1000),
        )

    def adapters(self):
        return list(self.loras)


class TinySetup(BaseModelSetup, ModelSetupDiffusionLossMixin):
    def __init__(self):
        BaseModelSetup.__init__(self, torch.device("cpu"), torch.device("cpu"), False)
        ModelSetupDiffusionLossMixin.__init__(self)
        self.checkpointed = False

    def create_parameters(self, *args):
        pass

    def setup_optimizations(self, *args):
        pass

    def setup_model(self, *args):
        pass

    def setup_train_device(self, model, config):
        model.train()

    def after_optimizer_step(self, *args):
        pass

    def predict(self, model, batch, config, train_progress, *, deterministic=False):
        latent = batch["latent_image"]
        seed = self._dpo_validation_seed
        if seed is None:
            seed = 0 if deterministic else train_progress.global_step
        generator = torch.Generator().manual_seed(seed)
        count = self._dpo_paired_half or len(latent)
        noise = torch.randn((count, *latent.shape[1:]), generator=generator)
        timestep = torch.randint(1, 1000, (count,), generator=generator)
        if self._dpo_paired_half:
            noise = torch.cat([noise, noise])
            timestep = torch.cat([timestep, timestep])
        inputs = 0.5 * (latent + noise)
        prediction = checkpoint(model.net, inputs, use_reentrant=False) if self.checkpointed else model.net(inputs)
        return {"predicted": prediction, "target": noise - latent, "loss_type": "target", "timestep": timestep}

    rlhf_logp_per_sample = BaseKrea2Setup.rlhf_logp_per_sample

    def calculate_loss(self, model, batch, data, config):
        return self._flow_matching_losses(batch, data, config, self.train_device).mean()

    def _write_dpo_pair_csv_log(self, **kwargs):
        pass

    def _write_dpo_bad_pair_csv_log(self, **kwargs):
        pass


class MixedRLHFTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(21)
        self.config = TrainConfig.default_values()
        self.config.model_type = ModelType.KREA_2
        self.config.training_method = TrainingMethod.LORA
        self.config.rlhf_enabled = True
        self.config.rlhf_dpo_ref_mode = DPORefMode.EXISTING_ADAPTER
        self.config.loss_scaler = LossScaler.NONE
        self.model = TinyModel()
        self.setup = TinySetup()
        self.setup.initialize_dpo_reference(self.model, self.config)
        self.progress = TrainProgress()
        self.batch = {
            "latent_image": torch.randn(2, 3, 4),
            "latent_image_rejected": torch.randn(2, 3, 4),
            "loss_weight": torch.ones(2),
            "dpo_is_paired": torch.tensor([False, True]),
            "concept_type": [ConceptType.STANDARD, ConceptType.STANDARD],
            "dpo_pair_key": ["", preference_pair_key("/concept/a", "001")],
        }
        self.trainer = object.__new__(GenericTrainer)
        self.trainer.model = self.model
        self.trainer.model_setup = self.setup
        self.trainer.config = self.config
        self.trainer.train_device = torch.device("cpu")
        self.trainer._dpo_metric_sums = {}
        self.trainer._dpo_metric_weight = 0.0
        self.trainer._dpo_timestep_sums = {}
        self.params = [p for adapter in self.model.adapters() for p in adapter.parameters() if p.requires_grad]

    def mixed_loss(self):
        return self.trainer._GenericTrainer__calculate_mixed_rlhf_training_loss(self.batch, self.progress)

    def pair_batch(self):
        return self.trainer._GenericTrainer__subbatch(self.batch, [1])

    def gradient(self, weights):
        self.config.rlhf_normal_weight, self.config.rlhf_dpo_weight, self.config.rlhf_chosen_weight = weights
        with self.setup.preference_dropout_disabled(self.model):
            loss = self.mixed_loss()
            grads = torch.autograd.grad(loss, self.params)
        return loss.detach(), torch.cat([g.flatten() for g in grads])

    def test_mixed_snapshot_backward_with_and_without_checkpointing(self):
        for checkpointed in (False, True):
            with self.subTest(checkpointed=checkpointed):
                self.setup.checkpointed = checkpointed
                for adapter in self.model.adapters():
                    adapter.dropout.p = 0.75
                _, grad = self.gradient((1.0, 1.0, 0.1))
                self.assertTrue(torch.isfinite(grad).all())
                self.assertGreater(grad.abs().sum().item(), 0)
                self.assertTrue(all(adapter.dropout.p == 0.75 for adapter in self.model.adapters()))
                for adapter, snapshot in zip(self.model.adapters(), self.setup._dpo_ref_params, strict=True):
                    for parameter, reference in zip(adapter.parameters(), snapshot, strict=True):
                        torch.testing.assert_close(parameter, reference)

    def test_three_weights_are_independent_in_values_and_gradients(self):
        parts = [self.gradient(weights) for weights in ((1, 0, 0), (0, 1, 0), (0, 0, 1))]
        actual_loss, actual_grad = self.gradient((2, 3, 4))
        torch.testing.assert_close(actual_loss, sum(c * part[0] for c, part in zip((2, 3, 4), parts, strict=True)))
        torch.testing.assert_close(actual_grad, sum(c * part[1] for c, part in zip((2, 3, 4), parts, strict=True)))

    def test_normal_only_batch_uses_normal_weight(self):
        self.batch["dpo_is_paired"].zero_()
        first, grad = self.gradient((1, 0, 0))
        second, grad2 = self.gradient((3, 7, 8))
        torch.testing.assert_close(second, 3 * first)
        torch.testing.assert_close(grad2, 3 * grad)

    def test_concept_weight_and_loss_scaler_do_not_change_reward_margin(self):
        with torch.no_grad():
            self.model.loras[0].lora_up.weight.add_(0.1)
        for objective in DPOObjective:
            with self.subTest(objective=objective):
                self.config.rlhf_dpo_objective = objective
                self.config.rlhf_chosen_weight = 0
                self.config.rlhf_dpo_weight = 1
                self.config.rlhf_dpo_chosen_reward_anchor = True
                self.config.rlhf_dpo_chosen_reward_anchor_weight = 0.1
                batch = self.pair_batch()
                self.config.loss_scaler = LossScaler.NONE
                first = self.setup.calculate_dpo_loss(self.model, batch, self.config, self.progress)
                g1 = torch.autograd.grad(first, self.params)
                before = self.setup.get_last_dpo_metrics()["reward_margin"]
                batch["loss_weight"] = torch.tensor([3.0])
                self.config.loss_scaler = LossScaler.BOTH
                self.config.batch_size, self.config.gradient_accumulation_steps = 2, 4
                second = self.setup.calculate_dpo_loss(self.model, batch, self.config, self.progress)
                g2 = torch.autograd.grad(second, self.params)
                self.assertEqual(before, self.setup.get_last_dpo_metrics()["reward_margin"])
                torch.testing.assert_close(second, first.detach() * 24)
                for a, b in zip(g1, g2, strict=True):
                    torch.testing.assert_close(b, a * 24)

    def test_all_native_flow_scores_ignore_sample_scaling(self):
        data = self.setup.predict(self.model, self.pair_batch(), self.config, self.progress)
        batch = self.pair_batch()
        for score_function in (BaseKrea2Setup.rlhf_logp_per_sample,
                               BaseFluxSetup.rlhf_logp_per_sample, BaseFlux2Setup.rlhf_logp_per_sample):
            self.config.loss_scaler = LossScaler.NONE
            batch["loss_weight"] = torch.ones(1)
            score = score_function(self.setup, self.model, batch, data, self.config)
            batch["loss_weight"] = torch.tensor([17.0])
            self.config.loss_scaler = LossScaler.BOTH
            self.config.batch_size, self.config.gradient_accumulation_steps = 7, 5
            torch.testing.assert_close(score_function(self.setup, self.model, batch, data, self.config), score)

    def test_dropout_does_not_create_reward_for_unchanged_policy(self):
        for adapter in self.model.adapters():
            adapter.dropout.p = 0.9
        for _ in range(3):
            self.setup.calculate_dpo_loss(self.model, self.pair_batch(), self.config, self.progress)
            self.assertEqual(self.setup.get_last_dpo_metrics()["chosen_reward"], 0.0)
            self.assertEqual(self.setup.get_last_dpo_metrics()["rejected_reward"], 0.0)

    def test_validation_is_repeatable_and_does_not_stage_curriculum_or_log_training(self):
        self.config.rlhf_dpo_objective = DPOObjective.ANCHORED_REJECT
        self.config.rlhf_dpo_hard_pair_curriculum = True
        with torch.no_grad():
            self.model.loras[0].lora_up.weight.add_(0.03)
        self.setup._write_dpo_pair_csv_log = Mock()
        self.setup._write_dpo_bad_pair_csv_log = Mock()
        first = self.setup.calculate_dpo_loss(self.model, self.pair_batch(), self.config, self.progress,
                                              validation=True, validation_seed=77)
        metrics = copy.deepcopy(self.setup.get_last_dpo_metrics())
        self.progress.global_step = 999
        self.setup.set_dpo_runtime_beta(900)
        second = self.setup.calculate_dpo_loss(self.model, self.pair_batch(), self.config, self.progress,
                                               validation=True, validation_seed=77)
        torch.testing.assert_close(first, second)
        self.assertEqual(metrics, self.setup.get_last_dpo_metrics())
        self.assertFalse(second.requires_grad)
        self.assertEqual(self.setup._dpo_curriculum_pending, {})
        self.setup._write_dpo_pair_csv_log.assert_not_called()
        self.setup._write_dpo_bad_pair_csv_log.assert_not_called()

    def test_curriculum_does_not_reduce_chosen_or_order_corrections(self):
        self.config.rlhf_dpo_objective = DPOObjective.ANCHORED_REJECT
        self.config.rlhf_chosen_weight = 0
        self.config.rlhf_dpo_anchored_rejected_weight = 0
        self.config.rlhf_dpo_anchored_chosen_target = 0.1
        self.config.rlhf_dpo_hard_pair_curriculum = False
        first = self.setup.calculate_dpo_loss(self.model, self.pair_batch(), self.config, self.progress)
        first_grad = torch.autograd.grad(first, self.params)
        self.config.rlhf_dpo_hard_pair_curriculum = True
        self.config.rlhf_dpo_hard_pair_curriculum_min_weight = 0.1
        second = self.setup.calculate_dpo_loss(self.model, self.pair_batch(), self.config, self.progress)
        second_grad = torch.autograd.grad(second, self.params)
        torch.testing.assert_close(first, second)
        for a, b in zip(first_grad, second_grad, strict=True):
            torch.testing.assert_close(a, b)

    def test_pair_ids_namespace_curriculum_state(self):
        keys = [preference_pair_key("/dataset/a", "001"), preference_pair_key("/dataset/b", "001")]
        self.assertNotEqual(*keys)
        self.setup._stage_dpo_curriculum_observations(
            {"dpo_pair_key": keys}, self.config, torch.tensor([-0.1, 0.2]),
        )
        self.assertEqual(len(self.setup._dpo_curriculum_pending), 2)

    def test_adaptive_beta_updates_only_on_flush_and_restores(self):
        self.trainer._dpo_beta_controller = DPOBetaController(300, warmup_steps=0)
        self.setup.set_dpo_runtime_beta(None)
        self.trainer._GenericTrainer__accumulate_dpo_metrics({"reward_margin": 0.1}, 2)
        self.trainer._GenericTrainer__accumulate_dpo_metrics({"reward_margin": 0.4}, 1)
        self.assertIsNone(self.setup._dpo_runtime_beta)
        self.trainer._GenericTrainer__flush_dpo_tensorboard_metrics(1)
        self.assertAlmostEqual(self.trainer._dpo_beta_controller.margin_ema, 0.2)
        self.assertIsNotNone(self.setup._dpo_runtime_beta)
        restored = DPOBetaController(300, warmup_steps=0)
        restored.load_state_dict(self.trainer._dpo_beta_controller.state_dict())
        self.assertEqual(restored.update(0.8), self.trainer._dpo_beta_controller.update(0.8))

    def test_validation_best_and_patience_are_connected(self):
        self.config.rlhf_dpo_save_best = True
        self.config.rlhf_dpo_patience_enabled = True
        self.config.rlhf_dpo_patience_value = 2
        self.trainer._dpo_best_validation = None
        self.trainer._dpo_validation_bad_steps = 0
        self.trainer._dpo_early_stop = False
        self.trainer.model_saver = Mock()
        with tempfile.TemporaryDirectory() as directory:
            self.config.workspace_dir = directory
            record = self.trainer._GenericTrainer__record_dpo_validation
            record({"objective_loss": 0.7}, self.progress)
            record({"objective_loss": 0.6}, self.progress)
            record({"objective_loss": 0.65}, self.progress)
            self.assertFalse(self.trainer._dpo_early_stop)
            record({"objective_loss": 0.65}, self.progress)
            self.assertTrue(self.trainer._dpo_early_stop)
            self.assertEqual(self.trainer.model_saver.save.call_count, 2)
            metadata = next((Path(directory) / "save").glob("*.metrics.json"))
            self.assertEqual(json.loads(metadata.read_text())["objective_loss"], 0.6)

    def test_validation_loop_evaluates_pairs_and_reuses_dataset(self):
        self.config.rlhf_dpo_validation = True
        self.config.rlhf_dpo_validation_samples = 2
        self.trainer._dpo_validation_started = False
        self.trainer._dpo_best_validation = None
        self.trainer._dpo_validation_bad_steps = 0
        self.trainer._dpo_early_stop = False
        self.trainer.callbacks = Mock()
        self.trainer.tensorboard = Mock()
        self.trainer.validation_data_loader = Mock()
        dataset = self.trainer.validation_data_loader.get_data_set.return_value
        dataset.approximate_length.return_value = 1
        self.trainer.validation_data_loader.get_data_loader.return_value = [self.pair_batch()]
        self.trainer._GenericTrainer__needs_validate = lambda progress: True
        self.trainer._GenericTrainer__needs_gc = lambda progress: False
        self.trainer._GenericTrainer__validate(self.progress)
        score = self.trainer._dpo_best_validation
        self.assertIsNotNone(score)
        self.progress.global_step = 100
        self.trainer._GenericTrainer__validate(self.progress)
        self.assertEqual(self.trainer._dpo_best_validation, score)
        dataset.start_next_epoch.assert_called_once()
        self.assertTrue(self.model.training)
        names = [call.args[0] for call in self.trainer.tensorboard.add_scalar.call_args_list]
        self.assertIn("dpo/validation/accuracy", names)
        self.assertIn("dpo/validation/chosen_error", names)

    def test_config_migration_and_recommended_defaults(self):
        fresh = TrainConfig.default_values()
        self.assertEqual((fresh.rlhf_normal_weight, fresh.rlhf_dpo_weight, fresh.rlhf_chosen_weight), (1, 1, 0.1))
        self.assertTrue(fresh.rlhf_dpo_beta_gradient_decouple)
        self.assertEqual(fresh.rlhf_dpo_beta_gradient_reference, 1)
        old = TrainConfig.default_values().from_dict({"__version": 11, "rlhf_supervised_mix": 0.37})
        self.assertEqual(old.rlhf_chosen_weight, 0.37)
        self.assertFalse(old.rlhf_dpo_beta_gradient_decouple)
        restored = TrainConfig.default_values().from_dict(fresh.to_dict())
        self.assertEqual(restored.to_dict(), fresh.to_dict())
        validate_rlhf_config(self.config)

    def test_ui_binds_three_weights(self):
        components = Mock()
        components.section_frame.return_value = object()
        BaseRLHFTabView(components).build_content(object(), Mock(), object())
        fields = [call.args[-1] for call in components.entry.call_args_list]
        for field in ("rlhf_normal_weight", "rlhf_dpo_weight", "rlhf_chosen_weight"):
            self.assertIn(field, fields)
        self.assertNotIn("rlhf_supervised_mix", fields)


class HoldoutTest(unittest.TestCase):
    def filter_rows(self, split):
        concept = {"path": "/dataset", "dpo_chosen_pattern": "chosen/{}.png",
                   "dpo_rejected_pattern": "rejected/{}.png", "include_subdirectories": True,
                   "_dpo_split": split, "_dpo_validation_percentage": 20, "_dpo_validation_seed": 12}
        rows = [f"/dataset/chosen/{i}.png" for i in range(100)] + ["/dataset/rejected/0.png"]
        module = FilterDPOChosenPaths()
        module._get_previous_length = lambda name: len(rows)
        module._get_previous_item = lambda variation, name, index: rows[index] if name == "image_path" else concept
        module.start(0)
        return {module.get_item(0, i)["image_path"] for i in range(module.length())}

    def test_real_filter_produces_disjoint_complete_split(self):
        training, validation = self.filter_rows("train"), self.filter_rows("validation")
        self.assertFalse(training & validation)
        self.assertEqual(len(training | validation), 100)
        self.assertTrue(training)
        self.assertTrue(validation)
        self.assertEqual(validation, self.filter_rows("validation"))

    def test_holdout_is_stable_across_repetitions_and_resolutions(self):
        key = preference_pair_key("/dataset", "001")
        decisions = [is_validation_pair(key, 20, 12) for _ in (512, 768, 1024, 512)]
        self.assertEqual(len(set(decisions)), 1)
        self.assertFalse(is_validation_pair(key, 0, 12))

    def test_validation_loader_includes_heldout_training_concepts(self):
        config = TrainConfig.default_values()
        config.train_device = "cpu"
        config.rlhf_enabled = config.rlhf_dpo_validation = True
        config.rlhf_dpo_validation_percentage = 10
        normal = ConceptConfig.default_values()
        paired = ConceptConfig.default_values()
        paired.path = "/paired"
        paired.dpo_chosen_pattern, paired.dpo_rejected_pattern = "chosen/{}.png", "rejected/{}.png"
        explicit = ConceptConfig.default_values()
        explicit.type = ConceptType.VALIDATION
        config.concepts = [normal, paired, explicit]
        loader = DataLoaderMgdsMixin()
        with patch("modules.dataLoader.mixin.DataLoaderMgdsMixin.MGDS") as mgds:
            loader._create_mgds(config, [], TrainProgress(), is_validation=True)
            concepts = mgds.call_args.args[1]
            self.assertEqual(len(concepts), 2)
            self.assertEqual(concepts[0]["_dpo_split"], "validation")
            self.assertFalse(concepts[0]["image"]["enable_crop_jitter"])
            self.assertEqual(concepts[1]["_dpo_split"], "")
            loader._create_mgds(config, [], TrainProgress(), is_validation=False)
            concepts = mgds.call_args.args[1]
            self.assertEqual(len(concepts), 2)
            self.assertEqual(concepts[1]["_dpo_split"], "train")


if __name__ == "__main__":
    unittest.main()
