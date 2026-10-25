"""Self-Flow for MOSS-Image. CPU tests with randomly initialized tiny models; no weights are downloaded."""
import tempfile
import unittest
from unittest.mock import Mock, patch

import modules.module.moss.self_flow as moss_self_flow
from modules.modelLoader.mixin.InternalModelLoaderMixin import InternalModelLoaderMixin
from modules.modelSampler.MOSSImageSampler import MOSSImageSampler
from modules.modelSaver.mixin.InternalModelSaverMixin import InternalModelSaverMixin
from modules.module.moss.self_flow import moss_self_flow_forward
from modules.module.SelfFlow import (
    SelfFlowEMA,
    resolve_self_flow_layers,
    stratified_token_indices,
    structural_alignment_loss,
)
from modules.trainer.GenericTrainer import GenericTrainer
from modules.util import create
from modules.util.config.SampleConfig import SampleConfig
from modules.util.enum.DPORefMode import DPORefMode
from modules.util.enum.LossWeight import LossWeight
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.TrainProgress import TrainProgress

import torch
from torch import nn

from test_moss_image import tiny_model, train_config


class SelfFlowEMATest(unittest.TestCase):
    def test_update_swap_and_state_roundtrip(self):
        parameter = nn.Parameter(torch.tensor([1.0, 2.0]))
        ema = SelfFlowEMA([parameter], decay=0.5)
        with torch.no_grad():
            parameter.add_(2.0)
        ema.update()
        torch.testing.assert_close(ema.ema_parameters[0], torch.tensor([2.0, 3.0]))

        # A weight change after the last update (e.g. a DPO reference setup) must survive the swap.
        with torch.no_grad():
            parameter.add_(1.0)
        version = parameter._version
        module = nn.Dropout()
        with ema.ema_weights([module]):
            torch.testing.assert_close(parameter.detach(), torch.tensor([2.0, 3.0]))
            self.assertFalse(module.training)
        torch.testing.assert_close(parameter.detach(), torch.tensor([4.0, 5.0]), rtol=0, atol=0)
        self.assertTrue(module.training)
        # Live autograd graphs stay valid: the swap does not bump the version counter.
        self.assertEqual(parameter._version, version)

        restored = SelfFlowEMA([parameter], decay=0.9, state_dict=ema.state_dict())
        torch.testing.assert_close(restored.ema_parameters[0], ema.ema_parameters[0])
        self.assertEqual(restored.optimization_steps, 1)
        self.assertEqual(restored.decay, 0.9)

    def test_update_refused_during_swap(self):
        parameter = nn.Parameter(torch.ones(1))
        ema = SelfFlowEMA([parameter], decay=0.5)
        with ema.ema_weights(), self.assertRaisesRegex(RuntimeError, "swapped"):
            ema.update()

    def test_layers_and_structural_loss(self):
        self.assertEqual(resolve_self_flow_layers(-1, -1, 42), (14, 34))
        with self.assertRaisesRegex(ValueError, "earlier"):
            resolve_self_flow_layers(3, 2, 42)
        indices = stratified_token_indices(100, 10, "cpu", torch.Generator().manual_seed(1))
        self.assertEqual(torch.div(indices, 10, rounding_mode="floor").tolist(), list(range(10)))
        feature = torch.randn(2, 16, 8)
        torch.testing.assert_close(structural_alignment_loss(feature, feature, 8), torch.zeros(2))
        self.assertTrue((structural_alignment_loss(feature, torch.randn(2, 16, 8), 8) > 0).all())


class MOSSSelfFlowForwardTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)
        self.transformer = tiny_model().transformer
        self.x = torch.randn(2, 128, 2, 3)
        self.text = torch.randn(2, 4, 8)
        self.mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])

    def test_block_per_token_modulation_matches_per_sample(self):
        block = self.transformer.layers[0]
        x, rope = torch.randn(2, 9, 32), torch.randn(1, 9, 8)
        modulation = torch.randn(2, 6, 32)
        expected = block(x, rope, modulation)
        actual = block(x, rope, modulation[:, None].expand(-1, 9, -1, -1))
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)

    def test_homogeneous_and_forced_per_token_paths_match_native(self):
        time = torch.tensor([20.0, 700.0])
        native = self.transformer(self.x, time, self.text, self.mask).sample
        fast = moss_self_flow_forward(self.transformer, self.x, time, time, self.text, self.mask).sample
        torch.testing.assert_close(fast, native, rtol=0, atol=1e-6)
        with patch.object(moss_self_flow, "_is_homogeneous", return_value=False):
            per_token = moss_self_flow_forward(self.transformer, self.x, time, time, self.text, self.mask).sample
        torch.testing.assert_close(per_token, native, rtol=1e-5, atol=1e-5)

    def test_token_timestep_is_local_and_feature_taps_agree(self):
        base = torch.tensor([300.0, 300.0])
        image = base[:, None].repeat(1, 6)
        changed = image.clone()
        changed[:, 0] = 900.0
        reference = moss_self_flow_forward(self.transformer, self.x, image, base, self.text, self.mask)
        other = moss_self_flow_forward(self.transformer, self.x, changed, base, self.text, self.mask)
        self.assertFalse(torch.allclose(reference.sample, other.sample))

        captured = moss_self_flow_forward(self.transformer, self.x, image, base, self.text, self.mask, capture_layer=1)
        stopped = moss_self_flow_forward(self.transformer, self.x, image, base, self.text, self.mask, stop_at_layer=1)
        self.assertIsNone(stopped.sample)
        self.assertEqual(captured.feature.shape, (2, 6, 32))
        torch.testing.assert_close(captured.feature, stopped.feature)
        torch.testing.assert_close(captured.sample, reference.sample)


class MOSSSelfFlowTrainingTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(1)
        self.model = tiny_model()
        self.config = train_config()
        self.config.self_flow_enabled = True
        self.config.self_flow_structural_enabled = True
        self.config.self_flow_structural_tokens = 4
        self.config.self_flow_ema_decay = 0.5
        self.config.self_flow_mask_ratio = 0.5
        self.model.train_config = self.config
        self.setup = create.create_model_setup(
            ModelType.MOSS_IMAGE, torch.device("cpu"), torch.device("cpu"), TrainingMethod.LORA,
        )

    def batch(self):
        return {
            "latent_image": torch.randn(2, 32, 4, 4),
            "latent_image_rejected": torch.randn(2, 32, 4, 4),
            "tokens": torch.tensor([[1, 2, 0, 0], [1, 2, 3, 0]]),
            "tokens_mask": torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]]),
            "text_encoder_hidden_state": torch.randn(2, 4, 8),
            "loss_weight": torch.ones(2),
            "dpo_is_paired": torch.tensor([False, True]),
            "concept_type": ["STANDARD"] * 2,
            "dpo_pair_key": ["", "moss/pair"],
        }

    def prepare(self):
        self.setup.setup_optimizations(self.model, self.config)
        self.setup.setup_model(self.model, self.config)
        self.setup.setup_train_device(self.model, self.config)

    def test_setup_creates_projector_group_and_ema(self):
        self.prepare()
        self.assertEqual((self.model.self_flow_student_layer, self.model.self_flow_teacher_layer), (1, 2))
        self.assertIn("self_flow_projector", self.model.parameters.unique_name_mapping)
        self.assertEqual(len(self.model.self_flow_ema.parameters), len(self.model.transformer_lora.parameters()))
        self.assertTrue(all(p.requires_grad for p in self.model.self_flow_projector.parameters()))

    def test_dual_timestep_forward_backward_and_ema_step(self):
        self.prepare()
        batch = self.batch()
        lora = self.model.transformer_lora.parameters()
        # Make the EMA teacher differ from the student; it must be active for the teacher forward only.
        with torch.no_grad():
            for ema in self.model.self_flow_ema.ema_parameters:
                ema.add_(0.25)
        student = [p.detach().clone() for p in lora]
        teacher_weights = []
        original = moss_self_flow.moss_self_flow_forward

        def spy(transformer, hidden_states, image_timestep, text_timestep, *args, **kwargs):
            if kwargs.get("stop_at_layer") is not None:
                teacher_weights.append([p.detach().clone() for p in lora])
                self.assertEqual(image_timestep.ndim, 1)
            else:
                self.assertEqual(image_timestep.shape, (2, 4))
            return original(transformer, hidden_states, image_timestep, text_timestep, *args, **kwargs)

        indices = iter((torch.tensor([100, 900]), torch.tensor([600, 50])))
        noise = torch.randn(2, 128, 2, 2)
        with patch("modules.modelSetup.BaseMOSSImageSetup.moss_self_flow_forward", side_effect=spy), \
                patch.object(self.setup, "_get_timestep_discrete", side_effect=lambda *a, **k: next(indices)), \
                patch.object(self.setup, "_create_noise", return_value=noise):
            output = self.setup.predict(self.model, batch, self.config, TrainProgress())

        for active, ema in zip(teacher_weights[0], self.model.self_flow_ema.ema_parameters, strict=True):
            torch.testing.assert_close(active, ema)
        for active, before in zip(lora, student, strict=True):
            torch.testing.assert_close(active.detach(), before, rtol=0, atol=0)

        token = output["self_flow_token_timestep"]
        self.assertTrue(set(token[0].tolist()) <= {100, 600} and set(token[1].tolist()) <= {900, 50})
        torch.testing.assert_close(output["timestep"], torch.tensor([100, 900]))
        self.assertEqual(output["self_flow_representation_loss"].shape, (2,))
        self.assertEqual(output["self_flow_structural_loss"].shape, (2,))

        loss = self.setup.calculate_loss(self.model, batch, output, self.config)
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in lora))
        self.assertTrue(all(p.grad is not None for p in self.model.self_flow_projector.parameters()))

        before = [p.clone() for p in self.model.self_flow_ema.ema_parameters]
        self.model.optimizer.step()
        self.setup.after_optimizer_step(self.model, self.config, TrainProgress())
        for ema, old, parameter in zip(self.model.self_flow_ema.ema_parameters, before, lora, strict=True):
            torch.testing.assert_close(ema, 0.5 * old + 0.5 * parameter.detach())

        tensorboard = Mock()
        self.model.train_progress = TrainProgress()
        self.setup._report_self_flow(self.model, self.config, tensorboard)
        names = {call.args[0] for call in tensorboard.add_scalar.call_args_list}
        self.assertTrue({"loss/self_flow_rep", "loss/self_flow_structural", "self_flow/cosine_similarity",
                         "performance/self_flow_teacher_ms"} <= names)

    def test_sigma_weighting_is_per_token(self):
        self.prepare()
        self.config.loss_weight_fn = LossWeight.SIGMA
        output = self.setup.predict(self.model, self.batch(), self.config, TrainProgress())
        weight = output["element_loss_weight"]
        self.assertEqual(weight.shape, (2, 1, 4, 4))
        expected = (output["self_flow_token_timestep"].float() + 1) / 1000
        torch.testing.assert_close(weight[:, 0, ::2, ::2].flatten(1), expected)

    def test_validation_dpo_and_no_grad_use_native_forward(self):
        self.prepare()
        batch = self.batch()
        with patch("modules.modelSetup.BaseMOSSImageSetup.moss_self_flow_forward") as forward:
            output = self.setup.predict(self.model, batch, self.config, TrainProgress(), deterministic=True)
            with torch.no_grad():
                self.setup.predict(self.model, batch, self.config, TrainProgress())
            with self.setup._dpo_stream_predict_context():
                self.setup.predict(self.model, batch, self.config, TrainProgress())
        forward.assert_not_called()
        self.assertNotIn("self_flow_representation_loss", output)

    def test_mixed_rlhf_normal_rows_use_self_flow_with_finite_gradients(self):
        self.prepare()
        self.config.rlhf_enabled = True
        self.config.rlhf_dpo_ref_mode = DPORefMode.NEW_ADAPTER
        self.setup.initialize_dpo_reference(self.model, self.config)
        trainer = object.__new__(GenericTrainer)
        trainer.model, trainer.model_setup, trainer.config = self.model, self.setup, self.config
        trainer.train_device = torch.device("cpu")
        trainer._dpo_metric_sums, trainer._dpo_timestep_sums = {}, {}
        trainer._dpo_metric_weight = 0.0
        self.setup._write_dpo_pair_csv_log = Mock()
        self.setup._write_dpo_bad_pair_csv_log = Mock()
        with patch("modules.modelSetup.BaseMOSSImageSetup.moss_self_flow_forward",
                   wraps=moss_self_flow_forward) as forward, self.setup.preference_dropout_disabled(self.model):
            loss = trainer._GenericTrainer__calculate_mixed_rlhf_training_loss(self.batch(), TrainProgress())
            loss.backward()
        self.assertEqual(forward.call_count, 2)  # teacher and student of the single normal row
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in self.model.transformer_lora.parameters()))

    def test_backup_roundtrip_restores_projector_and_ema(self):
        self.prepare()
        with torch.no_grad():
            self.model.self_flow_ema.ema_parameters[0].add_(3.0)
            self.model.self_flow_projector.net[0].weight.add_(1.0)
        saved = self.model.get_self_flow_state_dict()
        with tempfile.TemporaryDirectory() as root:
            self.model.ema = None
            self.model.train_progress = TrainProgress()
            InternalModelSaverMixin()._save_internal_data(self.model, root)
            restored = tiny_model()
            InternalModelLoaderMixin()._load_internal_data(restored, root)
        restored.lora_state_dict = self.model.transformer_lora.state_dict()
        restored.train_config = self.config
        restored.optimizer_state_dict = None
        setup = create.create_model_setup(
            ModelType.MOSS_IMAGE, torch.device("cpu"), torch.device("cpu"), TrainingMethod.LORA,
        )
        setup.setup_optimizations(restored, self.config)
        setup.setup_model(restored, self.config)
        torch.testing.assert_close(restored.self_flow_ema.ema_parameters[0], saved["ema"]["ema_parameters"][0])
        torch.testing.assert_close(restored.self_flow_projector.net[0].weight, saved["projector"]["net.0.weight"])
        self.assertIsNone(restored.self_flow_state_dict)

    def test_sampler_uses_ema_teacher_only_when_requested(self):
        self.prepare()
        with torch.no_grad():
            for ema in self.model.self_flow_ema.ema_parameters:
                ema.add_(1.0)
        lora = self.model.transformer_lora.parameters()
        student = [p.detach().clone() for p in lora]
        self.model.encode_text = Mock(return_value=(torch.randn(1, 4, 8), torch.ones(1, 4, dtype=torch.bool)))
        seen = []

        def forward(*args, **kwargs):
            seen.append(torch.equal(lora[0].detach(), self.model.self_flow_ema.ema_parameters[0]))
            return transformer_forward(*args, **kwargs)

        transformer_forward = self.model.transformer.forward
        sampler = MOSSImageSampler(torch.device("cpu"), torch.device("cpu"), self.model, ModelType.MOSS_IMAGE)
        sampler.save_sampler_output = Mock()
        for use_ema in (True, False):
            config = SampleConfig.default_values(ModelType.MOSS_IMAGE)
            config.width = config.height = 32
            config.diffusion_steps = 1
            config.cfg_scale = 1.0
            config.moss_uncond_skip_layers = ""
            config.self_flow_ema = use_ema
            seen.clear()
            with patch.object(self.model.transformer, "forward", side_effect=forward):
                sampler.sample(config, "unused")
            self.assertEqual(seen, [use_ema])
            for active, before in zip(lora, student, strict=True):
                torch.testing.assert_close(active.detach(), before, rtol=0, atol=0)

    def test_full_finetune_ema_covers_trained_transformer_weights(self):
        self.config.training_method = TrainingMethod.FINE_TUNE
        setup = create.create_model_setup(
            ModelType.MOSS_IMAGE, torch.device("cpu"), torch.device("cpu"), TrainingMethod.FINE_TUNE,
        )
        setup.setup_optimizations(self.model, self.config)
        setup.setup_model(self.model, self.config)
        trained = [p for p in self.model.transformer.parameters() if p.requires_grad]
        self.assertEqual(len(self.model.self_flow_ema.parameters), len(trained))
        batch = self.batch()
        output = setup.predict(self.model, batch, self.config, TrainProgress())
        setup.calculate_loss(self.model, batch, output, self.config).backward()
        self.assertTrue(any(p.grad is not None for p in trained))

    def test_unsupported_combinations_are_rejected(self):
        self.config.training_method = TrainingMethod.FINE_TUNE
        self.config.transformer.offload_fraction = 0.5
        setup = create.create_model_setup(
            ModelType.MOSS_IMAGE, torch.device("cpu"), torch.device("cpu"), TrainingMethod.FINE_TUNE,
        )
        with self.assertRaisesRegex(NotImplementedError, "layer offloading"):
            setup.setup_model(self.model, self.config)

        self.config.training_method = TrainingMethod.LORA
        self.config.transformer.offload_fraction = 0
        self.config.masked_training = True
        self.config.masked_prior_preservation_weight = 0.5
        with self.assertRaisesRegex(NotImplementedError, "prior preservation"):
            self.setup.setup_model(self.model, self.config)

        self.config.masked_training = False
        self.config.self_flow_mask_ratio = 0.75
        with self.assertRaisesRegex(ValueError, "mask ratio"):
            self.setup.setup_model(self.model, self.config)

        self.config.self_flow_mask_ratio = 0.25
        self.prepare()
        batch = self.batch()
        output = self.setup.predict(self.model, batch, self.config, TrainProgress())
        output["prior_target"] = output["target"].detach()
        with self.assertRaisesRegex(NotImplementedError, "prior prediction"):
            self.setup.calculate_loss(self.model, batch, output, self.config)


if __name__ == "__main__":
    unittest.main()
