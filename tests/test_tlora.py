"""T-LoRA (timestep-dependent LoRA rank) for MOSS Image. CPU only."""
import copy
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from modules.modelSampler.MOSSImageSampler import MOSSImageSampler
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.module.quantized import LinearW8A8 as w8a8
from modules.module.quantized.LinearSVD import BaseLinearSVD
from modules.ui.BaseLoraTabView import BaseLoraTabView
from modules.util import create
from modules.util.config.SampleConfig import SampleConfig
from modules.util.config.TrainConfig import QuantizationConfig
from modules.util.enum.DataType import DataType
from modules.util.enum.ModelType import ModelType, PeftType
from modules.util.enum.Optimizer import Optimizer
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.optimizer_util import change_optimizer
from modules.util.quantization_util import replace_linear_with_quantized_layers
from modules.util.TrainProgress import TrainProgress

import torch

from test_moss_image import tiny_model, train_config
from test_w8a8_compile import _cpu_mm_8bit


def _official_rank(timestep, rank, min_rank, total=1000):
    # get_rank_by_timestep from github.com/ControlGenAI/T-LoRA, flux/train.py
    return int((total - timestep) * (rank - min_rank) / total) + min_rank


class TLoRATest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        torch.set_num_threads(1)
        # W8A8's Triton backward kernel needs a GPU.
        self.mm = patch.object(w8a8, "mm_8bit", _cpu_mm_8bit)
        self.mm.start()
        self.addCleanup(self.mm.stop)

    def _setup(self, compile=False, rank=8, min_rank=1, model=None, tlora=True):
        model = model or tiny_model()
        config = train_config()
        config.lora_rank, config.lora_alpha = rank, rank
        config.tlora, config.tlora_min_rank = tlora, min_rank
        config.compile = compile
        model.train_config = config
        setup = create.create_model_setup(
            ModelType.MOSS_IMAGE, torch.device("cpu"), torch.device("cpu"), TrainingMethod.LORA,
        )
        setup.setup_optimizations(model, config)
        setup.setup_model(model, config)
        setup.setup_train_device(model, config)
        # LoRA up starts at zero; random weights make every rank component visible in outputs and gradients.
        for module in model.transformer_lora.lora_modules.values():
            torch.nn.init.normal_(module.lora_up.weight, std=0.1)
        return model, config, setup

    @staticmethod
    def _batch(size=1):
        return {
            "latent_image": torch.randn(size, 32, 4, 4), "tokens": torch.ones(size, 4, dtype=torch.long),
            "tokens_mask": torch.ones(size, 4), "text_encoder_hidden_state": torch.randn(size, 4, 8),
            "loss_weight": torch.ones(size), "concept_type": ["STANDARD"] * size,
        }

    def _predict(self, model, config, setup, batch, index):
        with patch.object(setup, "_get_timestep_discrete", return_value=torch.tensor(index)):
            return setup.predict(model, batch, config, TrainProgress())

    def test_rank_follows_the_official_schedule(self):
        model, _, _ = self._setup(rank=8, min_rank=2)
        times = torch.tensor([1.0, 250.0, 500.0, 999.0, 1000.0])
        model.set_tlora_timestep(times)
        mask = model.transformer_lora._rank_mask
        expected = [_official_rank(t, 8, 2) for t in times.tolist()]
        self.assertEqual(mask.sum(1).int().tolist(), expected)
        # Always the leading ranks.
        for row, count in zip(mask, expected, strict=True):
            self.assertEqual(row.tolist(), [1.0] * count + [0.0] * (8 - count))
        # Self-Flow passes per-token timesteps: a sample uses the mean noise level of its image tokens.
        model.set_tlora_timestep(torch.tensor([[1000.0, 1000.0, 1.0, 1.0]]))
        self.assertEqual(model.transformer_lora._rank_mask.sum().item(), _official_rank(500.5, 8, 2))

    def test_training_trains_only_the_ranks_of_its_timestep(self):
        model, config, setup = self._setup()
        output = self._predict(model, config, setup, self._batch(), [999])  # raw time 1000: pure noise
        setup.calculate_loss(model, self._batch(), output, config).backward()
        for module in model.transformer_lora.lora_modules.values():
            self.assertEqual(module.lora_down.weight.grad[1:].abs().sum().item(), 0.0)
            self.assertEqual(module.lora_up.weight.grad[:, 1:].abs().sum().item(), 0.0)
            self.assertGreater(module.lora_up.weight.grad[:, 0].abs().sum().item(), 0.0)

    def test_svd_quantized_layers_mask_only_the_lora_ranks(self):
        model, config, setup = self._setup(model=self._svd_model())
        svd = [m for m in model.transformer_lora.lora_modules.values() if isinstance(m.orig_module, BaseLinearSVD)]
        self.assertTrue(svd)
        batch = self._batch()
        output = self._predict(model, config, setup, batch, [999])
        setup.calculate_loss(model, batch, output, config).backward()
        for module in svd:
            self.assertEqual(module.lora_down.weight.grad[1:].abs().sum().item(), 0.0)
            self.assertEqual(module.lora_up.weight.grad[:, 1:].abs().sum().item(), 0.0)
            self.assertGreater(module.lora_up.weight.grad[:, 0].abs().sum().item(), 0.0)
        # The SVD residual stays unmasked: same output as a LoRA cut to its first rank without a mask.
        reference = copy.deepcopy(model)
        reference.train_config = copy.copy(config)
        reference.train_config.tlora = False
        reference.transformer_lora.set_rank_mask(None)
        for module in reference.transformer_lora.lora_modules.values():
            module.lora_up.weight.data[:, 1:] = 0
        with torch.no_grad():
            expected = self._predict(reference, reference.train_config, setup, batch, [999])["predicted"]
        torch.testing.assert_close(output["predicted"].detach(), expected)

    def test_masked_forward_equals_a_lora_cut_to_the_active_ranks(self):
        model, config, setup = self._setup()
        batch = self._batch(2)
        torch.manual_seed(0)
        masked = self._predict(model, config, setup, batch, [0, 600])["predicted"]
        for index, sample in ((0, 0), (600, 1)):
            rank = _official_rank(index + 1, 8, 1)
            reference = copy.deepcopy(model)
            reference.train_config = copy.copy(config)
            reference.train_config.tlora = False
            reference.transformer_lora.set_rank_mask(None)
            for module in reference.transformer_lora.lora_modules.values():
                module.lora_up.weight.data[:, rank:] = 0
            torch.manual_seed(0)
            expected = self._predict(reference, reference.train_config, setup, batch, [0, 600])["predicted"]
            torch.testing.assert_close(masked[sample], expected[sample])

    def test_positional_sampler_call_sets_a_mask_per_sample(self):
        model, _, _ = self._setup()
        hidden = torch.randn(2, 128, 2, 2)
        with torch.no_grad():
            model.transformer(hidden, torch.tensor(1000.0), torch.randn(2, 3, 8), torch.ones(2, 3, dtype=torch.bool))
        self.assertEqual(model.transformer_lora._rank_mask.sum(1).tolist(), [1.0, 1.0])

    @staticmethod
    def _svd_model():
        model = tiny_model()
        quantization = QuantizationConfig.default_values()
        quantization.svd_dtype, quantization.svd_rank, quantization.cache_dir = DataType.FLOAT_32, 2, None
        replace_linear_with_quantized_layers(
            model.transformer, DataType.INT_W8A8, keep_in_fp32_modules=[], quantization=quantization, copy_parameters=True,
        )
        return model

    def test_compiled_blocks_see_each_new_mask_without_recompiling(self):
        for svd in (False, True):
            with self.subTest(svd=svd):
                base = self._svd_model() if svd else tiny_model()
                compiled, config, setup = self._setup(compile=True, model=copy.deepcopy(base))
                eager, eager_config, _ = self._setup(compile=False, model=base)
                eager.transformer_lora.load_state_dict(compiled.transformer_lora.state_dict())
                batch = self._batch()
                torch._dynamo.reset()
                graphs = None
                for index in (999, 0, 500):
                    torch.manual_seed(1)
                    actual = self._predict(compiled, config, setup, batch, [index])
                    setup.calculate_loss(compiled, batch, actual, config).backward()
                    torch.manual_seed(1)
                    expected = self._predict(eager, eager_config, setup, batch, [index])["predicted"]
                    torch.testing.assert_close(actual["predicted"].detach(), expected.detach(), rtol=1e-4, atol=1e-4)
                    count = torch._dynamo.utils.counters["stats"]["unique_graphs"]
                    if graphs is None:
                        graphs = count
                    self.assertEqual(count, graphs)

    def test_without_tlora_training_and_sampling_use_every_rank(self):
        model, config, setup = self._setup(tlora=False)
        self.assertIsNone(model._tlora_hook)
        loras = list(model.transformer_lora.lora_modules.values())

        batch = self._batch()
        output = self._predict(model, config, setup, batch, [999])
        setup.calculate_loss(model, batch, output, config).backward()
        for module in loras:
            self.assertIsNone(module.rank_mask)
            self.assertGreater(module.lora_up.weight.grad[:, 1:].abs().sum().item(), 0.0)

        positive = torch.randn(1, 3, 8)
        mask = torch.ones(1, 3, dtype=torch.bool)
        model.encode_text = Mock(side_effect=[(positive, mask), (positive, mask)])
        sample_config = SampleConfig.default_values(ModelType.MOSS_IMAGE)
        sample_config.width = sample_config.height = 32
        sample_config.diffusion_steps, sample_config.cfg_scale, sample_config.moss_uncond_skip_layers = 2, 3.0, ""
        sampler = MOSSImageSampler(torch.device("cpu"), torch.device("cpu"), model, ModelType.MOSS_IMAGE)
        sampler.save_sampler_output = Mock()
        with patch.object(model.transformer, "forward", wraps=model.transformer.forward) as forward:
            sampler.sample(sample_config, "unused")
        self.assertGreater(forward.call_count, 0)
        self.assertTrue(all(module.rank_mask is None for module in loras))

    def test_switching_tlora_off_clears_a_previous_mask(self):
        model, config, _ = self._setup()
        model.set_tlora_timestep(torch.tensor([1000.0]))
        self.assertIsNotNone(model.transformer_lora._rank_mask)
        config.tlora = False
        model.set_tlora_timestep(torch.tensor([1000.0]))
        self.assertTrue(all(module.rank_mask is None for module in model.transformer_lora.lora_modules.values()))

    def test_sinksgd_adaptive_leaves_the_masked_ranks_untouched(self):
        model, config, setup = self._setup()
        config.optimizer.optimizer = Optimizer.SINKSGD_ADAPTIVE
        config.optimizer = change_optimizer(config)
        config.learning_rate = 1.0
        optimizer = create.create_optimizer(setup.create_parameters(model, config), None, config, model)
        loras = list(model.transformer_lora.lora_modules.values())
        before = [(m.lora_down.weight.detach().clone(), m.lora_up.weight.detach().clone()) for m in loras]
        batch = self._batch()
        output = self._predict(model, config, setup, batch, [999])  # pure noise: only rank 1 is active
        setup.calculate_loss(model, batch, output, config).backward()
        optimizer.step()
        for module, (down, up) in zip(loras, before, strict=True):
            torch.testing.assert_close(module.lora_down.weight.detach()[1:], down[1:], rtol=0, atol=0)
            torch.testing.assert_close(module.lora_up.weight.detach()[:, 1:], up[:, 1:], rtol=0, atol=0)
            self.assertFalse(torch.equal(module.lora_up.weight.detach()[:, 0], up[:, 0]))

    def test_rejects_unsupported_combinations(self):
        model = tiny_model()
        cases = {
            "DoRA": (lambda c: setattr(c, "lora_decompose", True), "without DoRA"),
            "LoHa": (lambda c: setattr(c, "peft_type", PeftType.LOHA), "LoRA type"),
            "min rank": (lambda c: setattr(c, "tlora_min_rank", 3), "between 1"),
            "other model": (lambda c: setattr(c, "model_type", ModelType.FLUX_DEV_1), "only connected"),
        }
        for name, (change, message) in cases.items():
            with self.subTest(name):
                config = train_config()
                config.tlora = True
                change(config)
                with self.assertRaisesRegex(ValueError, message):
                    LoRAModuleWrapper(model.transformer, "transformer", config, ["layers."])

    def test_lora_tab_offers_tlora_only_for_moss(self):
        for model_type, shown in ((ModelType.MOSS_IMAGE, True), (ModelType.FLUX_DEV_1, False)):
            labels = []
            components = SimpleNamespace(
                label=lambda master, row, column, text, labels=labels, **kwargs: labels.append(text),
                switch=lambda *args, **kwargs: None, entry=lambda *args, **kwargs: None,
                options_kv=lambda *args, **kwargs: None, path_entry=lambda *args, **kwargs: None,
            )
            controller = SimpleNamespace(
                train_config=SimpleNamespace(model_type=model_type), get_lora_weight_dtypes=list,
            )
            BaseLoraTabView(components).build_lora_options(None, controller, None, PeftType.LORA)
            self.assertEqual("T-LoRA" in labels and "T-LoRA Min Rank" in labels, shown)


if __name__ == "__main__":
    unittest.main()
