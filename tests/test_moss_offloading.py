"""MOSS offload traversal/recompute contracts; optional real CUDA transfers."""
import unittest
from unittest.mock import Mock, patch

import test_moss_image as fixtures

from modules.util.LayerOffloadConductor import LayerOffloadConductor
from modules.util.TrainProgress import TrainProgress

import torch


class RecordingConductor:
    """Exercise real checkpoint wrappers without pretending CPU is a CUDA device."""
    def __init__(self, *args):
        self.layers, self.starts, self.forward_calls, self.backward_calls = [], [], [], []
        self.pending = set()

    def add_layer(self, layer, indices):
        self.layers.append((layer, indices))

    def offload_activated(self):
        return True

    def start_forward(self, keep_graph):
        if self.pending:
            raise AssertionError("A new forward would discard unrecomputed activations")
        self.starts.append(keep_graph)
        self.keep_graph = keep_graph

    def before_layer(self, index, call, args):
        if torch.is_grad_enabled():
            self.pending.remove(call)
            self.backward_calls.append(index)
        else:
            self.forward_calls.append(index)
            if self.keep_graph:
                self.pending.add(call)
        return args

    def after_layer(self, *args):
        pass

    def to(self, device):
        pass


class MOSSOffloadingTest(unittest.TestCase):
    def setUp(self):
        self.f = fixtures.MOSSImageTest()
        self.f.setUp()

    def test_multirow_checkpoint_recompute_matches_no_offload_gradients(self):
        f = self.f
        parameters = f.attach_lora()
        batch = f.batch()
        baseline = f.setup.predict(f.model, batch, f.config, TrainProgress(), deterministic=True)
        baseline["predicted"].square().mean().backward()
        gradients = [p.grad.clone() for p in parameters]
        for p in parameters:
            p.grad = None
        f.config.transformer.offload_fraction = .5
        f.config.transformer.activation_offloading = True
        with patch("modules.util.checkpointing_util.LayerOffloadConductor", RecordingConductor):
            f.setup.setup_optimizations(f.model, f.config)
        result = f.setup.predict(f.model, batch, f.config, TrainProgress(), deterministic=True)
        torch.testing.assert_close(result["predicted"], baseline["predicted"])
        result["predicted"].square().mean().backward()
        for p, expected in zip(parameters, gradients, strict=True):
            torch.testing.assert_close(p.grad, expected)
        conductor = f.model.transformer_offload_conductor
        self.assertEqual(conductor.starts, [True])
        self.assertEqual(conductor.forward_calls, [0, 1, 2, 3, 4])
        self.assertEqual(conductor.backward_calls, [4, 3, 2, 1, 0])
        self.assertFalse(conductor.pending)
        with torch.no_grad():
            for _ in range(2):
                output = f.model.transformer(
                    torch.randn(2, 128, 2, 2), 500, batch["text_encoder_hidden_state"],
                    batch["tokens_mask"], skip_layers=[1],
                ).sample
                self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(conductor.starts, [True, False, False])
        self.assertEqual(conductor.forward_calls[5:], [0, 1, 2, 4] * 2)

    def test_real_qwen_language_layers_offload_in_prompt_order(self):
        f = self.f
        f.real_components()
        tokens, mask = f.batch()["tokens"], f.batch()["tokens_mask"]
        expected, _ = f.model.encode_text("cpu", tokens=tokens, tokens_mask=mask)
        f.config.text_encoder.offload_fraction = .5
        with patch("modules.util.checkpointing_util.LayerOffloadConductor", RecordingConductor):
            f.setup.setup_optimizations(f.model, f.config)
        actual, _ = f.model.encode_text("cpu", tokens=tokens, tokens_mask=mask)
        torch.testing.assert_close(actual, expected)
        conductor = f.model.text_encoder_offload_conductor
        self.assertEqual(len(conductor.layers), 2)
        self.assertEqual(conductor.starts, [False, False])
        self.assertEqual(conductor.forward_calls, [0, 1, 0, 1])
        self.assertFalse(conductor.pending)

    def test_skipped_layer_waits_for_its_newly_scheduled_transfer(self):
        conductor = object.__new__(LayerOffloadConductor)
        prefix = "_LayerOffloadConductor__"
        events = []
        values = {
            "is_active": True, "call_index_layer_index_map": {}, "is_forward_pass": True,
            "offload_activations": False, "offload_layers": True, "keep_graph": False,
            "wait_layer_transfer": lambda i: events.append(("wait", i)),
            "schedule_deferred_layers_to_temp": lambda **kwargs: None,
            "get_loaded_layers": lambda: [0],
            "temp_device": torch.device("cpu"), "train_device": torch.device("cuda"),
            "schedule_layer_to": lambda i, device, **kwargs: events.append(("load", i)),
        }
        strategy = Mock()
        strategy.get_layers_to_offload.return_value = []
        strategy.get_layers_to_load.return_value = [4, 5]
        values["offload_strategy"] = strategy
        for key, value in values.items():
            setattr(conductor, prefix + key, value)
        with torch.no_grad():
            conductor.before_layer(4, 1, ())
        self.assertEqual(events, [("wait", 4), ("load", 4), ("load", 5), ("wait", 4)])

    @unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA: CPU CI does not validate device transfers")
    def test_cuda_layer_activation_offload_training_and_skipped_sampling(self):
        f = self.f
        f.config.train_device, f.config.temp_device = "cuda", "cpu"
        f.config.transformer.offload_fraction = .5
        f.config.transformer.activation_offloading = True
        f.setup.train_device = torch.device("cuda")
        parameters = f.attach_lora()
        f.model.transformer_lora.to("cuda")
        f.setup.setup_optimizations(f.model, f.config)
        f.model.transformer_to(torch.device("cuda"))
        batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in f.batch().items()}
        for _ in range(2):
            result = f.setup.predict(f.model, batch, f.config, TrainProgress())
            result["predicted"].square().mean().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters))
            for p in parameters:
                p.grad = None
        with torch.no_grad():
            for skip in ([], [1], [0, 2], []):
                output = f.model.transformer(
                    torch.randn(2, 128, 2, 2, device="cuda"), 500,
                    batch["text_encoder_hidden_state"], batch["tokens_mask"], skip_layers=skip,
                ).sample
                self.assertTrue(torch.isfinite(output).all())
        f.model.transformer_to(torch.device("cpu"))
        self.assertTrue(all(p.device.type == "cpu" for p in f.model.transformer.parameters()))


if __name__ == "__main__":
    unittest.main()
