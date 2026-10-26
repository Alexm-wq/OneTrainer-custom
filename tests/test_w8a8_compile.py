"""W8A8 linear layers inside compiled, checkpointed blocks. CPU only; no weights are downloaded."""
import unittest
from unittest.mock import patch

import modules.util.quantization_util  # noqa: F401  (LinearW8A8 is imported through it)
from modules.module.quantized import LinearW8A8 as w8a8
from modules.util import create
from modules.util.enum.DataType import DataType
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.quantization_util import replace_linear_with_quantized_layers
from modules.util.TrainProgress import TrainProgress

import torch
from torch import nn

from test_moss_image import tiny_model, train_config


def _cpu_mm_8bit(a, b):
    # The Triton backward kernel needs a GPU. Like its autotuner, which keys a cache on the
    # shapes, this stand-in only accepts concrete sizes, so it cannot be traced symbolically.
    hash((a.shape, b.shape))
    return torch._int_mm(a, b)


class W8A8CompileTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.mm = patch.object(w8a8, "mm_8bit", _cpu_mm_8bit)
        self.mm.start()

    def tearDown(self):
        self.mm.stop()

    def _layer(self, dtype=torch.int8):
        linear = nn.Linear(64, 96)
        layer = w8a8.LinearW8A8(dtype, 64, 96)
        layer.load_state_dict(linear.state_dict(), strict=False)
        layer.compute_dtype = torch.bfloat16
        layer.quantize()
        return layer

    def _run(self, layer, x, grad_output=None):
        x = x.clone().requires_grad_(True)
        y = layer(x)
        if grad_output is None:
            y.float().square().sum().backward()
        else:
            y.backward(grad_output)
        return y.detach(), x.grad

    def test_matches_reference_math(self):
        layer = self._layer()
        x = torch.randn(40, 64, dtype=torch.bfloat16)
        y, grad = self._run(layer, x)
        expected = w8a8.int8_forward_tokenwise(x, layer.weight, layer.scale, layer.bias, torch.bfloat16)
        torch.testing.assert_close(y, expected, rtol=0, atol=0)
        expected_grad = w8a8.int8_backward_axiswise((2 * y.float()).to(torch.bfloat16), layer.weight, layer.scale)
        torch.testing.assert_close(grad, expected_grad, rtol=0, atol=0)

    def test_compiled_kernels_match_eager_for_changing_shapes(self):
        layer = self._layer()
        for rows in (40, 72, 56):
            x = torch.randn(rows, 64, dtype=torch.bfloat16)
            grad_output = torch.randn(rows, 96, dtype=torch.bfloat16)
            layer.compiled_kernels = False
            eager = self._run(layer, x, grad_output)
            layer.compiled_kernels = True
            compiled = self._run(layer, x, grad_output)
            # The fused bias add skips one intermediate bf16 rounding: at most one bf16 step apart.
            torch.testing.assert_close(compiled[0], eager[0], rtol=2 ** -7, atol=2 ** -7)
            torch.testing.assert_close(compiled[1], eager[1], rtol=0, atol=0)

    def test_compiled_lora_blocks_keep_no_matmul_results_for_backward(self):
        # torch 2.11 saved the int32 results of an autograd.Function forward in a compiled,
        # checkpointed block instead of recomputing them: 0.5 GiB per MOSS block at 1536px,
        # an OOM over 42 blocks. Eager checkpointing keeps none of them.
        model = tiny_model()
        config = train_config()
        config.compile = True
        model.train_config = config
        replace_linear_with_quantized_layers(model.transformer.layers, DataType.INT_W8A8, copy_parameters=True)
        config.transformer.weight_dtype = DataType.INT_W8A8
        setup = create.create_model_setup(
            ModelType.MOSS_IMAGE, torch.device("cpu"), torch.device("cpu"), TrainingMethod.LORA,
        )
        setup.setup_optimizations(model, config)
        setup.setup_model(model, config)
        setup.setup_train_device(model, config)
        # A second resolution recompiles with symbolic shapes, as multi-resolution buckets do.
        for step, size in enumerate((32, 48, 40)):
            batch = {
                "latent_image": torch.randn(1, 32, size, size), "tokens": torch.ones(1, 8, dtype=torch.long),
                "tokens_mask": torch.ones(1, 8), "text_encoder_hidden_state": torch.randn(1, 8, 8),
                "loss_weight": torch.ones(1), "concept_type": ["STANDARD"],
            }
            model.optimizer.zero_grad()
            saved = []

            def pack(tensor, saved=saved):
                saved.append(tensor)
                return tensor

            with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
                output = setup.predict(model, batch, config, TrainProgress(global_step=step))
                loss = setup.calculate_loss(model, batch, output, config)
            loss.backward()
        self.assertTrue(all(layer.compiled_kernels for layer in model.transformer.modules()
                            if isinstance(layer, w8a8.LinearW8A8)))
        self.assertFalse([tuple(t.shape) for t in saved if t.dtype == torch.int32])
        grads = [p.grad for p in model.transformer_lora.parameters()]
        self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))


if __name__ == "__main__":
    unittest.main()
