"""Focused checks for MOSS's new Qwen3.5 kernel dispatch. No model downloads."""
import contextlib
import inspect
import io
import os
import unittest
from unittest.mock import Mock, patch

from modules.module.quantized.LinearFp8 import LinearFp8
from modules.module.moss.text_encoder_kernels import (
    _GatedRMSNorm,
    _load_fla,
    _TextKernels,
    configure_moss_text_kernels,
)

import torch
import torch.nn.functional as F

from transformers import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5GatedDeltaNet,
    torch_chunk_gated_delta_rule,
)


def make_block(device="cpu", dtype=torch.float32, head_dim=16):
    config = Qwen3_5TextConfig(
        hidden_size=64, intermediate_size=128, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        layer_types=["linear_attention"], linear_num_key_heads=2,
        linear_num_value_heads=4, linear_key_head_dim=head_dim,
        linear_value_head_dim=head_dim,
    )
    return Qwen3_5GatedDeltaNet(config, 0).to(device=device, dtype=dtype).eval()


def quantize_projections(block):
    for name, module in list(block.named_children()):
        if isinstance(module, torch.nn.Linear):
            quantized = LinearFp8(
                module.in_features, module.out_features, bias=module.bias is not None,
                device=module.weight.device, dtype=module.weight.dtype,
            ).requires_grad_(False)
            quantized.load_state_dict(module.state_dict(), strict=False)
            quantized.quantize()
            setattr(block, name, quantized)


class MOSSKernelTest(unittest.TestCase):
    def test_real_fp8_projection_keeps_bf16_compute_for_kernel_dispatch(self):
        block = make_block(dtype=torch.bfloat16).requires_grad_(False)
        quantize_projections(block)
        configure_moss_text_kernels(block)
        self.assertEqual(block.in_proj_qkv.weight.dtype, torch.float8_e4m3fn)
        kernels = block.norm.kernels
        with patch.object(kernels, "operations", wraps=kernels.operations) as select, torch.no_grad():
            output = block(torch.randn(1, 7, 64, dtype=torch.bfloat16))
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(select.call_args_list)
        self.assertTrue(all(call.args[0].dtype == torch.bfloat16 for call in select.call_args_list))

    def test_real_fla_package_exports_expected_api(self):
        # Import the installed, pinned package; do not substitute dummy kernels.
        conv, chunk, norm = _load_fla()
        for fn, names in (
            (conv, ("x", "weight", "activation", "backend", "output_final_state")),
            (chunk, ("q", "k", "v", "g", "beta", "use_qk_l2norm_in_kernel")),
            (norm, ("x", "g", "weight", "eps", "activation")),
        ):
            self.assertTrue(set(names) <= set(inspect.signature(fn).parameters))

    def test_cpu_fallback_matches_reference_forward_and_gradients(self):
        torch.manual_seed(41)
        reference = make_block()
        target = make_block()
        target.load_state_dict(reference.state_dict())
        self.assertEqual(configure_moss_text_kernels(target), 1)
        # Non-chunk-aligned prompt lengths and multiple independent rows.
        for length in (1, 7, 65):
            with self.subTest(length=length):
                x = torch.randn(2, length, 64, requires_grad=True)
                y = x.detach().clone().requires_grad_(True)
                expected, actual = reference(x), target(y)
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
                expected.square().sum().backward()
                actual.square().sum().backward()
                torch.testing.assert_close(y.grad, x.grad, rtol=1e-5, atol=1e-6)

    def test_binding_preserves_parameters_and_checkpoint_and_is_idempotent(self):
        block = make_block()
        parameters = {name: id(p) for name, p in block.named_parameters()}
        state = {key: value.clone() for key, value in block.state_dict().items()}
        self.assertEqual(configure_moss_text_kernels(block), 1)
        self.assertEqual(configure_moss_text_kernels(block), 0)
        self.assertEqual(parameters, {name: id(p) for name, p in block.named_parameters()})
        self.assertEqual(set(state), set(block.state_dict()))
        for key, value in state.items():
            torch.testing.assert_close(block.state_dict()[key], value)
        self.assertIsInstance(block.norm, _GatedRMSNorm)

    def test_conv_layout_activation_and_no_final_state(self):
        kernels = _TextKernels()
        x = torch.randn(2, 17, 24).transpose(1, 2)
        weight, bias = torch.randn(24, 4), torch.randn(24)
        expected = F.silu(F.conv1d(x, weight[:, None], bias, padding=3, groups=24)[:, :, :17])

        def backend(**kwargs):
            self.assertEqual(kwargs["x"].shape, (2, 17, 24))
            self.assertEqual(kwargs["backend"], "triton")
            self.assertFalse(kwargs["output_final_state"])
            torch.testing.assert_close(kwargs["x"], x.transpose(1, 2))
            return expected.transpose(1, 2), None

        with patch.object(kernels, "operations", return_value=(backend, None, None)):
            actual = kernels.conv(x, weight, bias, "silu", seq_idx=None)
        torch.testing.assert_close(actual, expected)

    def test_delta_rule_preserves_gates_normalization_and_state_options(self):
        kernels = _TextKernels()
        q, k, v = [torch.randn(1, 7, 2, 16) for _ in range(3)]
        kwargs = {
            "g": -torch.rand(1, 7, 2), "beta": torch.rand(1, 7, 2),
            "initial_state": None, "output_final_state": False, "use_qk_l2norm_in_kernel": True,
        }
        backend = Mock(wraps=torch_chunk_gated_delta_rule)
        with patch.object(kernels, "operations", return_value=(None, backend, None)):
            actual, state = kernels.chunk(q, k, v, **kwargs)
        self.assertIsNone(state)
        backend.assert_called_once_with(q, k, v, **kwargs)
        torch.testing.assert_close(actual, torch_chunk_gated_delta_rule(q, k, v, **kwargs)[0])

    def test_cpu_and_fp32_do_not_import_or_run_cuda_kernels(self):
        kernels = _TextKernels()
        with patch("modules.module.moss.text_encoder_kernels._load_fla") as load:
            self.assertIsNone(kernels.operations(torch.zeros(1, dtype=torch.bfloat16)))
            fake_cuda_fp32 = Mock(device=torch.device("cuda"), dtype=torch.float32)
            self.assertIsNone(kernels.operations(fake_cuda_fp32))
            load.assert_not_called()

    def test_bf16_activations_select_fast_path_and_report_once(self):
        kernels = _TextKernels()
        tensor = Mock(device=torch.device("cuda:0"), dtype=torch.bfloat16)
        ops = (Mock(), Mock(), Mock())
        with patch("modules.module.moss.text_encoder_kernels._load_fla", return_value=ops):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertIs(kernels.operations(tensor), ops)
                self.assertIs(kernels.operations(tensor), ops)
            self.assertEqual(output.getvalue().count("FLA/Triton selected"), 1)

    def test_missing_dependency_and_explicit_disable_are_visible(self):
        tensor = Mock(device=torch.device("cuda:0"), dtype=torch.bfloat16)
        kernels = _TextKernels()
        with patch("modules.module.moss.text_encoder_kernels._load_fla", side_effect=ImportError("fla missing")) as load:
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertIsNone(kernels.operations(tensor))
                self.assertIsNone(kernels.operations(tensor))
            load.assert_called_once()
            self.assertIn("FLA unavailable", output.getvalue())
            self.assertEqual(output.getvalue().count("PyTorch text kernels"), 1)
        with patch.dict(os.environ, {"OT_MOSS_TEXT_KERNELS": "0"}):
            disabled = _TextKernels()
        with patch("modules.module.moss.text_encoder_kernels._load_fla") as load:
            self.assertIsNone(disabled.operations(tensor))
            load.assert_not_called()

    def test_runtime_kernel_failure_is_not_silently_hidden(self):
        kernels = _TextKernels()
        backend = Mock(side_effect=RuntimeError("kernel compilation failed"))
        q = torch.zeros(1, 7, 2, 16)
        with (
            patch.object(kernels, "operations", return_value=(None, backend, None)),
            self.assertRaisesRegex(RuntimeError, "kernel compilation failed"),
        ):
            kernels.chunk(q, q, q)


@unittest.skipUnless(torch.cuda.is_available() and torch.version.hip is None, "requires a real NVIDIA CUDA GPU")
class MOSSCUDAKernelTest(unittest.TestCase):
    def test_fla_prefill_matches_reference_with_real_qwen_head_dimensions(self):
        # This test must actually execute all three FLA kernels, not mocks.
        for dtype, fp8 in (
            (torch.bfloat16, False), (torch.float16, False), (torch.bfloat16, True),
        ):
            torch.manual_seed(42)
            reference = make_block("cuda", dtype, head_dim=128)
            target = make_block("cuda", dtype, head_dim=128)
            target.load_state_dict(reference.state_dict())
            if fp8:
                quantize_projections(reference)
                quantize_projections(target)
            with patch.dict(os.environ, {"OT_MOSS_TEXT_KERNELS": "0"}):
                configure_moss_text_kernels(reference)
            with patch.dict(os.environ, {"OT_MOSS_TEXT_KERNELS": "1"}):
                configure_moss_text_kernels(target)
            for length in (1, 7, 65, 257):
                with self.subTest(dtype=dtype, fp8=fp8, length=length), torch.no_grad():
                    x = torch.randn(1, length, 64, device="cuda", dtype=dtype)
                    self.assertIsNotNone(target.norm.kernels.operations(x))
                    expected, actual = reference(x), target(x)
                    torch.cuda.synchronize()
                    self.assertTrue(torch.isfinite(actual).all())
                    torch.testing.assert_close(actual.float(), expected.float(), rtol=0.03, atol=0.003)


if __name__ == "__main__":
    unittest.main()
