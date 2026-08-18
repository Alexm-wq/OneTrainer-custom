import unittest

import torch
import torch.nn as nn

from modules.module.MageFlowSelfFlow import (
    MageFlowSelfFlowEMA,
    mage_flow_forward,
    structural_alignment_loss,
)


class _FakeTime(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(1, dim, bias=False)

    def forward(self, timestep, hidden_states):
        return self.proj(timestep.float().reshape(-1, 1)).to(hidden_states.dtype)


class _FakeAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.img = nn.Linear(dim, dim, bias=False)
        self.txt = nn.Linear(dim, dim, bias=False)

    def forward(self, hidden_states, encoder_hidden_states, **kwargs):
        return self.img(hidden_states), self.txt(encoder_hidden_states)


class _FakeBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.img_mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        self.txt_mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        self.img_norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.txt_norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.img_norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.txt_norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = _FakeAttention(dim)
        self.img_mlp = nn.Linear(dim, dim, bias=False)
        self.txt_mlp = nn.Linear(dim, dim, bias=False)

    def _modulate(self, x, mod_params, cu_lens=None, seq_lens=None):
        shift, scale, gate = mod_params.chunk(3, dim=-1)
        if cu_lens is not None:
            lengths = cu_lens[1:] - cu_lens[:-1]
            shift = shift.repeat_interleave(lengths, dim=0).unsqueeze(0)
            scale = scale.repeat_interleave(lengths, dim=0).unsqueeze(0)
            gate = gate.repeat_interleave(lengths, dim=0).unsqueeze(0)
        return x * (1 + scale) + shift, gate


class _FakeFinalNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, 2 * dim)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)

    def forward(self, x, emb, cu_seqlens=None):
        params = self.linear(self.silu(emb).to(x.dtype))
        scale, shift = params.chunk(2, dim=-1)
        if cu_seqlens is not None:
            lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            scale = scale.repeat_interleave(lengths, dim=0).unsqueeze(0)
            shift = shift.repeat_interleave(lengths, dim=0).unsqueeze(0)
        return self.norm(x) * (1 + scale) + shift


class _FakeTransformer(nn.Module):
    def __init__(self, dim=8, depth=4):
        super().__init__()
        self.inner_dim = dim
        self.img_in = nn.Linear(dim, dim)
        self.txt_norm = nn.LayerNorm(dim)
        self.txt_in = nn.Linear(dim, dim)
        self.time_text_embed = _FakeTime(dim)
        self.transformer_blocks = nn.ModuleList([_FakeBlock(dim) for _ in range(depth)])
        self.norm_out = _FakeFinalNorm(dim)
        self.proj_out = nn.Linear(dim, dim)
        self.checkpoint = False

    def pos_embed(self, img_shapes, device):
        total = sum(shape[0][0] * shape[0][1] * shape[0][2] for shape in img_shapes)
        return torch.zeros(total, self.inner_dim // 2, dtype=torch.complex64, device=device)


class MageSelfFlowSmokeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234)
        self.model = _FakeTransformer()
        self.batch = 2
        self.img_per_sample = 4
        self.txt_per_sample = 3
        self.img = torch.randn(1, self.batch * self.img_per_sample, 8)
        self.txt = torch.randn(1, self.batch * self.txt_per_sample, 8)
        self.img_cu = torch.tensor([0, 4, 8], dtype=torch.int32)
        self.txt_cu = torch.tensor([0, 3, 6], dtype=torch.int32)
        self.shapes = [[(1, 2, 2)], [(1, 2, 2)]]

    def _forward(self, image_t, text_t, checkpoint=False):
        self.model.train(checkpoint)
        self.model.checkpoint = checkpoint
        return mage_flow_forward(
            transformer=self.model,
            img=self.img,
            txt=self.txt,
            image_timesteps=image_t,
            text_timesteps=text_t,
            img_shapes=self.shapes,
            img_cu_seqlens=self.img_cu,
            txt_cu_seqlens=self.txt_cu,
            capture_layer=1,
        )

    def test_homogeneous_tokenwise_equals_native_per_sample_conditioning(self):
        per_sample = torch.tensor([0.2, 0.7])
        native = self._forward(per_sample, per_sample).predicted
        tokenwise = torch.cat([
            per_sample[0].repeat(self.img_per_sample),
            per_sample[1].repeat(self.img_per_sample),
        ]).reshape(1, -1)
        self_flow = self._forward(tokenwise, per_sample).predicted
        torch.testing.assert_close(native, self_flow, rtol=1e-5, atol=1e-6)

    def test_tokenwise_forward_backward_and_checkpointing(self):
        image_t = torch.tensor([[0.2, 0.8, 0.2, 0.8, 0.7, 0.1, 0.7, 0.1]])
        text_t = torch.tensor([0.2, 0.7])
        result = self._forward(image_t, text_t, checkpoint=True)
        self.assertIsNotNone(result.feature)
        loss = result.predicted.square().mean() + result.feature.square().mean()
        loss.backward()
        grads = [p.grad for p in self.model.parameters() if p.requires_grad]
        self.assertTrue(any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads))

    def test_checkpointed_forward_matches_noncheckpointed_forward(self):
        image_t = torch.tensor([[0.2, 0.8, 0.2, 0.8, 0.7, 0.1, 0.7, 0.1]])
        text_t = torch.tensor([0.2, 0.7])
        direct = self._forward(image_t, text_t, checkpoint=False)
        checkpointed = self._forward(image_t, text_t, checkpoint=True)
        torch.testing.assert_close(direct.predicted, checkpointed.predicted, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(direct.feature, checkpointed.feature, rtol=1e-6, atol=1e-7)

    def test_teacher_early_exit_returns_selected_hidden_feature(self):
        per_sample = torch.tensor([0.2, 0.7])
        self.model.eval()
        result = mage_flow_forward(
            transformer=self.model,
            img=self.img,
            txt=self.txt,
            image_timesteps=per_sample,
            text_timesteps=per_sample,
            img_shapes=self.shapes,
            img_cu_seqlens=self.img_cu,
            txt_cu_seqlens=self.txt_cu,
            stop_layer=1,
        )
        self.assertEqual(result.predicted.shape, self.img.shape)
        self.assertIsNotNone(result.feature)
        torch.testing.assert_close(result.predicted, result.feature)

    def test_cpu_ema_teacher_swap_restores_student_and_updates(self):
        module = nn.Linear(4, 4, bias=False)
        ema = MageFlowSelfFlowEMA([module], decay=0.5)
        original = module.weight.detach().clone()
        with torch.no_grad():
            module.weight.add_(2.0)
        student = module.weight.detach().clone()
        ema.student_parameters = [student.float().cpu().clone()]
        with ema.teacher_parameters([module]):
            torch.testing.assert_close(module.weight, original)
        torch.testing.assert_close(module.weight, student)
        ema.update_after_optimizer_step()
        expected = original.float().cpu().mul(0.5).add(student.float().cpu(), alpha=0.5)
        torch.testing.assert_close(ema.ema_parameters[0], expected)
        self.assertEqual(ema.optimization_steps, 1)

    def test_cpu_ema_state_dict_round_trip(self):
        module = nn.Linear(4, 4, bias=False)
        ema = MageFlowSelfFlowEMA([module], decay=0.75)
        with torch.no_grad():
            module.weight.add_(1.25)
        ema.update_after_optimizer_step()
        state = ema.state_dict()

        restored = MageFlowSelfFlowEMA([module], decay=0.1, state_dict=state)
        self.assertEqual(restored.decay, ema.decay)
        self.assertEqual(restored.optimization_steps, ema.optimization_steps)
        for left, right in zip(restored.ema_parameters, ema.ema_parameters, strict=True):
            torch.testing.assert_close(left, right)

    def test_structural_loss_zero_for_identical_features(self):
        x = torch.randn(2, 16, 8)
        loss = structural_alignment_loss(x, x.clone(), sample_count=8)
        torch.testing.assert_close(loss, torch.zeros_like(loss), atol=1e-7, rtol=0)


if __name__ == "__main__":
    unittest.main()
