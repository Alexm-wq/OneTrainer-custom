import unittest

import torch

from mage_flow.models.mage_flow import MageFlow, MageFlowParams
from mage_flow.models.modules._attn_backend import set_attn_backend

from modules.model.MageFlowModel import MageFlowModel
from modules.module.MageFlowSelfFlow import mage_flow_forward


class MageOfficialRuntimeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(321)
        set_attn_backend("sdpa")
        self.model = MageFlow(MageFlowParams(
            in_channels=8,
            out_channels=8,
            context_in_dim=8,
            hidden_size=16,
            num_heads=2,
            depth=3,
            axes_dim=[2, 2, 4],
            checkpoint=False,
            patch_size=1,
        ))
        self.model.eval()

        self.batch = 2
        self.image_tokens_per_sample = 4
        self.text_tokens_per_sample = 3
        self.img = torch.randn(1, 8, 8)
        self.txt = torch.randn(1, 6, 8)
        self.img_cu = torch.tensor([0, 4, 8], dtype=torch.int32)
        self.txt_cu = torch.tensor([0, 3, 6], dtype=torch.int32)
        self.img_shapes = MageFlowModel.image_shapes(self.batch, 2, 2)
        self.timesteps = torch.tensor([0.2, 0.7])

    def _official(self):
        return self.model(
            img=self.img,
            txt=self.txt,
            timesteps=self.timesteps,
            img_shapes=self.img_shapes,
            img_cu_seqlens=self.img_cu,
            txt_cu_seqlens=self.txt_cu,
        )

    def _ours(self, image_timesteps, *, checkpoint=False):
        self.model.train(checkpoint)
        self.model.checkpoint = checkpoint
        return mage_flow_forward(
            transformer=self.model,
            img=self.img,
            txt=self.txt,
            image_timesteps=image_timesteps,
            text_timesteps=self.timesteps,
            img_shapes=self.img_shapes,
            img_cu_seqlens=self.img_cu,
            txt_cu_seqlens=self.txt_cu,
            capture_layer=1,
        )

    def test_scalar_forward_matches_official_mage_exact_path(self):
        self.model.eval()
        official = self._official()
        ours = self._ours(self.timesteps).predicted
        torch.testing.assert_close(ours, official, rtol=2e-5, atol=2e-6)

    def test_homogeneous_tokenwise_matches_official_mage(self):
        self.model.eval()
        official = self._official()
        tokenwise = torch.cat([
            self.timesteps[0].repeat(self.image_tokens_per_sample),
            self.timesteps[1].repeat(self.image_tokens_per_sample),
        ]).reshape(1, -1)
        ours = self._ours(tokenwise).predicted
        torch.testing.assert_close(ours, official, rtol=2e-5, atol=2e-6)

    def test_real_mage_tokenwise_checkpoint_backward(self):
        tokenwise = torch.tensor([[0.2, 0.8, 0.2, 0.8, 0.7, 0.1, 0.7, 0.1]])
        result = self._ours(tokenwise, checkpoint=True)
        loss = result.predicted.square().mean() + result.feature.square().mean()
        loss.backward()
        grads = [p.grad for p in self.model.parameters() if p.requires_grad]
        self.assertTrue(any(
            grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
            for grad in grads
        ))


if __name__ == "__main__":
    unittest.main()
