import unittest

import torch

from modules.model.MageFlowModel import MageFlowModel
from modules.util.enum.ModelType import ModelType


class MageFlowModelHelperTests(unittest.TestCase):
    def setUp(self):
        self.model = MageFlowModel(ModelType.MAGE_FLOW)

    def test_pack_unpack_latents_round_trip(self):
        latents = torch.randn(2, 128, 3, 5)
        packed = self.model.pack_latents(latents)
        self.assertEqual(tuple(packed.shape), (2, 15, 128))
        restored = self.model.unpack_latents(packed, 3, 5)
        torch.testing.assert_close(restored, latents)

    def test_cu_seqlens(self):
        cu = self.model._cu_seqlens([3, 5, 2], torch.device('cpu'))
        self.assertEqual(cu.dtype, torch.int32)
        torch.testing.assert_close(cu, torch.tensor([0, 3, 8, 10], dtype=torch.int32))

    def test_prepare_and_unprepare_packed_images_round_trip(self):
        tokens = torch.randn(3, 7, 128)
        packed, cu = self.model.prepare_packed_images(tokens)
        self.assertEqual(tuple(packed.shape), (1, 21, 128))
        torch.testing.assert_close(cu, torch.tensor([0, 7, 14, 21], dtype=torch.int32))
        restored = self.model.unprepare_packed_images(packed, 3)
        torch.testing.assert_close(restored, tokens)

    def test_prepare_packed_text_drops_padding_only(self):
        text = torch.randn(2, 5, 16)
        mask = torch.tensor([
            [True, True, True, False, False],
            [True, True, True, True, False],
        ])
        packed, cu = self.model.prepare_packed_text(text, mask)
        self.assertEqual(tuple(packed.shape), (1, 7, 16))
        torch.testing.assert_close(cu, torch.tensor([0, 3, 7], dtype=torch.int32))
        torch.testing.assert_close(packed[0, :3], text[0, :3])
        torch.testing.assert_close(packed[0, 3:], text[1, :4])

    def test_packed_image_shape_layout_matches_official_mage(self):
        shapes = self.model.image_shapes(3, 5, 7)
        self.assertEqual(shapes, [[(1, 5, 7), (1, 5, 7), (1, 5, 7)]])

    def test_model_type_contract(self):
        self.assertTrue(ModelType.MAGE_FLOW.is_mage_flow())
        self.assertTrue(ModelType.MAGE_FLOW.is_flow_matching())
        self.assertEqual(ModelType.MAGE_FLOW.model_parts(), ('transformer', 'text_encoder', 'vae'))


if __name__ == '__main__':
    unittest.main()
