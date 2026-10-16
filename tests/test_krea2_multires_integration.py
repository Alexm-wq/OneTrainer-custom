"""Exercise Krea/Kroma's real shared cache assembly with small CPU inputs."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from modules.dataLoader.Krea2BaseDataLoader import Krea2BaseDataLoader
from modules.dataLoader.cache.MultiResolutionDiskCache import MultiResolutionDiskCache
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod

from mgds.MGDS import MGDS
from mgds.OutputPipelineModule import OutputPipelineModule
from mgds.PipelineModule import PipelineState

import torch

from test_multi_resolution_cache import ImageSource, LatentProbe, concept


class KreaMultiResolutionIntegrationTest(unittest.TestCase):
    def test_krea_cache_assembly_with_and_without_aspect_bucketing(self):
        for bucketed in (True, False):
            with self.subTest(bucketed=bucketed), tempfile.TemporaryDirectory() as directory:
                config = TrainConfig.default_values()
                config.model_type = ModelType.KREA_2
                config.training_method = TrainingMethod.LORA
                config.text_encoder.train = False
                config.resolution = "64,128"
                config.cache_dir = directory
                config.aspect_ratio_bucketing = bucketed
                config.image_caching = True
                config.text_caching = True
                config.rlhf_enabled = False
                loader = object.__new__(Krea2BaseDataLoader)
                loader.train_device = torch.device("cpu")
                loader.temp_device = torch.device("cpu")
                model = SimpleNamespace(to=Mock(), vae_to=Mock(), eval=Mock())
                setup = SimpleNamespace(prepare_text_caching=Mock())
                modules = loader._cache_modules(config, model, setup)
                self.assertIsInstance(modules[0], MultiResolutionDiskCache)
                source = ImageSource([concept()])
                probe = LatentProbe()
                bucketing = loader._aspect_bucketing_in(config, 64)
                output = OutputPipelineModule(["latent_image", "crop_resolution", "tokens",
                                               "text_encoder_hidden_state"])
                state = PipelineState(1)
                try:
                    ds = MGDS(torch.device("cpu"), [concept()],
                              {"target_resolution": config.resolution, "target_frames": "1"},
                              [source, bucketing, probe, modules, output], 1, state)
                    seen = set()
                    for _ in range(2):
                        ds.start_next_epoch()
                        rows = list(ds)
                        self.assertEqual(len(rows), 1)
                        seen.add(rows[0]["crop_resolution"][-1])
                        self.assertEqual(rows[0]["latent_image"].shape[1], 1)
                        torch.testing.assert_close(rows[0]["tokens"], torch.tensor([10, 20]))
                    self.assertEqual(seen, {64, 128})
                    self.assertEqual(len(probe.calls), 2)
                    self.assertTrue(list(Path(directory).glob("image/*/variation-0/0.pt")))
                    setup.prepare_text_caching.assert_called_once()
                finally:
                    state.executor.shutdown(wait=True)


if __name__ == "__main__":
    unittest.main()
