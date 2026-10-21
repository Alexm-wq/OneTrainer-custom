"""Every MOSS training row must pair an image's latent with that image's own caption conditioning.

Each image has a unique solid color that survives the fake VAE and a unique caption, so the latent,
image path, prompt, tokens and text-encoder hidden state of every collated batch row can be
checked against each other. No model weights or GPU required.
"""
import json
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from modules.util import create  # noqa: F401

# isort: split
from modules.dataLoader.MOSSImageBaseDataLoader import MOSSImageBaseDataLoader
from modules.util.config.ConceptConfig import ConceptConfig
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.DataType import DataType
from modules.util.enum.ModelType import ModelType
from modules.util.moss_layout import empty_layout
from modules.util.TrainProgress import TrainProgress

import torch

from PIL import Image
from test_moss_layout import TinyModel

# (concept, sample, width, height): mixed aspect ratios exercise bucketing and batch regrouping.
SAMPLES = [
    (0, 0, 192, 128), (0, 1, 128, 192), (0, 2, 160, 160), (0, 3, 192, 128), (0, 4, 128, 192),
    (1, 0, 160, 160), (1, 1, 192, 128), (1, 2, 128, 192), (1, 3, 160, 160),
]


def color(concept, sample):
    return 40 + 20 * sample, 60 + 120 * concept, 200


class MOSSPairingTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for concept, sample, width, height in SAMPLES:
            folder = self.root / f"concept{concept}"
            folder.mkdir(exist_ok=True)
            stem = folder / f"s{sample}"
            Image.new("RGB", (width, height), color(concept, sample)).save(stem.with_suffix(".png"))
            if (concept, sample) == (1, 3):
                layout = empty_layout(width, height, f"c{concept} s{sample} layout scene")
                layout["items"] = [{"label": "object", "caption": "box", "bbox": [-20, -20, 20, 20]}]
                (folder / f"s{sample}.layout.json").write_text(json.dumps(layout), encoding="utf-8")
            else:
                stem.with_suffix(".txt").write_text(
                    f"c{concept} s{sample} first caption\nc{concept} s{sample} second caption", encoding="utf-8",
                )

    def make(self, image_cache, text_cache, cache):
        config = TrainConfig.default_values()
        config.model_type = ModelType.MOSS_IMAGE
        config.train_device = config.temp_device = "cpu"
        config.train_dtype = config.fallback_train_dtype = DataType.FLOAT_32
        config.base_model_name = "tiny-moss"
        config.text_encoder_sequence_length = 1024
        config.text_encoder.train = False
        config.image_caching, config.text_caching = image_cache, text_cache
        config.cache_dir = str(self.root / cache)
        config.resolution = "128,192"
        config.batch_size, config.caching_threads = 3, 2
        config.aspect_ratio_bucketing = True
        config.concepts = []
        for concept in range(2):
            concept_config = ConceptConfig.default_values()
            concept_config.path = str(self.root / f"concept{concept}")
            concept_config.seed = 7 + concept
            concept_config.image_variations = concept_config.text_variations = 2
            concept_config.balancing = 2.0 if concept else 1.0
            concept_config.image.enable_random_flip = True
            config.concepts.append(concept_config)
        self.configure(config)
        model = TinyModel()
        setup = SimpleNamespace(prepare_text_caching=lambda model, cfg: None)
        loader = MOSSImageBaseDataLoader(torch.device("cpu"), torch.device("cpu"), config, model, setup, TrainProgress())
        self.addCleanup(loader.get_data_set().loading_pipeline.modules[0]._state.executor.shutdown, wait=True)
        return loader

    def configure(self, config):
        pass

    def check_row(self, batch, row):
        path = Path(batch["image_path"][row])
        identity = (int(path.parent.name.removeprefix("concept")), int(path.stem.removeprefix("s")))
        # The fake VAE resizes the [-1, 1] image, so the solid color is the latent value.
        latent = batch["latent_image"][row]
        expected = torch.tensor([value / 127.5 - 1 for value in color(*identity)])
        torch.testing.assert_close(latent.mean(dim=(-2, -1)), expected, atol=1e-4, rtol=0)
        prompt = batch["prompt"][row]
        concept, sample = map(int, re.search(r"c(\d) s(\d)", prompt).groups())
        self.assertEqual((concept, sample), identity, prompt)
        encoded = TinyModel().tokenize(prompt, 1024)
        torch.testing.assert_close(batch["tokens"][row], encoded.input_ids[0])
        torch.testing.assert_close(batch["tokens_mask"][row], encoded.attention_mask[0])
        torch.testing.assert_close(batch["text_encoder_hidden_state"][row][:, 0], batch["tokens"][row].float())
        return identity

    def test_latent_and_text_conditioning_belong_to_the_same_image(self):
        self.rows_checked = 0
        for image_cache in (False, True):
            for text_cache in (False, True):
                with self.subTest(image_cache=image_cache, text_cache=text_cache):
                    loader = self.make(image_cache, text_cache, f"cache-{image_cache}-{text_cache}")
                    seen = []
                    for _ in range(3):
                        loader.get_data_set().start_next_epoch()
                        for batch in loader.get_data_loader():
                            self.assertEqual(len(batch["prompt"]), 3)
                            seen += [self.check_row(batch, row) for row in range(len(batch["prompt"]))]
                    # Rows from both concepts, captions and the layout reach collated batches.
                    self.assertEqual({concept for concept, _ in seen}, {0, 1})
                    self.rows_checked += len(seen)


if __name__ == "__main__":
    unittest.main()
