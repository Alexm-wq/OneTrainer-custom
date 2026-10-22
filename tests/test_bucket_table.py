"""Every aspect-bucket user follows the extended bucket table ported from preview 7e1899241d."""
import unittest

from modules.util import create  # noqa: F401

# isort: split
from modules.dataLoader import StableDiffusionFineTuneVaeDataLoader as vae_loader
from modules.dataLoader.cache.AspectBucketing import AspectBucketing
from modules.dataLoader.cache.SingleAspectCalculation import SingleAspectCalculation
from modules.dataLoader.mixin import DataLoaderText2ImageMixin as text_to_image
from modules.util.concept_stats import init_concept_stats

# The >=1 side of preview's aspect set; both orientations become buckets.
PREVIEW_ASPECTS = [
    1.0, 1.05, 1.10, 1.15, 1.20, 1.25, 1.3333333333, 1.40, 1.50, 1.60, 1.6666666667,
    1.75, 1.7777777778, 1.85, 2.0, 2.2, 2.4, 2.5, 2.75, 3.0, 3.5, 4.0,
]


class BucketTableTest(unittest.TestCase):
    def test_table_has_the_extra_preview_buckets(self):
        self.assertEqual([w / h for h, w in AspectBucketing.all_possible_input_aspects], PREVIEW_ASPECTS)

    def test_training_loaders_and_concept_stats_share_the_table(self):
        self.assertIs(text_to_image.AspectBucketing, AspectBucketing)
        self.assertIs(text_to_image.SingleAspectCalculation, SingleAspectCalculation)
        self.assertIs(vae_loader.AspectBucketing, AspectBucketing)
        self.assertIs(vae_loader.SingleAspectCalculation, SingleAspectCalculation)
        expected = sorted({round(ratio, 6) for aspect in PREVIEW_ASPECTS for ratio in (aspect, 1 / aspect)})
        self.assertEqual(sorted(init_concept_stats(True)["aspect_buckets"]), expected)
        self.assertEqual(len(expected), 43)


if __name__ == "__main__":
    unittest.main()
