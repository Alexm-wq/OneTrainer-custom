import sys
import types
import unittest


# AdaptiveDPODataset only needs these MGDS base classes for the policy tests.
# Stub them so the test remains runnable in the lightweight CI environment.
mgds = types.ModuleType("mgds")
pipeline_module = types.ModuleType("mgds.PipelineModule")
pipeline_module_types = types.ModuleType("mgds.pipelineModuleTypes")
random_access = types.ModuleType(
    "mgds.pipelineModuleTypes.SingleVariationRandomAccessPipelineModule"
)


class _PipelineModule:
    def __init__(self):
        pass


class _SingleVariationRandomAccessPipelineModule:
    pass


pipeline_module.PipelineModule = _PipelineModule
random_access.SingleVariationRandomAccessPipelineModule = (
    _SingleVariationRandomAccessPipelineModule
)
sys.modules.setdefault("mgds", mgds)
sys.modules.setdefault("mgds.PipelineModule", pipeline_module)
sys.modules.setdefault("mgds.pipelineModuleTypes", pipeline_module_types)
sys.modules.setdefault(
    "mgds.pipelineModuleTypes.SingleVariationRandomAccessPipelineModule",
    random_access,
)

from modules.dataLoader.dpo.AdaptiveDPODataset import AdaptiveDPODataset
from modules.util.enum.DPOObjective import DPOObjective


class AdaptiveDPODatasetPolicyTest(unittest.TestCase):
    def test_runtime_policy_enforces_25_percent_keep_floor(self):
        dataset = AdaptiveDPODataset(
            names=[],
            min_keep_probability=0.0,
        )
        self.assertEqual(dataset.min_keep_probability, 0.25)
        self.assertEqual(dataset._keep_probability(0.0, 1.0), 0.25)
        self.assertGreaterEqual(dataset._keep_probability(0.01, 1.0), 0.25)

    def test_user_can_raise_but_not_lower_keep_floor(self):
        low = AdaptiveDPODataset(names=[], min_keep_probability=0.10)
        high = AdaptiveDPODataset(names=[], min_keep_probability=0.40)
        self.assertEqual(low.min_keep_probability, 0.25)
        self.assertEqual(high.min_keep_probability, 0.40)

    def test_three_observation_warmup_is_hard_minimum(self):
        dataset = AdaptiveDPODataset(
            names=[],
            min_observations=1,
            min_keep_probability=0.25,
            default_objective=DPOObjective.LINEAR,
        )
        self.assertEqual(dataset.min_observations, 3)

        key = dataset._build_pair_key("/tmp/chosen.png", "/tmp/rejected.png")
        objective = str(DPOObjective.LINEAR)

        dataset.observe([(key, 1.0, objective)])
        self.assertIsNone(dataset._eligible_loss(key, objective))
        dataset.observe([(key, 1.0, objective)])
        self.assertIsNone(dataset._eligible_loss(key, objective))
        dataset.observe([(key, 1.0, objective)])
        self.assertIsNotNone(dataset._eligible_loss(key, objective))

    def test_user_can_raise_warmup_above_three(self):
        dataset = AdaptiveDPODataset(names=[], min_observations=5)
        self.assertEqual(dataset.min_observations, 5)


if __name__ == "__main__":
    unittest.main()
