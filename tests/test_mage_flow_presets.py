import json
import os
import tempfile
import unittest
from pathlib import Path

from modules.ui.TopBarController import TopBarController


ROOT = Path(__file__).resolve().parents[1]
PRESET_DIR = ROOT / "training_presets" / "Mage Flow"
SMOKE = PRESET_DIR / "#mage-flow 5090 Smoke Self-Flow DPO.json"
RECOMMENDED = PRESET_DIR / "#mage-flow 5090 Recommended Self-Flow DPO.json"


class MageFlowPresetTests(unittest.TestCase):
    @staticmethod
    def _load(path: Path) -> dict:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _assert_common(self, data: dict):
        self.assertEqual(data["base_model_name"], "SceneWorks/Mage-Flow-Base")
        self.assertEqual(data["model_type"], "MAGE_FLOW")
        self.assertEqual(data["training_method"], "LORA")
        self.assertEqual(data["output_model_format"], "DIFFUSERS_LORA")
        self.assertEqual(data["train_dtype"], "BFLOAT_16")
        self.assertEqual(data["output_dtype"], "BFLOAT_16")
        self.assertEqual(data["attention_mechanism"], "FLASH")
        # Compile is intentionally opt-in until the new Mage block compile path
        # has been exercised on the target RTX 5090. The code now honors this
        # toggle; presets keep the conservative first-run setting.
        self.assertFalse(data["compile"])
        self.assertTrue(data["image_caching"])
        self.assertTrue(data["text_caching"])
        self.assertTrue(data["transformer"]["gradient_checkpointing"])
        self.assertFalse(data["text_encoder"]["train"])
        self.assertEqual(data["timestep_distribution"], "UNIFORM")
        self.assertEqual(data["timestep_shift"], 6.0)
        self.assertFalse(data["dynamic_timestep_shifting"])

        self.assertTrue(data["self_flow_enabled"])
        self.assertAlmostEqual(data["self_flow_mask_ratio"], 0.25)
        self.assertAlmostEqual(data["self_flow_rep_weight"], 0.8)
        self.assertTrue(data["self_flow_structural_enabled"])
        self.assertAlmostEqual(data["self_flow_structural_weight"], 0.1)

        self.assertTrue(data["rlhf_enabled"])
        self.assertEqual(data["rlhf_mode"], "DPO")
        self.assertEqual(data["rlhf_dpo_objective"], "LINEAR")
        self.assertAlmostEqual(data["rlhf_dpo_linear_eta"], 0.01)
        self.assertAlmostEqual(data["rlhf_dpo_linear_ema_decay"], 0.995)
        self.assertAlmostEqual(data["rlhf_supervised_mix"], 1.0)
        self.assertTrue(data["rlhf_dpo_momentum_bypass"])
        self.assertFalse(data["rlhf_dpo_adaptive_dataset"])
        self.assertFalse(data["rlhf_dpo_hard_pair_curriculum"])

    def test_smoke_preset(self):
        data = self._load(SMOKE)
        self._assert_common(data)
        self.assertEqual(data["resolution"], "512")
        self.assertEqual(data["batch_size"], 1)
        self.assertEqual(data["gradient_accumulation_steps"], 1)
        self.assertEqual(data["lora_rank"], 16)
        self.assertEqual(data["text_encoder_sequence_length"], 256)
        self.assertAlmostEqual(data["self_flow_ema_decay"], 0.99)
        self.assertEqual(data["self_flow_structural_tokens"], 64)
        self.assertAlmostEqual(data["rlhf_dpo_beta"], 30.0)
        self.assertEqual(data["learning_rate_warmup_steps"], 0.0)

    def test_recommended_preset(self):
        data = self._load(RECOMMENDED)
        self._assert_common(data)
        self.assertEqual(data["resolution"], "1024")
        self.assertEqual(data["batch_size"], 1)
        self.assertEqual(data["gradient_accumulation_steps"], 2)
        self.assertEqual(data["lora_rank"], 64)
        self.assertEqual(data["text_encoder_sequence_length"], 512)
        self.assertAlmostEqual(data["self_flow_ema_decay"], 0.9999)
        self.assertEqual(data["self_flow_structural_tokens"], 256)
        self.assertAlmostEqual(data["rlhf_dpo_beta"], 100.0)
        self.assertEqual(data["learning_rate_warmup_steps"], 200.0)

    def test_ui_discovers_mage_flow_presets_outside_repo_cwd(self):
        controller = TopBarController(None)
        previous_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                os.chdir(temp_dir)
                tree = controller.load_preset_tree()
            finally:
                os.chdir(previous_cwd)

        top_level = dict(tree)
        self.assertIn("Mage Flow", top_level)
        mage_flow = dict(top_level["Mage Flow"])
        self.assertIn("#mage-flow 5090 Smoke Self-Flow DPO", mage_flow)
        self.assertIn("#mage-flow 5090 Recommended Self-Flow DPO", mage_flow)
        self.assertEqual(Path(mage_flow["#mage-flow 5090 Smoke Self-Flow DPO"]), SMOKE)
        self.assertEqual(Path(mage_flow["#mage-flow 5090 Recommended Self-Flow DPO"]), RECOMMENDED)


if __name__ == "__main__":
    unittest.main()
