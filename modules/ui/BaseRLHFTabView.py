from modules.util.enum.DPOObjective import DPOObjective
from modules.util.enum.DPORefMode import DPORefMode
from modules.util.enum.RLHFMode import RLHFMode


class BaseRLHFTabView:
    def __init__(self, components):
        self.components = components

    def build_content(self, frame, controller, ui_state):
        core = self.components.section_frame(frame, 0)

        self.components.label(
            core,
            0,
            0,
            "Enable RLHF",
            tooltip="Enable mixed normal and paired preference training.",
        )
        self.components.switch(core, 0, 1, ui_state, "rlhf_enabled")

        self.components.label(core, 1, 0, "RLHF Mode")
        self.components.options_kv(
            core,
            1,
            1,
            [("DPO", RLHFMode.DPO)],
            ui_state,
            "rlhf_mode",
        )

        self.components.label(
            core,
            2,
            0,
            "Objective",
            tooltip=(
                "DPO / Sigmoid is the standard two-sided preference objective. "
                "IPO targets a fixed reward margin. Anchored Reject independently "
                "pushes chosen reward above its target and rejected reward below "
                "its target, without a margin loss."
            ),
        )
        self.components.options_kv(
            core,
            2,
            1,
            [
                ("DPO / Sigmoid", DPOObjective.SIGMOID),
                ("IPO", DPOObjective.IPO),
                ("Anchored Reject", DPOObjective.ANCHORED_REJECT),
            ],
            ui_state,
            "rlhf_dpo_objective",
        )

        self.components.label(
            core,
            3,
            0,
            "Reference Mode",
            tooltip=(
                "New Adapter uses the base model as reference. Existing Adapter "
                "uses the fixed adapter snapshot saved with OT backups."
            ),
        )
        self.components.options_kv(
            core,
            3,
            1,
            [
                ("New Adapter / Base Reference", DPORefMode.NEW_ADAPTER),
                ("Existing Adapter Snapshot", DPORefMode.EXISTING_ADAPTER),
            ],
            ui_state,
            "rlhf_dpo_ref_mode",
        )

        self.components.label(core, 4, 0, "Beta")
        self.components.entry(core, 4, 1, ui_state, "rlhf_dpo_beta")

        self.components.label(
            core,
            5,
            0,
            "Beta Gradient Decouple",
            tooltip=(
                "Keeps beta's sigmoid saturation behavior while separately "
                "controlling the backward gradient scale."
            ),
        )
        self.components.switch(
            core,
            5,
            1,
            ui_state,
            "rlhf_dpo_beta_gradient_decouple",
        )

        self.components.label(core, 6, 0, "Beta Gradient Reference")
        self.components.entry(
            core,
            6,
            1,
            ui_state,
            "rlhf_dpo_beta_gradient_reference",
        )

        self.components.label(core, 7, 0, "Label Smoothing")
        self.components.entry(
            core,
            7,
            1,
            ui_state,
            "rlhf_dpo_label_smoothing",
        )

        self.components.label(
            core,
            8,
            0,
            "Supervised Mix",
            tooltip="Adds chosen-image supervised loss to the preference loss.",
        )
        self.components.entry(core, 8, 1, ui_state, "rlhf_supervised_mix")

        self.components.label(core, 9, 0, "IPO Tau")
        self.components.entry(core, 9, 1, ui_state, "rlhf_dpo_ipo_tau")

        self.components.label(
            core,
            10,
            0,
            "Adaptive Beta",
            tooltip="Adjust beta dynamically from observed DPO saturation.",
        )
        self.components.switch(
            core,
            10,
            1,
            ui_state,
            "rlhf_dpo_adaptive_beta",
        )

        anchored_reject = self.components.section_frame(frame, 1)

        self.components.label(
            anchored_reject,
            0,
            0,
            "Anchored Reject",
            tooltip=(
                "Independent one-sided chosen/rejected targets using Smooth-L1. "
                "No explicit reward-margin loss is used."
            ),
        )

        self.components.label(anchored_reject, 1, 0, "Chosen Target")
        self.components.entry(
            anchored_reject,
            1,
            1,
            ui_state,
            "rlhf_dpo_anchored_chosen_target",
        )

        self.components.label(anchored_reject, 2, 0, "Rejected Target")
        self.components.entry(
            anchored_reject,
            2,
            1,
            ui_state,
            "rlhf_dpo_anchored_rejected_target",
        )

        self.components.label(anchored_reject, 3, 0, "Chosen Weight")
        self.components.entry(
            anchored_reject,
            3,
            1,
            ui_state,
            "rlhf_dpo_anchored_chosen_weight",
        )

        self.components.label(anchored_reject, 4, 0, "Rejected Weight")
        self.components.entry(
            anchored_reject,
            4,
            1,
            ui_state,
            "rlhf_dpo_anchored_rejected_weight",
        )

        self.components.label(
            anchored_reject,
            5,
            0,
            "Huber Delta",
            tooltip=(
                "Smooth-L1 transition point. Large reward violations retain a "
                "bounded slope instead of becoming quadratic explosions."
            ),
        )
        self.components.entry(
            anchored_reject,
            5,
            1,
            ui_state,
            "rlhf_dpo_anchored_huber_delta",
        )

        self.components.label(
            anchored_reject,
            6,
            0,
            "Margin Target",
            tooltip=(
                "Require chosen_reward - rejected_reward to reach this value. "
                "The penalty is Smooth-L1 bounded by Huber Delta."
            ),
        )
        self.components.entry(
            anchored_reject,
            6,
            1,
            ui_state,
            "rlhf_dpo_anchored_margin_target",
        )

        self.components.label(
            anchored_reject,
            7,
            0,
            "Margin Weight",
            tooltip=(
                "Weight of the positive target-margin penalty. With the hard-"
                "pair curriculum enabled, this term is confidence-scaled."
            ),
        )
        self.components.entry(
            anchored_reject,
            7,
            1,
            ui_state,
            "rlhf_dpo_anchored_margin_weight",
        )

        self.components.label(
            anchored_reject,
            8,
            0,
            "Wrong-Order Weight",
            tooltip=(
                "Additional rescue penalty while rejected_reward is greater "
                "than chosen_reward. It is also confidence-scaled."
            ),
        )
        self.components.entry(
            anchored_reject,
            8,
            1,
            ui_state,
            "rlhf_dpo_anchored_wrong_order_weight",
        )

        self.components.label(
            anchored_reject,
            9,
            0,
            "Hard-Pair Curriculum",
            tooltip=(
                "Reduce the entire Anchored Reject gradient for close or "
                "incorrectly ranked pairs. Per-pair EMA state is saved in "
                "backups and restored exactly."
            ),
        )
        self.components.switch(
            anchored_reject,
            9,
            1,
            ui_state,
            "rlhf_dpo_hard_pair_curriculum",
        )

        self.components.label(anchored_reject, 10, 0, "Curriculum EMA")
        self.components.entry(
            anchored_reject,
            10,
            1,
            ui_state,
            "rlhf_dpo_hard_pair_curriculum_ema",
        )

        self.components.label(anchored_reject, 11, 0, "Minimum Weight")
        self.components.entry(
            anchored_reject,
            11,
            1,
            ui_state,
            "rlhf_dpo_hard_pair_curriculum_min_weight",
        )

        self.components.label(anchored_reject, 12, 0, "Full Margin")
        self.components.entry(
            anchored_reject,
            12,
            1,
            ui_state,
            "rlhf_dpo_hard_pair_curriculum_full_margin",
        )

        self.components.label(
            anchored_reject,
            13,
            0,
            "Bad Pair CSV",
            tooltip=(
                "Write only severe pair outliers to dpo_bad_pairs.csv. This does "
                "not add TensorBoard metrics."
            ),
        )
        self.components.switch(
            anchored_reject,
            13,
            1,
            ui_state,
            "rlhf_dpo_bad_pair_logging",
        )

        self.components.label(anchored_reject, 14, 0, "Bad Reward Violation")
        self.components.entry(
            anchored_reject,
            14,
            1,
            ui_state,
            "rlhf_dpo_bad_pair_reward_violation_threshold",
        )

        self.components.label(anchored_reject, 15, 0, "Bad Reward Change")
        self.components.entry(
            anchored_reject,
            15,
            1,
            ui_state,
            "rlhf_dpo_bad_pair_reward_change_threshold",
        )

        self.components.label(anchored_reject, 16, 0, "Bad Pair Loss")
        self.components.entry(
            anchored_reject,
            16,
            1,
            ui_state,
            "rlhf_dpo_bad_pair_loss_threshold",
        )

        anchor = self.components.section_frame(frame, 2)

        self.components.label(
            anchor,
            0,
            0,
            "Chosen Reward Anchor",
            tooltip=(
                "Adds chosen-side protection after the normal two-sided DPO "
                "objective. The rejected policy term remains active."
            ),
        )
        self.components.switch(
            anchor,
            0,
            1,
            ui_state,
            "rlhf_dpo_chosen_reward_anchor",
        )

        self.components.label(anchor, 1, 0, "Anchor Weight")
        self.components.entry(
            anchor,
            1,
            1,
            ui_state,
            "rlhf_dpo_chosen_reward_anchor_weight",
        )

        self.components.label(anchor, 2, 0, "Chosen Target")
        self.components.entry(
            anchor,
            2,
            1,
            ui_state,
            "rlhf_dpo_chosen_reward_target",
        )

        self.components.label(anchor, 3, 0, "Chosen Floor")
        self.components.entry(
            anchor,
            3,
            1,
            ui_state,
            "rlhf_dpo_chosen_reward_floor",
        )

        self.components.label(anchor, 4, 0, "Floor Multiplier")
        self.components.entry(
            anchor,
            4,
            1,
            ui_state,
            "rlhf_dpo_chosen_reward_floor_multiplier",
        )

        self.components.label(anchor, 5, 0, "Anchor Sharpness")
        self.components.entry(
            anchor,
            5,
            1,
            ui_state,
            "rlhf_dpo_chosen_reward_sharpness",
        )

        validation = self.components.section_frame(frame, 3)

        self.components.label(
            validation,
            0,
            0,
            "DPO Validation",
            tooltip="Reserve configured DPO pairs for validation.",
        )
        self.components.switch(
            validation,
            0,
            1,
            ui_state,
            "rlhf_dpo_validation",
        )

        self.components.label(validation, 1, 0, "Validation Percentage")
        self.components.entry(
            validation,
            1,
            1,
            ui_state,
            "rlhf_dpo_validation_percentage",
        )

        self.components.label(validation, 2, 0, "Patience Enabled")
        self.components.switch(
            validation,
            2,
            1,
            ui_state,
            "rlhf_dpo_patience_enabled",
        )

        self.components.label(validation, 3, 0, "Patience")
        self.components.entry(
            validation,
            3,
            1,
            ui_state,
            "rlhf_dpo_patience_value",
        )

        self.components.label(validation, 4, 0, "Save Best")
        self.components.switch(
            validation,
            4,
            1,
            ui_state,
            "rlhf_dpo_save_best",
        )

        self.components.label(
            validation,
            5,
            0,
            "Timestep Margin Logging",
            tooltip="Log DPO reward margins grouped by timestep.",
        )
        self.components.switch(
            validation,
            5,
            1,
            ui_state,
            "rlhf_dpo_timestep_margin_logging",
        )

        # OT_RLHF_PAIR_TOOLS_V1
        tools = self.components.section_frame(frame, 99)

        self.components.label(
            tools,
            0,
            0,
            "DPO Dataset Tools",
            tooltip=(
                "Utilities from the original RLHF PR for validating, reviewing, "
                "repairing, and bucket-checking chosen/rejected pairs."
            ),
        )

        self.components.button(
            tools,
            1,
            0,
            "Check Pairs",
            command=controller.check_pairs,
            tooltip=(
                "Validate pair keys, detect and optionally remove strays, "
                "flatten multiline captions, and check caption mismatches."
            ),
        )

        self.components.button(
            tools,
            1,
            1,
            "Review Pairs",
            command=controller.review_pairs,
            tooltip=(
                "Visually inspect chosen/rejected images side by side and "
                "remove bad or orphaned pairs."
            ),
        )

        self.components.button(
            tools,
            2,
            0,
            "Re-pair by Similarity",
            command=controller.repair_rejected,
            tooltip=(
                "Use DINOv2 similarity to reassign rejected images inside "
                "caption groups. Rejected files are renamed in place."
            ),
        )

        self.components.button(
            tools,
            2,
            1,
            "DPO Bucket Analysis",
            command=controller.bucket_analysis,
            tooltip=(
                "Show pair counts by aspect bucket and additions/removals "
                "needed for clean batch multiples."
            ),
        )
