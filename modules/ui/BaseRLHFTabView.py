from modules.util.enum.DPOObjective import DPOObjective
from modules.util.enum.DPORefMode import DPORefMode
from modules.util.enum.RLHFMode import RLHFMode


class BaseRLHFTabView:
    def __init__(self, components):
        self.components = components

    def build_content(self, frame, controller, ui_state):
        core = self.components.section_frame(frame, 0)
        mode_widgets = {}
        mode_frames = {}
        ref_var = ui_state.get_var("rlhf_dpo_ref_mode")
        try:
            initial_ref_mode = DPORefMode(ref_var.get())
        except ValueError:
            initial_ref_mode = DPORefMode.NEW_ADAPTER
        last_fixed_ref_mode = {
            "value": (
                initial_ref_mode
                if initial_ref_mode != DPORefMode.EMA_ADAPTER
                else DPORefMode.NEW_ADAPTER
            )
        }

        def set_value(name, value):
            ui_state.get_var(name).set(
                value if isinstance(value, bool) else str(value)
            )

        def refresh_reference(value):
            reference_mode = DPORefMode(value)
            objective = DPOObjective(
                ui_state.get_var("rlhf_dpo_objective").get()
            )
            if (
                reference_mode == DPORefMode.EMA_ADAPTER
                and objective != DPOObjective.LINEAR
            ):
                # Run after the option-menu's own selection callback returns;
                # otherwise both CTK and PySide suppress the nested variable
                # notification and leave the forbidden EMA choice displayed.
                fallback = str(last_fixed_ref_mode["value"])
                self.components.call_after(
                    core,
                    0,
                    lambda: ref_var.set(fallback),
                )
            elif reference_mode != DPORefMode.EMA_ADAPTER:
                last_fixed_ref_mode["value"] = reference_mode

        def refresh_objective(value=None):
            objective = DPOObjective(
                value
                if value is not None
                else ui_state.get_var("rlhf_dpo_objective").get()
            )
            linear = objective == DPOObjective.LINEAR
            sigmoid = objective == DPOObjective.SIGMOID
            ipo = objective == DPOObjective.IPO
            anchored = objective == DPOObjective.ANCHORED_REJECT
            balanced = objective == DPOObjective.BALANCED_REJECT

            current_ref = DPORefMode(ref_var.get())
            if linear:
                if current_ref != DPORefMode.EMA_ADAPTER:
                    last_fixed_ref_mode["value"] = current_ref
                if current_ref != DPORefMode.EMA_ADAPTER:
                    ref_var.set(str(DPORefMode.EMA_ADAPTER))

                # Disabled settings are also normalized so saving immediately
                # after an objective change produces a valid CLI/JSON config.
                set_value("rlhf_dpo_label_smoothing", 0.0)
                set_value("rlhf_dpo_adaptive_beta", False)
                set_value("rlhf_dpo_beta_gradient_decouple", False)
                set_value("rlhf_dpo_chosen_reward_anchor", False)
            elif current_ref == DPORefMode.EMA_ADAPTER:
                ref_var.set(str(last_fixed_ref_mode["value"]))

            enabled_by_name = {
                "reference": not linear,
                "beta": sigmoid or linear,
                "beta_gradient_decouple": sigmoid,
                "beta_gradient_reference": sigmoid,
                "label_smoothing": sigmoid,
                "ipo_tau": ipo,
                "adaptive_beta": sigmoid,
                "adaptive_dataset": True,
                "linear_eta": linear,
                "linear_ema_decay": linear,
            }
            for name, enabled in enabled_by_name.items():
                widget = mode_widgets.get(name)
                if widget is not None:
                    self.components.set_widget_enabled(widget, enabled)

            anchored_frame = mode_frames.get("anchored")
            if anchored_frame is not None:
                self.components.set_widget_enabled(anchored_frame, anchored)
            balanced_frame = mode_frames.get("balanced")
            if balanced_frame is not None:
                self.components.set_widget_enabled(balanced_frame, balanced)
            anchor_frame = mode_frames.get("anchor")
            if anchor_frame is not None:
                self.components.set_widget_enabled(
                    anchor_frame,
                    sigmoid or ipo,
                )

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
                "IPO targets a fixed reward margin. Anchored Reject keeps its "
                "existing anchor/margin behavior. Balanced Reject trains chosen "
                "normally and uses detached chosen improvement to budget a "
                "rejected-only suppression target. Linear DPO uses squared "
                "flow-error differences and a moving EMA adapter reference."
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
                ("Balanced Reject", DPOObjective.BALANCED_REJECT),
                ("Linear DPO", DPOObjective.LINEAR),
            ],
            ui_state,
            "rlhf_dpo_objective",
            command=refresh_objective,
        )

        self.components.label(
            core,
            3,
            0,
            "Reference Mode",
            tooltip=(
                "New Adapter uses the base model as reference. Existing Adapter "
                "uses the fixed adapter snapshot saved with OT backups. Linear "
                "DPO forces its separate CPU EMA reference."
            ),
        )
        mode_widgets["reference"] = self.components.options_kv(
            core,
            3,
            1,
            [
                ("New Adapter / Base Reference", DPORefMode.NEW_ADAPTER),
                ("Existing Adapter Snapshot", DPORefMode.EXISTING_ADAPTER),
                ("Linear-DPO EMA / CPU", DPORefMode.EMA_ADAPTER),
            ],
            ui_state,
            "rlhf_dpo_ref_mode",
            command=refresh_reference,
        )

        self.components.label(core, 4, 0, "Beta")
        mode_widgets["beta"] = self.components.entry(
            core, 4, 1, ui_state, "rlhf_dpo_beta"
        )

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
        mode_widgets["beta_gradient_decouple"] = self.components.switch(
            core,
            5,
            1,
            ui_state,
            "rlhf_dpo_beta_gradient_decouple",
        )

        self.components.label(core, 6, 0, "Beta Gradient Reference")
        mode_widgets["beta_gradient_reference"] = self.components.entry(
            core,
            6,
            1,
            ui_state,
            "rlhf_dpo_beta_gradient_reference",
        )

        self.components.label(core, 7, 0, "Label Smoothing")
        mode_widgets["label_smoothing"] = self.components.entry(
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
            tooltip=(
                "Weight of an additional chosen-image normal-training loss. "
                "With Flux2 Self-Flow enabled, this is the full normal "
                "Self-Flow objective (generation + representation + optional "
                "structural loss); rejected remains DPO-only. Anchored Reject "
                "and Balanced Reject always use chosen weight 1.0 regardless "
                "of this setting."
            ),
        )
        self.components.entry(core, 8, 1, ui_state, "rlhf_supervised_mix")

        self.components.label(core, 9, 0, "IPO Tau")
        mode_widgets["ipo_tau"] = self.components.entry(
            core, 9, 1, ui_state, "rlhf_dpo_ipo_tau"
        )

        self.components.label(
            core,
            10,
            0,
            "Linear Eta",
            tooltip=(
                "Minimum detached Linear-DPO utility. The paper default is "
                "0.01."
            ),
        )
        mode_widgets["linear_eta"] = self.components.entry(
            core,
            10,
            1,
            ui_state,
            "rlhf_dpo_linear_eta",
        )

        self.components.label(
            core,
            11,
            0,
            "Linear EMA Decay",
            tooltip=(
                "Decay of Linear-DPO's separate CPU FP32 adapter reference. "
                "The paper default is 0.995."
            ),
        )
        mode_widgets["linear_ema_decay"] = self.components.entry(
            core,
            11,
            1,
            ui_state,
            "rlhf_dpo_linear_ema_decay",
        )

        self.components.label(
            core,
            12,
            0,
            "Adaptive Beta",
            tooltip="Adjust beta dynamically from observed DPO saturation.",
        )
        mode_widgets["adaptive_beta"] = self.components.switch(
            core,
            12,
            1,
            ui_state,
            "rlhf_dpo_adaptive_beta",
        )

        self.components.label(
            core,
            13,
            0,
            "Adaptive DPO Dataset",
            tooltip=(
                "Use a saved EMA of pair difficulty to avoid wasting slots on "
                "already-solved examples. Linear-DPO uses a non-negative direct "
                "policy ranking difficulty rather than its signed training loss. "
                "Replacement pools are separated by objective, and unknown pairs "
                "are kept until they have observations."
            ),
        )
        mode_widgets["adaptive_dataset"] = self.components.switch(
            core,
            13,
            1,
            ui_state,
            "rlhf_dpo_adaptive_dataset",
        )

        self.components.label(
            core,
            14,
            0,
            "No Momentum DPO",
            tooltip=(
                "Keep DPO gradients out of the optimizer's persistent momentum. "
                "Normal training gradients continue to use optimizer momentum; "
                "DPO gradients are accumulated separately and applied through "
                "the momentum-free DPO update path. Disable this to use the "
                "optimizer's normal momentum for DPO as well."
            ),
        )
        self.components.switch(
            core,
            14,
            1,
            ui_state,
            "rlhf_dpo_momentum_bypass",
        )

        self.components.label(
            core,
            15,
            0,
            "DPO Gradient Scale",
            tooltip=(
                "Multiplies only the DPO-side backward gradient while preserving "
                "the exact forward rewards, margins, losses, curriculum values, "
                "and normal chosen/Self-Flow supervision. 1.0 is the previous "
                "strength; 0.25 gives quarter-strength preference gradients."
            ),
            wide_tooltip=True,
        )
        self.components.entry(
            core,
            15,
            1,
            ui_state,
            "rlhf_dpo_gradient_scale",
        )

        anchored_reject = self.components.section_frame(frame, 1)
        mode_frames["anchored"] = anchored_reject

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

        balanced_reject = self.components.section_frame(frame, 2)
        mode_frames["balanced"] = balanced_reject

        self.components.label(
            balanced_reject,
            0,
            0,
            "Balanced Reject",
            tooltip=(
                "Chosen receives one full normal Self-Flow loss. The preference "
                "gradient is rejected-only. Positive detached chosen reward sets "
                "the rejected suppression budget; if chosen reward is <= 0, "
                "rejected is only pushed back to the fixed-reference level."
            ),
            wide_tooltip=True,
        )

        self.components.label(
            balanced_reject,
            1,
            0,
            "Reject Balance",
            tooltip=(
                "Rejected target magnitude relative to positive detached chosen "
                "reward. 1.0 means chosen +0.10 budgets rejected to -0.10."
            ),
        )
        self.components.entry(
            balanced_reject,
            1,
            1,
            ui_state,
            "rlhf_dpo_balanced_reject_ratio",
        )

        self.components.label(
            balanced_reject,
            2,
            0,
            "Reject Weight",
            tooltip=(
                "Overall strength of the rejected-only Smooth-L1 objective. "
                "Chosen Self-Flow remains weight 1.0."
            ),
        )
        self.components.entry(
            balanced_reject,
            2,
            1,
            ui_state,
            "rlhf_dpo_balanced_reject_weight",
        )

        self.components.label(
            balanced_reject,
            3,
            0,
            "Huber Delta",
            tooltip="Smooth-L1 transition point for rejected-target violations.",
        )
        self.components.entry(
            balanced_reject,
            3,
            1,
            ui_state,
            "rlhf_dpo_balanced_huber_delta",
        )

        curriculum = self.components.section_frame(frame, 3)
        self.components.label(
            curriculum,
            0,
            0,
            "DPO Pair Curriculum",
            tooltip=(
                "Confidence-gate each pair's selected objective using a saved "
                "EMA of objective-appropriate competence. Sigmoid and "
                "Anchored Reject and Balanced Reject use reward margin; IPO respects its finite "
                "target; Linear-DPO uses the policy's direct chosen/rejected "
                "score gap so EMA-reference catch-up does not lower a good "
                "pair's weight. Applies to per-concept objective dispatches."
            ),
            wide_tooltip=True,
        )
        self.components.switch(
            curriculum,
            0,
            1,
            ui_state,
            "rlhf_dpo_hard_pair_curriculum",
        )

        self.components.label(curriculum, 1, 0, "Curriculum EMA")
        self.components.entry(
            curriculum,
            1,
            1,
            ui_state,
            "rlhf_dpo_hard_pair_curriculum_ema",
        )

        self.components.label(curriculum, 2, 0, "Minimum Weight")
        self.components.entry(
            curriculum,
            2,
            1,
            ui_state,
            "rlhf_dpo_hard_pair_curriculum_min_weight",
        )

        self.components.label(
            curriculum,
            3,
            0,
            "Full Competence",
            tooltip=(
                "EMA competence at which a pair reaches full objective "
                "weight. Values at or below zero use Minimum Weight. For IPO "
                "this is capped at IPO's theoretical target margin. For Linear "
                "Adaptive Dataset it also sets the softness scale of the direct "
                "chosen/rejected ranking difficulty."
            ),
        )
        self.components.entry(
            curriculum,
            3,
            1,
            ui_state,
            "rlhf_dpo_hard_pair_curriculum_full_margin",
        )

        anchor = self.components.section_frame(frame, 4)
        mode_frames["anchor"] = anchor

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

        validation = self.components.section_frame(frame, 5)

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

        diagnostics = self.components.section_frame(frame, 6)
        self.components.label(
            diagnostics,
            0,
            0,
            "Bad Pair CSV",
            tooltip=(
                "Write severe pair outliers to dpo_bad_pairs.csv for any DPO "
                "objective. This does not add TensorBoard metrics."
            ),
        )
        self.components.switch(
            diagnostics,
            0,
            1,
            ui_state,
            "rlhf_dpo_bad_pair_logging",
        )
        self.components.label(diagnostics, 1, 0, "Bad Reward Violation")
        self.components.entry(
            diagnostics,
            1,
            1,
            ui_state,
            "rlhf_dpo_bad_pair_reward_violation_threshold",
        )
        self.components.label(diagnostics, 2, 0, "Bad Reward Change")
        self.components.entry(
            diagnostics,
            2,
            1,
            ui_state,
            "rlhf_dpo_bad_pair_reward_change_threshold",
        )
        self.components.label(diagnostics, 3, 0, "Bad Pair Loss")
        self.components.entry(
            diagnostics,
            3,
            1,
            ui_state,
            "rlhf_dpo_bad_pair_loss_threshold",
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

        # Apply initial compatibility state after every target widget/frame
        # exists. Subsequent objective changes run through the same callback.
        refresh_objective()
