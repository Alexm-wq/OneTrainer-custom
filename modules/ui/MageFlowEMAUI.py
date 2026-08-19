from __future__ import annotations


def _controller_config(controller):
    """Return the TrainConfig exposed by either OneTrainer tab-controller shape."""
    config = getattr(controller, "config", None)
    if config is None:
        config = getattr(controller, "train_config", None)
    if config is None:
        raise AttributeError(
            f"{type(controller).__name__} exposes neither 'config' nor 'train_config'"
        )
    return config


def add_mage_self_flow_ema_controls(components, master, row, controller, ui_state):
    """Add Mage-only Self-Flow EMA residency controls to the Training tab."""
    config = _controller_config(controller)
    if not config.model_type.is_mage_flow():
        return None

    frame = components.section_frame(master, row)
    components.label(
        frame,
        0,
        0,
        "Mage EMA Residency",
        tooltip=(
            "Controls where Mage's training-only EMA/shadow tensors live. "
            "GPU residency removes CPU<->GPU parameter traffic at the cost of additional VRAM."
        ),
        wide_tooltip=True,
    )

    components.label(
        frame,
        1,
        0,
        "Self-Flow EMA on GPU",
        tooltip=(
            "Keep the Self-Flow teacher EMA and student shadow on the training GPU. "
            "This removes per-teacher-swap and per-update PCIe transfers. The EMA and shadow are FP32, "
            "so a BF16/FP16 adapter uses roughly 4x its serialized adapter size in additional VRAM here."
        ),
        wide_tooltip=True,
    )
    components.switch(frame, 1, 1, ui_state, "self_flow_ema_on_gpu")
    return frame


def add_mage_linear_dpo_ema_controls(components, master, row, controller, ui_state):
    """Add Mage-only DPO/Self-Flow experiment and EMA residency controls."""
    config = _controller_config(controller)
    if not config.model_type.is_mage_flow():
        return None

    frame = components.section_frame(master, row)
    components.label(
        frame,
        0,
        0,
        "Mage DPO / EMA",
        tooltip=(
            "Mage-specific DPO execution and Linear-DPO EMA residency controls."
        ),
        wide_tooltip=True,
    )

    components.label(
        frame,
        1,
        0,
        "DPO Pair Uses Self-Flow",
        tooltip=(
            "When enabled, chosen/rejected DPO policy and reference scoring use the dual-timestep Self-Flow path. "
            "Disable this to make the entire DPO preference pair use ordinary flow matching while keeping the "
            "separate chosen Supervised Mix forward on normal Self-Flow. This is useful for testing whether "
            "Self-Flow reconstruction/representation pressure on rejected samples weakens preference separation."
        ),
        wide_tooltip=True,
    )
    components.switch(frame, 1, 1, ui_state, "rlhf_dpo_self_flow")

    components.label(
        frame,
        2,
        0,
        "Linear-DPO EMA on GPU",
        tooltip=(
            "Keep the Linear-DPO FP32 EMA reference and native-dtype policy restore stash on the training GPU. "
            "This removes reference-swap PCIe traffic. For a BF16/FP16 adapter, expect roughly 3x the adapter's "
            "serialized size in additional VRAM for this DPO state."
        ),
        wide_tooltip=True,
    )
    components.switch(frame, 2, 1, ui_state, "rlhf_dpo_linear_ema_on_gpu")
    return frame
