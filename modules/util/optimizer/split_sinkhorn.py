"""Sinkhorn normalization per sub-matrix of fused layers, for SinkSGD_adv and SINKSGD_ADAPTIVE.

Some layers stack several projections in one weight, e.g. MOSS Image's FFN up and gate. Normalizing
such a matrix as a whole lets the column norms mix the parts. CMuon (Chen, Sun, Yuan, 2026) reports
that Muon converges much better on DiTs once fused weights are orthogonalized per part; this applies
the same split to the square-root Sinkhorn normalization of adv-optm's SinkSGD.
"""
import importlib
import re

from modules.module.LoRAModule import LoHaModule, LoRAModule

from torch import Tensor, nn

SINKHORN_CHUNKS = "_sinkhorn_chunks"


def _split_parameters(model) -> list[tuple[nn.Parameter, int]]:
    """Parameters whose rows are the stacked outputs of a fused transformer layer, with the part count."""
    chunks_by_pattern = getattr(model, "fused_output_chunks", dict)()
    if not chunks_by_pattern:
        return []
    patterns = [(re.compile(pattern), chunks) for pattern, chunks in chunks_by_pattern.items()]

    def chunks_for(name: str) -> int | None:
        name = name.replace(".checkpoint.", ".").removeprefix("checkpoint.")
        return next((chunks for pattern, chunks in patterns if pattern.fullmatch(name)), None)

    found = []
    lora = getattr(model, "transformer_lora", None)
    if lora is not None:
        for name, module in lora.lora_modules.items():
            chunks = chunks_for(name)
            if chunks is None:
                continue
            if isinstance(module, LoRAModule):
                found.append((module.lora_up.weight, chunks))
            elif isinstance(module, LoHaModule):
                found += [(module.hada_w1_a, chunks), (module.hada_w2_a, chunks)]
    transformer = getattr(model, "transformer", None)
    if transformer is not None:
        for name, module in transformer.named_modules():
            if isinstance(module, nn.Linear) and (chunks := chunks_for(name)) is not None:
                found.append((module.weight, chunks))
    return found


def configure_split_sinkhorn(model, enabled: bool):
    """Mark the parameters of fused layers for per-part Sinkhorn normalization, or clear the marks.

    SINKSGD_ADAPTIVE reads the marks itself; SinkSGD_adv needs install_adv_split_sinkhorn as well.
    """
    split = _split_parameters(model) if model is not None else []
    for parameter, chunks in split:
        setattr(parameter, SINKHORN_CHUNKS, chunks if enabled else None)
    if not enabled:
        return
    if not any(parameter.requires_grad for parameter, _ in split):
        raise ValueError(
            "Split Fused Layers: none of the trained parameters belongs to a fused layer. The fused layers are "
            "known for MOSS Image (FFN up+gate, text-refiner QKV, timestep modulation); LoKr and OFT are not split."
        )


def install_adv_split_sinkhorn():
    # The package exports the optimizer class under the module's name; patch the module itself.
    SinkSGD_adv = importlib.import_module("adv_optm.optim.SinkSGD_adv")
    original = SinkSGD_adv.apply_sr_sinkhorn
    if getattr(original, "split_fused", False):
        return

    def apply_sr_sinkhorn(update: Tensor, iters: int = 5, p: Tensor | None = None, ortho_project: bool = False) -> Tensor:
        chunks = getattr(p, SINKHORN_CHUNKS, None) if p is not None else None
        if not chunks or update.dim() < 2:
            return original(update, iters=iters, p=p, ortho_project=ortho_project)
        if update.shape[0] % chunks:
            raise ValueError(f"Split Fused Layers: {update.shape[0]} rows do not split into {chunks} parts.")
        for update_part, p_part in zip(update.chunk(chunks), p.chunk(chunks), strict=True):
            update_part.copy_(original(update_part, iters=iters, p=p_part, ortho_project=ortho_project))
        return update

    apply_sr_sinkhorn.split_fused = True
    SinkSGD_adv.apply_sr_sinkhorn = apply_sr_sinkhorn

