"""Strict in-place NeoPP weight updates and closure receipts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import torch


def model_state(model) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    model.pre_weight.state_dict(state)
    model.transformer_weights.state_dict(state)
    if hasattr(model, "post_weight"):
        model.post_weight.state_dict(state)
    return {
        name: tensor
        for name, tensor in state.items()
        if isinstance(tensor, torch.Tensor)
        # RMSWeight initializes an inactive LoRA/diff branch with a scalar
        # zero.  It is runtime state rather than a checkpoint parameter and
        # must not make a strict full-checkpoint closure impossible.  A real
        # registered diff has the parameter's non-scalar shape and remains in
        # the closure.
        and not (name.endswith((".diff", ".diff_b")) and tensor.ndim == 0)
    }


def closure(model) -> dict[str, dict[str, object]]:
    """Describe the online-update closure without reading weights back to CPU.

    Shape and dtype metadata are sufficient for routing the already-loaded
    model.  The data plane never reads live weights back to CPU.
    """

    return {
        name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
        for name, tensor in sorted(model_state(model).items())
    }


def update_weights(model, tensors: Mapping[str, torch.Tensor] | Iterable[tuple[str, torch.Tensor]], strict: bool = False) -> dict[str, object]:
    incoming = dict(tensors)
    current = model_state(model)
    unknown = sorted(set(incoming) - set(current))
    if unknown and strict:
        raise KeyError(f"unknown NeoPP tensors: {unknown[:5]}")
    if strict:
        missing = sorted(set(current) - set(incoming))
        if missing:
            raise KeyError(f"missing NeoPP tensors: {missing[:5]}")
    updated: list[str] = []
    with torch.no_grad():
        for name, source in incoming.items():
            if name not in current:
                continue
            target = current[name]
            if tuple(source.shape) != tuple(target.shape):
                raise ValueError(f"shape mismatch for {name}: {tuple(source.shape)} != {tuple(target.shape)}")
            if source.dtype != target.dtype:
                raise ValueError(f"dtype mismatch for {name}: {source.dtype} != {target.dtype}")
            target.copy_(source.to(target.device), non_blocking=True)
            updated.append(name)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    # NeoPP MoE keeps fused expert stacks derived from the source tensors.
    for block in getattr(model.transformer_weights, "blocks", []):
        mlp = getattr(block, "mlp_mot_gen", None)
        if mlp is not None and hasattr(mlp, "rebuild_fused_weights"):
            mlp.rebuild_fused_weights()
    return {
        "updated": sorted(updated),
        "ignored": unknown,
        "closure_size": len(current),
    }
