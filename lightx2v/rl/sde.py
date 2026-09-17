"""Hybrid ODE/SDE sampling used by SenseNova-U1.5 image rollouts.

The public NeoPP sampler advances flow time from 0 (noise) to 1 (data).  The
RL transition follows the Flow-GRPO convention, so ``sigma = 1 - t`` and the
NeoPP velocity is negated before applying the reverse-SDE formula.  All
likelihood arithmetic is deliberately float32.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable

import torch

_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)


def select_sde_indices(total_steps: int, window_start: int, window_end: int, selected_steps: int) -> tuple[int, ...]:
    if not 0 <= window_start < window_end <= total_steps:
        raise ValueError("SDE window must be a non-empty subset of total steps")
    width = window_end - window_start
    if not 1 <= selected_steps <= width:
        raise ValueError("selected SDE step count exceeds its window")
    if selected_steps == 1:
        return (window_start,)
    result = tuple(window_start + (position * (width - 1)) // (selected_steps - 1) for position in range(selected_steps))
    if len(set(result)) != selected_steps:
        raise RuntimeError("SDE window selection produced duplicate steps")
    return result


@dataclass(frozen=True)
class SdeRolloutConfig:
    noise_level: float = 0.7
    t_eps: float = 0.02
    window_start: int = 0
    window_end: int | None = None
    selected_steps: int | None = None
    indices: tuple[int, ...] | None = None

    def resolve_indices(self, total_steps: int) -> tuple[int, ...]:
        if not math.isfinite(self.noise_level) or self.noise_level <= 0:
            raise ValueError("noise_level must be a positive finite value")
        if not math.isfinite(self.t_eps) or not 0 < self.t_eps < 1:
            raise ValueError("t_eps must lie inside (0, 1)")
        if self.indices is not None:
            indices = tuple(int(index) for index in self.indices)
            if not indices or len(indices) != len(set(indices)):
                raise ValueError("indices must contain distinct SDE steps")
            if min(indices) < 0 or max(indices) >= total_steps:
                raise ValueError("indices contain an out-of-range SDE step")
            return tuple(sorted(indices))
        end = total_steps if self.window_end is None else self.window_end
        selected = end - self.window_start if self.selected_steps is None else self.selected_steps
        return select_sde_indices(total_steps, self.window_start, end, selected)


@dataclass
class SdeTrace:
    samples: list[torch.Tensor] = field(default_factory=list)
    next_samples: list[torch.Tensor] = field(default_factory=list)
    old_means: list[torch.Tensor] = field(default_factory=list)
    old_log_probs: list[torch.Tensor] = field(default_factory=list)
    timesteps: list[torch.Tensor] = field(default_factory=list)
    next_timesteps: list[torch.Tensor] = field(default_factory=list)
    scales: list[torch.Tensor] = field(default_factory=list)
    indices: list[int] = field(default_factory=list)
    final_latent: torch.Tensor | None = None
    sigma_max: float | None = None
    noise_level: float | None = None

    def as_tensors(self) -> dict[str, torch.Tensor]:
        if not self.samples or self.final_latent is None:
            raise ValueError("cannot serialize an empty SDE trace")
        return {
            "samples": torch.stack(self.samples).detach().cpu(),
            "next_samples": torch.stack(self.next_samples).detach().cpu(),
            "old_means": torch.stack(self.old_means).detach().cpu(),
            "old_log_probs": torch.stack(self.old_log_probs).detach().cpu(),
            "timesteps": torch.stack(self.timesteps).detach().float().cpu(),
            "next_timesteps": torch.stack(self.next_timesteps).detach().float().cpu(),
            "scales": torch.stack(self.scales).detach().cpu(),
            "indices": torch.tensor(self.indices, dtype=torch.int64),
            "final_latent": self.final_latent.detach().cpu(),
            "sigma_max": torch.tensor(float(self.sigma_max), dtype=torch.float32),
            "noise_level": torch.tensor(float(self.noise_level), dtype=torch.float32),
        }


def _broadcast(value: torch.Tensor | float, like: torch.Tensor) -> torch.Tensor:
    tensor = torch.as_tensor(value, device=like.device, dtype=like.dtype)
    if tensor.ndim == 0:
        return tensor.reshape(*([1] * like.ndim))
    if tensor.ndim == 1:
        if tensor.shape[0] != like.shape[0]:
            raise ValueError("per-sample timestep has the wrong batch size")
        return tensor.reshape(-1, *([1] * (like.ndim - 1)))
    return tensor


def transition(
    velocity: torch.Tensor,
    sample: torch.Tensor,
    t: torch.Tensor | float,
    t_next: torch.Tensor | float,
    sigma_max: float,
    noise_level: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if velocity.shape != sample.shape:
        raise ValueError("velocity and sample shapes differ")
    if not 0 < float(sigma_max) < 1:
        raise ValueError("sigma_max must lie inside (0, 1)")
    sample = sample.float()
    model_output = -velocity.float()
    sigma = _broadcast(1.0 - torch.as_tensor(t), sample)
    sigma_prev = _broadcast(1.0 - torch.as_tensor(t_next), sample)
    dt = sigma_prev - sigma
    if bool((dt >= 0).any()):
        raise ValueError("NeoPP flow timesteps must be strictly increasing")
    sigma_max_tensor = sample.new_tensor(float(sigma_max))
    std = torch.sqrt(sigma / (1 - torch.where(sigma == 1, sigma_max_tensor, sigma)))
    std = std * float(noise_level)
    mean = sample * (1 + std.square() / (2 * sigma) * dt) + model_output * (1 + std.square() * (1 - sigma) / (2 * sigma)) * dt
    return mean, std * torch.sqrt(-dt)


def recompute_log_prob(next_sample: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    next_sample = next_sample.detach().float()
    mean = mean.float()
    scale = scale.float()
    if next_sample.shape != mean.shape or not bool((scale > 0).all()):
        raise ValueError("invalid SDE likelihood geometry")
    value = -((next_sample - mean).square()) / (2 * scale.square()) - torch.log(scale) - _LOG_SQRT_2PI
    return value.expand_as(mean).mean(dim=tuple(range(1, mean.ndim)))


class HybridSdeState:
    """Mutable per-generation state attached to a NeoPP scheduler."""

    def __init__(self, config: SdeRolloutConfig, timesteps: torch.Tensor, seed: int | None, device: torch.device):
        if timesteps.ndim != 1 or timesteps.numel() < 2:
            raise ValueError("timesteps must contain at least two values")
        if not torch.isclose(timesteps[0].float(), timesteps.new_tensor(0.0).float(), atol=1e-7, rtol=0):
            raise ValueError("NeoPP RL rollout must start at t=0")
        if not torch.isclose(timesteps[-1].float(), timesteps.new_tensor(1.0).float(), atol=1e-7, rtol=0):
            raise ValueError("NeoPP RL rollout must end at t=1")
        self.indices = set(config.resolve_indices(timesteps.numel() - 1))
        self.noise_level = config.noise_level
        self.sigma_max = float(1.0 - timesteps[1].float())
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(0 if seed is None else int(seed))
        self.trace = SdeTrace(sigma_max=self.sigma_max, noise_level=self.noise_level)

    def advance(self, sample: torch.Tensor, velocity: torch.Tensor, t: torch.Tensor, t_next: torch.Tensor, index: int) -> torch.Tensor:
        if index not in self.indices:
            return (sample.float() + (t_next.float() - t.float()) * velocity.float()).to(sample.dtype)
        mean, scale = transition(velocity, sample, t, t_next, self.sigma_max, self.noise_level)
        noise = torch.randn(mean.shape, generator=self.generator, device=mean.device, dtype=torch.float32)
        next_sample = (mean + scale * noise).to(sample.dtype)
        self.trace.samples.append(sample.float().detach())
        self.trace.next_samples.append(next_sample.detach())
        self.trace.old_means.append(mean.detach())
        self.trace.old_log_probs.append(recompute_log_prob(next_sample.float(), mean, scale).detach())
        self.trace.timesteps.append(t.float().detach())
        self.trace.next_timesteps.append(t_next.float().detach())
        # ``scale`` is broadcastable over the latent.  Expanding it here made
        # every trace carry one redundant image-sized tensor per selected SDE
        # step (roughly 25% of the bundle) even though replay can broadcast the
        # compact value exactly.
        self.trace.scales.append(scale.detach())
        self.trace.indices.append(index)
        return next_sample

    def finish(self, final_latent: torch.Tensor) -> SdeTrace:
        self.trace.final_latent = final_latent.float().detach()
        return self.trace


def finite_trace(tensors: Iterable[torch.Tensor]) -> bool:
    return all(bool(torch.isfinite(tensor).all()) for tensor in tensors if tensor.is_floating_point())
