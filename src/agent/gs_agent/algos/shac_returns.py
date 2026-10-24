"""Return algebra for short-horizon pathwise actor-critic.

All arrays are [time, environment]. Terminal decisions are piecewise constant;
the actor retains derivatives through rewards and the frozen target value input.
"""

import torch


def short_horizon_return(
    rewards: torch.Tensor,
    next_values: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """One return per environment, stopping at its first episode boundary.

    Truncations bootstrap at their pre-reset state. Physical falls never do.
    Environments may keep simulating after the boundary until backward runs;
    those extra transitions must still be included in resource accounting.
    """
    if not (rewards.shape == next_values.shape == terminated.shape == truncated.shape):
        raise ValueError("All SHAC rollout arrays must have the same [time, env] shape")
    alive = torch.ones_like(rewards[0], dtype=torch.bool)
    result = torch.zeros_like(rewards[0])
    discount = 1.0
    for t in range(len(rewards)):
        result = result + discount * torch.where(alive, rewards[t], 0.0)
        boundary = terminated[t] | truncated[t]
        bootstrap = alive & ~terminated[t] & (truncated[t] | (t == len(rewards) - 1))
        result = result + discount * gamma * torch.where(bootstrap, next_values[t], 0.0)
        alive = alive & ~boundary
        discount *= gamma
    return result


@torch.no_grad()
def td_lambda_targets(
    rewards: torch.Tensor,
    next_values: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    gamma: float,
    trace_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Detached critic targets and a mask excluding post-terminal wasted steps."""
    if not (rewards.shape == next_values.shape == terminated.shape == truncated.shape):
        raise ValueError("All SHAC rollout arrays must have the same [time, env] shape")
    targets = torch.empty_like(rewards)
    continuation = next_values[-1]
    valid = torch.ones_like(terminated)
    for t in range(1, len(rewards)):
        valid[t] = valid[t - 1] & ~(terminated[t - 1] | truncated[t - 1])
    for t in reversed(range(len(rewards))):
        value = (1.0 - trace_lambda) * next_values[t] + trace_lambda * continuation
        value = torch.where(truncated[t], next_values[t], value)
        value = torch.where(terminated[t], 0.0, value)
        continuation = rewards[t] + gamma * value
        targets[t] = continuation
    return targets, valid
