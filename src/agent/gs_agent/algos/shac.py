"""Short-horizon actor-critic with analytic gradients through Genesis physics."""

import copy
import statistics
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from gs_agent.algos.config.schema import PPOArgs
from gs_agent.algos.ppo import PPO
from gs_agent.algos.shac_returns import short_horizon_return, td_lambda_targets

if TYPE_CHECKING:
    from gs_env.sim.envs.locomotion.differentiable_walking_env import DifferentiableWalkingEnv


class SHAC(PPO):
    """Reuse PPO's exact network construction; all policy updates are pathwise.

    The target critic freezes parameters while retaining its state derivative.
    A fall ends that environment's segment. Resets wait until backward; wasted
    post-boundary transitions are logged and counted in the physical budget.
    """

    def __init__(
        self,
        env: "DifferentiableWalkingEnv",
        cfg: PPOArgs,
        device: torch.device,
        target_retention: float = 0.4,
    ) -> None:
        super().__init__(env, cfg, device)
        self._target_critic = copy.deepcopy(self._critic).requires_grad_(False)
        self.target_retention = target_retention
        self._actor_obs, self._critic_obs = self.env.get_observations()

    def _build_rollouts(self) -> None:
        # SHAC stores a short autograd graph instead of PPO's GAE buffer.
        self._rollouts = None

    def train_one_iteration(self) -> dict[str, Any]:
        started = time.monotonic()
        critic_obs, rewards, next_values, falls, timeouts, entropies = [], [], [], [], [], []
        terms_buffer = []
        alive = torch.ones(self._num_envs, dtype=torch.bool, device=self.device)
        reset = torch.zeros_like(alive)
        valid_buffer = []
        for _ in range(self._num_steps):
            valid_buffer.append(alive.clone())
            critic_obs.append(self._critic_obs.detach())
            dist = self._actor.dist_from_obs(self._actor_obs)
            action = dist.rsample()
            entropy = dist.entropy().sum(-1)
            obs, value_obs, reward, terminal, timeout, terms = self.env.step_unreset(action)
            value = self._target_critic(value_obs).squeeze(-1)
            rewards.append(reward)
            next_values.append(value)
            falls.append(terminal)
            timeouts.append(timeout)
            entropies.append(entropy)
            terms_buffer.append({k: v.detach() for k, v in terms.items()})
            with torch.no_grad():
                self._curr_reward_sum += torch.where(alive, reward, 0.0)
                self._curr_ep_len += alive.to(torch.float32)
                ended = alive & (terminal | timeout)
                self._rewbuffer.extend(self._curr_reward_sum[ended].cpu().tolist())
                self._lenbuffer.extend(self._curr_ep_len[ended].cpu().tolist())
                self._curr_reward_sum[ended] = 0
                self._curr_ep_len[ended] = 0
                alive = alive & ~ended
                reset = reset | terminal | timeout
            self._actor_obs, self._critic_obs = obs, value_obs
        r, v = torch.stack(rewards), torch.stack(next_values)
        terminal, timeout = torch.stack(falls), torch.stack(timeouts)
        returns = short_horizon_return(r, v, terminal, timeout, self.cfg.gamma)
        valid = torch.stack(valid_buffer)
        entropy = torch.stack(entropies)[valid].mean()
        actor_loss = -returns.mean() / self._num_steps - self._current_entropy_coef * entropy
        if not torch.isfinite(actor_loss):
            raise FloatingPointError("Nonfinite SHAC actor loss")
        targets, critic_valid = td_lambda_targets(
            r.detach(), v.detach(), terminal, timeout, self.cfg.gamma, self.cfg.gae_lambda
        )
        self._actor_optimizer.zero_grad(set_to_none=True)
        self.env.backward_window(actor_loss)
        backbone_norm = torch.stack(
            [
                p.grad.norm()
                for name, p in self._actor.named_parameters()
                if name != "log_std" and p.grad is not None
            ]
        ).norm()
        actor_norm = torch.nn.utils.clip_grad_norm_(
            self._actor.parameters(), self.cfg.max_grad_norm, error_if_nonfinite=True
        )
        if actor_norm == 0:
            raise RuntimeError("SHAC actor gradient is zero")
        before = [p.detach().clone() for p in self._actor.parameters()]
        self._actor_optimizer.step()
        with torch.no_grad():
            parameter_delta = torch.stack(
                [
                    (parameter - old).norm()
                    for parameter, old in zip(self._actor.parameters(), before, strict=True)
                ]
            ).norm()
        backward_done = time.monotonic()
        x = torch.stack(critic_obs)[critic_valid]
        y = targets[critic_valid].unsqueeze(-1)
        losses = []
        for _ in range(self.cfg.num_epochs):
            permutation = torch.randperm(len(x), device=self.device)
            for indices in permutation.tensor_split(min(self.cfg.num_mini_batches, len(x))):
                loss = (self._critic(x[indices]) - y[indices]).square().mean()
                self._critic_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self._critic.parameters(), self.cfg.max_grad_norm, error_if_nonfinite=True
                )
                self._critic_optimizer.step()
                losses.append(float(loss.detach()))
        with torch.no_grad():
            for target, online in zip(
                self._target_critic.parameters(), self._critic.parameters(), strict=True
            ):
                target.lerp_(online, 1 - self.target_retention)
            if reset.any():
                self.env.reset_idx(torch.nonzero(reset).flatten())
            self._actor_obs, self._critic_obs = self.env.get_observations()
        return {
            "rollout": {
                "mean_reward": statistics.mean(self._rewbuffer) if self._rewbuffer else 0.0,
                "mean_length": statistics.mean(self._lenbuffer) if self._lenbuffer else 0.0,
            },
            "train": {
                "policy_loss": float(actor_loss.detach()),
                "value_loss": statistics.mean(losses),
                "entropy_loss": float(entropy.detach()),
                "actor_grad_norm": float(actor_norm),
                "backbone_grad_norm": float(backbone_norm),
                "actor_parameter_delta": float(parameter_delta),
                "mean_std": float(self._actor.action_std.mean().detach()),
                "learning_rate": self._current_lr,
                "entropy_coef": self._current_entropy_coef,
                "actor_updates": 1,
                "critic_updates": len(losses),
            },
            "speed": {
                **self.env.window_timings,
                "rollout_step": self._num_steps * self._num_envs,
                "useful_steps": int(valid.sum()),
                "rollout_backward_time": backward_done - started,
                "critic_time": time.monotonic() - backward_done,
            },
            "reward_terms": {
                k: float(torch.stack([t[k] for t in terms_buffer])[valid].mean())
                for k in terms_buffer[0]
            },
        }

    def save(self, path: Path) -> None:
        super().save(path)
        state = torch.load(path, map_location=self.device, weights_only=False)
        state["target_critic_state_dict"] = self._target_critic.state_dict()
        state["algorithm"] = "SHAC"
        torch.save(state, path)

    def load(self, path: Path, load_optimizer: bool = True) -> None:
        super().load(path, load_optimizer)
        state = torch.load(path, map_location=self.device, weights_only=False)
        self._target_critic.load_state_dict(
            state.get("target_critic_state_dict", state["critic_state_dict"])
        )
