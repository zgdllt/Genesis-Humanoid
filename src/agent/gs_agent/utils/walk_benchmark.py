"""A shared, frozen evaluation protocol and matched-quality comparison rules."""

# This adapter deliberately preserves the stock controller's internal state.
# ruff: noqa: SLF001

import contextlib
import random
from collections.abc import Iterator

import numpy as np
import torch

COMMANDS = [
    [0.0, 0.0, 0.0],
    [0.4, 0.0, 0.0],
    [0.8, 0.0, 0.0],
    [1.2, 0.0, 0.0],
    [-0.4, 0.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.8, 0.0, 0.6],
    [0.8, 0.0, -0.6],
]

PROTOCOL = {
    "task": "g1_walk",
    "num_envs": 4096,
    "iterations": 50000,
    "rollout_length": 24,
    "eval_interval": 1000,
    "eval_num_envs": 128,
    "eval_steps": 1000,
    "eval_seed": 73019,
    "eval_commands": COMMANDS,
    "eval_domain": "nominal, deterministic actions, no observation noise or pushes",
    "reference": "median of final five PPO evaluations after 50000 iterations",
    "absolute_quality": {"survival": 0.95, "velocity_rmse": 0.25, "yaw_rmse": 0.35},
    "matched_quality": {
        "return_fraction": 0.95,
        "survival_drop": 0.01,
        "velocity_factor": 1.05,
        "velocity_allowance": 0.01,
        "yaw_factor": 1.05,
        "yaw_allowance": 0.01,
        "slip_factor": 1.1,
        "slip_allowance": 0.02,
    },
    "confirmation": "three consecutive scheduled evaluations",
    "accounting": "all new control transitions, including post-fall SHAC waste; evaluation separate",
    "time": "wall time includes startup, compilation, training, logging and evaluations",
}


@contextlib.contextmanager
def independent_rng(seed: int) -> Iterator[None]:
    python_state, numpy_state = random.getstate(), np.random.get_state()
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        try:
            yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)


class WalkEvaluator:
    def __init__(self, num_envs: int = 128, steps: int = 1000) -> None:
        from gs_env.sim.envs.locomotion.differentiable_walking_env import (
            MatchedWalkingEnv,
            matched_walk_args,
        )

        self.num_envs, self.steps = num_envs, steps
        with independent_rng(PROTOCOL["eval_seed"]), torch.no_grad():
            self.env = MatchedWalkingEnv(
                args=matched_walk_args(False),
                num_envs=num_envs,
                device=torch.device("cuda"),
                eval_mode=True,
            )
        # The stock environment randomizes PD gains periodically even in eval
        # mode. Freeze them explicitly for this shared nominal evaluation.
        robot = self.env.robot
        robot._steps_to_randomize_pds = 10**15
        robot._steps_to_randomize_external_force = 10**15
        robot._external_force.zero_()
        robot._external_torque.zero_()
        self.commands = torch.tensor(COMMANDS, device="cuda")[
            torch.arange(num_envs, device="cuda") % len(COMMANDS)
        ]

    @torch.no_grad()
    def evaluate(self, actor: torch.nn.Module) -> dict[str, float]:
        with independent_rng(PROTOCOL["eval_seed"]):
            env = self.env
            env.scene.scene.reset()
            env.reset()
            env.time_since_random_push.zero_()
            env._action_buf.zero_()
            env._target_dof_vel_low_pass.zero_()
            env._robot._batched_dof_kp = env._robot._dof_kp[None, :].repeat(self.num_envs, 1)
            env._robot._batched_dof_kd = env._robot._dof_kd[None, :].repeat(self.num_envs, 1)
            env.commands[:] = self.commands
            alive = torch.ones(self.num_envs, dtype=torch.bool, device="cuda")
            returns = torch.zeros(self.num_envs, device="cuda")
            counts = torch.zeros_like(returns)
            velocity_error = torch.zeros_like(returns)
            yaw_error = torch.zeros_like(returns)
            slip = torch.zeros_like(returns)
            actor_obs, _ = env.get_observations()
            was_training = actor.training
            actor.eval()
            try:
                for _ in range(self.steps):
                    action, _ = actor(actor_obs, deterministic=True)
                    env.apply_action(action)
                    done = env.get_terminated()
                    reward, _ = env.get_reward()
                    returns += torch.where(alive, reward, 0.0)
                    velocity_error += torch.where(
                        alive,
                        (env.base_lin_vel[:, :2] - self.commands[:, :2]).square().sum(-1),
                        0.0,
                    )
                    yaw_error += torch.where(
                        alive, (env.base_ang_vel[:, 2] - self.commands[:, 2]).square(), 0.0
                    )
                    foot_slip = (env.feet_velocity[:, :, :2].norm(dim=-1) * env.foot_contact).mean(
                        -1
                    )
                    slip += torch.where(alive, foot_slip, 0.0)
                    counts += alive
                    alive &= ~done
                    env.update_history()
                    if done.any():
                        env.reset_idx(torch.nonzero(done).flatten())
                    env.commands[:] = self.commands
                    actor_obs, _ = env.get_observations()
            finally:
                actor.train(was_training)
            denominator = counts.sum().clamp_min(1.0)
            result = {
                "return": float(returns.mean()),
                "survival": float(alive.float().mean()),
                "velocity_rmse": float((velocity_error.sum() / denominator).sqrt()),
                "yaw_rmse": float((yaw_error.sum() / denominator).sqrt()),
                "foot_slip": float(slip.sum() / denominator),
                "mean_survival_seconds": float(counts.mean() * env.dt),
                "eval_transitions": self.num_envs * self.steps,
            }
            for i in range(len(COMMANDS)):
                selected = torch.arange(self.num_envs, device="cuda") % len(COMMANDS) == i
                if selected.any():
                    result[f"command_{i}_survival"] = float(alive[selected].float().mean())
            return result


def quality_matches(metrics: dict[str, float], reference: dict[str, float]) -> bool:
    absolute, matched = PROTOCOL["absolute_quality"], PROTOCOL["matched_quality"]
    return (
        metrics["survival"]
        >= max(absolute["survival"], reference["survival"] - matched["survival_drop"])
        and metrics["return"] >= reference["return"] * matched["return_fraction"]
        and metrics["velocity_rmse"]
        <= min(
            absolute["velocity_rmse"],
            reference["velocity_rmse"] * matched["velocity_factor"] + matched["velocity_allowance"],
        )
        and metrics["yaw_rmse"]
        <= min(
            absolute["yaw_rmse"],
            reference["yaw_rmse"] * matched["yaw_factor"] + matched["yaw_allowance"],
        )
        and metrics["foot_slip"]
        <= reference["foot_slip"] * matched["slip_factor"] + matched["slip_allowance"]
    )
