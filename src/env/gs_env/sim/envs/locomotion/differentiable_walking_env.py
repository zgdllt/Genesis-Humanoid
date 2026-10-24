"""G1 walk with functional Torch buffers and tracked rigid-state outputs.

The original WalkingEnv/PPO path is untouched. Discrete contact, termination and
randomization decisions are piecewise constant, as in short-horizon pathwise RL.
"""

# Functional adapter for the stock environment and controller internals.
# ruff: noqa: SLF001

import time
from typing import Any

import numpy as np
import torch

from gs_env.common.utils.math_utils import (
    quat_apply,
    quat_from_angle_axis,
    quat_inv,
    quat_mul,
    quat_to_euler,
)
from gs_env.sim.envs.config.schema import WalkingEnvArgs
from gs_env.sim.envs.locomotion.walking_env import WalkingEnv
from gs_env.sim.robots.config.schema import CtrlType


def _detached_copy(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _detached_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_detached_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_detached_copy(item) for item in value)
    return value


def matched_walk_args(requires_grad: bool) -> WalkingEnvArgs:
    from gs_env.sim.envs.config.registry import EnvArgsRegistry

    args = EnvArgsRegistry["g1_walk"].model_copy(deep=True)
    scene = args.scene_args
    sim_data = scene.sim_options.model_dump(exclude_unset=True)
    sim_data.update(requires_grad=requires_grad, substeps_local=sim_data.get("substeps", 1))
    sim = type(scene.sim_options)(**sim_data)
    rigid_data = scene.rigid_options.model_dump(exclude_unset=True)
    rigid_data.update(
        use_gjk_collision=True,
        box_box_detection=False,
        differentiable_contact=True,
        contact_pruning_tolerance=None,
        max_collision_pairs=64,
        max_contacts=128,
    )
    # The implicit contact adjoint assumes a converged solve. The stock f32
    # tolerance (1e-5) leaves force/foot-speed finite differences dominated by
    # solver residuals. Both algorithms use these same stricter numerics.
    rigid_data.update(iterations=100, tolerance=1e-8, ls_tolerance=1e-6)
    rigid = type(scene.rigid_options)(**rigid_data)
    import genesis as gs

    if rigid.friction_cone != gs.friction_cone.pyramidal or rigid.noslip_iterations != 0:
        raise ValueError("Walk output VJP requires pyramidal contacts without noslip iterations")
    for key in ("enable_torsional_friction", "use_hibernation"):
        if getattr(rigid, key, False):
            raise ValueError(f"Walk output VJP has not been validated with {key}")
    dr = args.robot_args.dr_args
    if dr.external_force_range is not None or dr.external_torque_range is not None:
        raise ValueError(
            "External wrench replay must be implemented before enabling continuous wrench randomization"
        )
    return args.model_copy(
        update={"scene_args": scene.model_copy(update={"sim_options": sim, "rigid_options": rigid})}
    )


class MatchedWalkingEnv(WalkingEnv):
    """Original controller with immediate overflow checks before reward or reset."""

    def apply_action(self, action: torch.Tensor) -> None:
        super().apply_action(action)
        self._scene.rigid_solver.check_errno()


class DifferentiableWalkingEnv(WalkingEnv):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._diff_ready = False
        super().__init__(*args, **kwargs)
        entity = self._robot.robot
        if entity.n_qs != entity.n_dofs + 1 or min(self._robot._dofs_idx_local) < 6:
            raise ValueError("This baseline supports G1's free base and scalar actuated joints")
        self._joint_qs = [entity.q_start + d + 1 for d in self._robot._dofs_idx_local]
        self._joint_dofs = [entity.dof_start + d for d in self._robot._dofs_idx_local]
        self._links = slice(entity.link_start, entity.link_start + entity.n_links)
        self._diff_ready = True
        self.update_buffers()

    def _robot_action(self, target: torch.Tensor) -> None:
        r = self._robot
        state = self._scene.rigid_solver.get_state(include_walk_outputs=False)
        # Clone gains because the original controller randomizes them in place.
        kp, kd = r._batched_dof_kp.clone(), r._batched_dof_kd.clone()
        desired = target[:, : r.dof_dim] + r._default_dof_pos - r._dof_pos + r._motor_offset
        if r.ctrl_type == CtrlType.DR_JOINT_POSITION_VELOCITY:
            force = kp * desired + kd * (
                -r._dof_vel + target[:, r.dof_dim :] * r._dof_feed_forward_ratio
            )
        elif r.ctrl_type == CtrlType.DR_JOINT_POSITION:
            force = kp * desired - kd * r._dof_vel
        else:
            raise ValueError(f"Unsupported controller {r.ctrl_type}")
        force = (force * r._motor_strength).clamp(-r._torque_limits, r._torque_limits)
        r._torque = force
        r.robot.control_dofs_force(force, dofs_idx_local=r._dofs_idx_local)
        # Preserve the original controller's one-physics-step measurement lag.
        r._dof_pos = state.qpos[:, self._joint_qs]
        r._dof_vel = state.dofs_vel[:, self._joint_dofs]
        r._steps_since_randomize_external_force += 1
        if r._steps_since_randomize_external_force >= r._steps_to_randomize_external_force:
            r._randomize_external_force()
            r._steps_since_randomize_external_force = 0
        r._apply_external_force()
        r._steps_since_randomize_pds += 1
        if r._steps_since_randomize_pds >= r._steps_to_randomize_pds:
            r._randomize_pds()
            r._steps_since_randomize_pds = 0

    def apply_action(self, action: torch.Tensor) -> None:
        action = action.to(self._device)
        self._action = action
        self._action_buf = torch.cat([self._action_buf[:, :, 1:], action.unsqueeze(-1)], dim=-1)
        target = self._action_buf[:, :, 0] * self.action_scale.unsqueeze(0)
        if self._args.robot_args.ctrl_type == CtrlType.DR_JOINT_POSITION_VELOCITY:
            velocity = (target - self._last_target_dof_pos) / self.dt
            self._target_dof_vel_low_pass = self._target_dof_vel_low_pass + self.low_pass_alpha * (
                velocity - self._target_dof_vel_low_pass
            )
            executable = torch.cat([target, self._target_dof_vel_low_pass], dim=-1)
        else:
            executable = target
        self._last_target_dof_pos = target
        self.torque = torch.zeros_like(action)
        for _ in range(self.decimation):
            self._pre_step()
            self._robot_action(executable)
            self._scene.scene.step(update_visualizer=False, refresh_visualizer=False)
            self.torque = torch.maximum(self.torque, self._robot.torque.abs())
        self._scene.rigid_solver.check_errno()
        self.update_buffers()
        self.feet_first_contact = (self.feet_air_time > 0.0) * self.foot_contact
        self.feet_air_time = self.feet_air_time + self.dt

    def update_buffers(self) -> None:
        if not self._diff_ready:
            super().update_buffers()
            return
        state = self._scene.rigid_solver.get_state()
        self.link_positions = state.links_pos[:, self._links]
        self.link_quaternions = state.links_quat[:, self._links]
        self.link_lin_velocities = state.links_vel[:, self._links]
        self.link_ang_velocities = state.links_ang[:, self._links]
        self.link_contact_forces = state.links_contact_force[:, self._links]
        self.base_pos = self.link_positions[:, 0]
        self.base_quat = self.link_quaternions[:, 0]
        relative = quat_mul(self.base_quat, quat_inv(self.base_default_quat))
        self.base_euler = quat_to_euler(relative)
        self.projected_gravity = quat_apply(quat_inv(self.base_quat), self.global_gravity)
        yaw_inv = quat_from_angle_axis(
            -self.base_euler[:, 2], torch.tensor([0.0, 0.0, 1.0], device=self._device)
        )
        self.base_lin_vel = quat_apply(yaw_inv, self.link_lin_velocities[:, 0])
        self.base_ang_vel = quat_apply(quat_inv(relative), self.link_ang_velocities[:, 0])
        feet = self._robot.foot_links_idx
        self.foot_contact_force = self.link_contact_forces[:, feet, 2]
        self.foot_contact = (self.foot_contact_force > 1.0).to(self.foot_contact_force.dtype)
        self.foot_contact_weighted = self.foot_contact_force / (
            self.robot.mass * self.scene.gravity
        )
        self.feet_position = self.link_positions[:, feet]
        self.feet_height = self.feet_position[:, :, 2]
        self.feet_velocity = self.link_lin_velocities[:, feet]
        feet_quat = self.link_quaternions[:, feet].reshape(-1, 4)
        self.feet_orientation = quat_apply(
            quat_inv(feet_quat),
            self.global_gravity[:, None, :].expand(-1, len(feet), -1).reshape(-1, 3),
        ).reshape(self.num_envs, len(feet), 3)
        self.foot_orientation = self.feet_orientation
        self.body_euler = quat_to_euler(self.link_quaternions[:, self._robot.body_link_idx])

    def update_history(self) -> None:
        # Commands used by earlier reward nodes must not be modified in place.
        self.commands = self.commands.clone()
        self.last_last_action = self.last_action
        self.last_action = self._action
        pushes = torch.nonzero(self.time_since_random_push >= self._random_push_time).flatten()
        if not self._eval_mode:
            self._random_push(envs_idx=pushes)
        self.time_since_random_push[pushes] = 0.0
        self.feet_air_time = self.feet_air_time * (1 - self.foot_contact)
        ids = torch.nonzero(self.time_since_resample > self._command_resample_time).flatten()
        self._resample_commands(envs_idx=ids)
        self.time_since_resample[ids] = 0.0

    def _random_push(self, envs_idx: torch.Tensor) -> None:
        if envs_idx.numel() == 0:
            return
        delta = torch.rand((len(envs_idx), 2), device=self._device) * 2.0 - 1.0
        state = self._scene.rigid_solver.get_state()
        velocity = state.links_vel[:, self._robot.robot.base_link_idx][envs_idx]
        pushed = torch.cat([velocity[:, :2] + delta, velocity[:, 2:]], dim=-1)
        self._robot.set_dofs_velocity(pushed, envs_idx=envs_idx, dofs_idx_local=[0, 1, 2])

    def step_unreset(
        self, action: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
    ]:
        self.apply_action(action)
        self.get_terminated()
        truncated = self.get_truncated().clone()
        # get_terminated includes timeouts; distinguish falls for bootstrapping.
        terms = self._extra_info["termination"]
        falls = (
            terms["tilt"] | terms["base_height"] | terms["contact_force"] | terms["floor_collision"]
        )
        reward, reward_terms = self.get_reward()
        self.update_history()
        actor_obs, critic_obs = self.get_observations()
        return actor_obs, critic_obs, reward, falls, truncated, reward_terms

    def backward_window(self, loss: torch.Tensor) -> None:
        # SimState omits collision warm starts/contact caches. Preserve the full
        # checkpoint so backward cannot change the next forward transition.
        solver = self._scene.rigid_solver
        started = time.monotonic()
        free_before_flush = torch.cuda.mem_get_info()[0]
        reserved_before_flush = torch.cuda.memory_reserved()
        # Even a CPU checkpoint needs temporary CUDA arrays in Quadrants'
        # to_numpy kernels. Release Torch's unused blocks before that copy,
        # not only before reverse/restore. The formal evaluator stays resident.
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        snapshot_started = time.monotonic()
        free_before_snapshot = torch.cuda.mem_get_info()[0]
        reserved_before_snapshot = torch.cuda.memory_reserved()
        try:
            with torch.no_grad():
                snapshot = solver.__getstate__()
        except RuntimeError as error:
            error.add_note(
                f"Snapshot CUDA memory: free before/after cache flush "
                f"{free_before_flush}/{free_before_snapshot} bytes; Torch reserved "
                f"{reserved_before_flush}/{reserved_before_snapshot} bytes"
            )
            raise
        pending = [_detached_copy(entity._tgt) for entity in solver.entities]
        captured = time.monotonic()
        # Quadrants allocates CUDA launch scratch outside Torch's allocator.
        # Release unused forward-workspace blocks before the reverse as well.
        torch.cuda.empty_cache()
        reverse_started = time.monotonic()
        self._scene.scene.backward(loss)
        torch.cuda.synchronize()
        reversed_at = time.monotonic()
        # Torch and Quadrants use different CUDA allocators. Backward releases
        # its graph into Torch's cache, but restoring a large solver snapshot
        # needs temporary Quadrants allocations. Return unused blocks first.
        torch.cuda.empty_cache()
        flushed = time.monotonic()
        solver.__setstate__(snapshot)
        for entity, targets in zip(solver.entities, pending, strict=True):
            entity._tgt = targets
        self.detach_buffers()
        torch.cuda.synchronize()
        self.window_timings = dict(
            pre_snapshot_flush_s=snapshot_started - started,
            snapshot_s=captured - snapshot_started,
            pre_reverse_flush_s=reverse_started - captured,
            reverse_s=reversed_at - reverse_started,
            allocator_flush_s=flushed - reversed_at,
            restore_s=time.monotonic() - flushed,
            snapshot_bytes=sum(value.nbytes for value in snapshot.arrays.values()),
            free_before_snapshot_flush_bytes=free_before_flush,
            free_after_snapshot_flush_bytes=free_before_snapshot,
            torch_reserved_before_snapshot_flush_bytes=reserved_before_flush,
            torch_reserved_after_snapshot_flush_bytes=reserved_before_snapshot,
        )

    def detach_buffers(self) -> None:
        for obj in (self, self._robot):
            for name, value in vars(obj).copy().items():
                if isinstance(value, torch.Tensor):
                    setattr(obj, name, value.detach().clone())
        self.foot_orientation = self.feet_orientation

    def capture(self, include_scratch: bool = False) -> tuple[Any, ...]:
        """Full forward replay snapshot for gradient/parity diagnostics."""
        tensors = {}
        scalars = {}
        for prefix, obj in (("env", self), ("robot", self._robot)):
            tensors[prefix] = {
                n: v.detach().clone() for n, v in vars(obj).items() if isinstance(v, torch.Tensor)
            }
            scalars[prefix] = {n: v for n, v in vars(obj).items() if type(v) in (int, float, bool)}
        with torch.no_grad():
            sim = self._scene.scene.get_state()
            rigid = self._scene.rigid_solver.__getstate__()
            if include_scratch:
                from genesis.utils import array_class
                from genesis.utils.misc import data_to_array

                rigid.kinds = frozenset(array_class.DataKind)
                rigid.arrays = {
                    item.name: np.array(getattr(item.owner, item.attr))
                    if isinstance(item, array_class.ScalarItem)
                    else data_to_array(item.array, to_torch=False, copy=True)
                    for item in self._scene.rigid_solver.data
                    if isinstance(item, array_class.DataItem)
                }
        return (
            sim,
            rigid,
            tensors,
            scalars,
            torch.random.get_rng_state(),
            torch.cuda.get_rng_state(),
            [_detached_copy(entity._tgt) for entity in self._scene.rigid_solver.entities],
        )

    def restore(self, snapshot: tuple[Any, ...]) -> None:
        sim, rigid, tensors, scalars, cpu_rng, cuda_rng, pending = snapshot
        self._scene.scene.reset(sim)
        self._scene.rigid_solver.__setstate__(rigid)
        for entity, targets in zip(self._scene.rigid_solver.entities, pending, strict=True):
            entity._tgt = _detached_copy(targets)
        for prefix, obj in (("env", self), ("robot", self._robot)):
            for name, value in tensors[prefix].items():
                setattr(obj, name, value.detach().clone())
            for name, value in scalars[prefix].items():
                setattr(obj, name, value)
        self.foot_orientation = self.feet_orientation
        torch.random.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng)
