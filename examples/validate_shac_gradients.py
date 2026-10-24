"""Fail-closed G1 gradient gate, including rewards, contact outputs and replay."""

# The diagnostic freezes the original controller's private randomization clocks.
# ruff: noqa: SLF001

import argparse
import hashlib
import json
from itertools import product
from pathlib import Path
from typing import Any

from gs_agent.utils.gpu_guard import ExclusiveGPU
from gs_agent.utils.shac_engine import activate


def validate(
    output: Path,
    horizons: list[int],
    epsilon: float,
    precision: str = "32",
    epsilons: list[float] | None = None,
    components: list[str] | None = None,
    solver_tolerance: float | None = None,
    line_search_tolerance: float | None = None,
    disable_multi_contact: bool = False,
    warmup_controls: int = 8,
) -> None:
    engine = activate()
    import genesis as gs
    import torch
    from genesis.utils.misc import qd_to_torch
    from gs_agent.utils.walk_evidence import evidence_context
    from gs_env.sim.envs.locomotion.differentiable_walking_env import (
        DifferentiableWalkingEnv,
        matched_walk_args,
    )

    gs.init(
        backend=gs.cuda,
        seed=0,
        performance_mode=True,
        use_deterministic_algorithms=True,
        precision=precision,
    )
    # The stock env allocates some explicit float32 buffers during construction.
    # Promote all of them together only after its initialization reset.
    torch.set_default_dtype(torch.float32)
    args = matched_walk_args(True)
    if solver_tolerance is not None:
        args.scene_args.rigid_options.tolerance = solver_tolerance
    if line_search_tolerance is not None:
        args.scene_args.rigid_options.ls_tolerance = line_search_tolerance
    if disable_multi_contact:
        args.scene_args.rigid_options.enable_multi_contact = False
    context = evidence_context(engine, args, precision)

    class PrecisionDiagnosticEnv(DifferentiableWalkingEnv):
        def reset_idx(self, envs_idx: torch.Tensor) -> None:
            # Promote before the stock reset clamps joint positions against
            # Genesis's float64 limits and assigns by environment indices.
            if precision == "64":
                torch.set_default_dtype(torch.float64)
                for obj in (self, self.robot):
                    for name, value in vars(obj).copy().items():
                        if isinstance(value, torch.Tensor) and value.is_floating_point():
                            setattr(obj, name, value.double())
            super().reset_idx(envs_idx)

    env = PrecisionDiagnosticEnv(args=args, num_envs=2, device=torch.device("cuda"), eval_mode=True)
    if precision == "64":
        for obj in (env, env.robot):
            for name, value in vars(obj).copy().items():
                if isinstance(value, torch.Tensor) and value.is_floating_point():
                    setattr(obj, name, value.double())
        torch.set_default_dtype(torch.float64)
        env.update_buffers()
    # Isolate derivatives at a nominal, non-fallen contact configuration. The
    # equivalence/replay check separately exercises training-time PD changes.
    env.robot._steps_to_randomize_pds = 10**15
    env.robot._steps_to_randomize_external_force = 10**15
    with torch.no_grad():
        for _ in range(warmup_controls):
            env.apply_action(torch.zeros(2, env.action_dim, device="cuda"))
            env.update_history()
        assert not env.get_terminated().any(), "Diagnostic fixture has already fallen"
    env.detach_buffers()
    initial = env.capture(include_scratch=True)
    # A planted foot has approximately zero speed derivative. Test the speed
    # output in flight to avoid subtracting two solver-residual-sized numbers.
    with torch.no_grad():
        raised = env.robot.base_pos.clone()
        raised[:, 2] += 2.0
        env.robot.set_state(
            pos=raised,
            quat=env.base_default_quat,
            dof_pos=env.default_dof_pos[None, :].repeat(env.num_envs, 1),
        )
        env.update_buffers()
    airborne = env.capture(include_scratch=True)
    checks = []
    pattern = torch.linspace(-0.7, 1.0, env.action_dim, device="cuda")[None, :]
    branch_trace = []
    geometry_diagnostics = []
    solver = env.scene.rigid_solver
    constraint_force = solver._func_constraint_force

    @torch.no_grad()
    def traced_constraint_force(*args: Any, **kwargs: Any) -> Any:
        result = constraint_force(*args, **kwargs)
        solver = env.scene.rigid_solver
        contacts = solver.collider.collider_state
        constraints = solver.constraint_solver.constraint_state
        counts = qd_to_torch(contacts.n_contacts).tolist()
        rows = qd_to_torch(constraints.n_constraints).tolist()
        order = qd_to_torch(contacts.contact_sort_idx)
        geom_a = qd_to_torch(contacts.contact_data.geom_a)
        geom_b = qd_to_torch(contacts.contact_data.geom_b)
        normal = qd_to_torch(contacts.contact_data.normal)
        detection_ids = qd_to_torch(contacts.diff_contact_input.plane_detection_id)
        active = qd_to_torch(constraints.active)
        geom_types = qd_to_torch(solver.dyn_info.geoms.type)
        geom_quaternions = qd_to_torch(solver.dyn_state.geoms.quat)
        minimum_radial = None
        signature = []
        for b, (count, n_rows) in enumerate(zip(counts, rows, strict=True)):
            ids = order[:count, b].long()
            cylinders = ids[
                (geom_types[geom_a[ids, b].long()] == int(gs.GEOM_TYPE.PLANE))
                & (geom_types[geom_b[ids, b].long()] == int(gs.GEOM_TYPE.CYLINDER))
            ]
            if len(cylinders):
                q = geom_quaternions[geom_b[cylinders, b].long(), b]
                w, x, y, z = q.unbind(-1)
                axis = torch.stack(
                    [2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], dim=-1
                )
                n = normal[cylinders, b]
                radial = n - (n * axis).sum(-1, keepdim=True) * axis
                value = float(radial.norm(dim=-1).min())
                minimum_radial = value if minimum_radial is None else min(minimum_radial, value)
            active_rows = (
                active[b, :n_rows]
                if solver.rigid_config.constraint_layout_batch_first
                else active[:n_rows, b]
            )
            signature.append(
                (
                    geom_a[ids, b].tolist(),
                    geom_b[ids, b].tolist(),
                    (normal[ids, b, 1].abs() < 0.5).tolist(),
                    detection_ids[b, ids].tolist(),
                    active_rows.tolist(),
                )
            )
        branch_trace.append(signature)
        geometry_diagnostics.append(
            dict(
                minimum_unperturbed_cylinder_radial_norm=minimum_radial,
                solver_residual_max=float(qd_to_torch(constraints.grad).abs().max()),
            )
        )
        return result

    solver._func_constraint_force = traced_constraint_force

    def rollout(
        parameter: torch.Tensor, horizon: int, component: str
    ) -> tuple[torch.Tensor, torch.Tensor, list, list]:
        branch_trace.clear()
        geometry_diagnostics.clear()
        total = torch.zeros((), device="cuda")
        contact_flags = []
        for t in range(horizon):
            obs, _ = env.get_observations()
            action = 0.03 * torch.tanh(
                parameter * pattern + 0.1 * obs[:, env.action_dim : 2 * env.action_dim]
            )
            _, _, reward, falls, timeout, _ = env.step_unreset(action)
            contact_flags.append(env.foot_contact.detach().clone())
            if component == "reward":
                loss = reward.mean()
            elif component == "foot_velocity":
                loss = (
                    env.feet_velocity[:, 0] * torch.tensor([0.4, -0.3, 0.7], device="cuda")
                ).sum() / 2
            elif component == "contact_force":
                # Break left/right support-force cancellation at a symmetric
                # standing pose; ground net force alone mostly measures weight.
                loss = (
                    env.foot_contact_force * torch.tensor([1.0, -0.4], device="cuda")
                ).sum() / 200
            else:
                raise ValueError(component)
            total = total + (0.99**t) * loss
        return total, torch.stack(contact_flags), branch_trace.copy(), geometry_diagnostics.copy()

    for horizon, component, delta in product(
        horizons, components or ("reward", "foot_velocity", "contact_force"), epsilons or [epsilon]
    ):
        fixture = airborne if component == "foot_velocity" else initial
        env.restore(fixture)
        parameter = torch.tensor(0.1, device="cuda", requires_grad=True)
        loss, flags, analytic_trace, analytic_geometry = rollout(parameter, horizon, component)
        analytic_value = float(loss.detach())
        env.backward_window(loss)
        analytic = float(parameter.grad)
        assert torch.isfinite(parameter.grad), f"Nonfinite {component} gradient"
        values = []
        same_contacts = True
        branch_changes = []
        repeated_values = []
        for _ in range(2):
            env.restore(fixture)
            with torch.no_grad():
                value, _, _, _ = rollout(parameter.detach(), horizon, component)
            repeated_values.append(float(value))
        for sign in (-1, 1):
            env.restore(fixture)
            with torch.no_grad():
                value, fd_flags, fd_trace, _ = rollout(
                    parameter.detach() + sign * delta, horizon, component
                )
            values.append(float(value))
            same_contacts &= torch.equal(flags, fd_flags)
            branch_changes.append(
                [i for i, (a, b) in enumerate(zip(analytic_trace, fd_trace, strict=True)) if a != b]
            )
        numerical = (values[1] - values[0]) / (2 * delta)
        error = abs(analytic - numerical) / max(abs(analytic), abs(numerical), 1e-5)
        replay_error = max(abs(value - analytic_value) for value in repeated_values)
        # Repeated identical inputs must describe the same forward function,
        # and the replay error must be small relative to the FD signal.
        replay_limit = min(
            1e-5 if precision == "32" else 1e-10,
            0.01 * max(abs(values[1] - values[0]), 1e-12),
        )
        passed = (
            abs(analytic) > 1e-8
            and error < 0.1
            and same_contacts
            and not any(branch_changes)
            and replay_error <= replay_limit
        )
        result = dict(
            horizon=horizon,
            component=component,
            analytic=analytic,
            analytic_value=analytic_value,
            geometry_diagnostics=analytic_geometry,
            repeated_values=repeated_values,
            replay_error=replay_error,
            replay_limit=replay_limit,
            perturbed_values=values,
            finite_difference=numerical,
            relative_error=error,
            same_contact_masks=same_contacts,
            same_active_sets=not any(branch_changes),
            changed_physics_steps=branch_changes,
            analytic_branch_sha256=hashlib.sha256(json.dumps(analytic_trace).encode()).hexdigest(),
            epsilon=delta,
            precision=precision,
            fixture="airborne" if component == "foot_velocity" else "grounded",
            fixture_protocol=f"{warmup_controls} nominal-PD warmup controls; fixed gains within diagnostic",
            passed=passed,
        )
        checks.append(result)
        print(json.dumps(result), flush=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                dict(**context, checks=checks, passed=all(c["passed"] for c in checks)),
                indent=2,
            )
        )
    assert all(c["passed"] for c in checks), (
        "Full walk gradient gate failed; no formal runs allowed"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("logs/shac/gradient_evidence.json"))
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 4, 24])
    parser.add_argument("--epsilon", type=float, default=0.06)
    parser.add_argument("--precision", choices=["32", "64"], default="32")
    parser.add_argument("--epsilons", type=float, nargs="+")
    parser.add_argument("--solver-tolerance", type=float)
    parser.add_argument("--line-search-tolerance", type=float)
    parser.add_argument("--disable-multi-contact", action="store_true")
    parser.add_argument("--warmup-controls", type=int, default=8)
    parser.add_argument(
        "--components", choices=["reward", "foot_velocity", "contact_force"], nargs="+"
    )
    options = parser.parse_args()
    with ExclusiveGPU():
        validate(
            options.output,
            options.horizons,
            options.epsilon,
            options.precision,
            options.epsilons,
            options.components,
            options.solver_tolerance,
            options.line_search_tolerance,
            options.disable_multi_contact,
            options.warmup_controls,
        )
