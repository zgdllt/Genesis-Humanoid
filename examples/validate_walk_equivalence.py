"""Stock-PPO forward parity, AD continuation identity, and contact-basis VJP."""

# Explicit inspection of the original environment's state is the point of this check.
# ruff: noqa: SLF001

import argparse
import json
from pathlib import Path

from gs_agent.utils.gpu_guard import ExclusiveGPU
from gs_agent.utils.shac_engine import activate


def validate(output: Path) -> None:
    engine = activate()
    import genesis as gs
    import torch
    from genesis.engine.solvers.rigid.constraint.backward import kernel_contact_output_vjp
    from genesis.utils.misc import qd_to_torch
    from gs_agent.utils.walk_evidence import evidence_context
    from gs_env.sim.envs.locomotion.differentiable_walking_env import (
        DifferentiableWalkingEnv,
        MatchedWalkingEnv,
        matched_walk_args,
    )
    from gs_env.sim.envs.locomotion.walking_env import WalkingEnv

    gs.init(backend=gs.cuda, seed=0, performance_mode=True, use_deterministic_algorithms=True)
    device = torch.device("cuda")
    args = matched_walk_args(False)
    context = evidence_context(engine, args)
    stock = MatchedWalkingEnv(args=args, num_envs=2, device=device)
    diff = DifferentiableWalkingEnv(args=matched_walk_args(True), num_envs=2, device=device)
    maxima = {}
    failures = []
    solver_diagnostics = []
    vjp_scales = {}

    def check(
        name: str, left: torch.Tensor, right: torch.Tensor, atol: float = 2e-5, rtol: float = 2e-4
    ) -> None:
        if left.dtype == torch.bool:
            assert torch.equal(left, right), name
            return
        maxima[name] = max(maxima.get(name, 0.0), float((left - right).abs().max()))
        try:
            torch.testing.assert_close(left, right, atol=atol, rtol=rtol, msg=name)
        except AssertionError:
            failures.append(name)
            print(
                json.dumps(
                    dict(
                        mismatch=name,
                        error=maxima[name],
                        left=left.detach().flatten()[:12].tolist(),
                        right=right.detach().flatten()[:12].tolist(),
                    )
                ),
                flush=True,
            )

    def observe(env: WalkingEnv) -> dict[str, torch.Tensor]:
        # Diagnostic reads must not advance the noise stream being compared.
        with torch.random.fork_rng(devices=[0]):
            actor, critic = env.get_observations()
        with torch.no_grad():
            state = env.scene.rigid_solver.get_state()
        return {
            "actor": actor.detach().clone(),
            "critic": critic.detach().clone(),
            "qpos": state.qpos.detach().clone(),
            "qd": state.dofs_vel.detach().clone(),
            "torque": env.torque.detach().clone(),
            "poses": env.link_positions.detach().clone(),
            "velocity": env.link_lin_velocities.detach().clone(),
            "force": env.link_contact_forces.detach().clone(),
            "foot_contact": env.foot_contact.detach().bool().clone(),
            "feet_first_contact": env.feet_first_contact.detach().bool().clone(),
        }

    def solver_summary(env: WalkingEnv) -> dict:
        solver = env.scene.rigid_solver
        state = solver.constraint_solver.constraint_state
        grad = qd_to_torch(state.grad)
        assert torch.isfinite(grad).all(), "Nonfinite constraint solver residual"
        return {
            "gradient_max": float(grad.abs().max()),
            "cost": qd_to_torch(state.cost).tolist(),
            "n_constraints": qd_to_torch(state.n_constraints).tolist(),
            "n_contacts": qd_to_torch(solver.collider.collider_state.n_contacts).tolist(),
        }

    @torch.no_grad()
    def transfer_forward_state() -> None:
        initial = stock.scene.scene.get_state()
        checkpoint = stock.scene.rigid_solver.__getstate__()
        diff.scene.scene.reset(initial)
        # AD builds additionally own adjoint caches and gradient arrays. Keep
        # those initialized while copying every shared forward-state field.
        compatible = diff.scene.rigid_solver.__getstate__()
        for name, value in checkpoint.arrays.items():
            if compatible.arrays[name].shape == value.shape:
                compatible.arrays[name] = value
            elif "_bw" not in name:
                raise ValueError(f"Forward-state layout differs: {name}")
        diff.scene.rigid_solver.__setstate__(compatible)
        # Scene.reset clears the tape but preserves queued tracked setters from
        # initialization. They must not replay the AD env's old random pose.
        for entity in diff.scene.rigid_solver.entities:
            entity._tgt.clear()
        for left, right in ((stock, diff), (stock.robot, diff.robot)):
            for name, value in vars(left).items():
                if isinstance(value, torch.Tensor):
                    setattr(right, name, value.clone())
                elif type(value) in (int, float, bool) and hasattr(right, name):
                    setattr(right, name, value)
        # Keep DifferentiableWalkingEnv's dispatch flag after state transfer.
        diff._diff_ready = True
        diff.foot_orientation = diff.feet_orientation

    with torch.no_grad():
        transfer_forward_state()
        initial_stock, initial_diff = observe(stock), observe(diff)
        for name in initial_stock:
            check(f"initial/{name}", initial_stock[name], initial_diff[name])
        assert not failures, f"Initial state transfer differs: {failures}"
        actions = 0.02 * torch.randn(16, 2, stock.action_dim, device=device)
        for step, action in enumerate(actions):
            if step:
                # Test the transition function at identical input states. A
                # free trajectory amplifies harmless float32 contact noise and
                # no longer isolates differences between implementations.
                transfer_forward_state()
            if step == 12:
                # Exercise the production push path once, with identical noise.
                stock.time_since_random_push[:] = stock._random_push_time
                diff.time_since_random_push[:] = diff._random_push_time
            cpu_rng, cuda_rng = torch.random.get_rng_state(), torch.cuda.get_rng_state()
            stock.apply_action(action)
            s_done = stock.get_terminated().clone()
            s_reward, s_terms = stock.get_reward()
            stock.update_history()
            s_obs = observe(stock)
            torch.random.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state(cuda_rng)
            diff.apply_action(action)
            d_done = diff.get_terminated().clone()
            d_reward, d_terms = diff.get_reward()
            diff.update_history()
            d_obs = observe(diff)
            solver_diagnostics.append(
                {"step": step, "native": solver_summary(stock), "ad": solver_summary(diff)}
            )
            check("done", s_done, d_done)
            check("reward", s_reward, d_reward)
            for name in s_terms:
                check(f"reward/{name}", s_terms[name], d_terms[name])
            for name in s_obs:
                # Native PPO retains its optimized float32 solver kernels. AD
                # uses scalar kernels, so require physical agreement rather
                # than bit identity after contact impulses.
                atol = 1.0 if name == "force" else (1e-4 if name in ("qpos", "poses") else 1e-3)
                check(name, s_obs[name], d_obs[name], atol=atol, rtol=0)
            assert not failures, f"Forward step {step} differs: {failures}"
            if step == 7:
                cpu_rng, cuda_rng = torch.random.get_rng_state(), torch.cuda.get_rng_state()
                stock.reset_idx(torch.tensor([0], device=device))
                torch.random.set_rng_state(cpu_rng)
                torch.cuda.set_rng_state(cuda_rng)
                diff.reset_idx(torch.tensor([0], device=device))
                left, right = observe(stock), observe(diff)
                for name in left:
                    check(f"subset_reset/{name}", left[name], right[name], atol=1e-3)
                assert not failures, f"Subset reset differs: {failures}"
    print(json.dumps({"forward_parity": True, "maximum_absolute_errors": maxima}), flush=True)

    # The reverse pass must leave the complete current forward state unchanged.
    diff.detach_buffers()
    initial = diff.capture()
    following = torch.full((2, diff.action_dim), 0.01, device=device)
    for horizon in (4, 24):

        def rollout(parameter: torch.Tensor, horizon: int = horizon) -> torch.Tensor:
            value = torch.zeros((), device=device)
            for t in range(horizon):
                if t == horizon - 1:
                    diff.time_since_random_push[:] = diff._random_push_time
                obs, _ = diff.get_observations()
                action = 0.01 * torch.tanh(parameter + 0.1 * obs[:, : diff.action_dim])
                _, _, reward, _, _, _ = diff.step_unreset(action)
                value = value + (0.99**t) * reward.mean()
            return value

        diff.restore(initial)
        with torch.no_grad():
            rollout(torch.tensor(0.1, device=device))
            expected_end = observe(diff)
            expected_pending = {
                key: {
                    name: value.detach().clone() if isinstance(value, torch.Tensor) else value
                    for name, value in values.items()
                }
                for key, values in diff.robot.robot._tgt.items()
            }
            assert any(key[0] == "set_dofs_velocity" for key in expected_pending), (
                "Missing pending push fixture"
            )
            cpu_rng, cuda_rng = torch.random.get_rng_state(), torch.cuda.get_rng_state()
            diff.apply_action(following)
            expected_next = observe(diff)
        diff.restore(initial)
        parameter = torch.tensor(0.1, device=device, requires_grad=True)
        loss = rollout(parameter)
        diff.backward_window(loss)
        with torch.no_grad():
            torch.random.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state(cuda_rng)
            actual_end = observe(diff)
            actual_pending = diff.robot.robot._tgt
            assert expected_pending.keys() == actual_pending.keys(), "Backward lost pending targets"
            for key, values in expected_pending.items():
                for name, value in values.items():
                    if isinstance(value, torch.Tensor):
                        check(f"pending/{name}", value, actual_pending[key][name])
                    else:
                        assert value == actual_pending[key][name]
            for name in actual_end:
                check(f"replay_h{horizon}/end/{name}", expected_end[name], actual_end[name])
            diff.apply_action(following)
            actual_next = observe(diff)
            for name in actual_next:
                check(f"replay_h{horizon}/next/{name}", expected_next[name], actual_next[name])
        assert parameter.grad is not None and torch.isfinite(parameter.grad)
        assert not failures, f"Continuation identity differs: {failures}"
    # Exercise both tangent-basis branches against an independent Torch VJP.
    solver = diff.scene.rigid_solver
    contacts = solver.collider.collider_state
    constraints = solver.constraint_solver.constraint_state
    with torch.no_grad():
        n_contacts = qd_to_torch(contacts.n_contacts)
        assert int(n_contacts[0]) > 0, "Contact-basis fixture has no contact"
        col = int(qd_to_torch(contacts.contact_sort_idx)[0, 0])
        la = int(qd_to_torch(contacts.contact_data.link_a)[col, 0])
        lb = int(qd_to_torch(contacts.contact_data.link_b)[col, 0])
        mu = qd_to_torch(contacts.contact_data.friction)[col, 0]
        start = int(
            qd_to_torch(constraints.n_constraints_equality)[0]
            + qd_to_torch(constraints.n_constraints_frictionloss)[0]
        )
        lambdas = qd_to_torch(constraints.efc_force)[start : start + 4, 0].clone()
        assert float(lambdas.norm()) > 1e-6, "Contact basis fixture is unloaded"
        vjp_scales["contact_lambda_norm"] = float(lambdas.norm())
        force_seed = qd_to_torch(solver.dyn_state.links.contact_force.grad)
        normal_seed = qd_to_torch(contacts.contact_data.normal.grad)
        normals = qd_to_torch(contacts.contact_data.normal)
        original = normals[col, 0].clone()
        seed = (
            torch.arange(force_seed.numel(), device=device, dtype=torch.float32).reshape(
                force_seed.shape
            )
            * 0.01
        )
        upstream = seed[lb, 0] - seed[la, 0]
    for normal_values in ([0.3, 0.2, 0.9327379], [0.2, 0.8, 0.5656854]):
        normal = torch.tensor(normal_values, device=device, requires_grad=True)
        n0, n1, n2 = normal.unbind()
        raw = (
            torch.stack([-n0 * n1, 1 - n1 * n1, -n2 * n1])
            if abs(normal_values[1]) < 0.5
            else (torch.stack([-n0 * n2, -n1 * n2, 1 - n2 * n2]))
        )
        b = raw / raw.norm()
        d1, d2 = torch.linalg.cross(b, normal), b
        basis = torch.stack(
            [-mu * d1 - normal, mu * d1 - normal, -mu * d2 - normal, mu * d2 - normal]
        )
        value = ((basis * lambdas[:, None]).sum(0) * upstream).sum()
        expected = torch.autograd.grad(value, normal)[0]
        assert float(expected.norm()) > 1e-6, "Contact basis oracle has zero upstream sensitivity"
        with torch.no_grad():
            solver.dyn_state.links.contact_force.grad.from_numpy(seed.cpu().numpy())
            contacts.contact_data.normal.grad.from_numpy(
                torch.zeros_like(normal_seed).cpu().numpy()
            )
            normals[col, 0] = normal.detach()
            contacts.contact_data.normal.from_numpy(normals.cpu().numpy())
            kernel_contact_output_vjp(
                solver.dyn_state, contacts, constraints, solver.rigid_config, 2
            )
            check(
                "contact_basis_vjp",
                expected,
                qd_to_torch(contacts.contact_data.normal.grad)[col, 0],
                atol=2e-4,
            )
            assert not failures, f"Contact basis VJP differs: {failures}"
    with torch.no_grad():
        normals[col, 0] = original
        contacts.contact_data.normal.from_numpy(normals.cpu().numpy())

    # Independently reverse lambda = -D * active * (J a - aref) in Torch. This
    # checks both added row VJPs without relying on finite differences across
    # changing contact sets. The existing implicit solve is tested end to end.
    jac = qd_to_torch(constraints.jac).detach().requires_grad_(True)
    acceleration = qd_to_torch(constraints.qacc).detach().requires_grad_(True)
    diagonal = qd_to_torch(constraints.efc_D).detach().requires_grad_(True)
    reference = qd_to_torch(constraints.aref).detach().requires_grad_(True)
    active = qd_to_torch(constraints.active)
    row_weights = torch.zeros_like(diagonal)
    sparsity = torch.zeros_like(jac)
    dof_counts = qd_to_torch(constraints.jac_n_dofs)
    dof_indices = qd_to_torch(constraints.jac_dofs_idx)
    counts = qd_to_torch(contacts.n_contacts).tolist()
    sorted_contacts = qd_to_torch(contacts.contact_sort_idx)
    link_a = qd_to_torch(contacts.contact_data.link_a)
    link_b = qd_to_torch(contacts.contact_data.link_b)
    friction = qd_to_torch(contacts.contact_data.friction)
    starts = qd_to_torch(constraints.n_constraints_equality) + qd_to_torch(
        constraints.n_constraints_frictionloss
    )
    for b_index, count in enumerate(counts):
        for logical in range(count):
            contact = int(sorted_contacts[logical, b_index])
            a_link, b_link = int(link_a[contact, b_index]), int(link_b[contact, b_index])
            upstream = -seed[a_link, b_index]
            if b_link >= 0:
                upstream = upstream + seed[b_link, b_index]
            n = normals[contact, b_index]
            n0, n1, n2 = n.unbind()
            raw = (
                torch.stack([-n0 * n1, 1 - n1 * n1, -n2 * n1])
                if abs(float(n1)) < 0.5
                else torch.stack([-n0 * n2, -n1 * n2, 1 - n2 * n2])
            )
            tangent2 = raw / raw.norm()
            tangent1 = torch.linalg.cross(tangent2, n)
            mu = friction[contact, b_index]
            basis = torch.stack(
                [-mu * tangent1 - n, mu * tangent1 - n, -mu * tangent2 - n, mu * tangent2 - n]
            )
            first = int(starts[b_index]) + 4 * logical
            row_weights[first : first + 4, b_index] = basis @ upstream
            for row in range(first, first + 4):
                columns = dof_indices[row, : int(dof_counts[row, b_index]), b_index].long()
                sparsity[row, columns, b_index] = 1
    residual = torch.einsum("cdb,db->cb", jac * sparsity, acceleration) - reference
    value = (-diagonal * active * residual * row_weights).sum()
    expected_grads = torch.autograd.grad(value, (acceleration, jac, diagonal, reference))
    with torch.no_grad():
        solver.dyn_state.links.contact_force.grad.from_numpy(seed.cpu().numpy())
        constraints.bw_w.from_numpy(residual.detach().cpu().numpy())
        for buffer in (
            constraints.dL_dqacc,
            constraints.dL_djac,
            constraints.dL_defc_D,
            constraints.dL_daref,
        ):
            buffer.fill(0)
        for stage in (0, 1):
            kernel_contact_output_vjp(
                solver.dyn_state, contacts, constraints, solver.rigid_config, stage
            )
        for name, expected, buffer in zip(
            ("acceleration", "jacobian", "diagonal", "reference"),
            expected_grads,
            (
                constraints.dL_dqacc,
                constraints.dL_djac,
                constraints.dL_defc_D,
                constraints.dL_daref,
            ),
            strict=True,
        ):
            actual = qd_to_torch(buffer)
            # Compare sensitivity to log D; absolute tolerance on dL/dD alone
            # could hide a zero gradient when contact stiffness is large.
            scale = diagonal.detach() if name == "diagonal" else 1.0
            expected_scaled, actual_scaled = expected * scale, actual * scale
            norm = float(expected_scaled.norm())
            assert norm > 1e-8, f"Contact {name} oracle has no sensitivity"
            relative = float((actual_scaled - expected_scaled).norm()) / norm
            vjp_scales[name] = dict(
                expected_norm=norm, actual_norm=float(actual_scaled.norm()), relative_error=relative
            )
            assert relative < 1e-4, f"Contact {name} relative VJP error: {relative}"
            check(f"contact_output_vjp/{name}", expected, actual, atol=2e-4)
        assert not failures, f"Contact row VJP differs: {failures}"
    result = dict(
        **context,
        forward_parity=True,
        replay_identity=True,
        contact_basis_vjp=True,
        contact_output_vjp=True,
        maximum_absolute_errors=maxima,
        solver_diagnostics=solver_diagnostics,
        vjp_scales=vjp_scales,
        forward_protocol="16 identical-state one-control-step transitions, including PD changes/reset/push",
        forward_tolerances={
            "position": 1e-4,
            "velocity_observation_torque": 1e-3,
            "force_N": 1.0,
            "relative": 0,
            "reward_absolute": 2e-5,
        },
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("logs/shac/equivalence_gate.json"))
    options = parser.parse_args()
    with ExclusiveGPU():
        validate(options.output)
