"""Primitive G1 state-gradient checks; not the complete SHAC acceptance gate."""

import argparse
import json
import os
import time
from pathlib import Path

from gs_agent.utils.gpu_guard import ExclusiveGPU


def probe(output: Path, horizons: list[int], num_envs: int) -> None:
    # The v1.4.5 zero-copy setters mutate cached Torch views across grad modes.
    # Use the kernel setter in both the AD probe and its finite differences.
    os.environ["GS_ENABLE_ZEROCOPY"] = "0"
    import genesis as gs
    import torch
    from gs_env.sim.envs.config.registry import EnvArgsRegistry
    from run_ppo_walking import create_gs_env

    gs.init(backend=gs.cuda, seed=0, performance_mode=True)
    args = EnvArgsRegistry["g1_walk"].model_copy(deep=True)
    scene_args = args.scene_args
    sim_data = scene_args.sim_options.model_dump(exclude_unset=True)
    # Genesis v1.4.5's target replay indexes complete checkpoint windows.
    # Its supported one-step window also avoids a large GPU tape at 4096 envs.
    sim_data.update(requires_grad=True, substeps_local=sim_data.get("substeps", 1))
    # Reconstruct: model_copy does not rerun Genesis' private _steps_local setup.
    sim = type(scene_args.sim_options)(**sim_data)
    rigid = scene_args.rigid_options.model_copy(
        update={"use_gjk_collision": True, "box_box_detection": False}
    )
    args = args.model_copy(
        update={
            "scene_args": scene_args.model_copy(update={"sim_options": sim, "rigid_options": rigid})
        }
    )
    env = create_gs_env(num_envs=num_envs, args=args, eval_mode=True)
    scene = env.scene.scene
    entity = env.robot.robot
    dofs = env.robot._dofs_idx_local  # noqa: SLF001
    global_dofs = [entity.dof_start + dof for dof in dofs]

    with torch.no_grad():
        for _ in range(24):
            env.step(torch.zeros(num_envs, env.action_dim, device="cuda"))
        initial = scene.get_state()

    checks = []
    for horizon in horizons:
        started = time.monotonic()
        scene.reset(initial)
        force = torch.full((num_envs, env.action_dim), 0.1, device="cuda", requires_grad=True)
        for _ in range(horizon * env.decimation):
            entity.control_dofs_force(force, dofs_idx_local=dofs)
            scene.step(update_visualizer=False)
        state = scene.rigid_solver.get_state()
        weights = torch.linspace(0.5, 1.5, len(dofs), device="cuda")
        loss = (state.dofs_vel[:, global_dofs] * weights).sum() / num_envs
        gradient = torch.autograd.grad(loss, force)[0]
        assert torch.isfinite(gradient).all() and gradient.norm() > 0
        direction = torch.ones_like(force) / force.numel() ** 0.5
        analytic = float((gradient * direction).sum())
        values = []
        epsilon = 0.01
        for sign in (-1, 1):
            scene.reset(initial)
            with torch.no_grad():
                perturbed = force.detach() + sign * epsilon * direction
                for _ in range(horizon * env.decimation):
                    entity.control_dofs_force(perturbed, dofs_idx_local=dofs)
                    scene.step(update_visualizer=False)
                state = scene.rigid_solver.get_state()
                values.append(float((state.dofs_vel[:, global_dofs] * weights).sum() / num_envs))
        numerical = (values[1] - values[0]) / (2 * epsilon)
        relative_error = abs(analytic - numerical) / max(abs(analytic), abs(numerical), 1e-6)
        item = {
            "horizon": horizon,
            "physics_steps": horizon * env.decimation,
            "analytic": analytic,
            "finite_difference": numerical,
            "relative_error": relative_error,
            "gradient_norm": float(gradient.norm()),
            "elapsed_s": time.monotonic() - started,
            "passed": relative_error < 0.1,
        }
        checks.append(item)
        print(json.dumps(item), flush=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"stage": "primitive_state", "checks": checks}, indent=2))
    assert all(item["passed"] for item in checks), "Primitive finite-difference checks failed"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("logs/shac/primitive_gradients.json"))
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 4, 16])
    parser.add_argument("--num-envs", type=int, default=1)
    options = parser.parse_args()
    with ExclusiveGPU():
        probe(options.output, options.horizons, options.num_envs)
