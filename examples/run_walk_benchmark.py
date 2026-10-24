"""Matched G1 walk PPO/SHAC runner with exclusive GPU use and SwanLab axes."""

# The runner owns algorithm checkpoints and diagnostics.
# ruff: noqa: SLF001

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Any

from gs_agent.utils.gpu_guard import ExclusiveGPU
from gs_agent.utils.shac_engine import activate

ROOT = Path(__file__).resolve().parents[1]


def flatten(values: dict[str, Any], prefix: str = "") -> dict[str, float]:
    out = {}
    for key, value in values.items():
        name = f"{prefix}/{key}" if prefix else key
        if isinstance(value, dict):
            out.update(flatten(value, name))
        elif isinstance(value, (int, float)):
            out[name] = value
    return out


def train(options: argparse.Namespace) -> None:
    started = time.monotonic()
    engine = activate()
    import genesis as gs
    import swanlab
    import torch
    from gs_agent.algos.config.registry import PPO_WALKING_MLP
    from gs_agent.algos.ppo import PPO
    from gs_agent.algos.shac import SHAC
    from gs_agent.utils.walk_benchmark import PROTOCOL, WalkEvaluator, independent_rng
    from gs_agent.utils.walk_evidence import evidence_context, require_gate, task_identity
    from gs_agent.wrappers.gs_env_wrapper import GenesisEnvWrapper
    from gs_env.sim.envs.locomotion.differentiable_walking_env import (
        DifferentiableWalkingEnv,
        MatchedWalkingEnv,
        matched_walk_args,
    )

    if not options.debug:
        if options.num_envs != 4096 or options.iterations != 50000:
            raise ValueError(
                "Formal protocol is 4096 environments x 50000 iterations; use --debug for pilots"
            )
    full_evaluation = not options.debug or getattr(options, "full_evaluation", False)
    evaluation_num_envs = PROTOCOL["eval_num_envs"] if full_evaluation else 16
    evaluation_steps = PROTOCOL["eval_steps"] if full_evaluation else 64
    evaluation_interval = PROTOCOL["eval_interval"] if full_evaluation else 2
    curriculum_iterations = PROTOCOL["iterations"] if full_evaluation else options.iterations
    run_dir = ROOT / options.output
    run_dir.mkdir(parents=True, exist_ok=False)
    gs.init(
        backend=gs.cuda, seed=options.seed, performance_mode=True, use_deterministic_algorithms=True
    )
    cfg = PPO_WALKING_MLP.model_copy(deep=True)
    args = matched_walk_args(options.algorithm == "shac")
    common, common_hash = task_identity(args)
    evidence = evidence_context(engine, args)
    if not options.debug:
        gate = json.loads((ROOT / "logs/shac/gradient_gate.json").read_text())
        require_gate(gate, evidence)
    metadata = dict(
        algorithm=options.algorithm,
        seed=options.seed,
        debug=options.debug,
        engine=engine,
        evidence=evidence,
        task=common,
        common_hash=common_hash,
        networks_and_optimizer=cfg.model_dump(mode="json"),
        protocol=PROTOCOL,
        num_envs=options.num_envs,
        iterations=options.iterations,
        shac_target_retention=0.4,
        shac_actor_lr_schedule="fixed",
        ppo_actor_lr_schedule="native KL-adaptive",
        physical_gpu=0,
        initialization_seed=100000 + options.seed,
        evaluation_execution=dict(
            num_envs=evaluation_num_envs,
            steps=evaluation_steps,
            interval=evaluation_interval,
            curriculum_iterations=curriculum_iterations,
        ),
    )
    (run_dir / "config.json").write_text(json.dumps(metadata, indent=2))
    label = "debug" if options.debug else "formal"
    run = swanlab.init(
        project="g1-walk-shac-vs-ppo",
        public=False,
        name=f"{label}-{options.algorithm}-n{options.num_envs}-seed{options.seed}",
        group=f"{label}-{common_hash[:12]}",
        job_type=label,
        tags=[label, options.algorithm],
        config=metadata,
        mode=options.swanlab_mode,
        log_dir=str(run_dir / "swanlab"),
    )
    url = run.url if options.swanlab_mode == "cloud" else None
    (run_dir / "swanlab.json").write_text(json.dumps({"url": url}, indent=2))
    print(json.dumps({"swanlab_url": url, "run_dir": str(run_dir)}), flush=True)
    swanlab.define_metric("quality_by_steps/*", x_axis="counters/train_env_steps")
    swanlab.define_metric("quality_by_time/*", x_axis="counters/wall_time_s")
    evaluator = None
    iteration, train_steps, eval_steps, active_time, eval_time = 0, 0, 0, 0.0, 0.0
    phase = "initializing"

    def record_state() -> None:
        (run_dir / "status.json").write_text(
            json.dumps(
                dict(
                    phase=phase,
                    iteration=iteration,
                    train_env_steps=train_steps,
                    eval_env_steps=eval_steps,
                    wall_time_s=time.monotonic() - started,
                    active_train_s=active_time,
                    eval_time_s=eval_time,
                    swanlab_url=url,
                ),
                indent=2,
            )
        )

    try:
        env_class = DifferentiableWalkingEnv if options.algorithm == "shac" else MatchedWalkingEnv
        env = env_class(args=args, num_envs=options.num_envs, device=torch.device("cuda"))
        if options.algorithm == "shac":
            env.reset()  # same second initialization reset as GenesisEnvWrapper
            wrapped = env
            algorithm_class = SHAC
        else:
            wrapped = GenesisEnvWrapper(env, device=torch.device("cuda"))
            algorithm_class = PPO
        with independent_rng(100000 + options.seed):
            algorithm = algorithm_class(env=wrapped, cfg=cfg, device=torch.device("cuda"))
        evaluator = WalkEvaluator(
            num_envs=evaluation_num_envs,
            steps=evaluation_steps,
        )

        def evaluate() -> None:
            nonlocal eval_steps, eval_time
            begin = time.monotonic()
            metrics = evaluator.evaluate(algorithm._actor)
            torch.cuda.synchronize()
            eval_time += time.monotonic() - begin
            eval_steps += metrics.pop("eval_transitions")
            counters = dict(
                train_env_steps=train_steps,
                eval_env_steps=eval_steps,
                forward_physics_substeps=train_steps
                * env.decimation
                * args.scene_args.sim_options.substeps,
                wall_time_s=time.monotonic() - started,
                active_train_s=active_time,
                eval_time_s=eval_time,
                iteration=iteration,
            )
            row = dict(iteration=iteration, counters=counters, quality=metrics)
            with (run_dir / "evaluations.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            payload = flatten(counters, "counters")
            payload.update(flatten(metrics, "quality_by_steps"))
            payload.update(flatten(metrics, "quality_by_time"))
            swanlab.log(payload, step=iteration)
            print(json.dumps(row), flush=True)

        evaluate()
        phase = "training"
        record_state()
        for iteration in range(1, options.iterations + 1):
            algorithm.current_iter = iteration
            algorithm.update_curriculum(iteration - 1, curriculum_iterations)
            torch.cuda.synchronize()
            begin = time.monotonic()
            metrics = algorithm.train_one_iteration()
            torch.cuda.synchronize()
            active_time += time.monotonic() - begin
            train_steps += options.num_envs * cfg.rollout_length
            if iteration % 10 == 0 or iteration == 1:
                payload = flatten(metrics)
                payload.update(
                    {
                        "counters/train_env_steps": train_steps,
                        "counters/wall_time_s": time.monotonic() - started,
                        "counters/active_train_s": active_time,
                    }
                )
                swanlab.log(payload, step=iteration)
                with (run_dir / "training.jsonl").open("a") as stream:
                    stream.write(json.dumps(dict(iteration=iteration, **payload)) + "\n")
                record_state()
                print(
                    json.dumps(
                        dict(
                            iteration=iteration,
                            train_env_steps=train_steps,
                            active_train_s=active_time,
                            **flatten(metrics, "metrics"),
                        )
                    ),
                    flush=True,
                )
            if iteration % evaluation_interval == 0 or iteration == options.iterations:
                for network in (algorithm._actor, algorithm._critic):
                    if not all(torch.isfinite(p).all() for p in network.parameters()):
                        raise FloatingPointError("Nonfinite actor/critic parameters")
                temporary = run_dir / "checkpoint.tmp.pt"
                algorithm.save(temporary)
                temporary.replace(run_dir / f"checkpoint_{iteration:05d}.pt")
                evaluate()
        phase = "complete"
        record_state()
    except BaseException:
        phase = "failed"
        record_state()
        (run_dir / "error.txt").write_text(traceback.format_exc())
        raise
    finally:
        swanlab.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", choices=["ppo", "shac"], required=True)
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--full-evaluation",
        action="store_true",
        help="Use the complete formal evaluator and curriculum schedule during a debug run",
    )
    parser.add_argument("--swanlab-mode", choices=["cloud", "offline", "disabled"], default="cloud")
    parser.add_argument("--output", required=True)
    options = parser.parse_args()
    with ExclusiveGPU():
        train(options)
