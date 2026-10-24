"""Compare complete formal runs at a predeclared mature-PPO quality level."""

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

from gs_agent.utils.walk_benchmark import PROTOCOL, quality_matches
from gs_agent.utils.walk_evidence import require_same_context, source_hashes


def load_run(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = json.loads((path / "config.json").read_text())
    status = json.loads((path / "status.json").read_text())
    if config["debug"] or config["num_envs"] != 4096 or config["iterations"] != 50000:
        raise ValueError(f"{path} is a pilot; it cannot establish formal sample efficiency")
    if status["phase"] != "complete" or status["iteration"] != 50000:
        raise ValueError(f"{path} has not completed the formal budget")
    rows = [json.loads(line) for line in (path / "evaluations.jsonl").read_text().splitlines()]
    validate_history(rows)
    return config, rows


def validate_history(rows: list[dict[str, Any]]) -> None:
    """Reject missing/duplicate scheduled evaluations and invalid accounting."""
    scheduled = list(range(0, PROTOCOL["iterations"] + 1, PROTOCOL["eval_interval"]))
    if [row["iteration"] for row in rows] != scheduled:
        raise ValueError("Incomplete or duplicate scheduled evaluations")
    previous = None
    for index, row in enumerate(rows):
        counters, quality = row["counters"], row["quality"]
        expected_steps = row["iteration"] * PROTOCOL["num_envs"] * PROTOCOL["rollout_length"]
        expected_eval = (index + 1) * PROTOCOL["eval_num_envs"] * PROTOCOL["eval_steps"]
        if (
            counters["train_env_steps"] != expected_steps
            or counters["eval_env_steps"] != expected_eval
        ):
            raise ValueError("Evaluation transition accounting differs from the fixed protocol")
        for key in ("wall_time_s", "active_train_s", "eval_time_s"):
            value = counters[key]
            if not math.isfinite(value) or value < 0 or (previous and value < previous[key]):
                raise ValueError(f"Invalid or nonmonotonic {key}")
        if previous and counters["wall_time_s"] <= previous["wall_time_s"]:
            raise ValueError("Evaluation wall time must increase")
        for key in ("return", "survival", "velocity_rmse", "yaw_rmse", "foot_slip"):
            if not math.isfinite(quality[key]):
                raise ValueError(f"Nonfinite evaluation {key}")
        if not 0 <= quality["survival"] <= 1 or any(
            quality[key] < 0 for key in ("velocity_rmse", "yaw_rmse", "foot_slip")
        ):
            raise ValueError("Invalid evaluation quality range")
        previous = counters


def attainment(rows: list[dict[str, Any]], reference: dict[str, float]) -> dict[str, Any] | None:
    consecutive = 0
    for row in rows:
        consecutive = consecutive + 1 if quality_matches(row["quality"], reference) else 0
        if consecutive >= 3:
            return row["counters"]
    return None


def publish_curves(
    ppo: dict[str, Any], rows: list[list[dict[str, Any]]], result: dict[str, Any], output: Path
) -> None:
    """Publish two directly viewable comparison figures after both formal runs."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import swanlab

    panels = [
        ("return", "Evaluation return"),
        ("survival", "20 s survival fraction"),
        ("velocity_rmse", "Velocity RMSE (m/s)"),
        ("yaw_rmse", "Yaw RMSE (rad/s)"),
        ("foot_slip", "Foot slip (m/s)"),
    ]
    images = {}
    for axis, label, divisor in (
        ("train_env_steps", "Training control transitions", 1),
        ("wall_time_s", "Total elapsed wall time (hours)", 3600),
    ):
        figure, axes = plt.subplots(3, 2, figsize=(11, 9), constrained_layout=True)
        for panel, (metric, title) in zip(axes.flat, panels, strict=False):
            for name, color, history in zip(
                ("PPO", "SHAC"), ("#2563eb", "#c2410c"), rows, strict=True
            ):
                x = [r["counters"][axis] / divisor for r in history]
                y = [r["quality"][metric] for r in history]
                panel.plot(x, y, label=name, color=color, linewidth=1.8)
            panel.axhline(
                result["reference"][metric],
                color="#6b7280",
                linestyle="--",
                linewidth=1,
                label="Mature PPO reference",
            )
            panel.set(xlabel=label, ylabel=title)
            panel.grid(alpha=0.2)
        summary = axes.flat[-1]
        summary.axis("off")
        status = (
            "High-quality PPO reference verified"
            if result["valid_high_quality_reference"]
            else "PPO reference below the fixed quality target"
        )
        gains = (
            f"Sample gain: {result['sample_gain']:.3f}x\nWall-time gain: {result['wall_time_gain']:.3f}x"
            if "sample_gain" in result
            else "Confirmed matched-quality attainment absent\nNo efficiency improvement established"
        )
        summary.text(
            0.05,
            0.8,
            f"4096 environments; seed {ppo['seed']}\n50,000 iterations per algorithm\n\n{status}\n\n{gains}",
            va="top",
            fontsize=11,
        )
        axes.flat[0].legend(fontsize=9)
        figure.suptitle("G1 walk: matched PPO and differentiable-physics SHAC")
        path = output.with_name(f"{output.stem}.{axis}.png")
        figure.savefig(path, dpi=180)
        figure.savefig(path.with_suffix(".svg"))
        plt.close(figure)
        images[axis] = path
    result["figures"] = {key: str(path) for key, path in images.items()}
    output.write_text(json.dumps(result, indent=2))
    run = swanlab.init(
        project="g1-walk-shac-vs-ppo",
        public=False,
        name=f"formal-comparison-seed{ppo['seed']}",
        group=f"formal-{ppo['common_hash'][:12]}",
        job_type="comparison",
        tags=["formal", "comparison"],
        mode="cloud",
        config={
            "source_runs": result["swanlab"],
            "protocol": ppo["protocol"],
            "reference": result["reference"],
            "valid_high_quality_reference": result["valid_high_quality_reference"],
        },
    )
    try:
        swanlab.log(
            {
                f"comparison/{axis}": swanlab.Image(
                    str(path), caption="Full formal histories; debug runs excluded"
                )
                for axis, path in images.items()
            }
        )
        result["direct_comparison_url"] = run.url
    finally:
        swanlab.finish()


def compare(ppo_path: Path, shac_path: Path, output: Path) -> None:
    ppo, ppo_rows = load_run(ppo_path)
    shac, shac_rows = load_run(shac_path)
    if ppo["common_hash"] != shac["common_hash"] or ppo["seed"] != shac["seed"]:
        raise ValueError("Comparison requires identical task configuration and paired seeds")
    require_same_context(ppo["evidence"], shac["evidence"])
    if ppo["protocol"] != PROTOCOL or ppo["evidence"]["source_hashes"] != source_hashes():
        raise ValueError("Current comparison code/protocol differs from the frozen experiment")
    for key in ("networks_and_optimizer", "protocol", "initialization_seed"):
        if ppo.get(key) != shac.get(key) or ppo.get(key) is None:
            raise ValueError(f"Comparison requires identical {key}")
    if ppo["algorithm"] != "ppo" or shac["algorithm"] != "shac":
        raise ValueError("Expected PPO reference and SHAC candidate")
    reference = {
        k: statistics.median(row["quality"][k] for row in ppo_rows[-5:])
        for k in ("return", "survival", "velocity_rmse", "yaw_rmse", "foot_slip")
    }
    absolute = PROTOCOL["absolute_quality"]
    valid_reference = reference["survival"] >= absolute["survival"] and (
        reference["velocity_rmse"] <= absolute["velocity_rmse"]
        and reference["yaw_rmse"] <= absolute["yaw_rmse"]
    )
    ppo_hit = attainment(ppo_rows, reference) if valid_reference else None
    shac_hit = attainment(shac_rows, reference) if valid_reference else None
    result = dict(
        reference=reference,
        valid_high_quality_reference=valid_reference,
        ppo_attainment=ppo_hit,
        shac_attainment=shac_hit,
        swanlab={
            "ppo": json.loads((ppo_path / "swanlab.json").read_text()),
            "shac": json.loads((shac_path / "swanlab.json").read_text()),
        },
    )
    if ppo_hit and shac_hit:
        result["sample_gain"] = ppo_hit["train_env_steps"] / shac_hit["train_env_steps"]
        result["wall_time_gain"] = ppo_hit["wall_time_s"] / shac_hit["wall_time_s"]
        result["fewer_steps_and_faster"] = (
            result["sample_gain"] > 1 and result["wall_time_gain"] > 1
        )
    else:
        result["fewer_steps_and_faster"] = False
        result["limitation"] = (
            "Mature high-quality reference or confirmed matched-quality attainment is absent"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    # Persist the local conclusion even if plotting or cloud publication fails.
    output.write_text(json.dumps(result, indent=2))
    try:
        publish_curves(ppo, [ppo_rows, shac_rows], result, output)
    except Exception as error:
        result["publication_error"] = str(error)
        output.write_text(json.dumps(result, indent=2))
        raise
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ppo", type=Path, required=True)
    parser.add_argument("--shac", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    opts = parser.parse_args()
    compare(opts.ppo, opts.shac, opts.output)
