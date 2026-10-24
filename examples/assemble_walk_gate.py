"""Aggregate independently saved evidence; never infer a pass from missing fields."""

import argparse
import json
import math
from pathlib import Path

from gs_agent.utils.walk_evidence import require_gate, require_same_context


def assemble(gradients: Path, equivalence: Path, actor_run: Path, output: Path) -> None:
    grad = json.loads(gradients.read_text())
    parity = json.loads(equivalence.read_text())
    config = json.loads((actor_run / "config.json").read_text())
    status = json.loads((actor_run / "status.json").read_text())
    actor = config["evidence"]
    require_same_context(grad, parity)
    require_same_context(grad, actor)
    if config["algorithm"] != "shac" or status["phase"] != "complete":
        raise ValueError("Actor evidence must come from a completed SHAC debug run")
    rows = [json.loads(line) for line in (actor_run / "training.jsonl").read_text().splitlines()]
    updates = [
        row
        for row in rows
        if all(
            math.isfinite(row.get(key, float("nan"))) and row[key] > 0
            for key in (
                "train/actor_grad_norm",
                "train/backbone_grad_norm",
                "train/actor_parameter_delta",
            )
        )
    ]
    required = {(h, c) for h in (1, 4, 24) for c in ("reward", "foot_velocity", "contact_force")}
    successful = {(c["horizon"], c["component"]) for c in grad["checks"] if c["passed"] is True}
    if grad.get("passed") is not True or not required.issubset(successful):
        raise ValueError("Complete reward/velocity/contact-force gradient checks have not passed")
    result = {
        **actor,
        "passed": True,
        "horizons": [1, 4, 24],
        "forward_parity": parity.get("forward_parity") is True,
        "replay_identity": parity.get("replay_identity") is True,
        "contact_basis_vjp": parity.get("contact_basis_vjp") is True,
        "contact_output_vjp": parity.get("contact_output_vjp") is True,
        "actor_update": bool(updates),
        "evidence_files": [str(gradients), str(equivalence), str(actor_run)],
    }
    require_gate(result, actor)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Full float32 walk gate passed: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gradients", type=Path, required=True)
    parser.add_argument("--equivalence", type=Path, required=True)
    parser.add_argument("--actor-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("logs/shac/gradient_gate.json"))
    opts = parser.parse_args()
    assemble(opts.gradients, opts.equivalence, opts.actor_run, opts.output)
