"""Source and runtime identity shared by diagnostics and formal comparisons."""

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
EXAMPLES = (
    "run_walk_benchmark.py",
    "run_serial_walk_comparison.py",
    "validate_shac_gradients.py",
    "validate_walk_equivalence.py",
    "assemble_walk_gate.py",
    "compare_walk_runs.py",
)


def source_hashes() -> dict[str, str]:
    paths = sorted((ROOT / "src").rglob("*.py"))
    paths += [ROOT / "examples" / name for name in EXAMPLES]
    paths += [ROOT / "uv.lock", ROOT / "pyproject.toml"]
    paths += sorted(
        path for path in (ROOT / "assets/robot/unitree_g1").rglob("*") if path.is_file()
    )
    return {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
    }


def engine_identity(engine: dict[str, Any]) -> dict[str, Any]:
    return {key: engine[key] for key in ("upstream", "patch_sha256", "source_hashes", "zero_copy")}


def task_identity(args: Any) -> tuple[dict[str, Any], str]:
    common = args.model_dump(mode="json")
    common["scene_args"]["sim_options"].pop("requires_grad", None)
    digest = hashlib.sha256(json.dumps(common, sort_keys=True).encode()).hexdigest()
    return common, digest


def evidence_context(engine: dict[str, Any], args: Any, precision: str = "32") -> dict[str, Any]:
    import torch

    device = torch.cuda.get_device_properties(0)
    return {
        "engine": engine_identity(engine),
        "source_hashes": source_hashes(),
        "common_hash": task_identity(args)[1],
        "runtime": {
            "precision": precision,
            "torch": torch.__version__,
            "python": platform.python_version(),
            "quadrants": importlib.metadata.version("quadrants"),
            "numpy": importlib.metadata.version("numpy"),
            "cuda": torch.version.cuda,
            "gpu": device.name,
            "gpu_uuid": str(device.uuid),
            "compute_capability": [device.major, device.minor],
        },
    }


def require_same_context(left: dict[str, Any], right: dict[str, Any]) -> None:
    for key in ("engine", "source_hashes", "common_hash", "runtime"):
        if not left.get(key) or left.get(key) != right.get(key):
            raise ValueError(f"Evidence identity differs or is absent: {key}")


def require_gate(gate: dict[str, Any], current: dict[str, Any]) -> None:
    require_same_context(gate, current)
    if current["runtime"]["precision"] != "32":
        raise ValueError("The formal benchmark requires float32 validation")
    for key in (
        "passed",
        "forward_parity",
        "replay_identity",
        "contact_basis_vjp",
        "contact_output_vjp",
        "actor_update",
    ):
        if gate.get(key) is not True:
            raise ValueError(f"Formal training requires successful {key} evidence")
    if not {1, 4, 24}.issubset(gate.get("horizons", [])):
        raise ValueError("Formal training requires 1/4/24-step gradient evidence")
