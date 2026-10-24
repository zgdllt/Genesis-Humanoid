import copy

import pytest
from gs_agent.utils.walk_evidence import require_gate, require_same_context


def context() -> dict:
    return {
        "engine": {"upstream": "pinned", "patch_sha256": "validated"},
        "source_hashes": {"algorithm.py": "hash"},
        "common_hash": "task",
        "runtime": {"precision": "32", "gpu_uuid": "GPU-test"},
    }


def test_formal_gate_rejects_missing_evidence_and_partial_horizon_coverage() -> None:
    current = context()
    gate = {
        **current,
        **{
            key: True
            for key in (
                "passed",
                "forward_parity",
                "replay_identity",
                "contact_basis_vjp",
                "contact_output_vjp",
                "actor_update",
            )
        },
        "horizons": [1, 4, 24],
    }
    require_gate(gate, current)
    for key in (
        "passed",
        "forward_parity",
        "replay_identity",
        "contact_basis_vjp",
        "contact_output_vjp",
        "actor_update",
    ):
        broken = gate.copy()
        broken.pop(key)
        with pytest.raises(ValueError, match=key):
            require_gate(broken, current)
    with pytest.raises(ValueError, match="1/4/24"):
        require_gate({**gate, "horizons": [1, 4]}, current)


@pytest.mark.parametrize("key", ["engine", "source_hashes", "common_hash", "runtime"])
def test_changed_identity_cannot_reuse_old_validation(key: str) -> None:
    changed = copy.deepcopy(context())
    changed[key] = "different"
    with pytest.raises(ValueError, match=key):
        require_same_context(context(), changed)


def test_double_precision_diagnostics_do_not_authorize_float32_training() -> None:
    double = context()
    double["runtime"]["precision"] = "64"
    with pytest.raises(ValueError, match="runtime"):
        require_same_context(double, context())
    with pytest.raises(ValueError, match="float32"):
        require_gate(double, double)
