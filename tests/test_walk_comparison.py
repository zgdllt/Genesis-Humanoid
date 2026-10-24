import copy

import pytest
from gs_agent.utils.walk_benchmark import PROTOCOL

from examples.compare_walk_runs import validate_history


def history() -> list[dict]:
    return [
        {
            "iteration": iteration,
            "counters": {
                "train_env_steps": iteration * 4096 * 24,
                "eval_env_steps": (index + 1) * 128 * 1000,
                "wall_time_s": index * 10 + 2,
                "active_train_s": index * 8,
                "eval_time_s": index + 1,
            },
            "quality": {
                "return": 1.0,
                "survival": 0.99,
                "velocity_rmse": 0.1,
                "yaw_rmse": 0.1,
                "foot_slip": 0.02,
            },
        }
        for index, iteration in enumerate(range(0, 50001, PROTOCOL["eval_interval"]))
    ]


def test_history_requires_all_scheduled_evaluations() -> None:
    rows = history()
    validate_history(rows)
    for broken in (rows[:-1], rows[:4] + rows[5:], rows[:5] + [rows[4]] + rows[5:]):
        with pytest.raises(ValueError, match="scheduled"):
            validate_history(broken)


@pytest.mark.parametrize(
    "area,key,value",
    [
        ("counters", "train_env_steps", 1),
        ("counters", "eval_env_steps", 1),
        ("counters", "wall_time_s", 0),
        ("counters", "active_train_s", float("nan")),
        ("quality", "velocity_rmse", float("inf")),
        ("quality", "survival", 1.1),
    ],
)
def test_history_rejects_invalid_counters_and_quality(area: str, key: str, value: float) -> None:
    rows = copy.deepcopy(history())
    rows[3][area][key] = value
    with pytest.raises(ValueError):
        validate_history(rows)
