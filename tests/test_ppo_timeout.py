from collections import deque
from types import SimpleNamespace

import pytest
import torch
from gs_agent.algos.ppo import PPO
from gs_agent.buffers.config.schema import GAEBufferKey


@pytest.mark.parametrize(("fall", "expected"), [(False, 3.0), (True, 1.0)])
def test_timeout_bootstrap_discount_and_fall_precedence(fall: bool, expected: float) -> None:
    obs = torch.zeros(1, 1)
    done = torch.ones(1, 1, dtype=torch.bool)
    env = SimpleNamespace(
        get_observations=lambda: (obs, obs),
        step=lambda _: (
            obs,
            torch.ones(1, 1),
            done,
            done,
            {
                "time_outs": done,
                "observations": {"critic": obs},
                "termination": {"tilt": torch.tensor([fall]), "any": torch.tensor([True])},
                "reward_terms": {"Total": torch.ones(1)},
            },
        ),
    )
    transitions = []
    algo = SimpleNamespace(
        env=env,
        cfg=SimpleNamespace(gamma=0.5),
        _actor=SimpleNamespace(forward_with_dist_params=lambda _: (obs, obs, obs, obs)),
        _critic=lambda _: torch.full((1, 1), 4.0),
        _rollouts=SimpleNamespace(append=transitions.append, set_final_value=lambda _: None),
        _curr_reward_sum=torch.zeros(1),
        _curr_ep_len=torch.zeros(1),
        _rewbuffer=deque(),
        _lenbuffer=deque(),
    )
    result = PPO._collect_rollouts(algo, 1)  # noqa: SLF001
    assert transitions[0][GAEBufferKey.REWARDS].item() == expected
    assert result["mean_reward"] == 1.0
