import torch
from gs_agent.algos.shac_returns import short_horizon_return, td_lambda_targets


def test_actor_bootstraps_horizon_and_timeout_but_not_falls() -> None:
    r = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [100.0, 100.0, 3.0]], requires_grad=True)
    v = torch.full_like(r, 10.0, requires_grad=True)
    falls = torch.zeros_like(r, dtype=torch.bool)
    timeouts = falls.clone()
    falls[1, 0] = True
    timeouts[1, 1] = True
    result = short_horizon_return(r, v, falls, timeouts, 0.5)
    torch.testing.assert_close(result, torch.tensor([2.0, 4.5, 4.0]))
    result.sum().backward()
    torch.testing.assert_close(r.grad[2], torch.tensor([0.0, 0.0, 0.25]))
    torch.testing.assert_close(v.grad[1], torch.tensor([0.0, 0.25, 0.0]))
    torch.testing.assert_close(v.grad[2], torch.tensor([0.0, 0.0, 0.125]))


def test_td_lambda_terminal_masks_and_no_graph() -> None:
    r = torch.ones(3, 2, requires_grad=True)
    v = torch.full_like(r, 4.0, requires_grad=True)
    falls = torch.zeros_like(r, dtype=torch.bool)
    timeout = falls.clone()
    falls[1, 0] = True
    timeout[1, 1] = True
    targets, valid = td_lambda_targets(r, v, falls, timeout, 0.5, 1.0)
    torch.testing.assert_close(targets[:2], torch.tensor([[1.5, 2.5], [1.0, 3.0]]))
    assert valid.tolist() == [[True, True], [True, True], [False, False]]
    assert not targets.requires_grad


def test_lambda_zero_is_one_step_td() -> None:
    r = torch.arange(6.0).view(3, 2)
    v = torch.full_like(r, 2.0)
    done = torch.zeros_like(r, dtype=torch.bool)
    targets, _ = td_lambda_targets(r, v, done, done, 0.9, 0.0)
    torch.testing.assert_close(targets, r + 1.8)
