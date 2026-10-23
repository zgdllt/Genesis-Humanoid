# ruff: noqa: SLF001
# Exercise randomization state across repeated resets without building a simulator.
from types import SimpleNamespace

import torch
from gs_env.sim.robots.leggedrobots import LeggedRobotBase


def test_repeated_mass_and_com_randomization_does_not_accumulate() -> None:
    mass_calls = []
    com_calls = []

    def set_mass(value: torch.Tensor, links: list[int], envs: torch.Tensor) -> None:
        assert links == [0]
        mass_calls.append((envs.clone(), value.clone()))

    def set_com(value: torch.Tensor, links: list[int], envs: torch.Tensor) -> None:
        assert links == [0]
        com_calls.append((envs.clone(), value.clone()))

    def set_friction(ratios: torch.Tensor, geoms: torch.Tensor, envs: torch.Tensor) -> None:
        assert ratios.device == envs.device == geoms.device

    robot = LeggedRobotBase.__new__(LeggedRobotBase)
    robot._device = torch.device("cpu")
    robot.body_link_idx = 0
    robot._args = SimpleNamespace(
        dr_args=SimpleNamespace(
            friction_range=(1.0, 1.0), mass_range=(1.0, 1.0), com_displacement_range=(0.01, 0.01)
        )
    )
    robot._robot = SimpleNamespace(
        solver=SimpleNamespace(n_geoms=1, set_geoms_friction_ratio=set_friction),
        set_links_mass=set_mass,
        set_links_COM=set_com,
    )
    robot._default_body_mass = torch.tensor([[10.0], [20.0]])
    robot._default_body_com = torch.full((2, 1, 3), 0.1)
    robot._friction_ratio = torch.zeros(2, 1)
    robot._added_mass = torch.zeros(2, 1)
    robot._com_displacement = torch.zeros(2, 3)

    for _ in range(2):
        robot._randomize_rigids(torch.tensor([0, 1]))
    robot._randomize_rigids(torch.tensor([1]))

    torch.testing.assert_close(mass_calls[0][1], torch.tensor([[11.0], [21.0]]))
    torch.testing.assert_close(mass_calls[1][1], mass_calls[0][1])
    torch.testing.assert_close(mass_calls[2][1], torch.tensor([[21.0]]))
    torch.testing.assert_close(com_calls[0][1], torch.full((2, 1, 3), 0.11))
    torch.testing.assert_close(com_calls[1][1], com_calls[0][1])
    torch.testing.assert_close(com_calls[2][1], torch.full((1, 1, 3), 0.11))
    torch.testing.assert_close(robot._default_body_mass, torch.tensor([[10.0], [20.0]]))
    torch.testing.assert_close(robot._default_body_com, torch.full((2, 1, 3), 0.1))
