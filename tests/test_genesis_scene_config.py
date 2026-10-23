from pathlib import Path

import pytest
from gs_env.sim.envs.config.registry import EnvArgsRegistry
from pydantic import ValidationError

from examples.utils import config_to_yaml, yaml_to_config

_SOLVER_FIELDS = ("tool_options", "rigid_options", "mpm_options", "fem_options", "sf_options")


@pytest.mark.parametrize("env_name", ("g1_walk", "g1_motion_teacher", "g1_motion"))
def test_environment_yaml_preserves_inherited_solver_timing(tmp_path: Path, env_name: str) -> None:
    original = EnvArgsRegistry[env_name]
    path = tmp_path / "env.yaml"
    config_to_yaml(original, path)
    restored = yaml_to_config(path, type(original))

    assert restored.scene_args.sim_options.dt == original.scene_args.sim_options.dt
    assert restored.scene_args.sim_options.substeps == original.scene_args.sim_options.substeps
    assert restored.reward_args == original.reward_args
    for field in _SOLVER_FIELDS:
        assert "dt" not in getattr(restored.scene_args, field).model_fields_set


@pytest.mark.parametrize("field", _SOLVER_FIELDS)
def test_legacy_null_and_explicit_numeric_solver_timing(field: str) -> None:
    original = EnvArgsRegistry["g1_walk"].scene_args
    data = original.model_dump()
    data[field]["dt"] = None
    restored = type(original).model_validate(data)
    assert "dt" not in getattr(restored, field).model_fields_set
    assert "dt" not in restored.model_dump()[field]

    data[field]["dt"] = 0.0025
    restored = type(original).model_validate(data)
    assert getattr(restored, field).dt == 0.0025
    assert restored.model_dump()[field]["dt"] == 0.0025


def test_old_scene_options_load_before_genesis_initialization() -> None:
    original = EnvArgsRegistry["g1_walk"].scene_args
    data = original.model_dump()
    data["rigid_options"].update(
        hibernation_thresh_acc=0.01, use_contact_island=False, contact_resolve_time=None
    )
    data["mpm_options"].update(use_sparse_grid=False, leaf_block_size=8)
    data["vis_options"].pop("split_envs")
    data["vis_options"].pop("rendered_envs_idx")
    data["vis_options"].update(env_separate_rigid=False, n_rendered_envs=2)
    viewer = data["viewer_options"]
    viewer["refresh_rate"] = 60
    viewer.pop("enable_help_text")
    viewer.pop("enable_default_keybinds")
    viewer.update(max_FPS=50, disable_help_text=False, disable_default_keybinds=True)

    restored = type(original).model_validate(data)
    assert restored.vis_options.split_envs is False
    assert tuple(restored.vis_options.rendered_envs_idx) == (0, 1)
    assert restored.viewer_options.refresh_rate == 50
    assert restored.viewer_options.enable_help_text is True
    assert restored.viewer_options.enable_default_keybinds is False


def test_conflicting_legacy_and_current_viewer_options_are_rejected() -> None:
    original = EnvArgsRegistry["g1_walk"].scene_args
    data = original.model_dump()
    data["viewer_options"]["max_FPS"] = 100
    with pytest.raises(ValidationError, match="Cannot specify both max_FPS and refresh_rate"):
        type(original).model_validate(data)
