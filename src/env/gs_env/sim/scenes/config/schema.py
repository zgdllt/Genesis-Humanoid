from typing import Any, TypeAlias

import genesis as gs
from gs_schemas.base_types import genesis_pydantic_config
from pydantic import (
    BaseModel,
    SerializerFunctionWrapHandler,
    ValidationInfo,
    field_serializer,
    field_validator,
)


class BaseSceneArgs(BaseModel):
    model_config = genesis_pydantic_config(frozen=True)
    center_envs_at_origin: bool
    compile_kernels: bool

    scene_type: str

    sim_options: gs.options.SimOptions
    tool_options: gs.options.ToolOptions
    rigid_options: gs.options.RigidOptions
    mpm_options: gs.options.MPMOptions
    fem_options: gs.options.FEMOptions
    sf_options: gs.options.SFOptions
    vis_options: gs.options.VisOptions
    viewer_options: gs.options.ViewerOptions

    show_viewer: bool
    show_FPS: bool

    @field_validator(
        "tool_options", "rigid_options", "mpm_options", "fem_options", "sf_options", mode="before"
    )
    @classmethod
    def _restore_solver_options(cls, value: Any, info: ValidationInfo) -> Any:
        if not isinstance(value, dict):
            return value
        options = dict(value)
        # In Genesis 1.4, an omitted dt inherits SimOptions; an explicit None does not.
        if options.get("dt") is None:
            options.pop("dt", None)
        if info.field_name == "rigid_options":
            options.pop("hibernation_thresh_acc", None)
            options.pop("use_contact_island", None)
            timeconst = options.pop("contact_resolve_time", None)
            if timeconst is not None:
                options.setdefault("constraint_timeconst", timeconst)
        elif info.field_name == "mpm_options":
            options.pop("use_sparse_grid", None)
            options.pop("leaf_block_size", None)
        return options

    @field_serializer(
        "tool_options", "rigid_options", "mpm_options", "fem_options", "sf_options", mode="wrap"
    )
    def _serialize_solver_options(
        self, value: Any, handler: SerializerFunctionWrapHandler
    ) -> dict[str, Any]:
        options = handler(value)
        if "dt" not in value.model_fields_set or value.dt is None:
            options.pop("dt", None)
        return options

    @field_validator("vis_options", "viewer_options", mode="before")
    @classmethod
    def _restore_visual_options(cls, value: Any, info: ValidationInfo) -> Any:
        if not isinstance(value, dict):
            return value
        options = dict(value)
        if info.field_name == "vis_options":
            count = options.pop("n_rendered_envs", None)
            if count is not None:
                if options.get("rendered_envs_idx") is not None:
                    raise ValueError("Cannot specify both n_rendered_envs and rendered_envs_idx.")
                options["rendered_envs_idx"] = tuple(range(count))
            aliases = (("env_separate_rigid", "split_envs", False),)
        else:
            # Old checkpoints stored refresh_rate's default alongside an explicit max_FPS.
            legacy_viewer = "disable_help_text" in options or "disable_default_keybinds" in options
            if legacy_viewer and options.get("max_FPS") is not None:
                options.pop("refresh_rate", None)
            aliases = (
                ("max_FPS", "refresh_rate", False),
                ("disable_help_text", "enable_help_text", True),
                ("disable_default_keybinds", "enable_default_keybinds", True),
            )
        for old, new, invert in aliases:
            legacy = options.pop(old, None)
            if legacy is not None:
                if options.get(new) is not None:
                    raise ValueError(f"Cannot specify both {old} and {new}.")
                options[new] = not legacy if invert else legacy
        return options


class FlatSceneArgs(BaseSceneArgs):
    model_config = genesis_pydantic_config(frozen=True)
    normal: tuple[float, float, float] = (0.0, 0.0, 1.0)


class CustomSceneArgs(BaseSceneArgs):
    model_config = genesis_pydantic_config(frozen=True)
    normal: tuple[float, float, float] = (0.0, 0.0, 1.0)
    remove_ground: bool = False
    objects: list[dict[Any, Any]] = []


SceneArgs: TypeAlias = FlatSceneArgs | CustomSceneArgs
