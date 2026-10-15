"""LightHW asset binding for the isolated training workspace.

The shared :mod:`rl_training.assets.deeprobotics` module provides the common
physical parameters.  This route clones that configuration and binds it to the
single canonical URDF under ``model_lighthw``.
"""

from copy import deepcopy
from pathlib import Path

from rl_training.assets import ISAACLAB_ASSETS_EXT_DIR
from rl_training.assets.deeprobotics import LIGHTHW_CFG

from .default_pose import LIGHTHW_DEFAULT_JOINT_POS, LIGHTHW_DEFAULT_SPAWN_HEIGHT


_REPO_ROOT = Path(ISAACLAB_ASSETS_EXT_DIR).resolve().parents[1]
LIGHTHW_WORKSPACE_URDF = (
    _REPO_ROOT / "model_lighthw" / "lighthw_urdf" / "urdf" / "lighthw.urdf"
).resolve()

# Physical fields (mass, inertias, actuator limits and collision conversion)
# come from LIGHTHW_CFG.  The current route owns its URDF path and original
# low-speed reference pose; the shared asset keeps the same legacy values.
LIGHTHW_ROUTE_CFG = deepcopy(LIGHTHW_CFG)
LIGHTHW_ROUTE_CFG.spawn.asset_path = str(LIGHTHW_WORKSPACE_URDF)
LIGHTHW_ROUTE_CFG.init_state.pos = (0.0, 0.0, LIGHTHW_DEFAULT_SPAWN_HEIGHT)
LIGHTHW_ROUTE_CFG.init_state.joint_pos = dict(LIGHTHW_DEFAULT_JOINT_POS)
