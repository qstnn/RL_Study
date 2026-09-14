"""LightHW asset binding for the isolated training workspace.

The shared :mod:`rl_training.assets.deeprobotics` module provides the common
physical parameters.  This route clones that configuration and binds it to the
single canonical URDF under ``model_lighthw``.
"""

from copy import deepcopy
from pathlib import Path

from rl_training.assets import ISAACLAB_ASSETS_EXT_DIR
from rl_training.assets.deeprobotics import LIGHTHW_CFG


_REPO_ROOT = Path(ISAACLAB_ASSETS_EXT_DIR).resolve().parents[1]
LIGHTHW_WORKSPACE_URDF = (
    _REPO_ROOT / "model_lighthw" / "lighthw_urdf" / "urdf" / "lighthw.urdf"
).resolve()

# Keep every physical field (mass, inertias, actuator limits, default pose,
# collision conversion flags) identical to LIGHTHW_CFG.  Only the source path
# is redirected to the copied, route-owned asset.
LIGHTHW_ROUTE_CFG = deepcopy(LIGHTHW_CFG)
LIGHTHW_ROUTE_CFG.spawn.asset_path = str(LIGHTHW_WORKSPACE_URDF)
