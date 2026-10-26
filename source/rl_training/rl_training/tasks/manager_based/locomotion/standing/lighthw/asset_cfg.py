"""独立站立资产：复用规范 URDF，只修改深拷贝后的仿真配置。"""

from copy import deepcopy
from pathlib import Path

from rl_training.assets import ISAACLAB_ASSETS_EXT_DIR
from rl_training.assets.deeprobotics import LIGHTHW_CFG

from .spec import DEFAULT_JOINT_POS, SPAWN_HEIGHT

REPO_ROOT = Path(ISAACLAB_ASSETS_EXT_DIR).resolve().parents[1]
STAND_ROUTE = REPO_ROOT / "model_lighthw/training/stand_route"
LIGHTHW_STAND_URDF = REPO_ROOT / "model_lighthw/lighthw_urdf/urdf/lighthw.urdf"

LIGHTHW_STAND_CFG = deepcopy(LIGHTHW_CFG)
LIGHTHW_STAND_CFG.spawn.asset_path = str(LIGHTHW_STAND_URDF)
# 转换产物独立存放，避免站立任务与 rough 共用可写 USD 缓存。
LIGHTHW_STAND_CFG.spawn.usd_dir = str(STAND_ROUTE / "outputs/usd")
LIGHTHW_STAND_CFG.spawn.usd_file_name = "lighthw_stand.usd"
LIGHTHW_STAND_CFG.init_state.pos = (0.0, 0.0, SPAWN_HEIGHT)
LIGHTHW_STAND_CFG.init_state.joint_pos = dict(DEFAULT_JOINT_POS)
LIGHTHW_STAND_CFG.init_state.joint_vel = {".*": 0.0}
for actuator in LIGHTHW_STAND_CFG.actuators.values():
    actuator.min_delay = 0
    actuator.max_delay = 0
LIGHTHW_STAND_CFG.actuators["joint"].stiffness = 80.0
LIGHTHW_STAND_CFG.actuators["joint"].damping = 3.0
LIGHTHW_STAND_CFG.actuators["wheel"].damping = 1.0
