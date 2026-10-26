"""Independent route 12: preserve route 11 physics and insert the 1.0 m/s stage."""

from __future__ import annotations

from isaaclab.utils import configclass

from .ppo_forward_speed_env_cfg import DeeproboticsLightHWPPOForwardSpeedEnvCfg
from .ppo_speed_bridge_core import BRIDGE_SPEEDS
from .ppo_speed_bridge_curriculum import LightHWSpeedBridgeLegActionCfg


@configclass
class DeeproboticsLightHWPPOSpeedBridgeEnvCfg(DeeproboticsLightHWPPOForwardSpeedEnvCfg):
    """先巩固0.8，再升至1.0与1.2；最终阶段保持，不自动扩大至3.2。"""

    def __post_init__(self):
        super().__post_init__()
        original = self.actions.joint_pos
        self.actions.joint_pos = LightHWSpeedBridgeLegActionCfg(
            asset_name=original.asset_name,
            joint_names=list(original.joint_names),
            preserve_order=original.preserve_order,
            use_default_offset=original.use_default_offset,
            scale=original.scale,
            clip=original.clip,
        )
        # 同一命令采样器从动作课程读取当前上限；只改可选阶段，不改采样分布。
        self.commands.base_velocity.stage_max_speeds = BRIDGE_SPEEDS
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, BRIDGE_SPEEDS[0])
        if type(self) is DeeproboticsLightHWPPOSpeedBridgeEnvCfg:
            print(
                "[LightHW PPO Speed Bridge] policy=60, critic=236, action=16; "
                "PPO-only; route11-stage2 transfer; flat 0.8 -> 1.0 -> 1.2 m/s; "
                "hold at 1.2; vy=0, wz=0, stop=25%; "
                "unchanged rewards, pose, leg limits, wheel mapping and quality gates; independent route 12"
            )
