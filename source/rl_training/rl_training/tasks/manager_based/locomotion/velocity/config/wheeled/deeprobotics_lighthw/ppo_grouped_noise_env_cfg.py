"""Independent route 13: change exploration while holding the speed curriculum."""

from __future__ import annotations

from isaaclab.utils import configclass

from .ppo_grouped_noise_core import GROUPED_NOISE_SPEEDS
from .ppo_grouped_noise_curriculum import LightHWGroupedNoiseLegActionCfg
from .ppo_speed_bridge_env_cfg import DeeproboticsLightHWPPOSpeedBridgeEnvCfg


@configclass
class DeeproboticsLightHWGroupedNoiseEnvCfg(DeeproboticsLightHWPPOSpeedBridgeEnvCfg):
    """先在相同1.0命令做短PPO对照；不改奖励、姿态、动作范围和质量门槛。"""

    def __post_init__(self):
        super().__post_init__()
        original = self.actions.joint_pos
        self.actions.joint_pos = LightHWGroupedNoiseLegActionCfg(
            asset_name=original.asset_name,
            joint_names=list(original.joint_names),
            preserve_order=original.preserve_order,
            use_default_offset=original.use_default_offset,
            scale=original.scale,
            clip=original.clip,
        )
        self.commands.base_velocity.stage_max_speeds = GROUPED_NOISE_SPEEDS
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, GROUPED_NOISE_SPEEDS[0])
        if type(self) is DeeproboticsLightHWGroupedNoiseEnvCfg:
            print(
                "[LightHW PPO Grouped Noise] policy=60, critic=236, action=16; PPO-only; "
                "route12-stage1 transfer; flat 0..1.0 m/s, hold at 1.0; vy=0, wz=0, stop=25%; "
                "unchanged rewards, pose, leg limits, wheel mapping and quality gates; independent route 13"
            )
