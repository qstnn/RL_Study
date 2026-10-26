"""Forward-speed PPO transfer route, with independent stop and contact checks."""

from __future__ import annotations

from isaaclab.utils import configclass

from .ppo_action_curriculum_env_cfg import DeeproboticsLightHWPPOActionCurriculumEnvCfg
from .ppo_forward_speed_curriculum import (
    LightHWForwardSpeedLegActionCfg,
    LightHWForwardSpeedVelocityCommandCfg,
)


@configclass
class DeeproboticsLightHWPPOForwardSpeedEnvCfg(DeeproboticsLightHWPPOActionCurriculumEnvCfg):
    """Learn forward 0..3.2 m/s and reliable stopping using only PPO.

    Start from the learned route-10 stage-2 policy. Its default-relative leg
    mapping and wheel scale are retained, while the permitted wheel target
    range expands explicitly for high speed. All 16 actions remain learned.
    """

    def __post_init__(self):
        super().__post_init__()
        original = self.actions.joint_pos
        self.actions.joint_pos = LightHWForwardSpeedLegActionCfg(
            asset_name="robot",
            joint_names=list(original.joint_names),
            preserve_order=True,
            use_default_offset=True,
            scale=1.0,
            clip=original.clip,
        )

        # 保留 scale=5，避免同一 raw action 突然变成不同轮速。
        # 轮半径 0.0875 m：3.2 m/s 约需 36.57 rad/s；37 低于 URDF 37.7。
        # 扩大原 ±10 的 clip 是显式迁移改动，由专用加载入口检查并记录。
        self.actions.joint_vel.scale = 5.0
        self.actions.joint_vel.clip = {".*_wheel_joint": (-37.0, 37.0)}
        self.commands.base_velocity = LightHWForwardSpeedVelocityCommandCfg(
            asset_name="robot",
            knee_joint_names=self.calf_joint_names,
            resampling_time_range=(8.0, 8.0),
            rel_standing_envs=0.0,
            rel_heading_envs=0.0,
            heading_command=False,
            heading_control_stiffness=0.0,
            debug_vis=False,
            lateral_range=(0.0, 0.0),
            yaw_range=(0.0, 0.0),
            zero_command_probability=0.25,
            minimum_abs_speed=0.08,
            anchor_probability=0.50,
            ranges=LightHWForwardSpeedVelocityCommandCfg.Ranges(
                lin_vel_x=(0.0, 0.2),
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(0.0, 0.0),
                heading=None,
            ),
        )

        # 本路线不以机身高度优化或卡住升阶；原默认角/出生高度仍作初始化。
        self.rewards.base_height_l2 = None
        # 提高前进速度的相对权重，扩宽速度核，让跨速度阶段仍有学习信号。
        self.rewards.track_lin_vel_xy_exp.weight = 2.5
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.50
        # 允许加减速载荷转移，不要求四轮等载；保留支撑和最弱轮奖励。
        self.rewards.wheel_contact_uniformity.weight = -0.005
        # 用户当前优先级是姿态基本正常、无非轮触地、能够可靠停车。
        self.rewards.undesired_contacts.weight = -2.0
        # 其余直立、yaw、动作平滑及停车项继承；停车速度惩罚仍为 -10。
        # policy=60/critic=236，平地1kHz/50Hz及原URDF均由父配置保留。
        if type(self) is DeeproboticsLightHWPPOForwardSpeedEnvCfg:
            print(
                "[LightHW PPO Forward Speed] policy=60, critic=236, action=16; "
                "PPO-only; route10-stage2 transfer; flat forward 0..3.2 m/s; "
                "vy=0, wz=0, stop=25%; moving yaw gate=0.08 rad/s; no height target; "
                "leg limits hip=0.04, thigh/calf=0.10 rad; wheel scale=5, clip=37 rad/s; "
                "independent route 11"
            )
