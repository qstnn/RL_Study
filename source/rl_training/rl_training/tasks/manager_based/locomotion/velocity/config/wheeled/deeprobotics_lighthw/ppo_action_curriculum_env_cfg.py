"""Independent PPO-only LightHW route with a bounded leg-action curriculum."""

from __future__ import annotations

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .flat_straight_stable_physics_stand_hold_precision_env_cfg import (
    DeeproboticsLightHWFlatStraightStablePhysicsStandHoldPrecisionEnvCfg,
)
from .ppo_action_curriculum import (
    LightHWActionCurriculumVelocityCommandCfg,
    LightHWLegPositionActionCfg,
    base_height_band_l2,
    bounded_action_rate_l2,
    record_ppo_curriculum_metrics,
    weakest_wheel_support,
)
from .rough_env_cfg import FOOT_LINK_PATTERN


@configclass
class DeeproboticsLightHWPPOActionCurriculumEnvCfg(
    DeeproboticsLightHWFlatStraightStablePhysicsStandHoldPrecisionEnvCfg
):
    """Learn all 16 actions with PPO, starting near the original leg posture.

    The actor has 60 observations and the privileged critic retains 236.
    Demonstrations, behavior cloning, fixed wheel commands, and command-to-wheel
    feedforward are not used.  The action term owns progression and checkpointed
    curriculum state; train and play instantiate this same environment class.
    """

    def __post_init__(self):
        super().__post_init__()

        # 保留原姿态和 1 kHz 物理/50 Hz 控制。课程仅限制默认角周围的
        # 目标偏移，clip 仍是 URDF 对应的绝对关节角，不能当作偏移范围。
        original_leg_action = self.actions.joint_pos
        self.actions.joint_pos = LightHWLegPositionActionCfg(
            asset_name="robot",
            joint_names=list(original_leg_action.joint_names),
            preserve_order=True,
            use_default_offset=True,
            scale=1.0,
            clip=original_leg_action.clip,
        )
        # 四轮始终由 policy 独立输出；±10 rad/s 覆盖本路线 ±0.6 m/s。
        # 映射是本路线专用，旧 checkpoint 即使维度相同也不能直接迁入。
        self.actions.joint_vel.scale = 5.0
        self.actions.joint_vel.clip = {".*_wheel_joint": (-10.0, 10.0)}

        self.commands.base_velocity = LightHWActionCurriculumVelocityCommandCfg(
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
            ranges=LightHWActionCurriculumVelocityCommandCfg.Ranges(
                lin_vel_x=(-0.60, 0.60),
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(0.0, 0.0),
                heading=None,
            ),
        )

        # 低速移动不应触发更强的静止姿态惩罚。三类腿关节统一门控；
        # 零命令机身保持项和六项静止门控继续继承 StandHoldPrecision。
        for name in ("hipx_joint_pos_penalty", "hipy_joint_pos_penalty", "knee_joint_pos_penalty"):
            term = getattr(self.rewards, name)
            term.params["command_threshold"] = 0.05
            term.params["stand_still_scale"] = 2.0
        # 已有按关节范围归一的左右对称项，避免重复累计旧 mirror 项。
        self.rewards.joint_mirror = None

        # 原姿态受力后约在 0.397~0.402 m 稳态；保留名义 0.416 m 的
        # 默认姿态，只在合理高度区间之外处罚，不强迫精确抬到名义高度。
        self.rewards.base_height_l2 = RewTerm(
            func=base_height_band_l2,
            weight=-3.0,
            params={
                "minimum": 0.395,
                "maximum": 0.425,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.rewards.action_rate_l2 = RewTerm(func=bounded_action_rate_l2, weight=-0.01)
        self.rewards.weakest_wheel_support = RewTerm(
            func=weakest_wheel_support,
            weight=0.20,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[FOOT_LINK_PATTERN]),
                "threshold": 5.0,
            },
        )
        # 返回零的记录项不改变奖励，仅收集独立课程判据和诊断量。
        # 非零权重避免被继承配置的零权重清理删除。
        self.rewards.ppo_curriculum_metrics = RewTerm(func=record_ppo_curriculum_metrics, weight=1.0)

        if type(self) is DeeproboticsLightHWPPOActionCurriculumEnvCfg:
            print(
                "[LightHW PPO Action Curriculum] policy=60, critic=236, action=16; "
                "PPO-only, original_pose=0.416 m, physics=1000 Hz, policy=50 Hz; "
                "initial vx=(-0.2, 0.2), maximum vx=(-0.6, 0.6), vy=0, wz=0, zero=25%; "
                "all 16 actions learned, no wheel feedforward; independent route 10"
            )
