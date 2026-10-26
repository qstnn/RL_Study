"""Independent PPO-only ablation: penalize saturated thigh/calf raw actions."""

from __future__ import annotations

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from .adapted_rewards import lighthw_joint_action_saturation_l1
from .ppo_grouped_noise_env_cfg import DeeproboticsLightHWGroupedNoiseEnvCfg
from .ppo_leg_desaturation_core import (
    DESATURATION_JOINT_NAMES,
    DESATURATION_RAW_LIMIT,
    DESATURATION_REWARD_NAME,
    DESATURATION_WEIGHT,
)
from .ppo_leg_desaturation_curriculum import LightHWLegDesaturationLegActionCfg


@configclass
class DeeproboticsLightHWLegDesaturationEnvCfg(DeeproboticsLightHWGroupedNoiseEnvCfg):
    """仅增加八个thigh/calf的raw hinge惩罚，不改变轮腿探索与物理动作映射。"""

    def __post_init__(self):
        super().__post_init__()
        original = self.actions.joint_pos
        self.actions.joint_pos = LightHWLegDesaturationLegActionCfg(
            asset_name=original.asset_name,
            joint_names=list(original.joint_names),
            preserve_order=original.preserve_order,
            use_default_offset=original.use_default_offset,
            scale=original.scale,
            clip=original.clip,
        )
        # 对全部命令激活；按解析后的动作关节名称取值，不假设关节在PhysX
        # 中的排列。惩罚采样raw动作超过95% tanh包络的部分；只作为PPO
        # 环境奖励，不添加监督loss、腿控制器或额外目标角约束。
        setattr(self.rewards, DESATURATION_REWARD_NAME, RewTerm(
            func=lighthw_joint_action_saturation_l1,
            weight=DESATURATION_WEIGHT,
            params={
                "action_term_name": "joint_pos",
                "joint_names": list(DESATURATION_JOINT_NAMES),
                "raw_limit": DESATURATION_RAW_LIMIT,
            },
        ))
        print(
            "[LightHW PPO Leg Desaturation] policy=60, critic=236, action=16; PPO-only; "
            "independent route14; flat 0..1.0 m/s, stop=25%; grouped noise/pose/actions unchanged; "
            f"added {DESATURATION_REWARD_NAME}: 8 thigh/calf, raw_limit={DESATURATION_RAW_LIMIT:.9f}, "
            f"weight={DESATURATION_WEIGHT:g}, all commands"
        )
