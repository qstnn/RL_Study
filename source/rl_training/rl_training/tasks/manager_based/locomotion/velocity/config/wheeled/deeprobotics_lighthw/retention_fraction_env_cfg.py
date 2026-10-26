"""独立75%旧技能保留任务；50%只用于验证与旧实现等价。"""
import torch
from isaaclab.utils import configclass
from isaaclab.managers import CurriculumTermCfg
from .spin_xy_env_cfg import LightHWSpinXYNarrowEnvCfg
from .omni_rough_commands import OmniRoughVelocityCommand, OmniRoughVelocityCommandCfg
from .omni_rough_env_cfg import OmniLegAction, OmniLegActionCfg, OmniRoughCurriculum
from .omni_rough_core import COMMAND_CONTRACT
from .safe_terrain_spec import columns_for_surface
from . import retention_fraction_core as allocation


class RetentionVelocityCommand(OmniRoughVelocityCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        # 复用原采样函数与随机调用顺序；只替换确定性的环境归属。
        self.retention.copy_(allocation.retention_mask(torch.arange(self.num_envs, device=self.device), cfg.retention_fraction))


@configclass
class RetentionVelocityCommandCfg(OmniRoughVelocityCommandCfg):
    class_type: type = RetentionVelocityCommand
    retention_fraction: float = 0.75


class RetentionLegAction(OmniLegAction):
    def _signature_payload(self):
        value = super()._signature_payload()
        value['command_distribution'] = allocation.command_contract(
            COMMAND_CONTRACT, self._env.cfg.commands.base_velocity.retention_fraction)
        return value


@configclass
class RetentionLegActionCfg(OmniLegActionCfg):
    class_type: type = RetentionLegAction


class RetentionCurriculum(OmniRoughCurriculum):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.retention_fraction = env.cfg.commands.base_velocity.retention_fraction
        self.retained.copy_(allocation.retention_mask(torch.arange(env.num_envs, device=env.device), self.retention_fraction))
        # 父类已经把stair_mask与旧偶数mask相交，必须从地形列重新构建，不能只继续取交集。
        positive = torch.isin(self.terrain.terrain_types,
            torch.tensor(columns_for_surface('pyramid_stairs'), device=env.device))
        self.stair_mask.copy_(positive & self.retained)
        self.state.stair.copy_(self.stair_mask)

    def snapshot(self):
        result = super().snapshot()
        result.update(version=allocation.COURSE_VERSION,
            retention_rule=allocation.retention_rule(self.retention_fraction),
            retention_fraction=self.retention_fraction,
            retained_environments=int(self.retained.sum()),
            positive_stair_retained_environments=int(self.stair_mask.sum()))
        return result


@configclass
class LightHWRetention75EnvCfg(LightHWSpinXYNarrowEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        old = self.commands.base_velocity
        self.commands.base_velocity = RetentionVelocityCommandCfg(**{
            k: v for k, v in vars(old).items() if k != 'class_type'})
        old = self.actions.joint_pos
        self.actions.joint_pos = RetentionLegActionCfg(**{
            k: v for k, v in vars(old).items() if k != 'class_type'})
        self.curriculum.terrain_levels = CurriculumTermCfg(func=RetentionCurriculum)
        print('[LightHW route42 effective] policy=60 critic=236 action=16; retention75%; '
              'spinXYstd0.5; zero25%; all16 PPO; source terrain levels restored exactly', flush=True)


@configclass
class LightHWRetention50CheckEnvCfg(LightHWRetention75EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.retention_fraction = 0.5
        print('[LightHW route42 validation override] retention50%; equivalence check only', flush=True)
