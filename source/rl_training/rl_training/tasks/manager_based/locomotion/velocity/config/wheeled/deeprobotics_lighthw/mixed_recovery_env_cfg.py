"""Route53 独立混合能力恢复：只改变任务/出生采样，原奖励和PPO物理接口保留。"""
from copy import deepcopy
import torch
from isaaclab.utils import configclass
from isaaclab.managers import CurriculumTermCfg, ManagerTermBase
from .right_lateral_env_cfg import LightHWRightLateralEnvCfg, RightLateralLegAction
from .omni_rough_commands import OmniRoughVelocityCommand, OmniRoughVelocityCommandCfg
from .lateral_tracking_env_cfg import LightHWLateralTrackingEnvCfg
from .safe_terrain_spec import columns_for_surface
from .stair_flight_state import bottom_origins
from .mixed_recovery_core import COMMAND_CONTRACT, VERSION, roles_for_ids, commands_from_uniform


class MixedRecoveryCommand(OmniRoughVelocityCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        with torch.inference_mode(False):
            self.roles = roles_for_ids(torch.arange(self.num_envs, device=self.device), self.num_envs)
            self.retention = (self.roles == 0) | (self.roles == 4)

    def _resample_command(self, env_ids):
        ids = self._ids(env_ids)
        if not ids.numel():
            return
        cmd, stop = commands_from_uniform(self.roles[ids], torch.rand((len(ids), 7), device=self.device))
        self.sampled_command[ids] = cmd
        self.is_standing_env[ids] = stop
        self.is_heading_env[ids] = False
        modes = torch.tensor([1, 4, 3, 5, 1], device=self.device)
        self.mode[ids] = torch.where(stop, 0, modes[self.roles[ids]])
        self._update_command()

    def _update_command(self):
        # 命令分组已经规定地形适用范围，不调用父类的保留组/heading混合规则。
        self.vel_command_b.copy_(self.sampled_command)
        self.vel_command_b[self.is_standing_env] = 0.


@configclass
class MixedRecoveryCommandCfg(OmniRoughVelocityCommandCfg):
    class_type: type = MixedRecoveryCommand


class MixedRecoveryLegAction(RightLateralLegAction):
    def _signature_payload(self):
        value = super()._signature_payload()
        value['command_distribution'] = deepcopy(COMMAND_CONTRACT)
        return value


class FixedMixedTaskSampling(ManagerTermBase):
    """每次原生reset前选择固定任务出生点；不晋级、不新增终止或控制逻辑。"""
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.terrain = env.scene.terrain
        with torch.inference_mode(False):
            ids = torch.arange(env.num_envs, device=env.device)
            self.roles = roles_for_ids(ids, env.num_envs)
            self.stairs = self.roles == 4
            flat = torch.tensor(columns_for_surface('flat'), device=env.device)
            self.columns = torch.full_like(ids, int(flat[len(flat) // 2]))
            self.columns[self.stairs] = columns_for_surface('pyramid_stairs')[0]
            levels = torch.tensor([0, 2, 4, 6], device=env.device)
            self.levels = torch.where(self.stairs, levels[ids % 4], 0)
            self.resets = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        env._mixed_task_sampling = self

    @torch.no_grad()
    def __call__(self, env, env_ids):
        ids = torch.arange(env.num_envs, device=env.device)[env_ids]
        t = self.terrain
        t.terrain_types[ids] = self.columns[ids]
        t.terrain_levels[ids] = self.levels[ids]
        t.env_origins[ids] = bottom_origins(t.terrain_origins, self.levels[ids], self.columns[ids], self.stairs[ids])
        self.resets[ids] += 1
        return t.terrain_levels.float().mean()

    def snapshot(self):
        return dict(version=VERSION, roles=self.roles.cpu().tolist(), levels=self.terrain.terrain_levels.cpu().tolist(),
                    columns=self.terrain.terrain_types.cpu().tolist(), reset_counts=self.resets.cpu().tolist(),
                    progression=False, source_flat_levels_restored=False)


@configclass
class LightHWMixedRecoveryEnvCfg(LightHWRightLateralEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # 完整原seed42地图保证与既有复测楼梯高度一致；出生只取平地/正向楼梯。
        self.scene.terrain.terrain_generator = LightHWLateralTrackingEnvCfg().scene.terrain.terrain_generator
        self.scene.terrain.max_init_terrain_level = 0
        self.curriculum.terrain_levels = CurriculumTermCfg(func=FixedMixedTaskSampling)
        self.actions.joint_pos.class_type = MixedRecoveryLegAction
        old = self.commands.base_velocity
        values = {k: v for k, v in vars(old).items() if k not in ('class_type', 'right_probability')}
        self.commands.base_velocity = MixedRecoveryCommandCfg(**values)
        self.commands.base_velocity.ranges.lin_vel_x = (0., 3.)
        self.commands.base_velocity.ranges.lin_vel_y = (-1., 1.)
        self.commands.base_velocity.ranges.ang_vel_z = (-1., 1.)
        print('[LightHW route53] policy=60 critic=236 action=16; all16 PPO; '
              '30%forward/20%yaw/20%lateral/10%combined/20%stairs;25%zero per role; '
              'fixed stair spawn levels0/2/4/6; original rewards/actions/PD/mirror/noise', flush=True)
