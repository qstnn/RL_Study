"""Route34：课程出口修复和只读失败拆分；原奖励/终止/动作不变。"""
import math
import torch
from isaaclab.utils import configclass
from isaaclab.managers import CurriculumTermCfg
from .safe_terrain_env_cfg import SafeTraversalCurriculum, progressive_random_terrain
from .safe_terrain_spec import GENERATOR, SURFACES, PROMOTION_TILT_DEG, CONTACT_LIMIT_N, columns_for_surface
from .terrain_traversal_state_v2 import TraversalStateV2
from .terrain_speed_commands import LightHWTerrainVelocityCommandCfg
from .moving_acceleration_env_cfg import LightHWMovingAccelerationEnvCfg
from .m20_full_rough_env_cfg import make_transfer_terrain
from .speed3_env_cfg import set_speed_cap
from .linear_leg_action import LightHWLinearURDFLegActionCfg
from . import linear_leg_spec

VERSION = 'lighthw_safe_traversal_exit_reasons_v2'
EPISODE_REASONS = ('nonwheel_contact', 'tilt_over45', 'nonfinite', 'wrong_assigned_column', 'outside_grid')
REASON_FIELDS = (*EPISODE_REASONS, 'native_unsafe_termination', 'command_distance_under12',
    'no_full_tile_after_enough_command', 'actual_level_increase', 'actual_level_decrease', 'demotion_clamped_at0')


class SafeTraversalCurriculumV2(SafeTraversalCurriculum):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.state = TraversalStateV2(env.num_envs, env.device, GENERATOR['num_rows'])
        with torch.inference_mode(False):
            self.reason_flags = torch.zeros((env.num_envs, len(EPISODE_REASONS)), dtype=torch.bool, device=env.device)
            self.reason_counts = torch.zeros((len(SURFACES), GENERATOR['num_rows'], len(REASON_FIELDS)), dtype=torch.long, device=env.device)

    @torch.no_grad()
    def observe(self, env):
        r = self.robot.data; position = r.root_pos_w
        size = position.new_tensor(self.terrain.cfg.terrain_generator.size)
        start = self.terrain.terrain_origins[0, 0, :2] - size * .5
        cell = torch.floor((position[:, :2] - start) / size).long()
        row, col = cell[:, 0], cell[:, 1]
        inside = (row >= 0) & (row < GENERATOR['num_rows']) & (col >= 0) & (col < GENERATOR['num_cols'])
        # 只读原终止compute已读取的50Hz缓存，不刷新history、不写奖励、不消耗随机数。
        forces = self.sensor.data.net_forces_w
        finite = (torch.isfinite(r.root_state_w).all(1) & torch.isfinite(r.joint_pos).all(1)
                  & torch.isfinite(r.joint_vel).all(1) & torch.isfinite(forces).all((1, 2)))
        nonwheel = (forces[:, self.nonwheel].norm(dim=-1) > CONTACT_LIMIT_N).any(1)
        tilt = r.projected_gravity_b[:, 2] > -math.cos(math.radians(PROMOTION_TILT_DEG))
        wrong = inside & (col != self.terrain.terrain_types)
        self.reason_flags.logical_or_(torch.stack((nonwheel, tilt, ~finite, wrong, ~inside), 1))
        command = env.command_manager.get_command('base_velocity')[:, 0]
        self.state.observe(row, col, inside, command > .05, nonwheel | tilt | ~finite,
            position[:, 0], command, env.step_dt, self.terrain.terrain_types)
        self.samples += 1

    @torch.no_grad()
    def __call__(self, env, env_ids):
        ids = torch.arange(env.num_envs, device=env.device)[env_ids]
        if env.common_step_counter == 0:
            value = super().__call__(env, env_ids)
            env._rough_initial_reset_audit['version'] = VERSION
            return value
        before = self.terrain.terrain_levels[ids].clone()
        kind = self.kinds[self.terrain.terrain_types[ids]]
        sampled = self.state.steps[ids] > 0
        _, _, down, _, _ = self.state.decisions(ids, before, kind == 0, env.termination_manager.terminated[ids])
        value = super().__call__(env, env_ids)
        after = self.terrain.terrain_levels[ids]
        enough = self.state.command_distance[ids] >= 12.
        incomplete = torch.where(before == 9, self.state.forward_distance[ids] < 4., ~self.state.qualified[ids])
        flags = torch.cat((self.reason_flags[ids], torch.stack((env.termination_manager.terminated[ids], ~enough,
            enough & incomplete, after > before, after < before, down & (before == 0) & (kind != 0)), 1)), 1)
        self.reason_counts.view(-1, len(REASON_FIELDS)).index_add_(0, kind * GENERATOR['num_rows'] + before,
            (flags & sampled[:, None]).long())
        return value

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        super().reset(env_ids)
        self.reason_flags[ids] = False

    def snapshot(self):
        values = super().snapshot()
        values.update(version=VERSION, reason_fields=list(REASON_FIELDS),
            cumulative_reasons_by_family_and_spawn_level=self.reason_counts.cpu().tolist(),
            reasons_can_overlap=True,
            note='50Hz diagnostic episodes; unsafe combines reasons. demotions includeslevel0 clamping; actual_level_decrease counts real decreases. Outside grid is recorded, not a new safety failure.')
        return values


@configclass
class LightHWTerrainSpeedEnvCfg(LightHWMovingAccelerationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        set_speed_cap(self, 3.)
        old = self.actions.joint_pos
        self.actions.joint_pos = LightHWLinearURDFLegActionCfg(**{k:v for k,v in vars(old).items() if k != 'class_type'})
        self.actions.joint_pos.scale = linear_leg_spec.SCALE.copy()
        self.actions.joint_pos.clip = linear_leg_spec.LIMITS.copy()
        self.scene.terrain.terrain_type = 'generator'
        self.scene.terrain.terrain_generator = make_transfer_terrain()
        self.scene.terrain.terrain_generator.sub_terrains['random_rough'].function = progressive_random_terrain
        self.scene.terrain.max_init_terrain_level = 0
        self.curriculum.terrain_levels = CurriculumTermCfg(func=SafeTraversalCurriculumV2)
        old = self.commands.base_velocity
        self.commands.base_velocity = LightHWTerrainVelocityCommandCfg(**{k:v for k,v in vars(old).items() if k != 'class_type'})
        print('[LightHW route34] policy=60, critic=236, action=16; linear URDF legs; wheel scale5; '
              'fixed terminal-tile exit, terrain-specific commands; PPO/rewards/PD/raw-noise unchanged', flush=True)


@configclass
class LightHWTerrainSpeedControlEnvCfg(LightHWTerrainSpeedEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.rough_speed_cap = 3.
