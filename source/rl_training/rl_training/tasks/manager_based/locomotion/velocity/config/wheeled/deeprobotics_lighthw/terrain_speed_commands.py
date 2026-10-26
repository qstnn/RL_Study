"""独立地形命令：平地高速，rough先通行；不改变腿轮动作映射。"""
import torch
from isaaclab.utils import configclass
from .ppo_forward_speed_curriculum import LightHWForwardSpeedVelocityCommand, LightHWForwardSpeedVelocityCommandCfg
from .safe_terrain_spec import columns_for_surface
from .terrain_speed_core import flat_at_position, terrain_command


class LightHWTerrainVelocityCommand(LightHWForwardSpeedVelocityCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        with torch.inference_mode(False):
            self._original_vx = torch.zeros(self.num_envs, device=self.device)
        self._flat_columns = columns_for_surface('flat')

    def terrain_masks(self):
        terrain = self._env.scene.terrain
        return flat_at_position(self.robot.data.root_pos_w, terrain.terrain_origins,
            terrain.cfg.terrain_generator.size, terrain.terrain_types, self._flat_columns)

    def _apply_terrain_cap(self, ids):
        flat, _ = self.terrain_masks()
        self.vel_command_b[ids, 0] = terrain_command(self._original_vx[ids], flat[ids],
            self.cfg.flat_speed_cap, self.cfg.rough_speed_cap, self.cfg.minimum_abs_speed)
        self.metrics['speed_curriculum_max'][ids] = torch.where(flat[ids], self.cfg.flat_speed_cap, self.cfg.rough_speed_cap)

    def _resample_command(self, env_ids):
        ids = self._env_ids_tensor(env_ids)
        # 原抽样顺序、25%零、50%运动上限锚点、vy/wz均保持。
        super()._resample_command(ids)
        self._original_vx[ids] = self.vel_command_b[ids, 0]
        self._apply_terrain_cap(ids)

    def _update_command(self):
        super()._update_command()
        # 穿过不同地形列时，下一控制步采用当前地形上限；保持原抽样分位。
        self._apply_terrain_cap(slice(None))


@configclass
class LightHWTerrainVelocityCommandCfg(LightHWForwardSpeedVelocityCommandCfg):
    class_type: type = LightHWTerrainVelocityCommand
    flat_speed_cap: float = 3.
    rough_speed_cap: float = 1.
