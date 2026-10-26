"""独立 route43：按真实台阶高度计一次性轮端净空奖励。"""
import torch
from isaaclab.managers import ManagerTermBase
from isaaclab.utils.math import quat_apply_yaw
from .safe_terrain_spec import columns_for_surface
from .stair_clearance_core import StairClearanceState, PARAMETERS, VERSION


class StairClearanceReward(ManagerTermBase):
    """用途：鼓励前进跨阶时适量轮端净空，不要求运动时四轮同时接触。

    激活条件：原 M20 正向楼梯，从底部出发、朝 +X 上楼，前后两帧命令
    vx 和世界 X 投影均 >0.05；本 episode 尚无采样到的非轮碰撞/失稳。
    返回：各轮本次安全跨阶奖金之和 / step_dt；RewardManager 再乘 dt，
    因而每轮每阶最高积分贡献 0.5*weight，整段最多 14*weight。
    地形真值仅用于奖励，未进入 policy 观测；不改变动作、命令或终止。
    """
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.robot = env.scene['robot']
        self.sensor = env.scene['contact_forces']
        self.terrain = env.scene.terrain
        names = [side + '_wheel_link' for side in ('FR', 'FL', 'RR', 'RL')]
        self.wheels = self.robot.find_bodies(names, preserve_order=True)[0]
        self.others = [i for i, name in enumerate(self.sensor.body_names) if not name.endswith('_wheel_link')]
        s = self.terrain.cfg.terrain_generator.sub_terrains['pyramid_stairs']
        risers = int(min((v - 2*s.border_width - s.platform_width)//(2*s.step_width)+1 for v in s.size))+1
        if risers != 7 or s.size != (8., 8.) or s.step_width != .3 or s.border_width != 1.:
            raise ValueError('Clearance reward requires the pinned native M20 seven-riser geometry')
        self.columns = torch.tensor(columns_for_surface('pyramid_stairs'), device=env.device)
        self.state = StairClearanceState(env.num_envs, env.device)
        with torch.inference_mode(False):
            # 只读诊断累计量不随 episode 清零，便于判断奖金是否实际触发。
            self.totals = torch.zeros(4, 3, device=env.device, dtype=torch.float64)
            self.calls = torch.zeros((), device=env.device, dtype=torch.long)

    def reset(self, env_ids=None):
        self.state.reset(env_ids)

    @torch.no_grad()
    def __call__(self, env):
        t = self.terrain
        center = t.terrain_origins[t.terrain_levels, t.terrain_types]
        cmd = env.command_manager.get_command('base_velocity')
        velocity = torch.zeros_like(cmd)
        velocity[:, :2] = cmd[:, :2]
        world_cmd = quat_apply_yaw(self.robot.data.root_quat_w, velocity)
        # 与原保护分离：这里只关闭新奖金，不额外 reset/终止；保持旧碰撞代价。
        force = self.sensor.data.net_forces_w_history[:, :, self.others].norm(dim=-1).amax((1, 2))
        gravity = self.robot.data.projected_gravity_b
        safe = torch.isfinite(force) & (force <= 1.) & torch.isfinite(gravity).all(1) & (-gravity[:, 2] >= 2**-.5)
        bonus = self.state.observe(self.robot.data.body_link_pos_w[:, self.wheels], center,
            torch.isin(t.terrain_types, self.columns), cmd[:, 0], world_cmd[:, 0], safe)
        self.totals[:, 0] += self.state.last_crossed.sum(0)
        self.totals[:, 1] += self.state.last_paid.sum(0)
        self.totals[:, 2] += torch.where(self.state.last_crossed, self.state.last_clearance, 0).sum(0)
        self.calls += 1
        return bonus / env.step_dt

    def snapshot(self):
        values = self.totals.detach().cpu().tolist()
        return dict(version=VERSION, calls=int(self.calls), wheel_order=['FR','FL','RR','RL'],
            fields=['crossings','paid_events','crossing_clearance_sum_m'], totals=values,
            parameters=PARAMETERS, sample_hz=50,
            note='Geometry at control samples, not an all-substep collision or gait certificate.')
