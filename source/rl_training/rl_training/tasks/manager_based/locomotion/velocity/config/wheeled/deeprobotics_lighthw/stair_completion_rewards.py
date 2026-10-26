"""整段上楼奖励：地形真值仅进入奖励，不扩充policy观测。"""
import torch
from isaaclab.managers import ManagerTermBase
from isaaclab.utils.math import quat_apply_yaw
from .safe_terrain_spec import columns_for_surface
from .stair_completion_core import StairCompletionState, PARAMETERS, VERSION


class StairCompletionReward(ManagerTermBase):
    """用途：让后轮完成过阶也成为必要条件，抑制前轮独占进展奖金。

    激活：底部出生的正向七级M20楼梯，过阶时身体/世界前进命令>0.05且安全；已合法跨完第7级后允许零命令留台完成。
    返回本步一次性奖金/step_dt，RewardManager再乘dt；整段积分最多10。
    保留原奖励、终止、命令和PPO，运动低载不关闭本项。50Hz状态和3ms
    接触历史不是全部1kHz过程的安全证明，独立回放继续按1kHz检查。
    """
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        self.robot=env.scene['robot'];self.sensor=env.scene['contact_forces'];self.terrain=env.scene.terrain
        self.wheels=self.robot.find_bodies([s+'_wheel_link' for s in ('FR','FL','RR','RL')],preserve_order=True)[0]
        self.others=[i for i,n in enumerate(self.sensor.body_names) if not n.endswith('_wheel_link')]
        s=self.terrain.cfg.terrain_generator.sub_terrains['pyramid_stairs']
        n=int(min((v-2*s.border_width-s.platform_width)//(2*s.step_width)+1 for v in s.size))+1
        if n!=7 or s.size!=(8.,8.) or s.step_width!=.3 or s.border_width!=1.: raise ValueError('Pinned M20 geometry required')
        self.columns=torch.tensor(columns_for_surface('pyramid_stairs'),device=env.device)
        self.state=StairCompletionState(env.num_envs,env.device)
        with torch.inference_mode(False):
            self.by_level=torch.zeros(10,4,device=env.device,dtype=torch.float64)
            self.by_riser=torch.zeros(7,device=env.device,dtype=torch.long)
            self.calls=torch.zeros((),device=env.device,dtype=torch.long)

    def reset(self,env_ids=None): self.state.reset(env_ids)

    @torch.no_grad()
    def __call__(self,env):
        t=self.terrain;center=t.terrain_origins[t.terrain_levels,t.terrain_types]
        cmd=env.command_manager.get_command('base_velocity');v=torch.zeros_like(cmd);v[:,:2]=cmd[:,:2]
        world=quat_apply_yaw(self.robot.data.root_quat_w,v)
        force=self.sensor.data.net_forces_w_history[:,:,self.others].norm(dim=-1).amax((1,2))
        gravity=self.robot.data.projected_gravity_b
        safe=torch.isfinite(force)&(force<=1)&torch.isfinite(gravity).all(1)&(-gravity[:,2]>=2**-.5)
        bonus=self.state.observe(self.robot.data.body_link_pos_w[:,self.wheels],center,
            torch.isin(t.terrain_types,self.columns),cmd[:,0],world[:,0],safe,env.step_dt)
        progress=self.state.last_progress;completed=self.state.last_completion
        values=torch.stack((progress,completed,bonus,self.state.eligible & ~self.state.invalid),1).double()
        self.by_level.index_add_(0,t.terrain_levels,values)
        self.by_riser.index_add_(0,(self.state.common_riser-1).clamp(0,6),progress)
        self.calls+=1
        return bonus/env.step_dt

    def snapshot(self):
        return dict(version=VERSION,calls=int(self.calls),parameters=PARAMETERS,sample_hz=50,
            fields=['common_riser_events','complete_flights','integrated_bonus','eligible_control_samples'],
            by_level=self.by_level.cpu().tolist(),by_riser=self.by_riser.cpu().tolist())
