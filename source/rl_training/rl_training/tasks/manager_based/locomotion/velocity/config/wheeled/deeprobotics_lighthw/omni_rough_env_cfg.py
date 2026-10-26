"""独立完整Rough基线：保留源直线/楼梯练习，同时学习M20多方向运动。"""
import math
from copy import deepcopy
import torch
from isaaclab.utils import configclass
from isaaclab.managers import CurriculumTermCfg
from .stair_flight_env_cfg import LightHWStairFlightsEnvCfg,StairFlightCurriculum
from .linear_leg_action import LightHWLinearURDFLegAction,LightHWLinearURDFLegActionCfg
from .omni_rough_core import COMMAND_CONTRACT,COURSE_VERSION,OmniTraversalState,retention_mask
from .omni_rough_commands import OmniRoughVelocityCommandCfg
from .omni_rough_monitor import OmniMonitor
from . import omni_rough_rewards as rewards


class OmniLegAction(LightHWLinearURDFLegAction):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        self.omni_monitor=OmniMonitor(env)

    def _signature_payload(self):
        value=super()._signature_payload()
        value['command_distribution']=deepcopy(COMMAND_CONTRACT)
        return value

    def _apply_command_stage(self,*,resample):
        # 固定动作范围；绝不能调用旧代码把新任务的vx改回0..cap。
        if resample:
            self._env.command_manager.get_term('base_velocity')._resample_command(
                torch.arange(self.num_envs,device=self.device))

    def record_quality(self):
        if self._training_curriculum:
            self._control_steps+=1
            self.omni_monitor.update()

    def advance_curriculum(self):
        if not self._training_curriculum:return {}
        if len(self.schedule.max_speeds)!=1:raise ValueError('Omni baseline has fixed actions, not a speed curriculum')
        due=self.schedule.window_updates+1>=self.schedule.evaluation_interval
        self.schedule.update(False if due else None)
        return {'total_updates':self.schedule.total_updates,'control_steps':self._control_steps}


@configclass
class OmniLegActionCfg(LightHWLinearURDFLegActionCfg):
    class_type:type=OmniLegAction


class OmniRoughCurriculum(StairFlightCurriculum):
    """保留组：原连续楼梯课程；多方向组：安全离开出生区域，不要求高速。"""
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        self.retained=retention_mask(torch.arange(env.num_envs,device=env.device))
        self.stair_mask.logical_and_(self.retained);self.state.stair.copy_(self.stair_mask)
        self.omni=OmniTraversalState(env.num_envs,env.device)
        with torch.inference_mode(False):self.omni_counts=torch.zeros_like(self.counters)

    @torch.no_grad()
    def observe(self,env):
        # 旧状态只用于retained；general不因走进其他列或转向而误罚。
        super().observe(env)
        t=self.terrain;r=self.robot.data;p=r.root_pos_w;f=self.sensor.data.net_forces_w
        size=p.new_tensor(t.cfg.terrain_generator.size)
        cell=torch.floor((p[:,:2]-(t.terrain_origins[0,0,:2]-size/2))/size).long()
        inside=(cell>=0).all(1)&(cell[:,0]<10)&(cell[:,1]<20)
        bad=(~torch.isfinite(r.root_state_w).all(1)|~torch.isfinite(r.joint_pos).all(1)
             |~torch.isfinite(r.joint_vel).all(1)|~torch.isfinite(f).all((1,2))
             |(f[:,self.nonwheel].norm(dim=-1)>1.).any(1)
             |(r.projected_gravity_b[:,2]>-math.cos(math.pi/4)))
        spawn=torch.stack((t.terrain_levels,t.terrain_types),1)
        self.omni.observe(p[:,:2],env.command_manager.get_command('base_velocity'),bad,spawn,cell,inside,
                          env.step_dt,t.terrain_origins[t.terrain_levels,t.terrain_types,:2])

    @torch.no_grad()
    def __call__(self,env,env_ids):
        ids=torch.arange(env.num_envs,device=env.device)[env_ids]
        retained=ids[self.retained[ids]];general=ids[~self.retained[ids]]
        if retained.numel():super().__call__(env,retained)
        t=self.terrain
        if env.common_step_counter and general.numel():
            levels=t.terrain_levels[general];kind=self.kinds[t.terrain_types[general]]
            next_level,up,down,unsafe,insufficient=self.omni.decisions(general,levels,kind==0,env.termination_manager.terminated[general])
            sampled=self.omni.steps[general]>0
            values=torch.stack((sampled,self.omni.qualified[general]&~unsafe,up,down,unsafe&sampled,insufficient),1).long()
            self.omni_counts.view(-1,6).index_add_(0,kind*10+levels,values)
            t.terrain_levels[general]=next_level
        t.env_origins[general]=t.terrain_origins[t.terrain_levels[general],t.terrain_types[general]]
        return t.terrain_levels.float().mean()

    def reset(self,env_ids=None):
        super().reset(env_ids)
        self.omni.reset(slice(None) if env_ids is None else env_ids)

    def snapshot(self):
        result=super().snapshot()
        result.update(version=COURSE_VERSION,retention_rule='even_environment_ids',
            retained_episode_counters=result.pop('cumulative_by_family_and_spawn_level'),
            omni_episode_counters=self.omni_counts.cpu().tolist(),
            omni_qualification='4m from spawn center, in spawn tile or first exit; >=1m commanded translation; whole episode safe',
            omni_demotion='unsafe or commanded translation>=8m without qualification',
            omni_low_load_is_not_failure=True,omni_turn_or_stop_without_translation_is_not_obstruction=True)
        return result


@configclass
class LightHWOmniRoughEnvCfg(LightHWStairFlightsEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        old=self.actions.joint_pos
        self.actions.joint_pos=OmniLegActionCfg(**{k:v for k,v in vars(old).items() if k!='class_type'})
        old=self.commands.base_velocity
        self.commands.base_velocity=OmniRoughVelocityCommandCfg(asset_name=old.asset_name,
            resampling_time_range=(8.,8.),rel_standing_envs=.25,rel_heading_envs=.2,
            heading_command=True,heading_control_stiffness=.5,debug_vis=old.debug_vis,
            knee_joint_names=old.knee_joint_names,ranges=OmniRoughVelocityCommandCfg.Ranges(
                lin_vel_x=(-2.,2.),lin_vel_y=(-1.,1.),ang_vel_z=(-1.,1.),heading=(-math.pi,math.pi)))
        self.curriculum.terrain_levels=CurriculumTermCfg(func=OmniRoughCurriculum)
        self.rewards.wheel_contact_support.func=rewards.moving_contact_support
        self.rewards.weakest_wheel_support.func=rewards.moving_weakest_support
        self.rewards.wheel_contact_uniformity.func=rewards.moving_contact_uniformity
        self.rewards.joint_acc_l2.func=rewards.moving_joint_acc_l2
        self.rewards.joint_acc_wheel_l2.func=rewards.moving_joint_acc_l2
        print('[LightHW route39] policy=60 critic=236 action=16; full PPO; 50% retained forward/stairs; '
              '50% M20 all directions; 25% zero; staircase/boxes planar command<=1m/s; '
              'reward gates use all 3 commanded axes; source optimizer/noise restored by route39 runner',flush=True)
