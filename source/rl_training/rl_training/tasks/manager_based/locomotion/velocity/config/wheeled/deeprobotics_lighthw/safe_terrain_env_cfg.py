"""Route33：只改变地形课程，完整M20终点、原PPO/动力学/奖励，恢复早期Rough线性腿映射。"""
import math
import torch
from isaaclab.utils import configclass
from isaaclab.managers import CurriculumTermCfg,ManagerTermBase
from isaaclab.terrains.height_field.hf_terrains import random_uniform_terrain
from .m20_full_rough_env_cfg import make_transfer_terrain
from .moving_acceleration_env_cfg import LightHWMovingAccelerationEnvCfg
from .speed3_env_cfg import set_speed_cap
from .safe_terrain_spec import (SPEED_CAP,SURFACES,GENERATOR,PROMOTION_TILT_DEG,CONTACT_LIMIT_N,
                                TERRAIN_VERSION,random_noise_range,columns_for_surface)
from .terrain_traversal_state import TraversalState
from .linear_leg_action import LightHWLinearURDFLegActionCfg
from . import linear_leg_spec


def progressive_random_terrain(difficulty,cfg):
    """仅随机起伏上界随课程变化；复制配置防止地形生成器原地覆盖最终M20范围。"""
    effective=cfg.copy()
    effective.noise_range=random_noise_range(difficulty,*cfg.noise_range,cfg.noise_step)
    return random_uniform_terrain(difficulty,effective)


class SafeTraversalCurriculum(ManagerTermBase):
    """每episode结束才变更出生level，使用原生终止计算之后的已有50Hz状态。

    不增加1kHz传感器读取：额外强制刷新接触history可能改变原奖励语义。
    包装器原样返回原生终止结果，不重置、不改命令、奖励或动作；采集不消耗RNG。
    """
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        self.state=TraversalState(env.num_envs,env.device)
        self.terrain=env.scene.terrain;self.robot=env.scene['robot'];self.sensor=env.scene['contact_forces']
        self.nonwheel=[i for i,n in enumerate(self.sensor.body_names) if 'wheel' not in n]
        with torch.inference_mode(False):
            self.kinds=torch.empty(GENERATOR['num_cols'],device=env.device,dtype=torch.long)
            for i,name in enumerate(SURFACES):self.kinds[columns_for_surface(name)]=i
            self.counters=torch.zeros((len(SURFACES),GENERATOR['num_rows'],6),device=env.device,dtype=torch.long)
        self.samples=0
        self.original=env.termination_manager.compute
        def compute():
            result=self.original()
            self.observe(env)
            return result
        env.termination_manager.compute=compute
        env._safe_terrain_curriculum=self

    @torch.no_grad()
    def observe(self,env):
        r=self.robot.data;position=r.root_pos_w
        size=position.new_tensor(self.terrain.cfg.terrain_generator.size)
        start=self.terrain.terrain_origins[0,0,:2]-size*.5
        cell=torch.floor((position[:,:2]-start)/size).long()
        row,col=cell[:,0],cell[:,1]
        inside=(row>=0)&(row<GENERATOR['num_rows'])&(col>=0)&(col<GENERATOR['num_cols'])
        # Base_link termination已经读取当前接触buffer；这里只读同一缓存。
        forces=self.sensor.data.net_forces_w
        finite=(torch.isfinite(r.root_state_w).all(1)&torch.isfinite(r.joint_pos).all(1)
                &torch.isfinite(r.joint_vel).all(1)&torch.isfinite(forces).all((1,2)))
        bad=(~finite|(forces[:,self.nonwheel].norm(dim=-1)>CONTACT_LIMIT_N).any(1)
             |(r.projected_gravity_b[:,2]>-math.cos(math.radians(PROMOTION_TILT_DEG))))
        command=env.command_manager.get_command('base_velocity')[:,0]
        self.state.observe(row,col,inside,command>.05,bad,position[:,0],command,env.step_dt,self.terrain.terrain_types)
        self.samples+=1

    @torch.no_grad()
    def __call__(self,env,env_ids):
        ids=torch.arange(env.num_envs,device=env.device)[env_ids]
        if env.common_step_counter==0:
            env._rough_initial_reset_audit=dict(version=TERRAIN_VERSION,guarded=True,initial_levels_unchanged=True)
            return self.terrain.terrain_levels.float().mean()
        levels=self.terrain.terrain_levels[ids]
        kind=self.kinds[self.terrain.terrain_types[ids]]
        next_level,up,down,unsafe,insufficient=self.state.decisions(
            ids,levels,kind==0,env.termination_manager.terminated[ids])
        sampled=self.state.steps[ids]>0
        values=torch.stack((sampled,self.state.qualified[ids]&~unsafe,up,down,unsafe&sampled,insufficient),1).long()
        self.counters.view(-1,6).index_add_(0,kind*GENERATOR['num_rows']+levels,values)
        # 显式clamp，不调用原生最高级随机回退；只在现有reset流程更换下一次出生位置。
        self.terrain.terrain_levels[ids]=next_level
        self.terrain.env_origins[ids]=self.terrain.terrain_origins[next_level,self.terrain.terrain_types[ids]]
        return self.terrain.terrain_levels.float().mean()

    def reset(self,env_ids=None):
        self.state.reset(slice(None) if env_ids is None else env_ids)

    def snapshot(self):
        return dict(version=TERRAIN_VERSION,sample_hz=50,observed_control_steps=self.samples,
            fields=['episodes','qualified_episodes','promotions','demotions','unsafe_episodes','insufficient_exposure'],
            families=list(SURFACES),cumulative_by_family_and_spawn_level=self.counters.cpu().tolist(),
            terrain_levels=self.terrain.terrain_levels.cpu().tolist(),
            note='Curriculum evidence only, not independent1kHz behavior acceptance; simulator state not restorable')


@configclass
class LightHWSafeTerrainEnvCfg(LightHWMovingAccelerationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        set_speed_cap(self,SPEED_CAP)
        # 用户选择早期线性映射，不是±0.08/±0.20的tanh包络。
        old=self.actions.joint_pos
        values={key:value for key,value in vars(old).items() if key!='class_type'}
        self.actions.joint_pos=LightHWLinearURDFLegActionCfg(**values)
        self.actions.joint_pos.scale=linear_leg_spec.SCALE.copy()
        self.actions.joint_pos.clip=linear_leg_spec.LIMITS.copy()
        self.scene.terrain.terrain_type='generator'
        self.scene.terrain.terrain_generator=make_transfer_terrain()
        self.scene.terrain.terrain_generator.sub_terrains['random_rough'].function=progressive_random_terrain
        self.scene.terrain.max_init_terrain_level=0
        self.curriculum.terrain_levels=CurriculumTermCfg(func=SafeTraversalCurriculum)
        print('[LightHW route33] policy=60, critic=236, action=16; source14799 full restore; '
              'safe traversal curriculum; linear hip scale0.08,thigh/calf0.14; URDF absolute clip; '
              'progressive random rough; vx0..3,25%zero,25%flat; all16 PPO',flush=True)
