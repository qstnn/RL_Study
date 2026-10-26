"""Route38：原M20连续楼梯，从楼梯脚下学习整段上楼；其他物理/奖励保持。"""
import torch
from isaaclab.utils import configclass
from isaaclab.managers import CurriculumTermCfg
from .easy_obstacle_env_cfg import LightHWEasyObstacleEnvCfg
from .terrain_speed_env_cfg import SafeTraversalCurriculumV2
from .safe_terrain_env_cfg import SafeTraversalCurriculum
from .safe_terrain_spec import columns_for_surface
from .stair_flight_state import StairFlightState, bottom_origins, VERSION, RISERS


class StairFlightCurriculum(SafeTraversalCurriculumV2):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        self.state=StairFlightState(env.num_envs,env.device)
        names=['FL_wheel_link','FR_wheel_link','RL_wheel_link','RR_wheel_link']
        self.wheel_ids=self.robot.find_bodies(names,preserve_order=True)[0]
        with torch.inference_mode(False):
            self.stair_mask=torch.isin(self.terrain.terrain_types,
                torch.tensor(columns_for_surface('pyramid_stairs'),device=env.device))
            self.flight_counts=torch.zeros((10,4),dtype=torch.long,device=env.device)
        self.state.stair.copy_(self.stair_mask)
        s=self.terrain.cfg.terrain_generator.sub_terrains['pyramid_stairs']
        risers=int(min((v-2*s.border_width-s.platform_width)//(2*s.step_width)+1 for v in s.size))+1
        if risers!=RISERS or s.size!=(8.,8.) or s.step_width!=.3 or s.border_width!=1.:
            raise ValueError('Flight milestones require the pinned native M20 geometry')

    @torch.no_grad()
    def observe(self,env):
        super().observe(env)
        center=self.terrain.terrain_origins[self.terrain.terrain_levels,self.terrain.terrain_types]
        self.state.observe_flight(self.robot.data.body_link_pos_w[:,self.wheel_ids],center,self.stair_mask,env.step_dt)

    @torch.no_grad()
    def __call__(self,env,env_ids):
        ids=torch.arange(env.num_envs,device=env.device)[env_ids]
        before=self.terrain.terrain_levels[ids].clone()
        if env.common_step_counter!=0:
            st=self.state;sampled=st.steps[ids]>0
            unsafe=st.unsafe[ids]|env.termination_manager.terminated[ids]|st.flight_invalid[ids]
            counts=torch.stack((sampled,st.flight_done[ids]&~unsafe,
                st.risers[ids],unsafe&sampled),1).long()*self.stair_mask[ids,None]
            self.flight_counts.index_add_(0,before,counts)
        # Base implementation uses StairFlightState.decisions; other families retain v2 decisions.
        value=SafeTraversalCurriculum.__call__(self,env,env_ids)
        t=self.terrain
        t.env_origins[ids]=bottom_origins(t.terrain_origins,t.terrain_levels[ids],t.terrain_types[ids],self.stair_mask[ids])
        if env.common_step_counter==0:env._rough_initial_reset_audit['version']=VERSION
        return value

    def snapshot(self):
        result=SafeTraversalCurriculum.snapshot(self)
        result.update(version=VERSION,flight_counter_fields=['episodes','safe_full_flights','sum_max_risers','unsafe_episodes'],
            flight_by_spawn_level=self.flight_counts.cpu().tolist(),num_risers=RISERS,
            positive_stair_spawn='bottom approach: centerX-3.6m, groundZ0',
            flight_qualification='ordered7risers, all wheels on landing0.2s, whole episode safe',
            other_families='unchanged v2 complete8m; inverse stairs remain centered in pit',
            moving_low_load_is_not_failure=True)
        return result


@configclass
class LightHWStairFlightsEnvCfg(LightHWEasyObstacleEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.curriculum.terrain_levels=CurriculumTermCfg(func=StairFlightCurriculum)
        print('[LightHW route38] policy=60 critic=236 action=16; native M20 seven-riser flights; '
              'positive stairs start at bottom; full flight promotion; all16 PPO; original rewards/actions/PD/noise',flush=True)
