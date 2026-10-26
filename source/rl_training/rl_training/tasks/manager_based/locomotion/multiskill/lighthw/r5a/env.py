"""独立 R5A J1：有限时域的一次起跳、落地和恢复。"""
from copy import deepcopy
import math
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz
from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_lighthw.asset_cfg import LIGHTHW_ROUTE_CFG
from . import spec
from .appearance import spawn_robot, terrain_class
from .protocol import JumpState, reference_pose


def robot_config():
    cfg=deepcopy(LIGHTHW_ROUTE_CFG).replace(prim_path='{ENV_REGEX_NS}/Robot')
    cfg.spawn.func=spawn_robot
    cfg.spawn.articulation_props.enabled_self_collisions=True
    cfg.spawn.articulation_props.solver_position_iteration_count=8
    cfg.spawn.articulation_props.solver_velocity_iteration_count=2
    for key,effort,velocity in (('joint',48.,12.57),('wheel',24.,37.7)):
        motor=cfg.actuators[key]
        motor.effort_limit=motor.effort_limit_sim=effort
        motor.velocity_limit=motor.velocity_limit_sim=velocity
        motor.min_delay=motor.max_delay=0
    cfg.actuators['joint'].stiffness=80.; cfg.actuators['joint'].damping=3.
    cfg.actuators['wheel'].stiffness=0.; cfg.actuators['wheel'].damping=1.
    return cfg


@configclass
class JumpSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = robot_config()
    terrain = TerrainImporterCfg(prim_path='/World/ground',terrain_type='plane',
        class_type=terrain_class(),collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.,dynamic_friction=1.,restitution=0.))
    contact_forces = ContactSensorCfg(prim_path='{ENV_REGEX_NS}/Robot/.*',update_period=.001,history_length=20)
    sky_light = AssetBaseCfg(prim_path='/World/sky',spawn=sim_utils.DomeLightCfg(intensity=750.,color=(1.,1.,1.)))


@configclass
class JumpLandingEnvCfg(DirectRLEnvCfg):
    scene: JumpSceneCfg = JumpSceneCfg(num_envs=512,env_spacing=3.,replicate_physics=True)
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(dt=.001,render_interval=20,
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.,dynamic_friction=1.,restitution=0.))
    decimation=20
    episode_length_s=4.
    is_finite_horizon=True
    action_space=16
    observation_space=285
    state_space=86
    reset_jitter=.01

    def __post_init__(self):
        super().__post_init__()
        self.viewer.origin_type='world'
        self.viewer.asset_name='robot'
        self.viewer.eye=(2.,-2.,1.2)
        self.viewer.lookat=(0.,0.,.45)


class JumpLandingEnv(DirectRLEnv):
    cfg: JumpLandingEnvCfg
    # Isaac Lab 2.2 RSL wrapper checks this field in addition to the critic Gym space.
    num_states = spec.CONTRACT["critic_dim"]

    def __init__(self,cfg,render_mode=None,**kwargs):
        self.physics_observer=None
        self.episode_sink=None
        self._last_capture_tick=-1
        super().__init__(cfg,render_mode,**kwargs)
        self.robot=self.scene['robot']; self.contacts=self.scene['contact_forces']
        self.joint_ids=self.robot.find_joints(list(spec.JOINTS),preserve_order=True)[0]
        self.leg_ids=self.robot.find_joints(list(spec.LEG_JOINTS),preserve_order=True)[0]
        self.wheel_ids=self.robot.find_joints(list(spec.WHEELS),preserve_order=True)[0]
        self.wheel_body_ids=self.robot.find_bodies(list(spec.WHEEL_BODIES),preserve_order=True)[0]
        self.contact_ids=self.contacts.find_bodies(list(spec.WHEEL_BODIES),preserve_order=True)[0]
        self.other_contacts=[i for i,n in enumerate(self.contacts.body_names) if n not in spec.WHEEL_BODIES]
        self.masses=self.robot.root_physx_view.get_masses().to(self.device)
        self.mass=self.masses.sum(-1)
        self.actions=torch.zeros(self.num_envs,16,device=self.device)
        self.previous_actions=torch.zeros_like(self.actions)
        self.targets=torch.zeros(self.num_envs,12,device=self.device)
        self.residual=torch.tensor(spec.RESIDUAL,device=self.device)
        self.history=torch.zeros(self.num_envs,5,57,device=self.device)
        self.history_reset=torch.ones(self.num_envs,dtype=torch.bool,device=self.device)
        self._history_step=-1
        self.origin_xy=torch.zeros(self.num_envs,2,device=self.device)
        self.state=JumpState(self.num_envs,self.device)
        self.episode_count=torch.zeros(self.num_envs,dtype=torch.long,device=self.device)
        self.last_reports=[]
        self.physics_finite=torch.tensor(True,device=self.device)
        self._protocol_tick=self._sim_step_counter
        self.episode_physics_steps=torch.zeros(self.num_envs,dtype=torch.long,device=self.device)
        self.policy_airborne_steps=torch.zeros(self.num_envs,device=self.device)
        self.policy_peak_support_force=torch.zeros(self.num_envs,device=self.device)
        self.airborne=torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)
        self.stable=torch.zeros_like(self.airborne)

    def _setup_scene(self):
        # InteractiveSceneCfg 已创建并复制所有资产，无需第二次 clone。
        pass

    def metrics(self):
        d=self.robot.data
        com=(d.body_com_pos_w*self.masses[:,:,None]).sum(1)/self.mass[:,None]
        com_vel=(d.body_com_lin_vel_w*self.masses[:,:,None]).sum(1)/self.mass[:,None]
        force=self.contacts.data.net_forces_w
        tilt=torch.acos((-d.projected_gravity_b[:,2]).clamp(-1.,1.))
        return dict(com=com,com_vel=com_vel,force_z=force[:,self.contact_ids,2],
            clearance=d.body_link_pos_w[:,self.wheel_body_ids,2]-.0875,
            tilt=tilt,base_z=d.root_link_pos_w[:,2],speed=com_vel[:,:2].norm(dim=-1),
            drift=(com[:,:2]-self.origin_xy).norm(dim=-1),
            body_force=force[:,self.other_contacts].norm(dim=-1).amax(-1),
            torque_peak=d.applied_torque.abs().amax(-1))

    def _capture(self,tick):
        if self.physics_observer is not None and tick!=self._last_capture_tick:
            self.physics_observer(self,tick)
            self._last_capture_tick=tick

    def _pre_physics_step(self,actions):
        if not torch.isfinite(actions).all():
            raise RuntimeError('R5A non-finite action')
        self.extras['log']={}
        self.physics_finite.fill_(True)
        self.policy_airborne_steps.zero_()
        self.policy_peak_support_force.zero_()
        self.previous_actions.copy_(self.actions)
        t=self.episode_length_buf*self.step_dt
        self.actions.copy_(actions.clamp(-1.,1.)*(t>=.5)[:,None])
        limits=self.robot.data.joint_pos_limits[:,self.leg_ids]
        self.targets.copy_((reference_pose(t)+self.actions[:,:12]*self.residual).clamp(limits[:,:,0],limits[:,:,1]))

    def _apply_action(self):
        # 记录上一步已完成的真实物理状态；本步驱动尚未写入，不混入未来目标。
        self._advance_protocol(self._sim_step_counter-1)
        self._capture(self._sim_step_counter-1)
        self.robot.set_joint_position_target(self.targets,joint_ids=self.leg_ids)
        self.robot.set_joint_velocity_target(self.actions[:,12:]*5.,joint_ids=self.wheel_ids)

    def _advance_protocol(self,tick):
        """只消费已完成且尚未消费的物理步；不受录像开关影响，不消费 reset 缓冲。"""
        if tick==self._protocol_tick:
            return
        if tick!=self._protocol_tick+1:
            raise RuntimeError(f'R5A physics event gap: {self._protocol_tick} -> {tick}')
        m=self.metrics()
        # Accumulate on device; synchronise once per policy step instead of per tensor at 1 kHz.
        self.physics_finite.logical_and_(torch.isfinite(torch.cat([v.reshape(-1) for v in m.values()])).all())
        self.episode_physics_steps.add_(1)
        t=self.episode_physics_steps*self.physics_dt
        self.origin_xy.copy_(torch.where((t<=.5)[:,None],m['com'][:,:2],self.origin_xy))
        m['drift']=(m['com'][:,:2]-self.origin_xy).norm(dim=-1)
        self.airborne,self.stable=self.state.update(self.physics_dt,t,m['com'][:,2],m['com_vel'][:,2],
            m['force_z'],m['clearance'],m['tilt'],m['base_z'],m['speed'],m['drift'],m['body_force'],m['torque_peak'])
        self.policy_airborne_steps.add_(self.airborne.float())
        peak=m['force_z'].clamp_min(0.).sum(-1)
        self.policy_peak_support_force.copy_(torch.maximum(self.policy_peak_support_force,peak))
        m['peak_support_bw']=self.policy_peak_support_force/(self.mass*9.81)
        self.current_metrics=m
        self._protocol_tick=tick

    def _get_dones(self):
        self._advance_protocol(self._sim_step_counter)
        self._capture(self._sim_step_counter)
        if not self.physics_finite:
            raise RuntimeError('R5A non-finite physical state')
        expected=self.episode_length_buf*self.cfg.decimation
        if not torch.equal(self.episode_physics_steps,expected):
            raise RuntimeError('R5A episode physical clock disagrees with policy clock')
        return self.state.failed.clone(), self.episode_length_buf>=self.max_episode_length

    def _get_rewards(self):
        """J1 奖励：起跳窗奖励向上 COM 速度，落地保持只在真实腾空之后激活。

        返回每环境每策略步的奖励。率项乘 dt，成功只在回合末确认，失败为一次性事件；
        不含全程四轮支撑奖励，不以增大机身高度代替腾空，不用于完整后空翻阶段。
        """
        m=self.current_metrics; t=self.episode_length_buf*self.step_dt
        launch=(t>=.85)&(t<=1.25)
        recovery=self.state.landed
        upright=torch.exp(-m['tilt'].square()/.12)
        upward=m['com_vel'][:,2].clamp(0.,1.5)/1.5
        rise=torch.exp(-(self.state.peak_rise-.08).square()/.0064)
        hold=(torch.exp(-(m['speed']/spec.CONTRACT['recovery_speed_scale_m_s']).square())
              *torch.exp(-(m['drift']/spec.CONTRACT['recovery_drift_scale_m']).square())
              *upright*(m['force_z']>5.).all(-1))
        # 触地后的缓冲窗口保留；1.5s 后减少轮部驱动与水平移动，鼓励落地后静止。
        braking=recovery & (t>=spec.CONTRACT['braking_start_s'])
        brake_cost=braking*(spec.CONTRACT['braking_speed_cost_rate']*(m['speed']/.05).square()
            +spec.CONTRACT['braking_wheel_cost_rate']*self.robot.data.joint_vel[:,self.wheel_ids].square().sum(-1))
        drift_cost=recovery*spec.CONTRACT['recovery_drift_cost_rate']*(m['drift']/.05).square()
        # 落地冲击用全部 1 kHz 接触历史，不能只取 50 Hz 末帧。3 倍体重为奖励软阈值，非硬件承载验收值。
        impact_cost=spec.CONTRACT['landing_force_cost_rate']*recovery*(m['peak_support_bw']-spec.CONTRACT['landing_force_cost_threshold_bw']).clamp_min(0.).square()
        torque=self.robot.data.applied_torque[:,self.leg_ids].square().sum(-1)
        rate=(2.*upright+4.*launch*upward+5.*(self.policy_airborne_steps/self.cfg.decimation)*rise+6.*recovery*hold
              -1e-5*torque-.01*(self.actions-self.previous_actions).square().sum(-1)
              -.1*m['drift'].square()-impact_cost-brake_cost-drift_cost)
        # 成功奖金只在正常回合末给出，不能先拿到奖金再滑走。
        terminal=self.episode_length_buf>=self.max_episode_length
        reward=rate*self.step_dt+spec.CONTRACT['terminal_success_bonus']*(terminal & self.state.success).float()-3.*self.state.failed.float()
        if not torch.isfinite(reward).all(): raise RuntimeError('R5A non-finite reward')
        return reward

    def _get_observations(self):
        d=self.robot.data; t=self.episode_length_buf*self.step_dt
        pos=(d.joint_pos-d.default_joint_pos)[:,self.joint_ids].clone()
        pos[:,[3,7,11,15]]=0.
        phase=torch.stack((torch.sin(t*math.pi/2),torch.cos(t*math.pi/2),torch.full_like(t,.8)),dim=-1)
        frame=torch.cat((d.root_ang_vel_b*.5,d.projected_gravity_b,pos,
                         d.joint_vel[:,self.joint_ids]*.05,self.actions,phase),dim=-1)
        if self._history_step!=self.common_step_counter:
            self.history[:,:-1].copy_(self.history[:,1:].clone())
            self.history[:,-1].copy_(frame)
            self._history_step=self.common_step_counter
        self.history[self.history_reset]=frame[self.history_reset,None,:].expand(-1,5,-1)
        self.history_reset.zero_()
        m=self.metrics()
        critic=torch.cat((frame,m['com_vel'],(m['com'][:,2]-self.state.baseline_z)[:,None],
             m['force_z']/self.mass[:,None]/9.81,(m['com'][:,:2]-self.origin_xy),
             self.state.critic(),m['base_z'][:,None],reference_pose(t)),dim=-1)
        return {'policy':self.history.flatten(1),'critic':critic}

    def _reset_idx(self,env_ids):
        if env_ids is None: env_ids=self.robot._ALL_INDICES
        elapsed=self.episode_length_buf[env_ids]*self.step_dt
        report_ids=env_ids[elapsed>0]
        if report_ids.numel():
            names=JumpState.FLOATS+JumpState.BOOLS
            values={n:getattr(self.state,n)[report_ids].detach().cpu().tolist() for n in names}
            indices=report_ids.cpu().tolist()
            counts=self.episode_count[report_ids].cpu().tolist()
            durations=(self.episode_length_buf[report_ids]*self.step_dt).cpu().tolist()
            reports=[dict(env_id=i,episode=counts[j],duration_s=durations[j],
                          **{n:values[n][j] for n in names}) for j,i in enumerate(indices)]
            self.last_reports=reports
            if self.episode_sink is not None: self.episode_sink(reports)
            self.extras['log']={f'R5A/{name}':getattr(self.state,name)[report_ids].float().mean()
                               for name in ('success','launched','landed','peak_rise','flight_time','stable_time','torque_peak','impact_peak')}
            self.episode_count[report_ids]+=1
        super()._reset_idx(env_ids)
        root=self.robot.data.default_root_state[env_ids].clone()
        root[:,:3]+=self.scene.env_origins[env_ids]
        noise=(torch.rand(len(env_ids),3,device=self.device)*2.-1.)*self.cfg.reset_jitter
        root[:,3:7]=quat_from_euler_xyz(noise[:,0],noise[:,1],noise[:,2])
        root[:,7:]=0.
        q=self.robot.data.default_joint_pos[env_ids].clone()
        self.robot.write_root_pose_to_sim(root[:,:7],env_ids)
        self.robot.write_root_velocity_to_sim(root[:,7:],env_ids)
        self.robot.write_joint_state_to_sim(q,torch.zeros_like(q),None,env_ids)
        self.actions[env_ids]=0.; self.previous_actions[env_ids]=0.
        self.targets[env_ids]=q[:,self.leg_ids]
        self.origin_xy[env_ids]=root[:,:2]
        self.state.reset(env_ids,root[:,2]-.085652)
        self.episode_physics_steps[env_ids]=0
        self.history_reset[env_ids]=True
        self.airborne[env_ids]=False; self.stable[env_ids]=False
