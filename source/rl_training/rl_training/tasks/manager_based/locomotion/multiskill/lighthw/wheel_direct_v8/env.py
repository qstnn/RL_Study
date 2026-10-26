"""R1单环境实现及保留终止前状态的分工况训练监控。"""
import math
import torch
from isaaclab.envs import ManagerBasedRLEnv
from . import core,mdp,spec,cart,torque


class UnifiedEnv(ManagerBasedRLEnv):
    def __init__(self,cfg,render_mode=None,**kwargs):
        n,device=cfg.scene.num_envs,cfg.sim.device
        for value in (cfg.leg_torque_limit,cfg.wheel_torque_limit):
            if not math.isfinite(value) or value<=0:
                raise ValueError("Actuator torque limits must be finite and positive")
        cfg.scene.robot.actuators["joint"].effort_limit=cfg.leg_torque_limit
        cfg.scene.robot.actuators["wheel"].effort_limit=cfg.wheel_torque_limit
        angle,mass,tow=core.assignments(n,spec.SLOPE_MIX,spec.PAYLOADS,spec.TOWED_MASSES,
                                      cfg.fixed_slope_deg,cfg.fixed_payload_kg,cfg.fixed_towed_kg)
        self.r1_slope_rad=torch.tensor(angle*math.pi/180,dtype=torch.float32,device=device)
        self.r1_payload_kg=torch.tensor(mass,dtype=torch.float32,device=device)
        self.r1_towed_kg=torch.tensor(tow,dtype=torch.float32,device=device)
        self.r1_tangent,self.r1_normal=core.plane_basis(self.r1_slope_rad)
        self.r1_axis_x=torch.tensor([1.,0.,0.],device=device).repeat(n,1)
        self.r1_axis_z=torch.tensor([0.,0.,1.],device=device).repeat(n,1)
        self.r1_cart_axis=torch.tensor([0.,1.,0.],device=device).repeat(n,1)
        self.r1_robot_hitch=torch.tensor([-.38,0.,cfg.hitch_height_m-.416223],device=device).repeat(n,1)
        self.r1_cart_hitch=torch.tensor([.35,0.,cfg.hitch_height_m-.30],device=device).repeat(n,1)
        self.r1_drawbar_force=torch.zeros(n,device=device)
        self.r1_drawbar_length=torch.ones(n,device=device)
        self.r1_drawbar_guard=torch.zeros(n,dtype=torch.bool,device=device)
        self.r1_drawbar_peak=torch.zeros(n,device=device)
        self.r1_rolling_torque=torch.zeros(n,4,device=device)
        self.r1_eval_phase=torch.zeros(n,dtype=torch.long,device=device)
        self.r1_eval_stop_time=torch.full((n,),-1.,device=device)
        self.r1_eval_origin=torch.zeros(n,3,device=device)
        self.r1_eval_progress=torch.zeros(n,device=device)
        self.r1_cart_origin=torch.zeros(n,3,device=device)
        self.r1_progress_before=torch.zeros(n,device=device)
        self.r1_physics_ready=False
        self._collect_transition=False
        self._terminal_metrics=None
        cfg.scene.terrain.fixed_slope_deg=cfg.fixed_slope_deg
        cfg.scene.terrain.fixed_payload_kg=cfg.fixed_payload_kg
        cfg.scene.terrain.fixed_towed_kg=cfg.fixed_towed_kg
        cfg.scene.terrain.physics_material.static_friction=cfg.ground_friction
        cfg.scene.terrain.physics_material.dynamic_friction=cfg.ground_friction
        cfg.sim.physics_material.static_friction=cfg.ground_friction
        cfg.sim.physics_material.dynamic_friction=cfg.ground_friction
        super().__init__(cfg,render_mode=render_mode,**kwargs)
        robot,trailer=self.scene["robot"],self.scene["trailer"]
        self.r1_base_id=robot.find_bodies("Base_link")[0][0]
        self.r1_cart_base_id=trailer.find_bodies("cart_base")[0][0]
        names=[f"cart_wheel_{i}" for i in range(4)]
        self.r1_cart_wheel_ids=trailer.find_bodies(names,preserve_order=True)[0]
        self.r1_cart_joint_ids=trailer.find_joints([name+"_joint" for name in names],preserve_order=True)[0]
        self.r1_cart_sensor_ids=self.scene["trailer_contacts"].find_bodies(names,preserve_order=True)[0]
        sensor=self.scene["contact_forces"]
        self.r1_wheel_sensor_ids=sensor.find_bodies([f"{leg}_wheel_link" for leg in ("FR","FL","RR","RL")],preserve_order=True)[0]
        self.r1_body_sensor_ids=[i for i in range(len(sensor.body_names)) if i not in self.r1_wheel_sensor_ids]
        self.r1_joint_ids=robot.find_joints(spec.CONTRACT["action_order"],preserve_order=True)[0]
        for asset in (robot,trailer):
            material=asset.root_physx_view.get_material_properties()
            material[:,:,:2]=cfg.ground_friction
            material[:,:,2]=0.
            asset.root_physx_view.set_material_properties(material,torch.arange(n,dtype=torch.long))
        self.r1_motor_limits=torque.actuator_limits(robot,validate=True)[:,self.r1_joint_ids]
        self.r1_torque_recorder=torque.PhysicsTorqueRecorder(self,robot,self.r1_joint_ids)
        robot.r1_torque_recorder=self.r1_torque_recorder
        self.r1_torque_window=torque.TorqueWindow(spec.CONTRACT["action_order"],self.physics_dt)
        self.r1_physics_ready=True
        self.r1_last_metrics=None
        self._monitor_sums=None
        self._monitor_count=None
        self._monitor_keys=None
        self._monitor_event_sums=None
        self._monitor_event_totals=None

    @torch.inference_mode()
    def _reset_idx(self,env_ids):
        if self._collect_transition:
            self._finish_task_at_timeout(env_ids)
            values=mdp.measurements(self,self.r1_transition_command)
            self._terminal_metrics=(env_ids,{key:value[env_ids].clone() for key,value in values.items()})
        super()._reset_idx(env_ids)

    def _finish_task_at_timeout(self,env_ids):
        """补记恰在timeout帧结束的完整协议窗；物理失败不记完成，不代表行为验收成功。

        Isaac先判终止并reset，再更新命令。若7秒保持/20秒静止恰在timeout帧结束，
        普通command.compute来不及产生事件，需在终止状态被重置前完成这项记账。
        """
        term=self.command_manager.get_term("base_velocity")
        eligible=self.reset_time_outs[env_ids]&(~self.reset_terminated[env_ids])&(~term.cycle_completed[env_ids])
        steps=self.episode_length_buf[env_ids]
        moving=term.requested_vx[env_ids]>0.
        held=moving&(self.r1_eval_phase[env_ids]==3)&((steps-term.phase_start_step[env_ids])>=round(7./self.step_dt))
        static=(~moving)&((steps-term.cycle_start_step[env_ids])>=round(20./self.step_dt))
        complete=eligible&(held|static)
        ids=env_ids[complete]
        term.completed_cycle_this_step[ids]=True
        term.completed_hold_this_step[ids]=held[complete]
        term.cycles_completed[ids]+=1
        term.hold_windows_completed[ids]+=held[complete].long()
        term.static_cycles_completed[ids]+=static[complete].long()
        term.cycle_completed[ids]=True

    @torch.inference_mode()
    def step(self,action):
        self.r1_drawbar_guard.zero_()
        self.r1_drawbar_peak.zero_()
        self._collect_transition=True
        self._terminal_metrics=None
        self.r1_transition_command=self.command_manager.get_command("base_velocity").clone()
        term=self.command_manager.get_term("base_velocity")
        self.r1_transition_target=term.target_vx.clone()
        transition_phase=self.r1_eval_phase.clone()
        self.r1_progress_before=term.joint_progress().clone()
        # 这些脉冲已在上一返回帧写入r1_last_metrics/监控。先消费，避免本步提前终止时
        # _reset_idx在下一次command.compute之前再次保存旧脉冲，造成重复计数。
        term.completed_cycle_this_step.zero_()
        term.completed_hold_this_step.zero_()
        self.r1_torque_recorder.begin()
        try:
            result=super().step(action)
        finally:
            self.r1_torque_recorder.enabled=False
        self.r1_torque_recorder.finish()
        self._collect_transition=False
        values=mdp.measurements(self,self.r1_transition_command)
        if self._terminal_metrics is not None:
            ids,saved=self._terminal_metrics
            for key in values:
                values[key][ids]=saved[key]
        values["terminated"]=result[2].float()
        values["time_out"]=result[3].float()
        if not bool(torch.isfinite(torch.stack(list(values.values()),dim=-1)).all()):
            raise RuntimeError("Non-finite R1 physical measurement")
        self.r1_last_metrics={key:value.detach().clone() for key,value in values.items()}
        # 诊断按真实执行该步动作的工况/阶段分组，重置帧使用已保存的终止前指标。
        self.extras["r1_condition"] = torch.stack((values["slope_deg"],values["payload_kg"],
                                                   values["towed_kg"],transition_phase),dim=-1)
        self._accumulate(values)
        return result

    def _accumulate(self,values):
        """每个物理转移记一次；协议完整窗计数与失败率/行为成功率严格分开。"""
        selected=("command_vx","target_vx","requested_vx","task_phase","phase_elapsed_s","cycle_elapsed_s",
                  "cycle_complete","hold_complete","cycles_completed","hold_windows_completed","static_cycles_completed",
                  "stand_drift_m","hold_elapsed_s","vx","vy","velocity_error","speed","yaw_rate_abs","tilt_deg","height",
                  "four_wheel","body_contact","torque_abs_max","torque_saturation","torque_peak_Nm_1kHz","torque_saturation_1kHz","torque_clipped_1kHz","torque_request_peak_Nm_1kHz","leg_target_clip",
                  "wheel_target_clip","drawbar_force_N","cart_tilt_deg","terminated","time_out",
                  "cart_progress_m","joint_progress_m","cart_vx","joint_progress_rate","successful_hold","termination_illegal_contact","termination_bad_orientation_2","termination_outside_lane","termination_cart_failure","termination_drawbar_ground","hitch_clearance_m")
        zero=(values["command_vx"].abs()<1e-4)&(values["target_vx"].abs()<1e-4)
        high=values["command_vx"]>=3.
        steep=self.r1_slope_rad.abs()>.43
        unloaded=(self.r1_payload_kg==0)&(self.r1_towed_kg==0)
        payload=self.r1_payload_kg>=15
        towing=self.r1_towed_kg>=40
        masks={"all":torch.ones(self.num_envs,dtype=torch.bool,device=self.device),"flat":self.r1_slope_rad==0,
               "uphill":self.r1_slope_rad>0,"downhill":self.r1_slope_rad<0,"unloaded":unloaded,
               "payload15":payload,"tow40":towing,"zero":zero,"moving":~zero,
               "unloaded_high_speed":unloaded&high,"payload15_slope_moving":payload&steep&~zero,
               "tow40_slope_moving":towing&steep&~zero,"payload15_hold":payload&zero,"tow40_hold":towing&zero,
               "phase_wait_or_static":values["task_phase"]==0,"phase_moving":values["task_phase"]==1,
               "phase_braking":values["task_phase"]==2,"phase_hold":values["task_phase"]==3}
        hard_up=(self.r1_towed_kg==40)&(self.r1_slope_rad> .43)
        hard_down=(self.r1_towed_kg==40)&torch.isclose(self.r1_slope_rad,torch.full_like(self.r1_slope_rad,-20.*math.pi/180.),atol=1.e-6)
        for name,mask in (("tow40_up25",hard_up),("tow40_down20",hard_down)):
            masks[name]=mask
            for phase,label in enumerate(("wait_static","move","brake","hold")):
                masks[name+"_"+label]=mask&(values["task_phase"]==phase)
        self.r1_torque_window.update(self.r1_torque_recorder,masks,self.r1_motor_limits)
        keys=list(masks)
        matrix=torch.stack([masks[key] for key in keys]).float()
        sums=matrix@torch.stack([values[key] for key in selected],dim=-1)
        counts=matrix.sum(-1)
        if self._monitor_sums is None:
            self._monitor_sums=torch.zeros_like(sums)
            self._monitor_count=torch.zeros_like(counts)
            self._monitor_keys=(keys,list(selected))
            self._monitor_event_sums=torch.zeros(len(keys),3,dtype=torch.long,device=self.device)
            self._monitor_event_totals=torch.zeros_like(self._monitor_event_sums)
        self._monitor_sums+=sums
        self._monitor_count+=counts
        # 脉冲求和才是完整循环次数；episode内累计count不能跨每帧再次相加。
        events=torch.stack((values["cycle_complete"],values["hold_complete"],
                            (values["cycle_complete"]-values["hold_complete"]).clamp_min(0.)),dim=-1)
        event_counts=(matrix@events).to(torch.long)
        self._monitor_event_sums+=event_counts
        self._monitor_event_totals+=event_counts

    @torch.inference_mode()
    def drain_metrics(self):
        """返回本统计窗均值和整数事件数；*_total是此env实例从创建起的累计事件。

        phase_*分组samples构成阶段分布，四组总样本数等于all。*_episode_mean仅为
        被采样状态中本episode计数的均值；*_events才可按更新求和，均不称存活率。
        """
        if self._monitor_sums is None:
            return []
        sums=self._monitor_sums.cpu().tolist()
        counts=self._monitor_count.cpu().tolist()
        groups,keys=self._monitor_keys
        events=self._monitor_event_sums.cpu().tolist()
        lifetime=self._monitor_event_totals.cpu().tolist()
        event_names=("cycle_complete","hold_complete","static_cycle_complete")
        episode_counts={"cycles_completed","hold_windows_completed","static_cycles_completed"}
        rows=[]
        for name,total,count,window_events,total_events in zip(groups,sums,counts,events,lifetime):
            means={(key+"_episode_mean" if key in episode_counts else key):(value/count if count else None)
                   for key,value in zip(keys,total)}
            row=dict(group=name,samples=int(count),**means)
            row.update({key+"_events":int(value) for key,value in zip(event_names,window_events)})
            row.update({key+"_total":int(value) for key,value in zip(event_names,total_events)})
            rows.append(row)
        self._monitor_sums.zero_();self._monitor_count.zero_();self._monitor_event_sums.zero_()
        return rows
