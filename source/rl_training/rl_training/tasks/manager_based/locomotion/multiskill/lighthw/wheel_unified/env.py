"""R1单环境实现及保留终止前状态的分工况训练监控。"""
import math
import torch
from isaaclab.envs import ManagerBasedRLEnv
from . import core,mdp,spec,cart


class UnifiedEnv(ManagerBasedRLEnv):
    def __init__(self,cfg,render_mode=None,**kwargs):
        n,device=cfg.scene.num_envs,cfg.sim.device
        angle,mass,tow=core.assignments(n,spec.SLOPE_MIX,spec.PAYLOADS,spec.TOWED_MASSES,
                                      cfg.fixed_slope_deg,cfg.fixed_payload_kg,cfg.fixed_towed_kg)
        self.r1_slope_rad=torch.tensor(angle*math.pi/180,dtype=torch.float32,device=device)
        self.r1_payload_kg=torch.tensor(mass,dtype=torch.float32,device=device)
        self.r1_towed_kg=torch.tensor(tow,dtype=torch.float32,device=device)
        self.r1_tangent,self.r1_normal=core.plane_basis(self.r1_slope_rad)
        self.r1_axis_x=torch.tensor([1.,0.,0.],device=device).repeat(n,1)
        self.r1_axis_z=torch.tensor([0.,0.,1.],device=device).repeat(n,1)
        self.r1_cart_axis=torch.tensor([0.,1.,0.],device=device).repeat(n,1)
        self.r1_robot_hitch=torch.tensor(cart.ROBOT_HITCH,device=device).repeat(n,1)
        self.r1_cart_hitch=torch.tensor(cart.CART_HITCH,device=device).repeat(n,1)
        self.r1_drawbar_force=torch.zeros(n,device=device)
        self.r1_drawbar_length=torch.ones(n,device=device)
        self.r1_drawbar_guard=torch.zeros(n,dtype=torch.bool,device=device)
        self.r1_drawbar_peak=torch.zeros(n,device=device)
        self.r1_rolling_torque=torch.zeros(n,4,device=device)
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
        self.r1_physics_ready=True
        self.r1_last_metrics=None
        self._monitor_sums=None
        self._monitor_count=None
        self._monitor_keys=None

    @torch.inference_mode()
    def _reset_idx(self,env_ids):
        if self._collect_transition:
            values=mdp.measurements(self,self.r1_transition_command)
            self._terminal_metrics=(env_ids,{key:value[env_ids].clone() for key,value in values.items()})
        super()._reset_idx(env_ids)

    @torch.inference_mode()
    def step(self,action):
        self.r1_drawbar_guard.zero_()
        self.r1_drawbar_peak.zero_()
        self._collect_transition=True
        self._terminal_metrics=None
        self.r1_transition_command=self.command_manager.get_command("base_velocity").clone()
        result=super().step(action)
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
        self._accumulate(values)
        return result

    def _accumulate(self,values):
        selected=("command_vx","vx","vy","velocity_error","speed","yaw_rate_abs","tilt_deg","height",
                  "four_wheel","body_contact","torque_abs_max","torque_saturation","leg_target_clip",
                  "wheel_target_clip","drawbar_force_N","cart_tilt_deg","terminated","time_out",
                  "termination_illegal_contact","termination_bad_orientation_2","termination_outside_lane","termination_cart_failure")
        zero=values["command_vx"].abs()<.05
        high=values["command_vx"]>=3.
        full=(self.r1_payload_kg>=15)&(self.r1_towed_kg>=40)&(self.r1_slope_rad.abs()>.43)
        masks={"all":torch.ones(self.num_envs,dtype=torch.bool,device=self.device),"flat":self.r1_slope_rad==0,
               "uphill":self.r1_slope_rad>0,"downhill":self.r1_slope_rad<0,"payload15":self.r1_payload_kg>=15,
               "tow40":self.r1_towed_kg>=40,"zero":zero,"moving":~zero,"high_speed":high,
               "full_target_moving":full&high,"full_target_hold":full&zero}
        keys=list(masks)
        matrix=torch.stack([masks[key] for key in keys]).float()
        sums=matrix@torch.stack([values[key] for key in selected],dim=-1)
        counts=matrix.sum(-1)
        if self._monitor_sums is None:
            self._monitor_sums=torch.zeros_like(sums)
            self._monitor_count=torch.zeros_like(counts)
            self._monitor_keys=(keys,list(selected))
        self._monitor_sums+=sums
        self._monitor_count+=counts

    @torch.inference_mode()
    def drain_metrics(self):
        if self._monitor_sums is None:
            return []
        sums=self._monitor_sums.cpu().tolist()
        counts=self._monitor_count.cpu().tolist()
        groups,keys=self._monitor_keys
        rows=[]
        for name,total,count in zip(groups,sums,counts):
            rows.append(dict(group=name,samples=int(count),**{key:(value/count if count else None) for key,value in zip(keys,total)}))
        self._monitor_sums.zero_();self._monitor_count.zero_()
        return rows
