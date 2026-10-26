"""特权支撑教师，只用于能力诊断和离线标签；训练/部署Actor禁止调用。

教师使用真实位姿、速度、接触几何、质量和牵引力。它不修改重力、摩擦、
外力或执行器，不给拖车助力，只输出与学生相同的16维腿位置/轮速动作。
"""
import torch
from isaaclab.utils.math import quat_apply

class SupportTeacher:
    def __init__(self,env):
        self.env=env.unwrapped
        self.robot=self.env.scene['robot']
        self.wheels=self.robot.find_bodies(['FR_wheel_link','FL_wheel_link','RR_wheel_link','RL_wheel_link'],preserve_order=True)[0]
        self.target_pos=None;self.previous_step=None;self.previous_v=None
        # Reachable front-short/rear-long poses with the four wheel centers fixed.
        import sys
        from pathlib import Path
        sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'unified_v3/diagnostics'))
        from support_ik import solve
        urdf=Path(__file__).resolve().parents[7]/'model_lighthw/lighthw_urdf/urdf/lighthw.urdf'
        poses=[solve(urdf,b) for b in (0.,1.,2.,3.,4.,5.)]
        self.pose_table=torch.tensor([v['q'] for v in poses],device=self.env.device,dtype=torch.float32)
        lean=solve(urdf,0.,.06)
        if lean['max_foot_error_m']>1e-4:raise RuntimeError('Tow support shift is unreachable')
        self.tow_pose=torch.tensor(lean['q'],device=self.env.device,dtype=torch.float32)

    @torch.no_grad()
    def __call__(self,obs):
        env,r=self.env,self.robot;device=env.device;n=env.num_envs
        root,vel=r.data.root_pos_w,r.data.root_lin_vel_w
        step=env.episode_length_buf
        command=env.command_manager.get_command('base_velocity')[:,0]
        if self.target_pos is None:
            self.target_pos=root.clone();self.previous_step=step.clone();self.previous_v=command.clone()
        reset=step<=self.previous_step
        self.target_pos[reset]=root[reset];self.previous_v[reset]=command[reset]
        stopped=(command.abs()<1e-4)&(self.previous_v.abs()>=1e-4)
        self.target_pos[stopped]=root[stopped]
        normal,tangent=env.r1_normal,env.r1_tangent
        lateral=torch.cross(normal,tangent,dim=-1)
        self.target_pos+=command[:,None]*tangent*env.step_dt
        dv=(command-self.previous_v)/env.step_dt
        self.previous_v=command.clone();self.previous_step=step.clone()
        mass=r.root_physx_view.get_masses().to(device)
        grav=torch.zeros(n,3,device=device);grav[:,2]=-9.81
        fg=mass[:,:,None]*grav[:,None,:]
        mg=torch.cross(r.data.body_com_pos_w-root[:,None,:],fg,dim=-1).sum(1)
        rh=quat_apply(r.data.root_quat_w,env.r1_robot_hitch)+root
        cart=env.scene['trailer']
        ch=quat_apply(cart.data.root_quat_w,env.r1_cart_hitch)+cart.data.root_pos_w
        direction=(rh-ch)/torch.linalg.vector_norm(rh-ch,dim=-1,keepdim=True).clamp_min(.1)
        ext=-direction*env.r1_drawbar_force[:,None]*(env.r1_towed_kg>0)[:,None]
        zaxis=quat_apply(r.data.root_quat_w,env.r1_axis_z)
        forward=quat_apply(r.data.root_quat_w,env.r1_axis_x)
        heading=torch.atan2((forward*lateral).sum(-1),(forward*tangent).sum(-1))
        bias_deg=(env.r1_slope_rad*180./torch.pi*.2).clamp(0.,5.)
        tow_shift=(env.r1_towed_kg/40.).clamp(0.,1.)*(env.r1_slope_rad/(25.*torch.pi/180.)).clamp(0.,1.)
        bias_deg=bias_deg*(1.-(env.r1_towed_kg>0).float())
        bias=bias_deg*torch.pi/180.
        desired_z=normal*torch.cos(bias)[:,None]+tangent*torch.sin(bias)[:,None]
        moment=80.*torch.cross(zaxis,desired_z,dim=-1)-8.*r.data.root_ang_vel_w-30.*heading[:,None]*normal
        error=root-self.target_pos
        xerr=(error*tangent).sum(-1);yerr=(error*lateral).sum(-1)
        # Avoid integrating unattainable moving positions into braking demands.
        xerr=torch.where(command.abs()>1e-4,torch.zeros_like(xerr),xerr)
        h=((root-env.scene.env_origins)*normal).sum(-1)
        acc=(100.*(.416223-h)-15.*(vel*normal).sum(-1))[:,None]*normal
        acc+=(dv-10.*xerr-8.*((vel*tangent).sum(-1)-command))[:,None]*tangent
        acc+=(-10.*yerr-8.*(vel*lateral).sum(-1))[:,None]*lateral
        force=mass.sum(-1)[:,None]*acc-fg.sum(1)-ext
        moment=moment-mg-torch.cross(rh-root,ext,dim=-1)
        norm=(force*normal).sum(-1).clamp_min(15.)
        tx=((force*tangent).sum(-1)/norm).clamp(-3.8,3.8)
        ty=((force*lateral).sum(-1)/norm).clamp(-.5,.5)
        fdir=normal+tx[:,None]*tangent+ty[:,None]*lateral
        feet=r.data.body_pos_w[:,self.wheels]-normal[:,None,:]*.0875
        arms=feet-root[:,None,:]
        cross=torch.cross(arms,fdir[:,None,:].expand(-1,4,-1),dim=-1)
        mat=torch.stack((torch.ones(n,4,device=device),(cross*tangent[:,None,:]).sum(-1),(cross*lateral[:,None,:]).sum(-1)),1)
        rhs=torch.stack((norm,(moment*tangent).sum(-1),(moment*lateral).sum(-1)),-1)
        normals=(mat.transpose(1,2)@torch.linalg.solve(mat@mat.transpose(1,2)+torch.eye(3,device=device)*1e-5,rhs[:,:,None])).squeeze(-1).clamp_min(0.)
        normals*=norm[:,None]/normals.sum(-1,keepdim=True).clamp_min(1.)
        forces=normals[:,:,None]*fdir[:,None,:]
        # Zero-net lateral pair supplies yaw moment without a hidden body wrench.
        yaw=((moment-torch.cross(arms,forces,dim=-1).sum(1))*normal).sum(-1)
        x=(arms*tangent[:,None,:]).sum(-1);x=x-x.mean(-1,keepdim=True)
        forces+=yaw[:,None,None]*x[:,:,None]/x.square().sum(-1)[:,None,None].clamp_min(.01)*lateral[:,None,:]
        moments=torch.cross(-normal[:,None,:].expand(-1,4,-1)*.0875,forces,dim=-1)
        # Floating-base generalized quantities contain six root DOFs before joints.
        jac=r.root_physx_view.get_jacobians()[:,self.wheels,:,6:][:,:,:,env.r1_joint_ids]
        gravity=r.root_physx_view.get_gravity_compensation_forces()[:,6:][:,env.r1_joint_ids]
        tau=gravity-(jac[:,:,:3,:]*forces[:,:,:,None]).sum((1,2))-(jac[:,:,3:,:]*moments[:,:,:,None]).sum((1,2))
        action=torch.empty(n,16,device=device)
        scale=torch.tensor([.08,.14,.14]*4,device=device)
        lo=bias_deg.floor().long();hi=(lo+1).clamp_max(5);blend=(bias_deg-lo.float())[:,None]
        ref=self.pose_table[lo]*(1.-blend)+self.pose_table[hi]*blend
        default=r.data.default_joint_pos[:,env.r1_joint_ids[:12]]
        ref=ref+(self.tow_pose-self.pose_table[0])*tow_shift[:,None]
        action[:,:12]=(ref-default+tau[:,:12]/r.actuators['joint'].stiffness)/scale
        action[:,12:]=(command[:,None]/.0875+tau[:,12:]/r.actuators['wheel'].damping)/5.
        if not torch.isfinite(action).all():raise RuntimeError('Non-finite teacher action')
        return action
