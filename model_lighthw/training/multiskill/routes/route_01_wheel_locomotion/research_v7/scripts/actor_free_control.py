"""Diagnostic posture/traction factorial; not the deployed Actor and no added external forces."""
import math
import xml.etree.ElementTree as ET
import numpy as np
import torch
from scipy.spatial.transform import Rotation

LEGS=('FR','FL','RR','RL')

def forward_and_jacobian(urdf,positions,*,with_angular=False):
    joints={x.get('name'):x for x in ET.parse(urdf).getroot().findall('joint')}
    def fk(leg,q):
        transform=np.eye(4);axes=[]
        for i,suffix in enumerate(('hip','thigh','calf','wheel')):
            j=joints[f'{leg}_{suffix}_joint'];origin=j.find('origin')
            fixed=np.eye(4);fixed[:3,3]=np.fromstring(origin.get('xyz'),sep=' ')
            fixed[:3,:3]=Rotation.from_euler('xyz',np.fromstring(origin.get('rpy','0 0 0'),sep=' ')).as_matrix()
            transform=transform@fixed
            if i<3:
                axes.append(transform[:3,:3]@np.fromstring(j.find('axis').get('xyz'),sep=' '))
                moving=np.eye(4);moving[:3,:3]=Rotation.from_rotvec(np.fromstring(j.find('axis').get('xyz'),sep=' ')*q[i]).as_matrix();transform=transform@moving
        return transform[:3,3],np.column_stack(axes)
    feet=[];jacobians=[];angular=[]
    for i,leg in enumerate(LEGS):
        q=np.asarray(positions[i*3:i*3+3]);point,axes=fk(leg,q);feet.append(point);angular.append(axes)
        jacobians.append(np.column_stack([(fk(leg,q+np.eye(3)[j]*1.e-5)[0]-fk(leg,q-np.eye(3)[j]*1.e-5)[0])/2.e-5 for j in range(3)]))
    result=(np.asarray(feet),np.asarray(jacobians))
    return (*result,np.asarray(angular)) if with_angular else result


def attitude_correction(gravity,omega,desired_pitch,kp=.8,kd=.15,max_angle=.10):
    """IMU feedback; positive nose-up pitch uses the opposite sign of body omega-y."""
    pitch=torch.atan2(-gravity[:,0],torch.sqrt(gravity[:,1].square()+gravity[:,2].square()))
    roll=torch.atan2(-gravity[:,1],-gravity[:,2])
    dp=(kp*(desired_pitch-pitch)+kd*omega[:,1]).clamp(-max_angle,max_angle)
    dr=(-kp*roll-kd*omega[:,0]).clamp(-max_angle,max_angle)
    rotation=torch.stack((dr,-dp,torch.zeros_like(dp)),dim=-1)
    return rotation,pitch,roll


def allocate_traction(normal,force,omega,vx,heading,yaw_rate,*,mu=4.,eta=.75,radius=.0875,slip_damping=.2):
    """Wheel-axis torque targets from contact normal loads; no force applied to robot/cart roots."""
    normal=normal.clamp_min(0.)
    total=normal.sum(-1).clamp_min(1.e-6)
    bounded=force.clamp(-eta*mu*total,eta*mu*total)
    shares=normal/total[:,None]
    longitudinal=bounded[:,None]*shares*radius
    slip=omega-vx[:,None]/radius
    turn=(-4.*heading-.5*yaw_rate).clamp(-1.,1.)
    desired=longitudinal-slip_damping*slip+turn[:,None]*normal.new_tensor([1.,-1.,1.,-1.])
    limits=(eta*mu*normal*radius).clamp_max(48.)
    desired=torch.maximum(torch.minimum(desired,limits),-limits)
    return desired,bounded,shares,limits,slip


class PhysicsRecorder:
    """Post-integration 1kHz sensing, including terminal substeps before reset; read-only."""
    def __init__(self,base,worker):
        from isaaclab.utils.math import quat_apply
        self.base=base;self.worker=worker;self.quat_apply=quat_apply;self.robot=base.scene['robot'];self.cart=base.scene['trailer'];self.cursor=0;self.last=-1
        self.wheels=self.robot.find_bodies([f'{leg}_wheel_link' for leg in LEGS],preserve_order=True)[0]
        self.masses=self.robot.root_physx_view.get_masses().to(base.device)
        self.weight=self.masses/self.masses.sum(-1,keepdim=True)
        # Isaac Lab ContactSensor contains normal force only. Read actual friction through a separate native view.
        from isaaclab.sim import get_current_stage
        from pxr import UsdPhysics
        stage=get_current_stage()
        ground=[str(prim.GetPath()) for prim in stage.Traverse() if str(prim.GetPath()).startswith(base.cfg.scene.terrain.prim_path+'/') and prim.HasAPI(UsdPhysics.CollisionAPI)]
        if not ground:raise RuntimeError('No terrain collision prim for native friction view')
        sensor=base.scene['contact_forces'];all_paths=list(sensor.body_physx_view.prim_paths)
        requested=[next(path for path in all_paths if f'/env_{i}/' in path and path.endswith('/'+leg+'_wheel_link')) for i in range(base.num_envs) for leg in LEGS]
        self.contact_view=sensor._physics_sim_view.create_rigid_contact_view(requested,filter_patterns=[ground]*len(requested),max_contact_data_count=len(requested)*128)
        actual=list(self.contact_view.sensor_paths)
        if len(actual)!=len(requested):raise RuntimeError('Friction view wheel count mismatch')
        self.contact_order=torch.tensor([actual.index(path) for path in requested],device=base.device)
        self.contact_offsets=torch.arange(64,device=base.device)
        self.contact_meta=dict(sensor_paths=actual,requested_paths=requested,ground_filters=ground,capacity=self.contact_view.max_contact_data_count,max_patch_points_per_pair=64)
        shapes=dict(applied_drawbar_force_w=(3,),contact_force_w=(4,3),friction_force_w=(4,3),native_ground_normal_w=(4,3),friction_patch_count=(4,),body_com_positions_w=(self.robot.num_bodies,3),whole_com_w=(3,),whole_com_velocity_w=(3,),body_pos_w=(3,),body_quat_w=(4,),body_omega_b=(3,),wheel_pos_w=(4,3),wheel_center_velocity_w=(4,3),wheel_angular_velocity_w=(4,3),hitch_robot_w=(3,),hitch_cart_w=(3,),drawbar_tension_interval_N=(),joint_pos=(16,),joint_velocity=(16,),whole_mass_kg=())
        self.buffers={k:torch.zeros((base.num_envs,base.cfg.decimation,*shape),device=base.device) for k,shape in shapes.items()}
        self.history={k:[] for k in shapes}
        original_sample=base.r1_torque_recorder.sample
        def interval_sample():
            before=base.r1_torque_recorder.cursor
            original_sample()
            if base.r1_torque_recorder.cursor>before:
                rh=self.quat_apply(self.robot.data.root_quat_w,base.r1_robot_hitch)+self.robot.data.root_pos_w
                ch=self.quat_apply(self.cart.data.root_quat_w,base.r1_cart_hitch)+self.cart.data.root_pos_w
                delta=rh-ch;direction=delta/delta.norm(dim=-1,keepdim=True).clamp_min(1.e-6)
                self.buffers['applied_drawbar_force_w'][:,before]=-base.r1_drawbar_force.clamp(-1500.,1500.)[:,None]*direction
        base.r1_torque_recorder.sample=interval_sample
        original_begin=base.r1_torque_recorder.begin
        def begin():self.cursor=0;original_begin()
        base.r1_torque_recorder.begin=begin
        original_update=base.scene.update
        def update(dt):
            original_update(dt)
            if base.r1_torque_recorder.enabled and base._sim_step_counter!=self.last:self.sample()
        base.scene.update=update
    def sample(self):
        b,r,c=self.base,self.robot,self.cart;k=self.cursor
        if k>=b.cfg.decimation:raise RuntimeError('Duplicate post-integration physical sample')
        values=dict(contact_force_w=b.scene['contact_forces'].data.net_forces_w[:,b.r1_wheel_sensor_ids],
          whole_com_w=(r.data.body_com_pos_w*self.weight[:,:,None]).sum(1),
          whole_com_velocity_w=(r.data.body_com_lin_vel_w*self.weight[:,:,None]).sum(1),
          body_pos_w=r.data.root_pos_w,body_quat_w=r.data.root_quat_w,body_omega_b=r.data.root_ang_vel_b,
          wheel_pos_w=r.data.body_pos_w[:,self.wheels],wheel_center_velocity_w=r.data.body_lin_vel_w[:,self.wheels],
          wheel_angular_velocity_w=r.data.body_ang_vel_w[:,self.wheels],
          hitch_robot_w=self.quat_apply(r.data.root_quat_w,b.r1_robot_hitch)+r.data.root_pos_w,
          hitch_cart_w=self.quat_apply(c.data.root_quat_w,b.r1_cart_hitch)+c.data.root_pos_w,
          drawbar_tension_interval_N=b.r1_drawbar_force,
          joint_pos=r.data.joint_pos[:,b.r1_joint_ids],joint_velocity=r.data.joint_vel[:,b.r1_joint_ids],whole_mass_kg=self.masses.sum(-1))
        friction,points,counts,starts=self.contact_view.get_friction_data(dt=b.physics_dt)
        counts=counts.long();starts=starts.long()
        valid=self.contact_offsets[None,None,:]<counts[:,:,None]
        indices=(starts[:,:,None]+self.contact_offsets[None,None,:]).clamp(0,len(friction)-1)
        patches=torch.where(valid[:,:,:,None],friction[indices],torch.zeros((),device=b.device))
        aggregate=patches.sum((1,2))[self.contact_order].reshape(b.num_envs,4,3)
        ground_normal=self.contact_view.get_contact_force_matrix(dt=b.physics_dt).sum(1)[self.contact_order].reshape(b.num_envs,4,3)
        values.update(friction_force_w=aggregate,native_ground_normal_w=ground_normal,
            friction_patch_count=counts.max(1).values[self.contact_order].reshape(b.num_envs,4),body_com_positions_w=r.data.body_com_pos_w)
        for name,value in values.items():self.buffers[name][:,k]=value
        self.cursor+=1;self.last=b._sim_step_counter
    def capture(self):
        if self.cursor!=self.base.cfg.decimation:raise RuntimeError(f'Incomplete contact/COM trace: {self.cursor}')
        for name,value in self.worker.cpu_measurements(self.buffers).items():
            if not np.isfinite(value).all():raise RuntimeError('Non-finite 1kHz '+name)
            self.history[name].append(value.copy())


class Coordinator:
    def __init__(self,base,cfg,cases,proof,offline_pose,urdf,recorder,*,kp=.8,kd=.15,velocity_gain=2.,slip_damping=.2,filter_tau=.04,wheel_mode='torque'):
        self.base=base;self.cfg=cfg;self.cases=cases;self.recorder=recorder;self.kp=kp;self.kd=kd;self.velocity_gain=velocity_gain;self.slip_damping=slip_damping;self.filter_tau=filter_tau;self.wheel_mode=wheel_mode
        self.ids=torch.tensor([i for i,c in enumerate(cases) if c['variant'] in ('attitude','traction','coordinated','coordinated_ff','coordinated_ff12')],device=base.device,dtype=torch.long)
        self.attitude=torch.tensor([c['variant'] in ('attitude','coordinated','coordinated_ff','coordinated_ff12') for c in cases],device=base.device)
        self.traction=torch.tensor([c['variant'] in ('traction','coordinated','coordinated_ff','coordinated_ff12') for c in cases],device=base.device)
        self.force_feedforward=torch.tensor([c['variant'] in ('coordinated_ff','coordinated_ff12') for c in cases],device=base.device)
        self.ff_limit=offline_pose.new_tensor([12. if c['variant']=='coordinated_ff12' else 4. for c in cases])
        self.pose=offline_pose;self.default=offline_pose.new_tensor([0.,.6,-1.5,0.,.6,-1.5,0.,-.6,1.5,0.,-.6,1.5]);self.scale=offline_pose.new_tensor([.08,.14,.14]*4)
        self.desired=offline_pose.new_tensor(proof['target_actual_joint_pos']);self.offset=self.default+offline_pose[:12]*self.scale-offline_pose.new_tensor(proof['source_actual_joint_pos'])
        feet,jac,angular=forward_and_jacobian(urdf,proof['target_actual_joint_pos'],with_angular=True);self.feet=offline_pose.new_tensor(feet);self.jac_inverse=offline_pose.new_tensor(np.linalg.inv(jac))
        # External force virtual work includes the contact lever below the wheel center.
        nominal_inverse=Rotation.from_rotvec([0.,math.radians(-3.),0.]).as_matrix()
        lever=nominal_inverse@np.array([0.,0.,-.0875])
        contact_jac=jac+np.cross(angular.transpose(0,2,1),lever).transpose(0,2,1)
        self.contact_jacobian=offline_pose.new_tensor(contact_jac)
        self.nominal_tangent_body=offline_pose.new_tensor(nominal_inverse@np.array([1.,0.,0.]))
        reference=__import__('pathlib').Path(__file__).resolve().parents[2]/'unified_v5/results/2026-10-19_coordinated_towing/factorial_velocity_01/seed43_case00_physics_trace.npz'
        with np.load(reference,allow_pickle=False) as z:
            tangent=np.array([math.cos(math.radians(25)),0.,math.sin(math.radians(25))])
            self.reference_traction=offline_pose.new_tensor((z['friction_force_w'][2500:3000]@tangent).mean(0))
        self.leg_pd_stiffness=float(cfg.scene.robot.actuators['joint'].stiffness)
        self.geometry=dict(contact_jacobian_body=contact_jac.tolist(),tangent_body=self.nominal_tangent_body.cpu().tolist(),reference_traction_N=self.reference_traction.cpu().tolist(),reference_path=str(reference),reference_sha256=__import__('hashlib').sha256(reference.read_bytes()).hexdigest(),feedforward_clip_Nm=self.ff_limit.cpu().tolist(),leg_pd_stiffness=self.leg_pd_stiffness)
        names=cfg.observations.policy.joint_vel.params['asset_cfg'].joint_names
        indices=[list(names).index(f'{leg}_wheel_joint') for leg in LEGS]
        self.velocity_columns=torch.tensor([189+j for j in indices],device=base.device)
        scale=cfg.observations.policy.joint_vel.scale
        self.velocity_scale=torch.as_tensor(1. if scale is None else scale,device=base.device).expand(16)[indices]
        self.filtered_normal=torch.zeros((base.num_envs,4),device=base.device)
        self.mass=recorder.masses.sum(-1)
    def apply(self,obs,action,cmd,actual,t):
        b=self.base;n=b.num_envs
        omega=obs[:,12:15]/torch.as_tensor(self.cfg.observations.policy.base_ang_vel.scale,device=b.device)
        gravity=obs[:,27:30]
        rotation,pitch,roll=attitude_correction(gravity,omega,b.r1_slope_rad-math.radians(3.),self.kp,self.kd)
        ramp_start=getattr(self,'ramp_start',1.5)
        ramp_duration=getattr(self,'ramp_duration',1.)
        ramp=((t-ramp_start)/ramp_duration).clamp(0.,1.)
        foot_delta=-torch.cross(rotation[:,None,:].expand(-1,4,-1),self.feet[None].expand(n,-1,-1),dim=-1)
        dq=torch.einsum('lij,nlj->nli',self.jac_inverse,foot_delta).reshape(n,12).clamp(-.12,.12)*ramp[:,None]
        desired=self.desired[None]+dq*self.attitude[:,None]
        position_error_limit=getattr(self,'position_error_limit',.05)
        target=actual+(desired-actual).clamp(-position_error_limit,position_error_limit)+self.offset
        ids=self.ids
        action[ids,:12]=(target[ids]-self.default)/self.scale
        heading=obs[:,289];yaw=omega[:,2];corr=(-8.*heading-yaw).clamp(-2.,2.)
        action[ids,12:]=self.pose[None,12:]+cmd[ids,None]/.0875/5.+corr[ids,None]*obs.new_tensor([1.,-1.,1.,-1.])/5.
        if self.recorder.cursor==b.cfg.decimation:
            measured=(self.recorder.buffers['contact_force_w']*b.r1_normal[:,None,None,:]).sum(-1).clamp_min(0.).mean(1)
        else:measured=torch.zeros_like(self.filtered_normal)
        alpha=b.step_dt/(self.filter_tau+b.step_dt)
        reset=t==0.;self.filtered_normal[reset]=measured[reset]
        self.filtered_normal=torch.lerp(self.filtered_normal,measured,alpha)
        r=b.scene['robot'];vx=(r.data.root_com_lin_vel_w*b.r1_tangent).sum(-1)
        wheel=obs[:,self.velocity_columns]/self.velocity_scale
        # Measured drawbar reaction along the lane plus robot gravity and bounded speed feedback.
        from isaaclab.utils.math import quat_apply
        rh=quat_apply(r.data.root_quat_w,b.r1_robot_hitch)+r.data.root_pos_w
        cart=b.scene['trailer'];ch=quat_apply(cart.data.root_quat_w,b.r1_cart_hitch)+cart.data.root_pos_w
        delta=rh-ch;direction=delta/delta.norm(dim=-1,keepdim=True).clamp_min(1.e-6)
        drawbar_along=b.r1_drawbar_force*(direction*b.r1_tangent).sum(-1)
        # The passive cart initially shares the downhill slip with the robot,
        # so measured spring tension is low before motion begins. Preload from
        # the known mass and slope instead of waiting for the spring to stretch.
        cart_gravity=b.r1_towed_kg*9.81*torch.sin(b.r1_slope_rad)
        cart_rolling=b.r1_towed_kg*9.81*self.cfg.rolling_coefficient*torch.cos(b.r1_slope_rad)
        anticipated_drawbar=torch.maximum(drawbar_along,cart_gravity+cart_rolling)
        force_ff=self.mass*9.81*torch.sin(b.r1_slope_rad)+anticipated_drawbar
        force=force_ff+(self.mass+b.r1_towed_kg)*self.velocity_gain*(cmd-vx)
        torque,bounded,shares,limits,slip=allocate_traction(self.filtered_normal,force,wheel,vx,heading,yaw,slip_damping=self.slip_damping)
        traction_sign=getattr(self,'traction_sign',1.)
        effective=traction_sign*torque
        if self.wheel_mode=='velocity':
            # Keep native velocity-servo damping, then bound the instantaneous request by the contact cap.
            effective=traction_sign*(torque+2.*(cmd[:,None]/.0875-wheel)).clamp(-limits,limits)
        raw=(wheel+effective/2.).clamp(-50.,50.)/5.
        blend=(ramp*self.traction)[:,None]
        action[:,12:]=torch.lerp(action[:,12:],raw,blend)
        delta_traction=effective/.0875-self.reference_traction[None]
        delta_force=delta_traction[:,:,None]*self.nominal_tangent_body
        leg_feedforward=-torch.einsum('lji,nlj->nli',self.contact_jacobian,delta_force).reshape(n,12)
        leg_feedforward=leg_feedforward.clamp(-self.ff_limit[:,None],self.ff_limit[:,None])*self.force_feedforward[:,None]*ramp[:,None]
        action[:,:12]+=leg_feedforward/self.leg_pd_stiffness/self.scale
        return action,dict(control_leg_feedforward_Nm=leg_feedforward,control_rotation=rotation,control_pitch=pitch,control_roll=roll,control_desired_q=desired,
            control_normal_N=self.filtered_normal.clone(),control_force_ff=force_ff,control_force_request=force,control_force_bounded=bounded,
            control_torque_target=torque,control_effective_torque_target=effective,control_torque_contact_cap=limits,control_shares=shares,control_slip_rad_s=slip,
            control_wheel_estimate=wheel,control_true_vx=vx,control_blend=blend.squeeze(-1))
