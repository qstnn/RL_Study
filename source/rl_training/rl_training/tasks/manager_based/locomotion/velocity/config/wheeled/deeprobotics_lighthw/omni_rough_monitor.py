"""只读50Hz分组指标，不修改动作、奖励、命令或原生PPO更新。"""
import torch

GROUPS=('retained_flat_0to1','retained_flat_1to2','retained_flat_2to3','retained_rough','retained_stop','forward','reverse','lateral',
        'yaw','mixed','heading','omni_stop')
FIELDS=('samples','command_vx','actual_vx','command_vy','actual_vy','command_wz','actual_wz',
        'velocity_xy_error','yaw_error','planar_speed','four_wheel_support','tilt_deg',
        'nonwheel_contact','wheel_force_peak','height_above_origin','leg_acc_E',
        'leg_target_URDF_clip_fraction')
WHEEL_FIELDS=('raw','target','actual','torque','absolute_mechanical_power','vertical_load')


class OmniMonitor:
    def __init__(self,env):
        self.env=env
        with torch.inference_mode(False):
            self.sums=torch.zeros(len(GROUPS),len(FIELDS),device=env.device,dtype=torch.float64)
            self.wheels=torch.zeros(len(GROUPS),len(WHEEL_FIELDS),4,device=env.device,dtype=torch.float64)
        self.wheel_ids=None

    @torch.no_grad()
    def update(self):
        env=self.env;r=env.scene['robot'].data;c=env.command_manager.get_term('base_velocity')
        leg=env.action_manager.get_term('joint_pos');wheel=env.action_manager.get_term('joint_vel')
        sensor=env.scene['contact_forces']
        if self.wheel_ids is None:
            self.wheel_ids=sensor.find_bodies([n.replace('_joint','_link') for n in wheel._joint_names],preserve_order=True)[0]
            self.other=[i for i,n in enumerate(sensor.body_names) if not n.endswith('_wheel_link')]
        command=c.command;force=sensor.data.net_forces_w;load=force[:,self.wheel_ids,2]
        flat=torch.isin(c.terrain_columns(),torch.as_tensor(c._flat,device=env.device))
        stop=command.norm(dim=1)<=.05
        group=torch.full((env.num_envs,),11,device=env.device,dtype=torch.long)
        for index,(lo,hi) in enumerate(((0.,1.),(1.,2.),(2.,3.))):
            group[c.retention&~stop&flat&(command[:,0]>lo)&(command[:,0]<=hi)]=index
        group[c.retention&~stop&~flat]=3;group[c.retention&stop]=4
        general=~c.retention&~stop
        group[general&(c.mode==2)&(command[:,0]>0)]=5
        group[general&(c.mode==2)&(command[:,0]<0)]=6
        for mode,index in ((3,7),(4,8),(5,9),(6,10)):group[general&(c.mode==mode)]=index
        target=leg._offset+leg.raw_actions*leg._scale
        clipped=((target<=leg._clip[:,:,0])|(target>=leg._clip[:,:,1])).float().mean(1)
        values=torch.stack((torch.ones_like(command[:,0]),command[:,0],r.root_lin_vel_b[:,0],
            command[:,1],r.root_lin_vel_b[:,1],command[:,2],r.root_ang_vel_b[:,2],
            (command[:,:2]-r.root_lin_vel_b[:,:2]).norm(dim=1),(command[:,2]-r.root_ang_vel_b[:,2]).abs(),
            r.root_lin_vel_b[:,:2].norm(dim=1),(load>5.).all(1).float(),
            torch.rad2deg(torch.acos((-r.projected_gravity_b[:,2]).clamp(-1,1))),
            (force[:,self.other].norm(dim=-1)>1.).any(1).float(),force[:,self.wheel_ids].norm(dim=-1).amax(1),
            r.root_pos_w[:,2]-env.scene.env_origins[:,2],r.joint_acc[:,leg._joint_ids].square().mean(1),clipped),1)
        self.sums.index_add_(0,group,torch.nan_to_num(values,nan=1e6,posinf=1e6,neginf=-1e6).double())
        torque=r.applied_torque[:,wheel._joint_ids];actual=r.joint_vel[:,wheel._joint_ids]
        values=torch.stack((wheel.raw_actions,wheel.processed_actions,actual,torque,(torque*actual).abs(),load),1)
        self.wheels.index_add_(0,group,torch.nan_to_num(values,nan=1e6,posinf=1e6,neginf=-1e6).double())

    def consume(self):
        rows=self.sums.cpu().tolist();wheels=self.wheels.cpu().tolist();result={}
        for i,name in enumerate(GROUPS):
            n=rows[i][0]
            result[name]={'samples':int(n),**{k:v/max(n,1) for k,v in zip(FIELDS[1:],rows[i][1:])},
                'wheels':{k:[v/max(n,1) for v in row] for k,row in zip(WHEEL_FIELDS,wheels[i])}}
        self.sums.zero_();self.wheels.zero_()
        return result
