"""50Hz训练起停遥测，独立缓冲，不参与奖励/门槛/动作或PPO更新。

随机命令下用相同时间定义分组；1kHz独立评估才用于持续低载和起停验收。
"""
import torch

PHASES=('moving_first2s','moving_steady','braking_first2s','hold_after_motion_2to8s','fixed_zero_2to8s')
FIELDS=('samples','command_vx','actual_vx','speed_error','four_wheel_support','nonwheel_contact','tilt_deg')


class ReferenceWheelMonitor:
    """每轮/命令阶段的50Hz真实控制链；只操作自身累加缓冲，不消费RNG。"""
    FIELDS=('raw','target_rad_s','actual_rad_s','torque_nm','load_n')
    def __init__(self,device):
        self.sum=torch.zeros((len(PHASES),4,5),device=device,dtype=torch.float64)
        self.maximum=torch.zeros_like(self.sum)
        self.count=torch.zeros(len(PHASES),device=device,dtype=torch.float64)

    def update(self,transition,command,raw,target,actual,torque,load):
        moving=command[:,0]>.05
        age=transition.age;seen=transition.seen_motion
        masks=torch.stack((moving&(age<2),moving&(age>=2),~moving&seen&(age<2),
            ~moving&seen&(age>=2)&(age<8),~moving&~seen&(age>=2)&(age<8)),dim=1)
        values=torch.stack((raw,target,actual,torque,load),dim=-1)
        self.sum.add_(torch.einsum('np,nwf->pwf',masks.float(),values).double())
        self.count.add_(masks.sum(0).double())
        for i in range(len(PHASES)):
            maximum=torch.where(masks[:,i,None,None],values.abs(),0).amax(0)
            self.maximum[i].copy_(torch.maximum(self.maximum[i],maximum))

    def consume(self):
        sums=self.sum.cpu().tolist();counts=self.count.cpu().tolist();maximum=self.maximum.cpu().tolist()
        result={}
        for i,phase in enumerate(PHASES):
            result[f'{phase}/samples']=counts[i]
            if not counts[i]:continue
            for j,wheel in enumerate(('FR','FL','RR','RL')):
                for k,field in enumerate(self.FIELDS):
                    result[f'{phase}/{wheel}/{field}_mean']=sums[i][j][k]/counts[i]
                    result[f'{phase}/{wheel}/{field}_max_abs']=maximum[i][j][k]
        self.sum.zero_();self.count.zero_();self.maximum.zero_()
        return result


class ReferenceTransitionMonitor:
    def __init__(self,num_envs,device,dt):
        self.dt=dt
        self.previous=torch.zeros((num_envs,3),device=device)
        self.age=torch.zeros(num_envs,device=device)
        self.seen_motion=torch.zeros(num_envs,device=device,dtype=torch.bool)
        self.initialized=torch.zeros_like(self.seen_motion)
        self.pending_brake=torch.zeros_like(self.seen_motion)
        self.hold_ready=torch.zeros_like(self.seen_motion)
        self.stable_steps=torch.zeros(num_envs,device=device,dtype=torch.long)
        self.distance=torch.zeros(num_envs,device=device)
        self.last_xy=torch.zeros((num_envs,2),device=device)
        self.anchor=torch.zeros_like(self.last_xy)
        self.low_steps=torch.zeros((num_envs,4),device=device,dtype=torch.long)
        self.stats=torch.zeros((len(PHASES),len(FIELDS)),device=device,dtype=torch.float64)
        self.max_displacement=torch.zeros(len(PHASES),device=device)
        self.brake_totals=torch.zeros(4,device=device,dtype=torch.float64)
        self.low_max=torch.zeros(4,device=device)

    def reset(self,env_ids=None):
        ids=slice(None) if env_ids is None else env_ids
        for name in ('previous','age','seen_motion','initialized','pending_brake','hold_ready',
                     'stable_steps','distance','last_xy','anchor','low_steps'):
            getattr(self,name)[ids]=0

    def update(self,command,velocity,xy,gravity,wheel_fz,nonwheel_force):
        moving=command[:,0]>.05
        changed=(command-self.previous).abs().amax(-1)>1e-5
        new_stop=(self.previous[:,0]>.05)&~moving&self.initialized
        self.age.copy_(torch.where(changed,0.,self.age+self.dt))
        self.seen_motion.logical_or_(moving)
        self.hold_ready.logical_and_(~changed)
        self.pending_brake.logical_and_(~changed)
        self.pending_brake.logical_or_(new_stop)
        self.stable_steps.masked_fill_(changed,0)
        self.distance.masked_fill_(changed,0)
        increment=(xy-self.last_xy).norm(dim=-1)
        self.distance.add_(torch.where(self.pending_brake&self.initialized,increment,0.))
        speed=velocity[:,:2].norm(dim=-1)
        self.stable_steps.copy_(torch.where(self.pending_brake&(speed<.05),self.stable_steps+1,0))
        confirmed=self.pending_brake&(self.stable_steps*self.dt>=.2)
        self.brake_totals.add_(torch.stack((new_stop.sum(),confirmed.sum(),
            torch.where(confirmed,self.age+self.dt,0.).sum(),torch.where(confirmed,self.distance,0.).sum())).double())
        self.pending_brake.logical_and_(~confirmed)
        settled_stop=~moving&(self.age>=2)&(self.age<8)
        anchoring=settled_stop&~self.hold_ready
        self.anchor[anchoring]=xy[anchoring]
        self.hold_ready.logical_or_(anchoring)
        displacement=(xy-self.anchor).norm(dim=-1)
        masks=torch.stack((moving&(self.age<2),moving&(self.age>=2),
            ~moving&self.seen_motion&(self.age<2),settled_stop&self.seen_motion,
            settled_stop&~self.seen_motion),dim=1)
        features=torch.stack((torch.ones_like(speed),command[:,0],velocity[:,0],
            torch.where(moving,abs(velocity[:,0]-command[:,0]),speed),
            (wheel_fz>5).all(-1).float(),(nonwheel_force>1).any(-1).float(),
            torch.rad2deg(torch.acos((-gravity[:,2]).clamp(-1,1)))),dim=1)
        self.stats.add_((masks.float().T@features).double())
        for i in (3,4):
            value=torch.where(masks[:,i],displacement,0.).max()
            self.max_displacement[i].copy_(torch.maximum(self.max_displacement[i],value))
        self.low_steps.copy_(torch.where(wheel_fz<=5,self.low_steps+1,0))
        self.low_max.copy_(torch.maximum(self.low_max,self.low_steps.max(0).values*self.dt))
        self.previous.copy_(command);self.last_xy.copy_(xy);self.initialized.fill_(True)

    def consume(self):
        rows=self.stats.cpu().tolist();maximum=self.max_displacement.cpu().tolist()
        result={}
        for phase,row,max_displacement in zip(PHASES,rows,maximum):
            result[f'{phase}/samples']=row[0]
            if row[0]:
                result.update({f'{phase}/{key}':value/row[0] for key,value in zip(FIELDS[1:],row[1:])})
                if phase.startswith(('hold','fixed')):result[f'{phase}/max_displacement_m']=max_displacement
        candidates,completed,time,distance=self.brake_totals.cpu().tolist()
        result.update(brake_candidates=candidates,brake_confirmations=completed)
        if completed:
            result.update(brake_time_to_confirmation_s=time/completed,brake_path_to_confirmation_m=distance/completed)
        for name,value in zip(('FR','FL','RR','RL'),self.low_max.cpu().tolist()):
            result[f'{name}/max_continuous_low_load_s_at_50Hz']=value
        self.stats.zero_();self.max_displacement.zero_();self.brake_totals.zero_();self.low_max.zero_()
        return result
