"""独立关节参考探针：只施加受原驱动限制的目标，不修改根状态。"""
import torch


class Reference:
    def __init__(self, env, profiles):
        self.env=env
        self.profiles=profiles
        self.params={k:torch.tensor([p[k] for p in profiles],device=env.device) for k in profiles[0] if isinstance(profiles[0][k],(int,float))}
        self.stand=torch.tensor([0.,.6,-1.5,0.,.6,-1.5,0.,-.6,1.5,0.,-.6,1.5],device=env.device).expand(env.num_envs,-1)
        self.released=torch.zeros(env.num_envs,dtype=torch.bool,device=env.device)
        self.landed=self.released.clone();self.apex=self.released.clone()
        self.release_t=torch.zeros(env.num_envs,device=env.device)
        self.apex_t=self.release_t.clone();self.landing_t=self.release_t.clone()
        self.rotation=self.release_t.clone();self.previous_angle=self.release_t.clone()
        self.last_tick=-1
        self.clear_steps=torch.zeros(env.num_envs,dtype=torch.long,device=env.device)

    def update(self,t,m):
        e=self.env;d=e.robot.data
        # 后空翻进度：机身前向轴相对世界 XZ 的连续角；侧翻另行统计。
        from isaaclab.utils.math import matrix_from_quat
        r=matrix_from_quat(d.root_quat_w)
        angle=torch.atan2(r[:,2,0],r[:,0,0])
        delta=torch.atan2(torch.sin(angle-self.previous_angle),torch.cos(angle-self.previous_angle))
        self.rotation += delta
        self.previous_angle.copy_(angle)
        forces=e.contacts.data.net_forces_w.norm(dim=-1)
        clear=(forces<2.).all(-1)&(m['clearance']>0.).all(-1)
        candidate=(t>=.65)&clear&(m['com_vel'][:,2]>.25)
        self.clear_steps=torch.where(candidate,self.clear_steps+1,0)
        release=(~self.released)&(self.clear_steps>=7)
        self.release_t=torch.where(release,t-6*e.physics_dt,self.release_t);self.released|=release
        apex=self.released&~self.apex&(m['com_vel'][:,2]<=0.)
        self.apex_t=torch.where(apex,t,self.apex_t);self.apex|=apex
        land=self.released&~self.landed&((forces>=2.).any(-1))&(t>self.release_t)
        self.landing_t=torch.where(land,t,self.landing_t);self.landed|=land

    def pose(self,t):
        p=self.params;stand=self.stand
        crouch=stand.clone();extend=stand.clone();pose=stand.clone()
        for j,sign in ((0,1),(3,1),(6,-1),(9,-1)):
            rear=j>=6
            thigh=p['crouch_thigh']+(p['rear_crouch_offset'] if rear else 0.)
            calf=p['crouch_calf']-(p['rear_crouch_offset'] if rear else 0.)
            crouch[:,j+1]=sign*thigh;crouch[:,j+2]=sign*calf
            extend[:,j+1]=sign*p['extend_thigh'];extend[:,j+2]=sign*p['extend_calf']
            start=p['push_start']+(p['rear_delay'] if rear else 0.)
            earliest=start+torch.minimum(p['knee_delay'],torch.zeros_like(t))
            # 先完整下蹲，可选择保留短暂停顿；每个关节起蹬处连续。
            end=earliest-p['hold']
            alpha=((t-p['crouch_start'])/(end-p['crouch_start'])).clamp(0.,1.)
            for off,duration,delay in ((1,p['thigh_duration'],0.),(2,p['knee_duration'],p['knee_delay'])):
                k=j+off;down=stand[:,k]+(crouch[:,k]-stand[:,k])*alpha
                up=crouch[:,k]+(extend[:,k]-crouch[:,k])*((t-start-delay)/duration).clamp(0.,1.)
                pose[:,k]=torch.where(t>=start+delay,up,down)
                # 主动收回一对支撑腿，检查另一对蹬伸是否能保留旋转冲量。
                enabled=p['unload_rear'] if rear else p['unload_front']
                unload_start=p['push_start']+p['unload_time']
                target=sign*(p['tuck_thigh'] if off==1 else p['tuck_calf'])
                unload_alpha=((t-unload_start)/p['unload_duration']).clamp(0.,1.)
                unload=pose[:,k]+(target-pose[:,k])*unload_alpha
                pose[:,k]=torch.where(enabled>0,unload,pose[:,k])
        # 空中收腿固定为独立目标，避免加深预蹲时把收腿推到关节限位外。
        tuck=stand.clone()
        for j,sign in ((0,1),(3,1),(6,-1),(9,-1)):
            tuck[:,j+1]=sign*p['tuck_thigh'];tuck[:,j+2]=sign*p['tuck_calf']
        age=t-self.release_t
        tucked=extend+(tuck-extend)*(age/p['tuck_duration']).clamp(0.,1.)[:,None]
        # 竖跳在顶点展开；旋转探针到指定累计角度后展开。
        prepare=self.apex & (t>=self.apex_t+p['apex_delay'])
        prepare=torch.where(p['flip']>0,self.rotation>=p['open_angle'],prepare)
        if not hasattr(self,'prepare_t'):self.prepare_t=torch.full_like(t,-1.)
        self.prepare_t=torch.where(prepare&(self.prepare_t<0),t,self.prepare_t)
        expanded=tuck+(stand-tuck)*((t-self.prepare_t)/p['prepare_duration']).clamp(0.,1.)[:,None]
        tucked=torch.where((self.prepare_t>=0)[:,None],expanded,tucked)
        pose=torch.where(self.released[:,None],tucked,pose)
        age=t-self.landing_t
        cushion=stand+.25*(crouch-stand)
        landing=stand+(cushion-stand)*(age/.08).clamp(0.,1.)[:,None]
        recover=cushion+(stand-cushion)*((age-.08)/.32).clamp(0.,1.)[:,None]
        landing=torch.where((age>.08)[:,None],recover,landing)
        return torch.where(self.landed[:,None],landing,pose)


def baseline():
    return dict(name='continuous_baseline',legacy=0,crouch_thigh=.95,crouch_calf=-1.95,
        crouch_start=.5,push_start=.85,hold=0.,thigh_duration=.16,knee_duration=.05,knee_delay=-.075,
        rear_delay=0.,rear_crouch_offset=0.,extend_thigh=.05,extend_calf=-.10,
        tuck_thigh=1.23,tuck_calf=-2.31,tuck_duration=.06,apex_delay=0.,prepare_duration=.08,
        flip=0,open_angle=4.8,unload_rear=0,unload_front=0,unload_time=0.,unload_duration=.06)
