"""CPU重放封存轨迹的真实状态机/奖励；不运行仿真，不修改模型。

bash r6.sh python -m r5a_auto_v5.audit_rewards --trace PATH --output DIR
用于相同轨迹比较v4/v5评分与首次奖励时刻，不能据此声称策略已经改善。
"""
import argparse,ast,importlib,json,math
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import torch
from .dynamics import centroidal,rotation
from .io import sha,write

def replay(trace,version):
 a=dict(np.load(trace,allow_pickle=False));meta=json.loads(trace.with_suffix('.json').read_text())
 spec=importlib.import_module(version+'.spec');protocol=importlib.import_module(version+'.protocol')
 shaping=importlib.import_module(version+'.shaping');recovery=importlib.import_module(version+'.recovery')
 source=Path(spec.__file__).with_name('env.py')
 node=next(n for c in ast.parse(source.read_text()).body if isinstance(c,ast.ClassDef) and c.name=='AutoFlipEnv' for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='_get_rewards')
 ns=dict(torch=torch,math=math,C=spec.CONTRACT,recovery_terms=recovery.recovery_terms)
 ns.update({k:getattr(shaping,k) for k in ('flight_shaping','landing_progress','flight_quality','reward_potentials','preparation_potential')})
 exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),ns)
 _,_,h,locked,_,_=centroidal(a['body_com_position'],a['body_com_velocity'],a['body_link_quaternion'],a['body_angular_velocity'],meta['masses'],meta['inertias_link_axes'])
 axis=rotation(a['root_state'][0,3:7])[:,1];iy=np.einsum('i,tij,j->t',axis,locked,axis)
 H=-h@axis;omega=-a['root_state'][:,10:13]@axis
 tt=lambda x:torch.tensor(np.asarray(x),dtype=torch.float32).unsqueeze(0)
 s=protocol.FlipState(1);s.reset(torch.tensor([0]),tt(a['root_state'][0,2]-.085652),tt(a['root_state'][0,:2]),tt(a['root_state'][0,3:7]))
 e=NS(state=s,step_dt=.02,physics_dt=.001,mass=torch.tensor([sum(meta['masses'])]),
  leg_ids=[meta['joint_names'].index(n) for n in spec.LEG_JOINTS],wheel_ids=[meta['joint_names'].index(n) for n in spec.WHEELS],
  max_episode_length=300,episode_rewards={k:torch.zeros(1) for k in spec.CONTRACT['reward_weights']},extras={'log':{}})
 mismatches={k:0 for k in ('launched','landed','failed','success')};history=[];angle_error=0.;air_seen=torch.zeros(1,dtype=torch.bool)
 for step in range((len(a['t'])-1)//20):
  begin=step*20;end=begin+20;s.landing_event.zero_()
  if hasattr(s,'valid_air_event'):s.valid_air_event.zero_()
  e.old_potential=shaping.reward_potentials(s,e.mass).clone();e.old_preparation=shaping.preparation_potential(s).clone()
  if hasattr(s,'extension_peak'):
   e.old_extension=s.extension_peak.clone();e.old_launch_score=s.launch_score_peak.clone();e.old_air_rotation=s.air_rotation_peak.clone()
  e.previous_actions=tt(a['actions'][begin]);e.actions=tt(a['actions'][begin+1]);e.peak_support=torch.zeros(1)
  e.phase_switch=s.launched&~air_seen;air_seen=s.launched.clone()
  for i in range(begin+1,end+1):
   m=dict(com=tt(a['com'][i]),com_vel=tt(a['com_velocity'][i]),force=tt(a['contact_force'][i]),force_z=tt(a['wheel_force_z'][i]),
    clearance=tt(a['wheel_clearance'][i]),tilt=tt(a['tilt'][i]),base_z=tt(a['base_height'][i]),body_force=tt(a['body_force'][i]),
    torque_peak=tt(np.abs(a['applied_torque'][i]).max()),H_backward=tt(H[i]),omega_backward=tt(omega[i]),locked_inertia=tt(iy[i]))
   s.update(.001,torch.tensor([i],dtype=torch.float32)*.001,m,tt(a['root_state'][i,3:7]))
   for k in mismatches:mismatches[k]+=int(bool(getattr(s,k)[0])!=bool(a['online_'+k][i]))
   angle_error=max(angle_error,abs(float(s.angle[0])-float(a['online_angle'][i])))
  e.current_metrics=m;e.episode_length_buf=torch.tensor([step+1]);e.episode_physics_steps=torch.tensor([end])
  # 范数不依赖世界/机身坐标，因此这里只重放角速度范数罚项。
  e.robot=NS(data=NS(applied_torque=tt(a['applied_torque'][end]),joint_vel=tt(a['joint_velocity'][end]),root_ang_vel_b=tt(a['root_state'][end,10:13])))
  reward=ns['_get_rewards'](e)
  history.append(dict(t=end*.001,total=float(reward[0]),terms={k:float(v[0]) for k,v in e.reward_terms.items()}))
 assert not any(mismatches.values()) and angle_error<2e-4,(mismatches,angle_error)
 return dict(version=version,trace_sha256=sha(trace),physics_samples=len(a['t']),flag_mismatches=mismatches,angle_max_error=angle_error,
  total=sum(float(v[0]) for v in e.episode_rewards.values()),components={k:float(v[0]) for k,v in e.episode_rewards.items()},
  first_positive_s={k:next((r['t'] for r in history if r['terms'][k]>1e-8),None) for k in e.episode_rewards},state=s.report(0),history=history)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--trace',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--compare-v4',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 for v in (['r5a_auto_v4','r5a_auto_v5'] if a.compare_v4 else ['r5a_auto_v5']):
  result=replay(a.trace,v);write(a.output/(v+'.json'),result)
  print(json.dumps({k:val for k,val in result.items() if k not in ('history','state')},ensure_ascii=False))
if __name__=='__main__':main()
