"""候选原始轨迹的CPU奖励复算；不启动Isaac，不执行回收包中的Python源码。

passed仅表示来源/原协议/数值复算成立，不表示新奖励已产生更好机器人行为。
"""
from pathlib import Path
from types import SimpleNamespace as NS
import ast, importlib, json, math
import numpy as np
import torch
from .spec import CONTRACT as C, ROOT, LEG_JOINTS
from .io import read, write, sha


def _close(actual, saved):
    if isinstance(actual, dict):
        for key, value in actual.items():
            if key in saved:_close(value, saved[key])
    elif isinstance(actual, list):
        if len(actual)!=len(saved):raise ValueError('Report list length mismatch')
        for v,w in zip(actual,saved):_close(v,w)
    elif isinstance(actual, float):
        if not math.isclose(actual,saved,rel_tol=2e-5,abs_tol=2e-5):raise ValueError(f'Report mismatch {actual} != {saved}')
    elif actual!=saved:raise ValueError(f'Report mismatch {actual} != {saved}')


def predict(observation,state):
    """纯CPU重放原三头Actor，避免把模型散列关联误当执行行为一致。"""
    obs=torch.as_tensor(observation)
    def branch(name):
        x=obs
        for i in (0,2,4,6):
            x=torch.nn.functional.linear(x,state[f'{name}.{i}.weight'],state[f'{name}.{i}.bias'])
            if i!=6:x=torch.nn.functional.elu(x)
        return x
    with torch.inference_mode():
        raw=torch.where((obs[:,-8]<.5)[:,None],branch('ground_actor'),branch('actor'))
        raw=torch.where((obs[:,-7]>.5)[:,None],branch('landing_actor'),raw)
        return torch.tanh(raw.clamp(-3,3)).numpy()


def replay(a,meta,version):
    """用实际执行的1kHz机械量重放状态机/50Hz奖励，旧在线三项代价另核对。
    仅从已安装、本地受版本散列保护的env.py提取奖励方法，避免导入仿真环境。
    """
    if version not in ('v4','v5'):raise ValueError('Unsupported reward version')
    package='r5a_guided_'+version
    cfg=importlib.import_module(package+'.spec').CONTRACT
    protocol=importlib.import_module(package+'.protocol')
    shaping=importlib.import_module(package+'.shaping')
    landing=importlib.import_module(package+'.landing')
    recovery=importlib.import_module(package+'.recovery')
    dynamics=importlib.import_module(package+'.dynamics')
    source=ROOT/package/'env.py'
    node=next(n for c in ast.parse(source.read_text()).body if isinstance(c,ast.ClassDef) and c.name=='AutoFlipEnv' for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='_get_rewards')
    ns=dict(torch=torch,math=math,C=cfg,landing=landing,recovery_terms=recovery.recovery_terms)
    ns.update({k:getattr(shaping,k) for k in ('flight_shaping','landing_progress','flight_quality','reward_potentials','preparation_potential')})
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),ns)
    _,_,h,inertias,_,_=dynamics.centroidal(a['body_com_position'],a['body_com_velocity'],a['body_link_quaternion'],a['body_angular_velocity'],meta['masses'],meta['inertias_link_axes'])
    axis=dynamics.rotation(a['root_state'][:1,3:7])[0,:,1]
    H=-h@axis;omega=-a['root_state'][:,10:13]@axis;I=np.einsum('i,tij,j->t',axis,inertias,axis)
    tt=lambda v:torch.as_tensor(np.asarray(v),dtype=torch.float32).unsqueeze(0)
    s=protocol.FlipState(1);s.reset(torch.tensor([0]),tt(a['com'][0,2]),tt(a['com'][0,:2]),tt(a['root_state'][0,3:7]))
    leg=[meta['joint_names'].index(n) for n in LEG_JOINTS]
    wheel=[i for i,n in enumerate(meta['joint_names']) if 'wheel' in n]
    e=NS(state=s,step_dt=.02,physics_dt=.001,mass=tt(sum(meta['masses'])),leg_ids=leg,wheel_ids=wheel,max_episode_length=300,
         episode_rewards={k:torch.zeros(1) for k in cfg['reward_weights']},extras={'log':{}})
    flags={k:0 for k in ('launched','landed','failed','success')};cost_error={k:0. for k in ('prelanding_speed_cost','touchdown_spin_cost','landing_speed_cost')}
    angle_error=0.;history=[];seen=torch.zeros(1,dtype=torch.bool)
    for step in range((len(a['t'])-1)//20):
        begin=step*20;end=begin+20;s.landing_event.zero_();s.valid_air_event.zero_()
        e.old_potential=shaping.reward_potentials(s,e.mass).clone();e.old_preparation=shaping.preparation_potential(s).clone()
        e.old_extension=s.extension_peak.clone();e.old_launch_score=s.launch_score_peak.clone();e.old_air_rotation=s.air_rotation_peak.clone();e.old_compact=s.compact_potential.clone();e.old_landing_costs=landing.costs(s).clone()
        e.previous_actions=tt(a['actions'][begin]);e.actions=tt(a['actions'][begin+1]);e.phase_switch=s.launched&~seen;seen=s.launched.clone()
        for i in range(begin+1,end+1):
            m=dict(com=tt(a['com'][i]),com_vel=tt(a['com_velocity'][i]),force=tt(a['contact_force'][i]),force_z=tt(a['wheel_force_z'][i]),clearance=tt(a['wheel_clearance'][i]),tilt=tt(a['tilt'][i]),base_z=tt(a['base_height'][i]),body_force=tt(a['body_force'][i]),torque_peak=tt(abs(a['applied_torque'][i]).max()),H_backward=tt(H[i]),omega_backward=tt(omega[i]),locked_inertia=tt(I[i]),leg_velocity=tt(a['joint_velocity'][i,leg]),root_angular_velocity=tt(a['root_state'][i,10:13]))
            # Match production int64 tick -> float32 multiply, including gate edges.
            s.update(.001,torch.tensor([i],dtype=torch.long)*cfg['physics_dt'],m,tt(a['root_state'][i,3:7]))
            for k in flags:flags[k]+=int(bool(getattr(s,k)[0])!=bool(a['online_'+k][i]))
            for k in cost_error:
                if 'online_'+k not in a:raise ValueError('Missing online reward cost '+k)
                cost_error[k]=max(cost_error[k],abs(float(getattr(s,k)[0])-float(a['online_'+k][i])))
            angle_error=max(angle_error,abs(float(s.angle[0])-float(a['online_angle'][i])))
        e.current_metrics=m;e.episode_length_buf=torch.tensor([step+1]);e.episode_physics_steps=torch.tensor([end])
        e.robot=NS(data=NS(applied_torque=tt(a['applied_torque'][end]),joint_vel=tt(a['joint_velocity'][end]),root_ang_vel_b=tt(a['root_state'][end,10:13])))
        r=ns['_get_rewards'](e);history.append(dict(t=end*.001,total=float(r[0]),**{k:float(v[0]) for k,v in e.reward_terms.items()}))
    if any(flags.values()) or angle_error>2e-4 or max(cost_error.values())>2e-5:raise ValueError(f'Online replay mismatch {flags}, {angle_error}, {cost_error}')
    components={k:float(v[0]) for k,v in e.episode_rewards.items()}
    result=dict(total=sum(components.values()),discounted_total=sum(row['total']*cfg['gamma']**i for i,row in enumerate(history)),components=components,flag_mismatches=flags,angle_error=angle_error,online_landing_cost_error=cost_error,steps=len(history),physical_steps=len(a['t'])-1)
    if not all(math.isfinite(v) for v in components.values()):raise ValueError('Nonfinite reward')
    return result,history


def formula_checks():
    """合成信号只检验公式性质，不作为新物理轨迹/成功行为。"""
    from .shaping import impact_potential
    from .landing import anchor_cost_rates
    x=torch.tensor([0.,6.,26.,77.81148,104.944,246.,300.,600.])
    y=impact_potential(x)
    assert y[0]==y[1]==0 and (torch.diff(y[1:])>0).all()
    assert abs(float(torch.diff(y).sum()-y[-1]))<1e-5
    s=NS(qualified_landing=torch.ones(4,dtype=torch.bool),failed=torch.zeros(4,dtype=torch.bool),landing_t=torch.ones(4),landing_xy=torch.zeros(4,2))
    m=dict(com=torch.tensor([[0.,0.,0.],[.04,0.,0.],[.10,0.,0.],[.15,0.,0.]]),com_vel=torch.zeros(4,3))
    p,v=anchor_cost_rates(s,m,torch.ones(4)*1.3)
    assert (torch.diff(p)>0).all() and (p<1).all() and (v<1).all()
    for time in (.9,1.,1.249):assert all(torch.count_nonzero(k)==0 for k in anchor_cost_rates(s,m,torch.ones(4)*time))
    returns=-2*m['com'][:,:2];returns*= (.2/returns.norm(dim=-1).clamp_min(1e-8)).clamp_max(1)[:,None]
    m['com_vel'][:,:2]=returns;_,towards=anchor_cost_rates(s,m,torch.ones(4)*1.3)
    m['com_vel'][:,:2]=-returns;_,away=anchor_cost_rates(s,m,torch.ones(4)*1.3)
    assert (towards[1:]<v[1:]).all() and (v[1:]<away[1:]).all()
    rates=anchor_cost_rates(s,m,torch.ones(4)*1.3)
    integrated=[sum(float(rates[0][2])*dt/6 for _ in range(round(1/dt))) for dt in (.001,.002,.02)]
    assert max(integrated)-min(integrated)<1e-10
    return dict(passed=True,scope='formula counterfactuals only, not physical replay',impact_BW=x.tolist(),weighted_impact=(C['reward_weights']['impact_rate']*y).tolist(),anchor_monotone=True,return_better_than_stopping_or_outward=True,grace_period_zero=True,dt_integral=integrated,high_impact_no_saturation=True)


def audit_session_rewards(session_root,output_dir):
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    root=Path(session_root).resolve()
    result=dict(passed=False,source_session=str(root),inputs={},scope='CPU replay of existing v4 physical trajectories under old and proposed rewards; not new behavior',cases=[],errors=[])
    try:
        torch.set_num_threads(1)
        from .contact_audit import audit_session
        from r5a_guided_v4.spec import CONTRACT as OLD
        from r5a_guided_v4.record import judge
        from r5a_guided_v4.protocol import targets
        from r5a_guided_v4.io import implementation as old_implementation
        contact=audit_session(session_root)
        if not contact['comparison_complete']:raise ValueError('Complete baseline and final contact evidence required')
        result['inputs']=dict(contact['inputs'])
        original_implementation=old_implementation()
        needed=[x for x in contact['cases'] if x['role'] in ('baseline','final')]
        if len({x['model_sha256'] for x in needed})<2:raise ValueError('Final candidate must be a genuinely trained checkpoint, not a second baseline')
        for case in needed:
            paths={k:Path(v['path']) for k,v in case['files'].items()}
            a=dict(np.load(paths['trace'],allow_pickle=False));meta=read(paths['metadata'])
            if meta['contract']!=OLD:raise ValueError('Trajectory contract differs from installed v4')
            judged=judge(a,meta);_close(judged,read(paths['report']))
            if not (judged['success'] and judged['online_agrees'] and judged['physics_consistent'] and len(a['t'])==6001):raise ValueError('Candidate failed original full protocol/physics check')
            with torch.serialization.safe_globals([np.core.multiarray._reconstruct,np.ndarray,np.dtype,type(np.dtype('uint32'))]):
                ck=torch.load(paths['checkpoint'],map_location='cpu',weights_only=True)
            if ck['infos']['contract']!=OLD or ck['infos']['completed_updates']!=case['completed_updates']:raise ValueError('Checkpoint contract/update mismatch')
            if ck['infos'].get('implementation_sha256')!=original_implementation:raise ValueError('V4 checkpoint implementation differs from local replay code')
            if case['role']=='final' and ck['infos']['completed_updates']<=0:raise ValueError('Final candidate has no completed updates')
            state=ck['model_state_dict']
            if not all(torch.isfinite(v).all() for v in state.values()):raise ValueError('Nonfinite model')
            policy=paths['trace'].with_name(paths['trace'].name.replace('_trace.npz','_policy.npz'))
            if not policy.resolve().is_relative_to(root):raise ValueError('Policy trace escapes session')
            result['inputs'][str(policy.relative_to(root))]=sha(policy)
            pol=dict(np.load(policy,allow_pickle=False))
            if not all(np.isfinite(v).all() for v in pol.values()):raise ValueError('Nonfinite policy trace')
            if pol['observation'].shape!=(300,420) or pol['raw_action'].shape!=(300,16):raise ValueError('Incomplete 50 Hz policy data')
            err=float(abs(predict(pol['observation'],state)-pol['raw_action']).max())
            if err>5e-6:raise ValueError('Model outputs differ from recorded executed candidate')
            executed=pol['raw_action'].copy();executed[:20]=0
            if not np.array_equal(executed,a['actions'][1::20]):raise ValueError('Executed actions differ from policy')
            if abs(targets(torch.tensor(executed)).numpy()-a['leg_targets'][1::20]).max()>2e-6:raise ValueError('Targets differ from policy')
            versions={}
            for version in ('v4','v5'):
                r,h=replay(a,meta,version);versions[version]=r
                write(out/(case['label']+'_'+version+'_rewards.json'),dict(summary=r,history=h))
            if abs(versions['v4']['components']['terminal_success']-versions['v5']['components']['terminal_success'])>1e-6:raise ValueError('Success reward changed')
            result['cases'].append(dict(role=case['role'],label=case['label'],checkpoint_sha256=sha(paths['checkpoint']),policy_sha256=sha(policy),actor_forward_max_error=err,original_judgment=judged,rewards=versions))
        result['formula_checks']=formula_checks()
        result['passed']=True
        result['interpretation']='Evidence and scoring verified. Candidate score rankings describe recorded actions; they do not prove v5 training improvement.'
    except Exception as error:
        result['errors'].append(f'{type(error).__name__}: {error}')
    write(out/'reward_audit.json',result)
    lines=['# V5 recorded reward audit','',f"Passed: {result['passed']}",'',result['scope'],'',*result['errors']]
    for r in result['cases']:
        lines.append(f"{r['role']} {r['label']}: v4 total={r['rewards']['v4']['total']:.5f}, v5 total={r['rewards']['v5']['total']:.5f}; physical actions unchanged.")
    (out/'REWARD_AUDIT.md').write_text('\n'.join(lines)+'\n')
    return result
