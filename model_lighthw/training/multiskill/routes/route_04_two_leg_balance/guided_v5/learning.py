"""已验证教师示范→原观测Actor→独立自主PPO，所有阶段显式启动。"""
import argparse
from pathlib import Path
import random
import traceback
import numpy as np
import torch
from common_guided import (HERE, contract, sources, now, read, write, sha, require_sources,
                           require_teacher)


def actor_network():
    return torch.nn.Sequential(torch.nn.Linear(305,256),torch.nn.ELU(),torch.nn.Linear(256,256),
                               torch.nn.ELU(),torch.nn.Linear(256,128),torch.nn.ELU(),torch.nn.Linear(128,16))


def finite(value):
    if isinstance(value,torch.Tensor) and not torch.isfinite(value).all(): raise ValueError('Nonfinite learned tensor')
    if isinstance(value,dict):
        for v in value.values(): finite(v)
    if isinstance(value,(list,tuple)):
        for v in value: finite(v)


def teacher_evidence(folder):
    """重新从物理轨迹验收，不仅信任成功标记；两个布局都需完整任务成功。"""
    from verify_trace import verify
    from common import asset_hashes
    folder=Path(folder).resolve()
    result=require_teacher(folder)
    if read(folder/'asset_manifest.json')!=asset_hashes(): raise ValueError('Teacher physical assets changed')
    if result['layout']!='pair' or len(result['cases'])<6: raise ValueError('Need repeated two-slot teacher trials')
    verify(folder)
    if result['cases']!=read(folder/'evaluation.json')['cases']: raise ValueError('Teacher reports differ')
    if not all(c['held'] and c['success'] and c['timeout'] and not c['failed'] for c in result['cases']):
        raise ValueError('Partial/failed teacher episode cannot become a successful demonstration')
    reference=Path(result['reference'])
    require_sources(reference)
    if read(reference/'asset_manifest.json')!=asset_hashes(): raise ValueError('Reference physical assets changed')
    if sha(reference/'result.json')!=result['reference_sha256']: raise ValueError('Teacher reference changed')
    ref=read(reference/'result.json'); verify(reference)
    if ref['cases']!=read(reference/'evaluation.json')['cases']: raise ValueError('Reference reports differ')
    if (ref['layout']!='original' or ref['candidate']!=result['candidate'] or len(ref['cases'])<3
            or not all(c['held'] and c['success'] and c['timeout'] and not c['failed'] for c in ref['cases'])):
        raise ValueError('Original-layout teacher did not pass')
    for name,digest in ref['trace_hashes'].items():
        if sha(reference/name)!=digest: raise ValueError('Original-layout trace changed')
    return result


def archived_learning_sources(folder):
    """只读导入修正数据：允许学习脚本版本变化，21份物理源码及资产必须完全一致。"""
    from common import asset_hashes
    folder=Path(folder); recorded=read(folder/'source_manifest.json')
    learning_key=str((HERE/'learning.py').relative_to(__import__('common_guided').REPO))
    if set(recorded)!=set(sources(True)) or {k:v for k,v in recorded.items() if k!=learning_key}!=sources(False):
        raise ValueError('Correction import changed physical/observation/controller sources')
    if read(folder/'contract.json')!=contract() or read(folder/'asset_manifest.json')!=asset_hashes():
        raise ValueError('Correction import changed contract or physical assets')
    if any(sha(folder/'source_snapshot'/k)!=v for k,v in recorded.items()):
        raise ValueError('Archived learning source snapshot changed')
    return recorded


def correction_evidence(folder):
    """旧Actor只用于重算已记录动作，不迁移权重/优化器，也不把失败标为成功。"""
    from verify_trace import verify
    folder=Path(folder).resolve();recorded=archived_learning_sources(folder)
    result=read(folder/'result.json');request=read(folder/'request.json')
    if result.get('kind')!='autonomous_evaluation' or not request.get('actor'):
        raise ValueError('Correction needs recorded autonomous BC Actor trajectories')
    actor=Path(request['actor']);value=torch.load(actor,map_location='cpu',weights_only=False)
    if (sha(actor)!=result['actor_sha256'] or value.get('sources')!=recorded or value.get('contract')!=contract()
            or value.get('kind')!='bc_actor'):
        raise ValueError('Correction Actor provenance mismatch')
    original=Path(value['dataset']);archived_learning_sources(original)
    if sha(original/'dataset.npz')!=value['dataset_sha256']: raise ValueError('Correction Actor dataset changed')
    teacher_evidence(read(original/'result.json')['teacher'])
    verify(folder)
    if result['cases']!=read(folder/'evaluation.json')['cases']: raise ValueError('Correction reports differ')
    model=actor_network();finite(value['actor_state_dict']);model.load_state_dict(value['actor_state_dict'])
    torch.set_num_threads(4);error=0.
    for name,digest in result['trace_hashes'].items():
        if sha(folder/name)!=digest: raise ValueError('Correction trace changed')
        with np.load(folder/name,allow_pickle=False) as z,torch.no_grad():
            x=torch.from_numpy(z['policy_observation']).float()
            error=max(error,float(np.abs(model(x).numpy()-z['policy_action']).max()))
    if error>3e-6: raise ValueError('Recorded Actor actions do not match checkpoint')
    return dict(folder=str(folder),result_sha256=sha(folder/'result.json'),actor=str(actor),actor_sha256=sha(actor),
                source_manifest_sha256=sha(folder/'source_manifest.json'),action_reconstruction_error=error,
                successful_demonstration=False)


def corrected_trace_arrays(data,candidate,geometry):
    """按控制前一物理帧重算教师标签；使用真值产生标签，但输入保持原305维。"""
    from types import SimpleNamespace
    from teacher import Teacher
    obs=data['policy_observation'];count=obs.shape[1]
    teacher=Teacher([candidate]*count,geometry,'cpu')
    origin=torch.zeros(count,2);seen=torch.zeros(count,dtype=torch.bool)
    labels=[];active=[];times=[]
    for step in range(len(obs)):
        # 第j个动作在t=0.02j施加；记录帧从0.001s开始，不能使用动作执行后的第20j帧。
        index=max(0,20*step-1)
        metrics={k:torch.from_numpy(data[k][index].copy()) for k in ('gravity','ang_vel','wheel_pos','com','com_vel','force_z')}
        t=torch.zeros(count) if step==0 else torch.from_numpy(data['task_t'][index].copy())
        entered_hold=(t>=4.)&~seen;origin[entered_hold]=metrics['com'][entered_hold,:2];seen|=entered_hold
        base=SimpleNamespace(metrics=lambda:metrics,state=SimpleNamespace(hold_origin=origin))
        with torch.no_grad():labels.append(teacher(base,t).numpy().copy())
        # 终止由前一控制步结束时确认；前一物理帧active尚为真，下一控制帧已自动reset。
        # 当前控制步第一物理帧保存的是动作前相同的active，避免把reset后动作收入纠偏数据。
        active.append(data['active'][20*step].copy());times.append(t.numpy().copy())
    use=np.stack(active);times=np.stack(times)
    return dict(observations=obs[use],actions=np.stack(labels)[use],phase=obs[use,-6:-1].argmax(-1),
                episode=np.broadcast_to(np.arange(count),use.shape)[use],time=times[use],
                is_correction=np.ones(int(use.sum()),dtype=np.int64))


def dataset_check(folder,archive=False):
    folder=Path(folder).resolve()
    if archive: archived_learning_sources(folder)
    else: require_sources(folder,learning=True)
    meta=read(folder/'result.json')
    if meta.get('kind')!='demonstrations': raise ValueError('Not a guided demonstration dataset')
    if sha(folder/'dataset.npz')!=meta['dataset_sha256']: raise ValueError('Dataset changed')
    if sha(Path(meta['teacher'])/'result.json')!=meta['teacher_result_sha256']: raise ValueError('Teacher evidence changed')
    teacher_evidence(meta['teacher'])
    for evidence in meta.get('corrections',[]):
        actual=correction_evidence(evidence['folder'])
        if any(actual[k]!=v for k,v in evidence.items() if k!='action_reconstruction_error'):
            raise ValueError('Correction evidence changed')
    with np.load(folder/'dataset.npz',allow_pickle=False) as z:
        data={k:z[k] for k in z.files}
    validate_arrays(data)
    return data,meta


def validate_arrays(data):
    """输入只能是原305维观测；阶段/时钟标签只用于采样权重，不拼入Actor。"""
    x,y=data['observations'],data['actions']
    if x.ndim!=2 or x.shape[1]!=305 or y.shape!=(len(x),16): raise ValueError('Actor data must be N x 305 -> N x 16')
    if not len(x) or not all(np.isfinite(v).all() for v in data.values()): raise ValueError('Empty/nonfinite dataset')
    if np.abs(y).max()>1.000001: raise ValueError('Demonstration action exceeds original action envelope')
    if not np.array_equal(data['phase'],x[:,-6:-1].argmax(-1)): raise ValueError('Phase label differs from actor observation')
    if not set(range(5)).issubset(set(data['phase'].tolist())): raise ValueError('Full protocol phases missing')


def load_actor(path):
    path=Path(path).resolve(); value=torch.load(path,map_location='cpu',weights_only=False)
    if value.get('kind')!='bc_actor' or value.get('contract')!=contract() or value.get('sources')!=sources(True):
        raise ValueError('Actor version/source mismatch; no legacy implicit transfer')
    finite(value['actor_state_dict']);actor_network().load_state_dict(value['actor_state_dict'],strict=True)
    dataset=Path(value['dataset'])
    if sha(dataset/'dataset.npz')!=value['dataset_sha256']: raise ValueError('Actor dataset provenance mismatch')
    dataset_check(dataset)
    return value


def load_checkpoint(path):
    value=torch.load(path,map_location='cpu',weights_only=False); info=value.get('infos',{})
    if info.get('contract')!=contract() or info.get('source_manifest')!=sources(True):
        raise ValueError('Checkpoint is not this exact guided workflow')
    from common import asset_hashes
    if info.get('asset_manifest')!=asset_hashes(): raise ValueError('Checkpoint physical assets changed')
    finite(value['model_state_dict']);finite(value['optimizer_state_dict'])
    lineage=info.get('initialization',{})
    if not lineage.get('actor') or not lineage.get('validation'): raise ValueError('Missing verified Actor lineage')
    if sha(lineage['actor'])!=lineage['actor_sha256']: raise ValueError('Initialization Actor changed')
    if sha(Path(lineage['validation'])/'result.json')!=lineage['validation_sha256']:
        raise ValueError('Initialization validation changed')
    load_actor(lineage['actor']);require_autonomous(lineage['validation'],lineage['actor'])
    return value


def require_autonomous(validation,actor_path):
    from verify_trace import verify
    validation=Path(validation).resolve();require_sources(validation,learning=True)
    result=read(validation/'result.json')
    if result.get('kind')!='autonomous_evaluation' or result.get('actor_sha256')!=sha(actor_path):
        raise ValueError('Autonomous validation is not for this initialization Actor')
    verify(validation)
    if result['cases']!=read(validation/'evaluation.json')['cases']: raise ValueError('Evaluation reports differ')
    if len(result['cases'])<6 or not all(c['held'] and c['timeout'] and not c['failed'] for c in result['cases']):
        raise ValueError('Actor has not independently repeated strict 2 s hold; PPO remains locked')
    for name,digest in result['trace_hashes'].items():
        if sha(validation/name)!=digest: raise ValueError('Autonomous trace changed')
    return result


def preflight(a):
    if a.mode=='collect':
        if a.teacher is None: raise ValueError('collect requires --teacher with repeated full-protocol evidence')
        teacher_evidence(a.teacher)
        if a.validation: correction_evidence(a.validation)
        if a.dataset:
            _,meta=dataset_check(a.dataset,archive=True)
            if Path(meta['teacher']).resolve()!=a.teacher.resolve(): raise ValueError('Parent dataset uses another teacher')
    elif a.mode=='pretrain':
        if a.dataset is None: raise ValueError('pretrain requires --dataset')
        dataset_check(a.dataset)
    elif a.mode in ('evaluate','train'):
        if (a.actor is None)==(a.checkpoint is None): raise ValueError('Provide exactly one --actor or --checkpoint')
        if a.actor:
            load_actor(a.actor)
            if a.mode=='train':
                if a.validation is None: raise ValueError('train requires explicit --validation from autonomous actor evaluation')
                require_autonomous(a.validation,a.actor)
        else:
            saved=load_checkpoint(a.checkpoint)
            if a.mode=='train' and saved['infos'].get('completed_updates',0)<1:
                raise ValueError('Use --actor plus verified autonomous evaluation for fresh guided training')
            if a.mode=='train' and saved['infos']['completed_updates']+a.updates>contract()['formal_ppo_budget']:
                raise ValueError('Initial guided PPO budget is 200 total attempts; review before expanding')


def collect(run,request):
    teacher=Path(request['teacher']); result=teacher_evidence(teacher)
    features=[]; labels=[]; phases=[]; episodes=[]; step_times=[]
    for episode,case in enumerate(result['cases']):
        with np.load(teacher/case['trace'],allow_pickle=False) as z:
            i=case['trace_env'];use=z['policy_active'][:,i]
            obs=z['policy_observation'][use,i];action=z['policy_action'][use,i]
        features.append(obs);labels.append(action);phases.append(obs[:,-6:-1].argmax(-1))
        episodes.append(np.full(len(obs),episode,dtype=np.int64));step_times.append(np.arange(len(obs),dtype=np.float32)*.02)
    data=dict(observations=np.concatenate(features),actions=np.concatenate(labels),phase=np.concatenate(phases),
              episode=np.concatenate(episodes),time=np.concatenate(step_times),
              is_correction=np.zeros(sum(map(len,phases)),dtype=np.int64))
    corrections=[]
    if request.get('dataset'):
        data,meta=dataset_check(request['dataset'],archive=True);corrections=meta.get('corrections',[]).copy()
        data.setdefault('is_correction',np.zeros(len(data['phase']),dtype=np.int64))
    if request.get('validation'):
        evidence=correction_evidence(request['validation'])
        if evidence in corrections: raise ValueError('Correction trajectories already included')
        folder=Path(request['validation']);geometry=read(folder/'geometry.json')
        for path in sorted(folder.glob('trace_seed*.npz')):
            with np.load(path,allow_pickle=False) as z:
                new=corrected_trace_arrays({k:z[k] for k in z.files},result['candidate'],geometry)
            new['episode']+=int(data['episode'].max())+1
            data={k:np.concatenate((v,new[k])) for k,v in data.items()}
        corrections.append(evidence)
    validate_arrays(data)
    np.savez_compressed(run/'dataset.npz',**data)
    write(run/'result.json',dict(kind='demonstrations',at=now(),teacher=str(teacher),
        teacher_result_sha256=sha(teacher/'result.json'),dataset_sha256=sha(run/'dataset.npz'),
        frames=len(data['phase']),episodes=int(data['episode'].max())+1,observation_dim=305,action_dim=16,
        successful_teacher_frames=int((data['is_correction']==0).sum()),correction_frames=int(data['is_correction'].sum()),
        corrections=corrections,parent_dataset=request.get('dataset'),
        parent_dataset_sha256=sha(Path(request['dataset'])/'dataset.npz') if request.get('dataset') else None,
        phase_counts=np.bincount(data['phase'],minlength=5).tolist(),ppo_updates=0,
        note='Successful teacher trajectories plus explicitly marked counterfactual labels on Actor states; corrections are not success evidence'))


def imitation_weights(phase,times):
    counts=torch.bincount(phase,minlength=5).clamp_min(1)
    weight=1./counts[phase]
    # 起身/保持优先；阶段切换±0.12s另加权，输入维度不变。
    weight*=torch.where((phase==1)|(phase==2),2.,1.)
    switch=torch.stack([(times-t).abs()<=.12 for t in (1.,4.,7.,10.)]).any(0)
    weight*=torch.where(switch,2.,1.)
    return weight/weight.mean()


def fit_actor(data,epochs,seed):
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    torch.set_num_threads(4)
    model=actor_network()
    x=torch.from_numpy(data['observations']).float();y=torch.from_numpy(data['actions']).float()
    phase=torch.from_numpy(data['phase']).long();times=torch.from_numpy(data['time']).float()
    weights=imitation_weights(phase,times)
    if 'is_correction' in data:
        weights*=torch.where(torch.from_numpy(data['is_correction']).bool(),2.,1.)
        weights/=weights.mean()
    optimizer=torch.optim.Adam(model.parameters(),lr=3e-4)
    history=[]
    # 留出的固定重复轨迹不能称为独立验证集；这里只报告监督拟合，行为另行GPU验证。
    for epoch in range(epochs):
        order=torch.randperm(len(x))
        for batch in order.split(512):
            prediction=model(x[batch]);loss=((prediction-y[batch]).square().mean(-1)*weights[batch]).mean()
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        if (epoch+1)%10==0 or epoch==epochs-1:
            with torch.no_grad():
                error=(model(x)-y).square().mean(-1)
                history.append(dict(epoch=epoch+1,weighted_mse=float((error*weights).mean()),
                    phase_mse=[float(error[phase==i].mean()) for i in range(5)]))
    finite(model.state_dict())
    return model,history


def pretrain(run,request):
    data,meta=dataset_check(request['dataset'])
    model,history=fit_actor(data,request['epochs'],request['seed'])
    path=run/'actor.pt'
    torch.save(dict(kind='bc_actor',contract=contract(),sources=sources(True),actor_state_dict=model.state_dict(),
        dataset=str(Path(request['dataset']).resolve()),dataset_sha256=meta['dataset_sha256'],
        teacher_result_sha256=meta['teacher_result_sha256'],seed=request['seed'],epochs=request['epochs'],
        critic_transferred=False,optimizer_transferred=False,behavior_accepted=False),path)
    write(run/'result.json',dict(kind='actor_imitation',actor=str(path),actor_sha256=sha(path),history=history,
        ppo_updates=0,autonomous_behavior_tested=False,behavior_accepted=False))


def gpu(run,request):
    from physics_worker import setup
    from common import asset_hashes
    from two_leg_balance_split_v4.spec import runner_config
    app=env=None
    try:
        app,env=setup(run,request['num_envs'] if request['mode']=='train' else 2,request['seed'])
        from worker import make_runner,evaluate
        from rsl_rl.runners import OnPolicyRunner
        from verify_trace import verify
        base=make_runner(run,contract())
        initialization=(dict(actor=request['actor'],actor_sha256=sha(request['actor']),
            validation=request.get('validation'),
            validation_sha256=sha(Path(request['validation'])/'result.json') if request.get('validation') else None)
            if request.get('actor') else load_checkpoint(request['checkpoint'])['infos']['initialization'])
        class GuidedRunner(base):
            def save(self,path,infos=None):
                path=Path(path)
                if path.parent==Path(self.log_dir) and path.name.startswith('model_'):
                    if self.completed_updates%self.checkpoint_every: return
                    path=path.with_name(f'model_{self.completed_updates:06d}.pt')
                info=dict(contract=contract(),source_manifest=sources(True),asset_manifest=asset_hashes(),
                    completed_updates=self.completed_updates,learning_rate=self.alg.learning_rate,
                    torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
                    python_rng=random.getstate(),numpy_rng=np.random.get_state(),physics_state_saved=False,
                    initialization=initialization,
                    resumed_from=request.get('checkpoint'))
                OnPolicyRunner.save(self,str(path),info)
        cfg=runner_config(request['seed']);cfg['save_interval']=request['save_every']
        runner=GuidedRunner(env,cfg,log_dir=str(run/'training'),device=env.device)
        runner.logger_type='tensorboard'
        b=env.unwrapped
        def sink(rows):
            import json
            with (run/'episodes.jsonl').open('a') as f:
                for row in rows: f.write(json.dumps(row,allow_nan=False)+'\n')
        b.episode_sink=sink
        if request.get('actor'):
            saved=load_actor(request['actor']);runner.alg.policy.actor.load_state_dict(saved['actor_state_dict'])
            with torch.no_grad():
                runner.alg.policy.log_std[:12].fill_(math_log(.05));runner.alg.policy.log_std[12:].fill_(math_log(.03))
            assert not runner.alg.optimizer.state
            for k,v in runner.alg.policy.actor.state_dict().items():
                torch.testing.assert_close(v.cpu(),saved['actor_state_dict'][k],atol=0,rtol=0)
        else:
            saved=load_checkpoint(request['checkpoint']);info=runner.load(request['checkpoint'],load_optimizer=request['mode']=='train')
            runner.completed_updates=runner.current_learning_iteration=info['completed_updates']
            runner.alg.learning_rate=info['learning_rate']
            for name,value in saved['model_state_dict'].items():
                torch.testing.assert_close(runner.alg.policy.state_dict()[name].cpu(),value.cpu(),atol=0,rtol=0)
            for group in runner.alg.optimizer.param_groups: group['lr']=info['learning_rate']
            if request['mode']=='train':
                if runner.alg.optimizer.state_dict()['param_groups']!=saved['optimizer_state_dict']['param_groups']:
                    raise ValueError('Restored Adam parameter groups differ')
                for key,state in saved['optimizer_state_dict']['state'].items():
                    for name,value in state.items():
                        if isinstance(value,torch.Tensor):
                            torch.testing.assert_close(runner.alg.optimizer.state_dict()['state'][key][name].cpu(),value.cpu(),atol=0,rtol=0)
                torch.set_rng_state(info['torch_rng'].cpu());torch.cuda.set_rng_state_all(info['cuda_rng'])
                random.setstate(info['python_rng']);np.random.set_state(info['numpy_rng'])
                write(run/'resume_check.json',dict(model_exact=True,adam_exact=True,parameter_groups_exact=True,
                    rng_restored=True,completed_updates=runner.completed_updates,physics_state_saved=False))
        if request['mode']=='train':
            runner.save(run/'initial.pt')
            runner.learn(num_learning_iterations=request['updates'],init_at_random_ep_len=False)
            runner.save(run/'final.pt')
            write(run/'result.json',dict(kind='autonomous_ppo',new_updates=request['updates'],completed_updates=runner.completed_updates,
                checkpoint=str(run/'final.pt'),checkpoint_sha256=sha(run/'final.pt'),frozen_evaluations=0,
                review_status='awaiting_user_request',behavior_accepted=False))
        else:
            result=evaluate(env,runner,run,dict(eval_seeds=request['seeds'],headless=True),app)
            result.update(kind='autonomous_evaluation',verification=verify(run),
                actor_sha256=sha(request['actor']) if request.get('actor') else None,
                checkpoint_sha256=sha(request['checkpoint']) if request.get('checkpoint') else None,
                trace_hashes={p.name:sha(p) for p in run.glob('trace_seed*.npz')})
            write(run/'result.json',result)
        require_sources(run,learning=True)
        if runner.writer is not None: runner.writer.close()
    except BaseException as exc:
        # Isaac关闭可能掩盖Python退出状态，因此必须在关闭App前保存异常。
        write(run/'worker_error.json',dict(error=str(exc),traceback=traceback.format_exc()))
        raise
    finally:
        try:
            if env is not None:env.close()
        except BaseException as exc:
            write(run/'worker_error.json',dict(error=str(exc),traceback=traceback.format_exc()))
            raise
        finally:
            if app is not None:app.close()


def math_log(x):
    import math
    return math.log(x)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);run=p.parse_args().run.resolve()
    request=read(run/'request.json')
    try:
        require_sources(run,learning=True)
        if request['mode']=='collect':collect(run,request)
        elif request['mode']=='pretrain':pretrain(run,request)
        else:gpu(run,request)
        require_sources(run,learning=True)
    except BaseException as exc:
        write(run/'worker_error.json',dict(error=str(exc),traceback=traceback.format_exc()));raise


if __name__=='__main__':main()
