"""固定高目标训练：候选冻结比较、Critic预热、分块PPO、质量保护和单目录回收。"""
import argparse
import json
from pathlib import Path
import shutil
import sys
import traceback
sys.path.insert(0,str(Path(__file__).resolve().parents[7]/"offline"))
from station import ROOT,read,write,stamp,now,digest,session_lock,capture_environment,child,export_session,checkpoint_from_run

SCRIPTS=ROOT/'model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/highjump/scripts'
sys.path.insert(0,str(SCRIPTS))
from launch import spec,checkpoint
from migrate import migrate,SOURCE_SHA
from quality import summarize,compare,rank,initialization_ok


def mark(run,role):write(run/'review_role.json',dict(role=role))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--num-envs','--num_envs',type=int,default=512)
    p.add_argument('--actor-updates',type=int,default=spec.CONTRACT['training_actor_updates'])
    p.add_argument('--quality-interval',type=int,default=spec.CONTRACT['quality_interval'])
    p.add_argument('--seed',type=int,default=42);p.add_argument('--device',default='cuda:0')
    p.add_argument('--eval-seeds',type=int,nargs='+',default=[43,44,45])
    p.add_argument('--eval-episodes',type=int,default=4);p.add_argument('--play-episodes',type=int,default=3)
    p.add_argument('--baseline-only',action='store_true')
    a=p.parse_args(argv)
    if not 1<=a.actor_updates<=500 or not 1<=a.quality_interval<=50:p.error('Actor updates 1..500; quality interval 1..50')
    if min(a.num_envs,a.eval_episodes,a.play_episodes)<=0:p.error('Counts must be positive')
    source=SCRIPTS.parent/'initialization/source_v1.pt'
    if not source.is_file() or digest(source)!=SOURCE_SHA:raise ValueError('Bundled reviewed original model missing or corrupt')
    profile=dict(read(ROOT/'offline/routes.json')['routes']['05a'])
    profile['entry']=str((SCRIPTS/'launch.py').relative_to(ROOT))
    profile.update(name='R5A J1 v3 固定高目标起跳与停稳',contract_version=spec.VERSION,status='integrated_J1_highjump_v3')
    session=ROOT/'offline_sessions/route_05a'/('J1_highjump_v3_'+stamp());session.mkdir(parents=True,exist_ok=False)
    state=dict(schema_version=1,route='05a',stage=spec.CONTRACT['stage'],profile=profile,kind='highjump',created_at=now(),
        status='starting',runs=[],stages=[],targets={k:v for k,v in spec.CONTRACT.items() if k.startswith('target_')},height_curriculum=False,
        requested_actor_updates=a.actor_updates,critic_warmup_updates=spec.CONTRACT['critic_warmup_updates'],
        formal_behavior_acceptance='requires_review')
    code=0
    with session_lock(session):
        print('[HIGHJUMP SESSION]',session,flush=True)
        try:
            state['status']='running';write(session/'session.json',state);capture_environment(session)
            warm=migrate(source,session/'migration',a.seed)
            common=['--headless','--seed',str(a.seed),'--device',a.device]
            quick=['--eval_seeds',*[str(s) for s in a.eval_seeds],'--eval_episodes','1']
            candidates=[]
            for name,path in [('preserved_actor',warm),('zero_residual',warm.with_name('reference_start.pt'))]:
                run,result=child(session,state,profile,'evaluate',['--checkpoint',str(path),*common,*quick],name='baseline_'+name)
                mark(run,'baseline_'+name);candidates.append((name,path,result))
                print('[HIGHJUMP BASELINE]',name,json.dumps(summarize(result)),flush=True)
            viable=[c for c in candidates if initialization_ok(c[2])]
            if not viable:raise RuntimeError('No initializer retains a valid jump and terminal hold; training not started. Copy the USB diagnostics back.')
            chosen=max(viable,key=lambda c:rank(c[2]));name,initial,baseline=chosen
            best=initial;best_result=baseline;current=initial
            state.update(initialization_variant=name,initial_checkpoint_sha256=digest(initial),best_checkpoint=str(best))
            write(session/'baseline_selection.json',dict(selected=name,checkpoint_sha256=digest(initial),
                candidates={n:summarize(r) for n,_,r in candidates},new_reference_contract=spec.VERSION))
            history=[];bad_streak=0;done=0
            if not a.baseline_only:
                run,_=child(session,state,profile,'train',['--checkpoint',str(current),*common,'--num_envs',str(a.num_envs),
                    '--max_iterations',str(spec.CONTRACT['critic_warmup_updates'])],name='critic_warmup')
                current=checkpoint_from_run(run)
                old=checkpoint(initial)['model_state_dict'];new=checkpoint(current)['model_state_dict']
                import torch
                if not all(torch.equal(v,new[k]) for k,v in old.items() if k.startswith('actor.') or k=='log_std'):
                    raise RuntimeError('Critic warmup changed Actor/noise')
                write(session/'warmup_verification.json',dict(actor_exactly_preserved=True,noise_exactly_preserved=True,
                    completed_updates=checkpoint(current)['infos']['completed_updates']))
                while done<a.actor_updates:
                    count=min(a.quality_interval,a.actor_updates-done)
                    run,_=child(session,state,profile,'train',['--checkpoint',str(current),*common,'--num_envs',str(a.num_envs),
                        '--max_iterations',str(count)],name=f'train_actor_{done+count:04d}')
                    current=checkpoint_from_run(run);done+=count
                    run,result=child(session,state,profile,'evaluate',['--checkpoint',str(current),*common,*quick],name=f'quality_{done:04d}')
                    mark(run,'quality')
                    gate=compare(result,baseline);bad_streak=0 if gate['eligible'] else bad_streak+1
                    if gate['eligible'] and rank(result)>rank(best_result):best=current;best_result=result
                    history.append(dict(actor_update_attempts=done,checkpoint=str(current),sha256=digest(current),**gate))
                    state.update(completed_actor_update_attempts=done,last_checkpoint=str(current),best_checkpoint=str(best),bad_streak=bad_streak)
                    write(session/'quality_history.json',history);write(session/'session.json',state)
                    print('[HIGHJUMP QUALITY]',done,json.dumps(gate),flush=True)
                    if bad_streak>=spec.CONTRACT['quality_consecutive_failures']:
                        state['stop_reason']='consecutive_quality_regression';break
                selected=session/'selected_best.pt';shutil.copy2(best,selected)
                assert digest(selected)==digest(best)
                state.update(checkpoint=str(selected),checkpoint_sha256=digest(selected),
                    selected_actor_update_attempts=max(0,checkpoint(selected)['infos']['completed_updates']-spec.CONTRACT['critic_warmup_updates']))
                full=['--eval_seeds',*[str(s) for s in a.eval_seeds],'--eval_episodes',str(a.eval_episodes)]
                run,final=child(session,state,profile,'evaluate',['--checkpoint',str(selected),*common,*full],name='final_evaluate')
                mark(run,'final')
                run,video=child(session,state,profile,'play',['--checkpoint',str(selected),*common,'--record-video',
                    '--play-episodes',str(a.play_episodes)],name='final_video');mark(run,'final')
                write(session/'comparison.json',dict(baseline=summarize(baseline),final=summarize(final),video=summarize(video),
                    chosen_initializer=name,selected_sha256=digest(selected),formal_behavior_acceptance='requires_review'))
                (session.parent/'LATEST_HIGHJUMP_CHECKPOINT.txt').write_text(str(selected.relative_to(ROOT))+'\n')
                print('[HIGHJUMP SELECTED]',selected,flush=True)
            state['status']='stopped_on_regression' if state.get('stop_reason') else 'completed'
        except KeyboardInterrupt:code=130;state.update(status='interrupted',error='KeyboardInterrupt')
        except Exception as error:
            code=1;state.update(status='failed',error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc())
            print('[HIGHJUMP ERROR]',state['error'],file=sys.stderr,flush=True)
        finally:
            state['finished_at']=now();write(session/'session.json',state)
            try:export_session(session,state)
            except Exception as error:
                code=1;write(session/'collection_error.json',dict(error=str(error),traceback=traceback.format_exc()))
                print('[USB REVIEW ERROR]',error,'raw session:',session,file=sys.stderr)
    return code


if __name__=='__main__':raise SystemExit(main())
