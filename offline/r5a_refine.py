"""Offline J1 v2: explicit migration, frozen baseline, bounded refinement, one USB folder."""
import argparse
import json
from pathlib import Path
import statistics
import sys
import traceback

from station import ROOT, read, write, stamp, now, digest, session_lock, capture_environment, child, export_session, checkpoint_from_run

SCRIPTS=ROOT/'model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/scripts'
sys.path.insert(0,str(SCRIPTS))
from migrate import migrate


def find_source(explicit):
    if explicit:return Path(explicit).resolve(strict=True)
    expected=read(SCRIPTS.parent/'compat/v1.json')['reviewed_final_checkpoint_sha256']
    candidates=list((ROOT/'offline_sessions/route_05a').glob('*/train/training/model_999.pt'))
    matches=[p for p in candidates if digest(p)==expected]
    if not matches:raise ValueError('Reviewed v1 model_999.pt not found. Supply --checkpoint /absolute/path/to/original/model_999.pt')
    return sorted(matches)[0].resolve()


def baseline_gate(result):
    cases=result.get('cases',[])
    if not cases:raise ValueError('Baseline has no cases')
    bad=[i for i,c in enumerate(cases) if not (c.get('uniform_physics_trace') and c.get('reason')=='timeout'
        and c.get('duration_s',0)>=3.999 and c.get('launched') and c.get('landed') and not c.get('failed')
        and c.get('peak_rise',0)>=.05 and c.get('flight_time',0)>=.10-1e-6)]
    if bad:raise ValueError(f'Frozen baseline lost jump/contact validity in cases {bad}; refinement was not started')
    return dict(passed=True,cases=len(cases),required='full 4s, real jump and landing, no physical failure',
                terminal_hold_required_for_this_gate=False,terminal_hold_is_training_objective=True)


def brief(result):
    cases=result['cases']
    return dict(cases=len(cases),terminal_successes=sum(c['success'] for c in cases),
        ever_stable_cases=sum(c.get('ever_success',False) for c in cases),
        median_impact_N=statistics.median(c['impact_peak'] for c in cases),
        maximum_impact_N=max(c['impact_peak'] for c in cases),
        median_terminal_stable_s=statistics.median(c['stable_time'] for c in cases),
        minimum_airtime_s=min(c['flight_time'] for c in cases))


def mark_role(run,role):
    write(run/'review_role.json',dict(role=role))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--num-envs','--num_envs',type=int,default=512)
    parser.add_argument('--max-iterations','--max_iterations',type=int,default=300)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--eval-seeds',type=int,nargs='+',default=[43,44,45])
    parser.add_argument('--eval-episodes',type=int,default=4)
    parser.add_argument('--play-episodes',type=int,default=3)
    parser.add_argument('--baseline-only',action='store_true')
    args=parser.parse_args(argv)
    if not 1<=args.max_iterations<=300:parser.error('This refinement stage is bounded to 1..300 updates')
    if min(args.num_envs,args.eval_episodes,args.play_episodes)<=0:parser.error('Counts must be positive')
    source=find_source(args.checkpoint)
    profile=read(ROOT/'offline/routes.json')['routes']['05a']
    session=ROOT/'offline_sessions/route_05a'/('J1_v2_'+stamp());session.mkdir(parents=True,exist_ok=False)
    state=dict(schema_version=1,route='05a',profile=profile,kind='refine',created_at=now(),
        status='starting',runs=[],stages=[],source_checkpoint=str(source),source_sha256=digest(source),
        requested_v2_updates=0 if args.baseline_only else args.max_iterations,behavior_acceptance='requires_review')
    code=0
    with session_lock(session):
        print('[SESSION]',session,flush=True)
        try:
            state['status']='running';write(session/'session.json',state)
            capture_environment(session)
            frozen,warm=migrate(source,session/'migration',args.seed)
            common=['--headless','--seed',str(args.seed),'--device',args.device]
            eval_args=['--eval_seeds',*[str(v) for v in args.eval_seeds],'--eval_episodes',str(args.eval_episodes)]
            base_run,baseline=child(session,state,profile,'evaluate',
                ['--checkpoint',str(frozen),*common,*eval_args],name='baseline_evaluate')
            mark_role(base_run,'baseline')
            gate=baseline_gate(baseline);write(session/'baseline_gate.json',gate)
            print('[R5A BASELINE]',json.dumps(brief(baseline)),flush=True)
            if not args.baseline_only:
                training,result=child(session,state,profile,'train',
                    ['--checkpoint',str(warm),*common,'--num_envs',str(args.num_envs),
                     '--max_iterations',str(args.max_iterations)],name='train')
                state['training_run']=str(training)
                checkpoint=checkpoint_from_run(training)
                state.update(checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint))
                write(session/'session.json',state)
                run,final=child(session,state,profile,'evaluate',['--checkpoint',str(checkpoint),*common,*eval_args],name='final_evaluate')
                mark_role(run,'final')
                run,video=child(session,state,profile,'play',['--checkpoint',str(checkpoint),*common,
                    '--record-video','--play-episodes',str(args.play_episodes)],name='final_video')
                mark_role(run,'final')
                comparison=dict(baseline=brief(baseline),final=brief(final),video=brief(video),
                    final_checkpoint_sha256=digest(checkpoint),formal_behavior_acceptance='requires_review',
                    full_backflip_implemented=False)
                write(session/'comparison.json',comparison)
                print('[R5A COMPARISON]',json.dumps(comparison,ensure_ascii=False),flush=True)
            state['status']='completed'
        except KeyboardInterrupt:
            code=130;state.update(status='interrupted',error='KeyboardInterrupt')
        except Exception as error:
            code=1;state.update(status='failed',error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc())
            print('[R5A REFINE ERROR]',state['error'],file=sys.stderr,flush=True)
        finally:
            state['finished_at']=now();write(session/'session.json',state)
            try:export_session(session,state)
            except Exception as error:
                code=1;write(session/'collection_error.json',dict(error=str(error),traceback=traceback.format_exc()))
                print('[USB REVIEW ERROR]',error,'raw session:',session,file=sys.stderr)
    return code


if __name__=='__main__':raise SystemExit(main())
