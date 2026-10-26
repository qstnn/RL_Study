"""Offline training station: route adapters, automatic checks, and one USB review folder."""
import argparse
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import traceback

from review_bundle import ROOT, collect, digest, now, read, safe_child, stable_copy, stamp, verify, write
from review_bundle import collect as legacy_collect, verify as legacy_verify
sys.path.insert(0,str(ROOT/"model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/offline"))
import r5a_review_bundle


def collect(runs,route,session=None):
    return r5a_review_bundle.collect(runs,route,session) if route=='05a' else legacy_collect(runs,route,session=session)


def verify(folder):
    marker=read(Path(folder)/'review_manifest.json',{})
    return r5a_review_bundle.verify(folder) if marker.get('schema')=='r5a_usb_review_v1' else legacy_verify(folder)


@contextmanager
def session_lock(session):
    with (session/'.station.lock').open('a+') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f'Session is still in use: {session}')
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def capture_environment(session):
    folder=session/'environment'; folder.mkdir()
    with (folder/'doctor.log').open('w') as log:
        result=subprocess.run([sys.executable, str(ROOT/'offline/doctor.py'), '--output', str(folder/'doctor.json')],
                              cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    info={'recorded_at':now(),'python':sys.executable,'python_version':sys.version,'platform':platform.platform(),
          'hostname':platform.node(),'project':str(ROOT),'isaac_sim':os.environ.get('R6_ISAAC_SIM'),
          'dependency_directory':os.environ.get('R6_DEPS')}
    try:
        gpu=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv'],
                           capture_output=True,text=True,timeout=15)
        info['gpu']=gpu.stdout; info['gpu_query_returncode']=gpu.returncode
    except (OSError,subprocess.TimeoutExpired) as error:
        info['gpu_query_error']=str(error)
    write(folder/'machine.json',info)
    if result.returncode:
        raise RuntimeError(f'Environment check failed; see {folder/"doctor.log"}')


def run_stage(session, state, name, command):
    stage={'name':name,'command':command,'started_at':now(),'status':'running'}
    state['stages'].append(stage); write(session/'session.json',state)
    print('[STATION]',name,flush=True)
    process=None
    try:
        with (session/(name+'.log')).open('w') as log:
            process=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                     text=True,errors='replace',bufsize=1,start_new_session=True)
            stage['pid']=process.pid; write(session/'session.json',state)
            for line in process.stdout:
                log.write(line); log.flush(); print(line,end='',flush=True)
            stage['returncode']=process.wait()
        stage['status']='completed' if stage['returncode']==0 else 'failed'
    except KeyboardInterrupt:
        stage.update(status='interrupted',returncode=130)
        raise
    finally:
        if process and process.poll() is None:
            os.killpg(process.pid,signal.SIGINT if stage.get('status')=='interrupted' else signal.SIGTERM)
            try: process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL); process.wait()
        stage['finished_at']=now(); write(session/'session.json',state)
    if stage['returncode']:
        raise RuntimeError(f'{name} failed with code {stage["returncode"]}; training data is preserved')


def child(session, state, profile, mode, options, name=None):
    name=name or mode
    out=session/name
    if out.exists(): raise ValueError(f'Run already exists: {out}')
    state['runs'].append(str(out)); write(session/'session.json',state)
    command=[sys.executable,str(safe_child(ROOT,profile['entry'])),mode,'--route',state['route'],
             '--run-dir',str(out),*options]
    run_stage(session,state,name,command)
    result=read(out/'result.json',{})
    meta=read(out/'run.json',{})
    if result.get('status')!='completed' or meta.get('status')!='completed' or not (out/'runtime_check.json').is_file():
        raise RuntimeError(f'{mode}: child exited without complete runtime/result evidence')
    return out,result


def checkpoint_from_run(run, explicit=None):
    run=Path(run).resolve()
    if explicit:
        return Path(explicit).resolve(strict=True)
    result=read(run/'training_result.json',read(run/'result.json',{}))
    saved=result.get('checkpoint')
    if not saved:
        raise ValueError('No final model in this run; supply --checkpoint explicitly for an interrupted run')
    candidates=[run/'training'/Path(saved).name,Path(saved)]
    expected=result.get('checkpoint_sha256')
    for candidate in candidates:
        if candidate.is_file() and (not expected or digest(candidate)==expected):
            return candidate.resolve()
    raise ValueError('Final checkpoint missing or changed; inspect the original run')


def parameters(request, names):
    values=[]
    for name in names:
        if name not in request: continue
        flag='--payload-kg' if name=='payload_kg' else '--'+name
        value=request[name]
        values += [flag]+[str(v) for v in value] if isinstance(value,list) else [flag,str(value)]
    return values


def post_checks(session,state,profile,training,checkpoint,overrides):
    request=read(training/'request.json',{})
    shared=parameters(request,profile['shared_parameters'])
    evaluation=parameters(request,profile['evaluation_parameters'])
    options=['--checkpoint',str(checkpoint),*shared]
    suffix=stamp()
    if digest(checkpoint)!=state['checkpoint_sha256']: raise RuntimeError('Selected checkpoint changed before evaluation')
    child(session,state,profile,'evaluate',[*options,*evaluation,*profile['evaluate_args'],*overrides],name='evaluate_'+suffix)
    if digest(checkpoint)!=state['checkpoint_sha256']: raise RuntimeError('Selected checkpoint changed before video')
    child(session,state,profile,'play',[*options,*profile['video_args']],name='video_'+suffix)


def export_session(session,state):
    runs=[Path(p) for p in state.get('runs',[]) if Path(p).is_dir()]
    folder=collect(runs,state['route'],session=session)
    print('\n[USB REVIEW] '+str(folder),flush=True)
    print('[USB REVIEW] Copy this entire folder back. No additional log/model selection is needed.',flush=True)
    write(session/'latest_review.json',{'folder':str(folder),'created_at':now()})
    return folder


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('routes','train','review','play','collect','verify'))
    parser.add_argument('--route',default='06')
    parser.add_argument('--session',type=Path)
    parser.add_argument('--run',type=Path)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--folder',type=Path)
    args,extra=parser.parse_known_args(argv)
    profiles=read(ROOT/'offline/routes.json')['routes']
    if args.command=='routes':
        for key,p in profiles.items(): print(key,p['name'],p['status'])
        print('Other routes: collect supports existing runs; automated training requires a route adapter.')
        return 0
    if args.command=='verify':
        if not args.folder: parser.error('verify requires --folder')
        print(verify(args.folder)); return 0
    if args.command=='collect':
        if extra: parser.error('Unexpected collect arguments: '+' '.join(extra))
        if args.session:
            session=args.session.resolve(strict=True)
            with session_lock(session): export_session(session,read(session/'session.json'))
        elif args.run:
            print('[USB REVIEW]',collect([args.run],args.route))
        else: parser.error('collect requires --session or --run')
        return 0
    if args.route not in profiles:
        parser.error('No automated adapter for route '+args.route+'; use collect for existing run data')
    if '--run-dir' in extra or any(x.startswith('--run-dir=') for x in extra):
        parser.error('--run-dir is managed by the station; use --session for replay/review')
    profile=profiles[args.route]
    if args.session:
        if args.command=='train': parser.error('Every train command creates a new session; resume with --checkpoint')
        session=args.session.resolve(strict=True)
        state=read(session/'session.json')
        if not state or state['route']!=args.route: parser.error('Session route mismatch')
    else:
        session=ROOT/'offline_sessions'/('route_'+args.route)/stamp()
        session.mkdir(parents=True,exist_ok=False)
        state={'schema_version':1,'route':args.route,'profile':profile,'kind':args.command,'created_at':now(),
               'status':'starting','runs':[],'stages':[],'behavior_acceptance':'not_evaluated'}
    code=0
    with session_lock(session):
        print('[SESSION] '+str(session),flush=True)
        try:
            state.update(status='running',latest_command=args.command,active_profile=profile); write(session/'session.json',state)
            if not (session/'environment').exists(): capture_environment(session)
            if args.command=='train':
                options=[*profile['train_defaults'],*extra]
                if args.checkpoint: options += ['--checkpoint',str(args.checkpoint.resolve(strict=True))]
                training,result=child(session,state,profile,'train',options)
                state['training_run']=str(training)
                checkpoint=checkpoint_from_run(training)
                state['checkpoint']=str(checkpoint); state['checkpoint_sha256']=digest(checkpoint)
                write(session/'session.json',state)
                post_checks(session,state,profile,training,checkpoint,[])
            elif args.command=='review':
                training=Path(args.run or state.get('training_run','')).resolve()
                if not (training/'request.json').exists(): raise ValueError('review requires --run or a session with training data')
                if str(training) not in state['runs']: state['runs'].insert(0,str(training))
                state['training_run']=str(training)
                checkpoint=checkpoint_from_run(training,args.checkpoint)
                state['checkpoint']=str(checkpoint); state['checkpoint_sha256']=digest(checkpoint)
                post_checks(session,state,profile,training,checkpoint,extra)
            else:
                checkpoint=Path(args.checkpoint or state.get('checkpoint','')).resolve(strict=True)
                if not checkpoint.is_file(): raise ValueError('play requires --checkpoint or --session with a final model')
                options=[]
                if state.get('training_run'):
                    options=parameters(read(Path(state['training_run'])/'request.json',{}),profile['shared_parameters'])
                # Standalone play can link the original training run when the model is in its training/ directory.
                elif checkpoint.parent.name=='training' and (checkpoint.parent.parent/'request.json').is_file():
                    state['training_run']=str(checkpoint.parent.parent)
                    if state['training_run'] not in state['runs']: state['runs'].append(state['training_run'])
                    options=parameters(read(checkpoint.parent.parent/'request.json',{}),profile['shared_parameters'])
                state['checkpoint']=str(checkpoint); state['checkpoint_sha256']=digest(checkpoint)
                child(session,state,profile,'play',['--checkpoint',str(checkpoint),*options,'--record-video',*extra],name='play_'+stamp())
            state['status']='completed'
        except KeyboardInterrupt:
            code=130; state.update(status='interrupted',error='KeyboardInterrupt')
        except Exception as error:
            code=1; state.update(status='failed',error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc())
            print('[STATION ERROR]',state['error'],file=sys.stderr,flush=True)
        finally:
            state['finished_at']=now(); write(session/'session.json',state)
            try: export_session(session,state)
            except Exception as error:
                code=1
                write(session/'collection_error.json',{'error':str(error),'traceback':traceback.format_exc()})
                print('[USB REVIEW ERROR]',error,'; raw session preserved:',session,file=sys.stderr)
    return code


if __name__=='__main__':
    raise SystemExit(main())
