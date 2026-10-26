"""R1 V7：物理对照→姿态筛选→动态教师。任一阶段失败就保存报告、停在该阶段。"""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from research_common import *
from gates import physics_gate,teacher_gate
from settings import GROUND_FRICTION
from friction import calculate as calculate_friction
import protocol_v7


def execute(out,name,cases,*,position=4,velocity=1,seconds=None,pose_file=None,pose_index=0,seed=43,noise=False,checkpoint=BASELINE,actor_inference="row",teacher_options=None):
    case_file=out/'requests'/(name+'.json');launch.write_json(case_file,cases)
    directory=out/name;command=[sys.executable,str(ROOT/'scripts/worker.py'),'--output',str(directory),'--cases',str(case_file),
        '--position-iterations',str(position),'--velocity-iterations',str(velocity),'--seed',str(seed),'--checkpoint',str(checkpoint),'--actor-inference',actor_inference]
    if seconds is not None:command+=['--seconds',str(seconds)]
    if pose_file:command+=['--pose-file',str(pose_file),'--pose-index',str(pose_index)]
    if teacher_options and pose_file:command+=['--teacher-options',str(teacher_options)]
    if noise:command+=['--noise']
    print('START',name,'log:',out/(name+'.log'),flush=True)
    with (out/(name+'.log')).open('w') as log:
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,TERM='xterm'))
        try:code=process.wait()
        except BaseException:
            launch.stop_owned_processes(process);raise
    if code:
        raise RuntimeError(f'{name}: worker exit {code}; inspect {out/(name+".log")}')
    print('SAVED',name,flush=True);return directory


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('research','physics','smoke','evaluate'))
    p.add_argument('--actor-inference',choices=('row','batch'),default='row')
    p.add_argument('--output',type=Path);p.add_argument('--checkpoint',type=Path,default=BASELINE)
    p.add_argument('--teacher-options',type=Path,help='Explicit diagnostic teacher configuration; no automatic promotion')
    p.add_argument('--suite',choices=('boundary','full'),default='boundary')
    a=p.parse_args()
    out=(a.output or ROOT/'runs'/time.strftime('%Y-%m-%d_%H-%M-%S')).resolve()
    out.mkdir(parents=True,exist_ok=False);snapshot(out)
    checkpoint=a.checkpoint.resolve(strict=True)
    launch.write_json(out/'request.json',dict(mode=a.mode,created_at=launch.now(),checkpoint=str(checkpoint),checkpoint_sha256=launch.sha256(checkpoint),
        ppo_updates=0,actor_inference=a.actor_inference,ground_friction=GROUND_FRICTION,teacher_required_before_student=True,source_physics='wheel_unified_v6',solver_candidates=[[4,1],[16,8],[32,16]],
        note='No automatic PPO or motor selection. Formal experiments are started by the user.'))
    teacher_options=None
    if a.teacher_options:
        teacher_options=out/'teacher_options.json'
        launch.write_json(teacher_options,json.loads(a.teacher_options.read_text()))
    stage='friction_calculation'
    try:
        calculate_friction(out/'friction.json')
        if a.mode=='evaluate':
            stage='frozen_actor_evaluation'
            directory=execute(out,'evaluation',protocol_v7.cases(suite=a.suite),checkpoint=checkpoint,actor_inference=a.actor_inference)
            launch.write_json(out/'result.json',dict(stage=stage,report=str(directory/'evaluation.json'),suite=a.suite,ppo_updates=0))
            return 0
        from pose_search import search
        if a.mode=='smoke':
            stage='bounded_integration_check';poses=search(out/'poses.json')
            if not poses['selected']:raise RuntimeError('No feasible static pose candidate')
            execute(out,'shortcheck',protocol_v7.cases(suite='teacher'),seconds=3.,pose_file=out/'poses.json',checkpoint=checkpoint,actor_inference=a.actor_inference,teacher_options=teacher_options)
            launch.write_json(out/'result.json',dict(stage=stage,software_check_complete=True,behavior_passed=False,ppo_updates=0))
            return 0
        stage='physics_controls';solver=None;physical=[]
        for position,velocity in ((4,1),(16,8),(32,16)):
            trials=[]
            def static(slope):return dict(slope_deg=slope,payload_kg=0.,towed_kg=40.,speed=0.,goal_distance_m=1.)
            layouts=[[static(25.)],[static(-20.)],[static(s) for s in (25.,-20.,25.,-20.)],[static(s) for s in (-20.,25.,-20.,25.)]]
            for layout,cases in enumerate(layouts):
                for repeat in range(2):
                    trials.append(execute(out,f'physics_p{position}_v{velocity}_layout{layout}_repeat{repeat}',cases,
                        position=position,velocity=velocity,seconds=2.8,checkpoint=checkpoint,actor_inference=a.actor_inference))
            gate=physics_gate(trials);gate.update(position_iterations=position,velocity_iterations=velocity)
            physical.append(gate);launch.write_json(out/'physics_gate.json',dict(candidates=physical,passed=gate['passed']))
            if gate['passed']:solver=(position,velocity);break
        if solver is None:
            launch.write_json(out/'result.json',dict(stage=stage,status='failed_physics_repeatability',next='Inspect first divergent 1kHz contact and state samples; teacher/student/PPO not started.',ppo_updates=0))
            return 2
        if a.mode=='physics':
            launch.write_json(out/'result.json',dict(stage=stage,status='physics_screen_passed',solver_iterations=solver,
                teacher_ready=False,student_training_ready=False,ppo_ready=False,ppo_updates=0))
            return 0
        stage='pose_screen';poses=search(out/'poses.json')
        if not poses['selected']:raise RuntimeError('No feasible static pose; inspect poses.json')
        stage='teacher';gates=[];selected=None
        for index,pose in enumerate(poses['selected']):
            runs=[];candidate_failed=False
            for seed in (43,44,45):
                for layout in (0,1):
                    cases=protocol_v7.cases(suite='teacher')
                    if layout:cases=list(reversed(cases))
                    directory=execute(out,f'teacher_pose{index}_seed{seed}_layout{layout}',cases,position=solver[0],velocity=solver[1],
                        pose_file=out/'poses.json',pose_index=index,seed=seed,noise=True,checkpoint=checkpoint,actor_inference=a.actor_inference,teacher_options=teacher_options)
                    runs.append(directory)
                    report=json.loads((directory/'evaluation.json').read_text())
                    if not all(c['success'] and c['complete'] for c in report['cases']):
                        candidate_failed=True
                        print('CANDIDATE_FAILED',index,'report:',directory/'evaluation.json',flush=True)
                        break
                if candidate_failed:break
            gate=teacher_gate(runs);gate['stopped_after_failed_trial']=candidate_failed;gates.append(gate)
            launch.write_json(out/'teacher_candidates.json',gates)
            if gate['passed']:selected=gate;break
        ready=dict(teacher_screen_passed=selected is not None,student_training_ready=False,ppo_ready=False,
            pose_gate_passed=True,physics_gate_passed=True,solver_iterations=solver,selected=selected,
            next='If teacher screen passes, validate initial-state perturbations and native servo torque error, then collect balanced deployed-observation demonstrations; otherwise diagnose failed stage.',
            scope='Preliminary teacher screen, not full R1 or hardware acceptance. No student/PPO is silently started.')
        launch.write_json(out/'teacher_readiness.json',ready)
        launch.write_json(out/'result.json',dict(stage=stage,status='teacher_screen_passed' if selected else 'teacher_failed',
            teacher_readiness=str(out/'teacher_readiness.json'),ppo_updates=0))
        return 0 if selected else 2
    except BaseException as exc:
        launch.write_json(out/'run_error.json',dict(stage=stage,type=type(exc).__name__,message=str(exc),ppo_updates=0))
        raise

if __name__=='__main__':raise SystemExit(main())
