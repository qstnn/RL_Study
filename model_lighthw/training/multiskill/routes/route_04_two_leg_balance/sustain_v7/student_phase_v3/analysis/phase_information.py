"""从原305维输入与真实轨迹检查阶段信息；影子状态仅诊断，不生成动作标签。"""
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import torch
from scipy.spatial import cKDTree

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE.parent/'student_segments_v2_1'))
from common_segments import read,write,sha,parent,require
from teacher_sustain import SustainTeacher


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    dataset=HERE.parent/'student_segments_v2_1/datasets/2026-10-23_teacher_segments'
    require(dataset);meta=read(dataset/'dataset.json')
    assert sha(dataset/'dataset.npz')==meta['dataset_sha256']
    with np.load(dataset/'dataset.npz') as z: saved={k:z[k] for k in z.files}
    rows=[];labels=np.full(len(saved['stage']),-1,np.int64);blend=np.zeros(len(labels));trigger=np.zeros(len(labels),bool)
    paths=sorted({Path(e['run'])/e['trace'] for e in meta['episodes']})
    paths += [HERE.parent/'student_v1/runs/2026-10-22_bc_02_autonomous/trace_seed43.npz',
              HERE.parent/'student_segments_v2_1/runs/2026-10-23_bc_03_autonomous/trace_autonomous_seed43.npz']
    for path in paths:
        with np.load(path) as z:d={k:z[k] for k in z.files}
        teacher=SustainTeacher([parent.candidate(.8)]*2,read(path.parent/'geometry.json'),'cpu')
        stages=[];mix=[];latched=[];teacher_actions=[]
        for k in range(len(d['policy_time'])):
            j=max(0,k*20-1)
            metrics={key:torch.from_numpy(d[key][j]) for key in ('gravity','ang_vel','wheel_pos','com','com_vel','force_z')}
            mock=SimpleNamespace(metrics=lambda:metrics,
                physics_steps=torch.round(torch.from_numpy(d['policy_physics_time'][k])/.001).long(),physics_dt=.001,
                state=SimpleNamespace(hold_origin=torch.from_numpy(d['policy_hold_origin'][k]),return_time=torch.from_numpy(d['policy_return_time'][k])))
            t=torch.from_numpy(d['policy_time'][k]);action=teacher(mock,t)
            u=((t-teacher.trigger_t)/.8).clamp(0,1)
            phase=torch.where(t<1,0,torch.where(~teacher.triggered,1,torch.where(u<1,2,3)))
            stages.append(phase.numpy());mix.append(u.numpy());latched.append(teacher.triggered.numpy().copy());teacher_actions.append(action.numpy())
        stages=np.stack(stages);mix=np.stack(mix);latched=np.stack(latched);teacher_actions=np.stack(teacher_actions)
        obs=d['policy_observation'].reshape(-1,2,5,61)
        pitch=np.arctan2(obs[:,:,-1,3],-obs[:,:,-1,5])
        state_gate=(pitch<=np.deg2rad(-5.)) & (obs[:,:,-1,55]<.5) & (obs[:,:,-1,60]>.5)
        commanded=obs[:,:,-1,55:60].argmax(-1)
        active=d['policy_active']
        # 0时刻没有保存reset前物理样本；不将首物理步误当独立reset读回。
        np.testing.assert_allclose(obs[1:,:,-1,3:6][active[1:]],d['gravity'][19::20][:-1][active[1:]],atol=1e-6)
        executed=active & d['policy_takeover']
        teacher_error=float(abs(teacher_actions[executed]-d['policy_teacher_action'][executed]).max()) if executed.any() else None
        if teacher_error is not None:assert teacher_error<2e-5
        confusion=np.zeros((5,4),int)
        np.add.at(confusion,(commanded[active],stages[active]),1)
        for i,e in enumerate(meta['episodes']):
            if path!=Path(e['run'])/e['trace']:continue
            selected=saved['episode']==i;idx=saved['policy_index'][selected];slot=e['slot']
            np.testing.assert_array_equal(saved['observation'][selected],d['policy_observation'][idx,slot])
            labels[selected]=stages[idx,slot];blend[selected]=mix[idx,slot];trigger[selected]=latched[idx,slot]
        rows.append(dict(trace=str(path),sha256=sha(path),active_frames=int(active.sum()),
            teacher_executed_frames=int(executed.sum()),teacher_action_replay_error=teacher_error,
            commanded_vs_teacher_stage=confusion.tolist(),
            observed_pitch_vs_latched_mismatch=int((state_gate[active]!=latched[active]).sum()),
            mismatch_frames=[dict(slot=int(s),policy_index=int(k),t=float(d['policy_physics_time'][k,s]))
                            for k,s in np.argwhere(active & (state_gate!=latched))][:20]))
    assert (labels>=0).all()
    obs=saved['observation'].reshape(-1,5,61);pitch=np.arctan2(obs[:,-1,3],-obs[:,-1,5])
    gate=(pitch<=np.deg2rad(-5.)) & (obs[:,-1,55]<.5) & (obs[:,-1,60]>.5)
    # 跨整组轨迹最近邻，避免同轨迹临近帧或两槽重复形成伪验证。
    families=np.array([next(i for i,p in enumerate(paths) if p==Path(meta['episodes'][e]['run'])/meta['episodes'][e]['trace']) for e in saved['episode']])
    neighbors=[]
    x=saved['observation'].astype(np.float64)
    for f in np.unique(families):
        train=families!=f;test=~train
        scale=np.maximum(x[train].std(0),.05)
        distance,index=cKDTree(x[train]/scale).query(x[test]/scale)
        train_ids=np.flatnonzero(train);test_ids=np.flatnonzero(test)
        reference=train_ids[index]
        transition=np.isin(labels[test],[1,2])
        neighbors.append(dict(heldout_family=int(f),samples=int(test.sum()),
            teacher_stage_accuracy=float((labels[reference]==labels[test]).mean()),
            transition_stage_accuracy=float((labels[reference][transition]==labels[test][transition]).mean()),
            blend_progress_mae=float(abs(blend[reference]-blend[test])[transition].mean()),
            action_max_error_p95=float(np.percentile(abs(saved['action'][reference]-saved['action'][test]).max(-1),95)),
            normalized_distance_p95=float(np.percentile(distance/np.sqrt(305),95))))
    result=dict(dataset=str(dataset),dataset_sha256=meta['dataset_sha256'],source_sha256=sha(Path(__file__)),
        features=dict(angular_velocity='0:3 scaled0.2',gravity='3:6',leg_position='6:18',joint_velocity='18:34 scaled0.05',
            previous_action='34:50',support_command='50:52',target_gravity='52:55',commanded_phase='55:60',stand_request='60',history='5x61; 80ms lookback'),
        teacher_stage_counts=np.bincount(labels,minlength=4).tolist(),dataset_pitch_gate_mismatch=int((gate!=trigger).sum()),
        trace_checks=rows,leave_trace_family_out=neighbors,
        conclusion='Command phase alone is not actual capture state. Pitch gate sufficiency is checked only on archived trajectories; blend age is not globally observable from a finite history. Nearest-neighbor values are descriptive, not autonomous proof.',
        proposed_single_factor='shared unchanged trunk, two initialized-identical output heads selected by current observed pitch<=-5deg; no new observation or privileged feature',
        unexecuted_action_labels_added=0,ppo_admission=False)
    write(HERE/'validation/phase_information.json',result)
    np.savez_compressed(HERE/'validation/phase_diagnostic_labels.npz',teacher_stage=labels,blend_progress=blend,triggered=trigger,observed_gate=gate)
    print({k:result[k] for k in ['teacher_stage_counts','dataset_pitch_gate_mismatch','leave_trace_family_out']})
    for r in rows:print(r['trace'],r['observed_pitch_vs_latched_mismatch'],r['commanded_vs_teacher_stage'])


if __name__=='__main__':main()
