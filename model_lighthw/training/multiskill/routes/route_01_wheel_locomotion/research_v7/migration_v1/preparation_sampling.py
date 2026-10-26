"""离线准备阶段采样；只重分组，绝不添加Actor输入或生成成功标签。"""
import csv
import json
from pathlib import Path

import numpy as np

from .contract import CONTRACT, sha256
from .dataset import EvidenceError


VERSION = 'r1_offline_preparation_balance_v1'


def preparation_stages(policy, original_stage, entry_duration_s, csv_time, csv_phase, csv_command):
    """实际控制前时钟/进入记录分组；命令开始后的原stage逐项保留。"""
    stage = np.asarray(original_stage)
    n = len(stage); dt = CONTRACT['policy_dt']
    keys = ('teacher_time_s','phase','command_vx','ordinary_entry_path_time_s','ordinary_entry_prepared')
    if n < 1 or stage.shape != (n,) or any(key not in policy for key in keys):
        raise EvidenceError('Preparation balance needs complete recorded teacher time/path/prepared fields')
    fields = {key: np.asarray(policy[key]) for key in keys}
    csv_time, csv_phase, csv_command = map(np.asarray,(csv_time,csv_phase,csv_command))
    if any(value.shape != (n,) or not np.isfinite(value).all() for value in (*fields.values(),csv_time,csv_phase,csv_command)):
        raise EvidenceError('Preparation sampling record lengths/nonfinite fields differ')
    time, phase, command = (fields[key] for key in keys[:3])
    path, prepared = (fields[key] for key in keys[3:])
    if (not np.allclose(time,np.arange(n)*dt,atol=1.e-7,rtol=0.)
            or not np.allclose(csv_time,time+dt,atol=1.e-7,rtol=0.)
            or not np.array_equal(phase,np.r_[0,csv_phase[:-1]])
            or not np.allclose(command,csv_command,atol=1.e-7,rtol=0.)):
        raise EvidenceError('Recorded teacher time/phase/command does not align with actual transition CSV')
    duration = float(entry_duration_s)
    if (not np.isfinite(duration) or duration <= 0. or duration > time[-1]
            or not np.isin(phase,[0,1,2,3]).all() or not np.isin(prepared,[0,1]).all()
            or not np.allclose(path,np.minimum(time,duration),atol=1.e-7,rtol=0.)):
        raise EvidenceError('Unreviewed entry duration/path clock/phase/prepared semantics')
    before = phase == 0
    if not np.isin(stage[before],['prepare','wait']).all() or np.any(command[before] != 0.):
        raise EvidenceError('Pre-command preparation disagrees with existing stage or command')
    if np.isin(stage[~before],['prepare','wait']).any():
        raise EvidenceError('Post-command stages cannot be relabeled as preparation')
    entry = before & (time < duration-1.e-8)
    if prepared[entry].any():
        raise EvidenceError('Prepared flag precedes completion of the declared entry path')
    capture = before & ~entry & (prepared == 0)
    result = stage.astype('<U40',copy=True)
    result[entry] = 'prepare_path_entry'
    result[capture] = 'prepare_capture'
    result[before & ~entry & (prepared == 1)] = 'prepare_wait'
    if 'ordinary_entry_downhill_phase' in policy:
        downhill = np.asarray(policy['ordinary_entry_downhill_phase'])
        if downhill.shape != (n,) or not np.isfinite(downhill).all() or not np.isin(downhill,[0,1,2,3,4]).all():
            raise EvidenceError('Malformed recorded downhill capture phases')
        if np.any(capture & (downhill == 4)):
            raise EvidenceError('Downhill finished phase disagrees with unprepared capture state')
        for value,label in [(1,'roll'),(2,'brake'),(3,'settle')]:
            result[capture & (downhill == value)] = 'prepare_capture_'+label
    if not np.array_equal(result[~before],stage[~before]):
        raise AssertionError('Sampling altered commanded stages')
    return result


def prepare_sampling(arrays, teacher_manifest, *, enabled=False):
    """默认原组/原sampler不动；启用时只读已认证档案并返回新group_id副本。"""
    metadata = dict(version=VERSION,balance_preparation=bool(enabled),
        algorithm='uniform_case_stage_group_then_uniform_member_frame',
        actor_inputs_changed=False,action_labels_changed=False,success_labels_changed=False,
        episode_split_changed=False,group_metadata_is_deployment_input=False)
    if not enabled:
        return arrays, metadata
    n=len(arrays['observation'])
    for key in ('episode_id','group_id','case_key','stage'):
        if np.asarray(arrays[key]).shape != (n,):
            raise EvidenceError('Preparation sampling metadata length differs: '+key)
    episodes=teacher_manifest['episodes']
    if set(np.unique(arrays['episode_id'])) != set(range(len(episodes))):
        raise EvidenceError('Preparation sampling requires exact complete episode mapping')
    assigned=np.full(n,-1,dtype=np.int64)
    groups={}; details={}; sources=[]
    for episode_id,episode in enumerate(episodes):
        if episode.get('complete_teacher_pass') is not True or episode.get('successful_labels_are_executed') is not True:
            raise EvidenceError('Preparation grouping cannot create successful teacher labels')
        run=Path(episode['directory']);rows=np.flatnonzero(arrays['episode_id']==episode_id)
        names=('request.json','seed43_case00_policy_trace.npz','seed43_case00.csv')
        expected={name:episode['source_files'].get(name) for name in names}
        if any(not digest or sha256(run/name)!=digest for name,digest in expected.items()):
            raise EvidenceError('Preparation grouping source SHA differs from verified episode')
        request=json.loads((run/'request.json').read_text())
        with np.load(run/'seed43_case00_policy_trace.npz',allow_pickle=False) as data:
            policy=dict(data)
        csv_rows=list(csv.DictReader((run/'seed43_case00.csv').open()))
        csv_values={key:np.array([float(row[key]) for row in csv_rows]) for key in ('t','eval_phase','command_vx')}
        if (len(rows)!=episode['samples'] or policy['observation'].shape != arrays['observation'][rows].shape
                or not np.array_equal(policy['observation'],arrays['observation'][rows])
                or not np.array_equal(policy['executed_action'],arrays['action'][rows])):
            raise EvidenceError('Preparation records do not match the complete accepted episode frames')
        label=preparation_stages(policy,arrays['stage'][rows],request['entry_duration_s'],
            csv_values['t'],csv_values['eval_phase'],csv_values['command_vx'])
        time=np.asarray(policy['teacher_time_s'])
        for name in dict.fromkeys(label.tolist()):
            member=np.flatnonzero(label==name);key=(str(arrays['case_key'][rows[0]]),name)
            if not np.all(arrays['case_key'][rows] == key[0]):
                raise EvidenceError('An episode changed its nominal case key')
            group=groups.setdefault(key,len(groups));assigned[rows[member]]=group
            record=details.setdefault(group,dict(group_id=group,case_key=key[0],stage=name,count=0,members=[]))
            record['count']+=len(member)
            record['members'].append(dict(episode_id=episode_id,directory=str(run),count=len(member),
                frame_indices=member.tolist(),teacher_times_s=time[member].tolist(),
                first_time_s=float(time[member[0]]),last_time_s=float(time[member[-1]])))
        if any(sha256(run/name)!=digest for name,digest in expected.items()):
            raise EvidenceError('Preparation grouping source changed during reading')
        sources.append(dict(episode_id=episode_id,directory=str(run),files=expected,
            entry_duration_s=float(request['entry_duration_s']),policy_dt_s=CONTRACT['policy_dt']))
    if (assigned<0).any():
        raise EvidenceError('Incomplete preparation group assignment')
    result=dict(arrays,group_id=assigned)
    records=[details[key] for key in sorted(details)]
    counts={key:sum(record['case_key']==key for record in records) for key in sorted(set(arrays['case_key']))}
    metadata.update(groups=records,group_count=len(records),source_records=sources,
        probability_per_group=1./len(records),nominal_case_probability={key:value/len(records) for key,value in counts.items()},
        nominal_probability_scope='All accepted episodes; fitting separately records actual training groups after complete-episode split',
        note='Only sampling group_id changes. Original stage/observation/action/success/episode arrays remain exact.')
    return result,metadata
