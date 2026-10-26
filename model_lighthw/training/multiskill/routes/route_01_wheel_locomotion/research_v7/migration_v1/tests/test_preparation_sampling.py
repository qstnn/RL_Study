"""离线分组CPU测试；临时档案均为合成接口数据，不是成功物理证据。"""
from copy import deepcopy
import csv
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from migration_v1.preparation_sampling import prepare_sampling, preparation_stages
from migration_v1.bc import balanced_indices, split_episodes
from migration_v1.contract import sha256
from migration_v1.dataset import EvidenceError


def trace():
    n=40;time=np.arange(n)*.02;phase=np.r_[np.zeros(30),np.ones(10)]
    command=np.r_[np.zeros(30),np.arange(1,11)*.001]
    policy=dict(teacher_time_s=time,phase=phase,command_vx=command,
        ordinary_entry_path_time_s=np.minimum(time,.2),ordinary_entry_prepared=(time>=.5).astype(float),
        ordinary_entry_downhill_phase=np.r_[np.ones(15),np.full(5,2.),np.full(5,3.),np.full(15,4.)])
    stage=np.where(phase==0,'prepare','start')
    return policy,stage,time+.02,np.r_[phase[1:],phase[-1]],command.copy()


@pytest.fixture
def signed_synthetic(tmp_path):
    arrays=dict(observation=np.zeros((80,300),np.float32),action=np.zeros((80,16),np.float32),
        episode_id=np.repeat([0,1],40),group_id=np.r_[np.zeros(30),np.ones(10),np.full(30,2),np.full(10,3)].astype(np.int64),
        case_key=np.repeat(['up','down'],40),stage=np.tile(np.r_[np.full(30,'prepare'),np.full(10,'start')],2),
        successful=np.ones(80,dtype=bool))
    arrays['observation'][:,299]=1.
    episodes=[]
    for index in range(2):
        policy,_,times,phases,commands=trace()
        if index==0:del policy['ordinary_entry_downhill_phase']
        rows=arrays['episode_id']==index
        policy.update(observation=arrays['observation'][rows],executed_action=arrays['action'][rows])
        run=tmp_path/f'synthetic_only_{index}';run.mkdir()
        (run/'request.json').write_text(json.dumps(dict(entry_duration_s=.2,artifact_kind='synthetic_interface_only')))
        np.savez_compressed(run/'seed43_case00_policy_trace.npz',**policy)
        with (run/'seed43_case00.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=['t','eval_phase','command_vx']);writer.writeheader()
            writer.writerows(dict(t=t,eval_phase=p,command_vx=c) for t,p,c in zip(times,phases,commands))
        episodes.append(dict(directory=str(run),samples=40,complete_teacher_pass=True,
            successful_labels_are_executed=True,source_files={name:sha256(run/name) for name in
                ('request.json','seed43_case00_policy_trace.npz','seed43_case00.csv')}))
    return arrays,dict(schema='synthetic_interface_only_not_physical_evidence',episodes=episodes)


def test_disabled_is_exact_original_arrays_and_sampling_without_reading_sources(signed_synthetic):
    arrays,_=signed_synthetic
    original=deepcopy(arrays)
    result,meta=prepare_sampling(arrays,{},enabled=False)
    assert result is arrays and not meta['balance_preparation']
    a=balanced_indices(original['group_id'],np.arange(80),1000,np.random.default_rng(43))
    b=balanced_indices(result['group_id'],np.arange(80),1000,np.random.default_rng(43))
    assert np.array_equal(a,b)
    assert all(np.array_equal(arrays[key],original[key]) for key in arrays)


def test_entry_capture_prepared_boundaries_and_post_command_preserved():
    policy,stage,time,phase,command=trace()
    result=preparation_stages(policy,stage,.2,time,phase,command)
    assert list(result[[9,10,14,15,19,20,24,25,29,30]]) == [
        'prepare_path_entry','prepare_capture_roll','prepare_capture_roll','prepare_capture_brake',
        'prepare_capture_brake','prepare_capture_settle','prepare_capture_settle','prepare_wait','prepare_wait','start']
    assert np.array_equal(result[30:],stage[30:])


@pytest.mark.parametrize('change',['length','time','csv_time','csv_phase','command','prepared','path','early_prepared','downhill_phase'])
def test_misaligned_or_invalid_real_record_semantics_rejected(change):
    policy,stage,time,phase,command=trace()
    if change=='length':policy['teacher_time_s']=policy['teacher_time_s'][:-1]
    elif change=='time':policy['teacher_time_s'][10]+=.02
    elif change=='csv_time':time[10]+=.02
    elif change=='csv_phase':phase[0]=1
    elif change=='command':command[5]=.1
    elif change=='prepared':policy['ordinary_entry_prepared'][20]=.5
    elif change=='path':policy['ordinary_entry_path_time_s'][5]=.3
    elif change=='early_prepared':policy['ordinary_entry_prepared'][2]=1.
    else:policy['ordinary_entry_downhill_phase'][20]=7.
    with pytest.raises(EvidenceError):
        preparation_stages(policy,stage,.2,time,phase,command)


def test_grouping_changes_only_sampler_metadata_and_records_exact_members(signed_synthetic):
    arrays,manifest=signed_synthetic;before=deepcopy(arrays)
    result,meta=prepare_sampling(arrays,manifest,enabled=True)
    assert meta['group_count']==10
    for key in arrays:
        assert np.array_equal(arrays[key],before[key])
        if key!='group_id':assert result[key] is arrays[key]
    assert sum(group['count'] for group in meta['groups'])==80
    for group in meta['groups']:
        for member in group['members']:
            assert member['count']==len(member['frame_indices'])==len(member['teacher_times_s'])
            assert np.allclose(np.asarray(member['frame_indices'])*.02,member['teacher_times_s'])
    assert meta['nominal_case_probability']=={'down':.6,'up':.4}
    assert not meta['actor_inputs_changed'] and not meta['success_labels_changed']
    assert np.array_equal(split_episodes(arrays)[0],split_episodes(result)[0])
    with pytest.raises(EvidenceError,match='required uphill/downhill'):
        split_episodes(result,[1])


def test_stage_sampler_balances_small_groups_and_honors_whole_episode_eligibility(signed_synthetic):
    arrays,manifest=signed_synthetic
    result,_=prepare_sampling(arrays,manifest,enabled=True)
    rows=balanced_indices(result['group_id'],np.arange(80),12000,np.random.default_rng(43))
    counts=np.bincount(result['group_id'][rows])
    assert max(abs(counts-1200))<180
    eligible=np.flatnonzero(arrays['episode_id']==0)
    selected=balanced_indices(result['group_id'],eligible,2000,np.random.default_rng(44))
    assert set(selected)<=set(eligible)


@pytest.mark.parametrize('change',['source','rows','label_provenance','episode_length'])
def test_source_or_episode_mismatch_cannot_generate_sampling_groups(signed_synthetic,change):
    arrays,manifest=signed_synthetic
    if change=='source':
        (Path(manifest['episodes'][0]['directory'])/'request.json').write_text('{}')
    elif change=='rows':arrays['action'][0,0]=1.
    elif change=='label_provenance':manifest['episodes'][0]['complete_teacher_pass']=False
    else:manifest['episodes'][0]['samples']=39
    with pytest.raises(EvidenceError):
        prepare_sampling(arrays,manifest,enabled=True)
