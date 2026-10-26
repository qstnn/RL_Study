"""CPU synthetic readback/diagnostic boundary checks; never physical acceptance."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from migration_v1 import admission,retention_protocol as protocol,retention_physics_probe as probe,retention_worker as worker
from migration_v1.contract import REPO,sha256
from migration_v1.dataset import EvidenceError


@pytest.fixture
def readbacks(tmp_path):
    data=json.loads((Path(__file__).parent/'fixtures/ordinary_observation_excerpt.json').read_text())
    runtime=deepcopy(data['runtime']);initial=deepcopy(data['initial'])
    profile=dict(data['request_metadata']['execution_physics'],robot_solver=[16,4],cart_solver=[16,4],
        robot_hitch_local_m=runtime['robot_hitch_local_m'],cart_hitch_local_m=runtime['cart_hitch_local_m'])
    robot=REPO/'model_lighthw/lighthw_urdf/urdf/lighthw.urdf'
    cart=tmp_path/'synthetic_cart.urdf';cart.write_text('synthetic asset identity only')
    profile.update(robot_urdf_sha256=sha256(robot),cart_urdf_sha256=sha256(cart))
    case=protocol.select_case(1)
    runtime.update(slopes_deg=[0.],payloads_kg=[0.],towed_masses_kg=[0.],physical_active_cart_masses_kg=[],
        unloaded_acceleration_mps2=1.,loaded_acceleration_mps2=.3)
    initial.update(robot_root_state=[0.,0.,.416223,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
        cart_root_state=[0.,500.,1.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
        env_origin=[0.,0.,0.],joint_velocity_action_order=[0.]*16,cart_joint_velocity=[0.]*4,
        robot_tensor_device='cuda:0',simulation_device='cuda:0',command_vx=0.,episode_step=0,
        requested_vx=3.1,goal_distance_m=1.)
    declared={link.get('name'):float(link.find('inertial/mass').get('value'))
        for link in ET.parse(robot).getroot().findall('link') if link.find('inertial/mass') is not None}
    names=list(declared);base=names.index('Base_link');masses=np.array(list(declared.values()))
    com=np.zeros((len(names),7));com[base,:3]=runtime['payload_examples'][0]['base_com_m']
    inertia=np.broadcast_to(np.eye(3),(len(names),3,3)).copy();inertia[base]=runtime['payload_examples'][0]['inertia_body_frame']
    physical=dict(body_names=names,masses_kg=masses,coms=com,inertias=inertia,
        unloaded_masses_kg=masses.copy(),unloaded_inertias=inertia.copy())
    solver={'/physicsScene':{'physxScene:maxPositionIterationCount':16,'physxScene:maxVelocityIterationCount':4}}
    for name in ('Robot/Base_link','Trailer/cart_base'):
        solver['/World/envs/env_0/'+name]={'physxArticulation:solverPositionIterationCount':16,
            'physxArticulation:solverVelocityIterationCount':4}
    materials=dict(assets={name:[[[3.,3.]]] for name in ('robot','trailer')},
        terrain={'synthetic':dict(static=3.,dynamic=3.,combine='min')})
    return dict(runtime=runtime,initial=initial,solver=solver,materials=materials,
        assets=dict(robot=robot,cart=cart),profile=profile,case=case,physical=physical)


def test_default_profile_and_complete_runtime_contract_unchanged(readbacks):
    source=deepcopy(readbacks['profile']);actual,diagnostic=protocol.execution_profile(source)
    assert actual==source and actual is not source and diagnostic is None
    result=protocol.check_runtime(**readbacks)
    assert result['runtime_profile_verified'] and result['diagnostic_physics_override'] is None
    assert not result['autonomous_behavior_accepted']


@pytest.mark.parametrize('variant,solver',[('mu4_solver16_4',[16,4]),('mu4_solver4_1',[4,1])])
def test_two_named_contrasts_use_actual_readbacks_keep_source_profile(readbacks,variant,solver):
    source=deepcopy(readbacks['profile']);actual,diagnostic=protocol.execution_profile(source,variant)
    assert source['ground_friction']==3. and source['robot_solver']==[16,4]
    assert actual['ground_friction']==4. and actual['robot_solver']==solver
    assert diagnostic['current_mu3_retention_admission_eligible'] is False
    readbacks.update(profile=actual,source_profile=source,diagnostic_physics_override=variant)
    readbacks['runtime']['ground_friction']=4.
    for asset in readbacks['materials']['assets'].values():asset[0][0]=[4.,4.]
    readbacks['materials']['terrain']['synthetic']=dict(static=4.,dynamic=4.,combine='min')
    for path,values in readbacks['solver'].items():
        prefix='physxScene:max' if path=='/physicsScene' else 'physxArticulation:solver'
        values[prefix+'PositionIterationCount']=solver[0];values[prefix+'VelocityIterationCount']=solver[1]
    result=protocol.check_runtime(**readbacks)
    assert result['diagnostic_physics_override']==diagnostic
    readbacks['materials']['assets']['robot'][0][0][0]=3.
    with pytest.raises(EvidenceError,match='asset friction differs'):
        protocol.check_runtime(**readbacks)


@pytest.mark.parametrize('override',['mu5','mu3_solver4_1','',False,1,{'ground_friction':4.}])
def test_arbitrary_override_is_rejected(readbacks,override):
    with pytest.raises((EvidenceError,TypeError)):
        protocol.execution_profile(readbacks['profile'],override)


def test_mu4_cannot_masquerade_as_unmodified_source_or_default_retention(readbacks):
    readbacks['profile']['ground_friction']=4.
    with pytest.raises(EvidenceError,match='source must remain'):
        protocol.check_runtime(**readbacks)
    with pytest.raises(EvidenceError,match='source must remain'):
        protocol.execution_profile(readbacks['profile'],'mu4_solver4_1')


def test_profile_rejects_undeclared_other_physics_change(readbacks):
    source=deepcopy(readbacks['profile'])
    actual,_=protocol.execution_profile(source,'mu4_solver16_4');actual['rolling_coefficient']=.001
    readbacks.update(profile=actual,source_profile=source,diagnostic_physics_override='mu4_solver16_4')
    with pytest.raises(EvidenceError,match='Executed profile differs'):
        protocol.check_runtime(**readbacks)


def test_probe_prepare_adds_own_execution_source_and_calls_strict_original_loader(monkeypatch,tmp_path,readbacks):
    source=deepcopy(readbacks['profile'])
    manifest=dict(teacher_bundle=dict(episodes=[dict(physics_profile=source,case={}),dict(physics_profile=source,case={})]))
    sentinel=object();calls=[]
    def load(checkpoint,**kwargs):calls.append((checkpoint,kwargs));return sentinel,manifest
    monkeypatch.setattr(worker,'load_student',load)
    output=tmp_path/'uncreated'
    prepared=probe.prepare(tmp_path/'candidate.pt',1,output,variant='mu4_solver4_1')
    assert prepared['policy'] is sentinel and len(calls)==1 and calls[0][1]==dict(device='cpu',inference_mode='row')
    assert prepared['profile']==source and prepared['profile']['ground_friction']==3.
    assert prepared['execution_profile']['ground_friction']==4.
    assert prepared['execution_profile']['robot_solver']==[4,1]
    assert str(Path(probe.__file__).resolve()) in prepared['sources']
    assert not output.exists()


def test_missing_or_unknown_probe_variant_cannot_fall_back_to_standard_run(monkeypatch,tmp_path):
    monkeypatch.setattr(worker,'prepare',lambda *args,**kwargs:pytest.fail('must reject before loading'))
    with pytest.raises(ValueError,match='named diagnostic'):
        probe.prepare(tmp_path/'missing.pt',1,tmp_path/'output',variant=None)


def test_admission_rejects_diagnostic_kind_even_if_saved_success_claims_pass():
    request=dict(artifact_kind='diagnostic_retention_physics_probe',success=True,
        diagnostic_physics_override=dict(variant='mu4_solver16_4'))
    with pytest.raises(EvidenceError,match='non-student artifact'):
        admission._verify_identity(request,None,None,{},True)


def test_probe_keeps_original_behavior_thresholds_and_worker_execution():
    assert probe.worker.run_simulation is worker.run_simulation
    source=Path(protocol.PROTOCOL_SOURCE).read_text()
    assert "metrics['cruise_mae_mps']<=.15" in source
    assert "review = protocol.original.summarize(rows,case,reason)" in Path(worker.__file__).read_text()
