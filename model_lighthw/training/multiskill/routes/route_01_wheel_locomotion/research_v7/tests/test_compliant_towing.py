import ast
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

V7 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V7/'scripts'))
from compliant_towing import (DrawbarParameters, diagnostic_parameters, drawbar_force,
    native_force_function, estimate_robot_force, apply_cart_forces, NATIVE, freeze_sources)
from compliant_cart_audit import drawbar_errors, apply_variant_audit, declared_parameters
from real_cart_audit import drawbar_errors as original_errors


@pytest.mark.parametrize('enabled,k,c,state,live', [
    (False,500.,75.,'ordinary',True), (True,500.,75.,'loaded',True),
    (True,500.,75.,'ordinary',False), (True,2000.,150.,'ordinary',True),
    (True,0.,75.,'ordinary',True), (True,float('nan'),75.,'ordinary',True),
    (True,500.,-1.,'ordinary',True)])
def test_diagnostic_is_explicit_ordinary_live_and_finite(enabled,k,c,state,live):
    with pytest.raises(ValueError): diagnostic_parameters(enabled,k,c,state,live)


def test_default_remains_native_and_metadata_keeps_original_guards():
    assert diagnostic_parameters(False,2000.,150.,'loaded',False) is None
    p=diagnostic_parameters(True,500.,75.,'ordinary',True)
    assert p == DrawbarParameters(500.,75.)
    assert p.metadata()['force_clip_N'] == 1500.
    assert p.metadata()['maximum_length_failure_m'] == 1.6
    assert not p.metadata()['original_teacher_dataset_eligible']


def sample_arrays():
    data=json.loads((V7/'tests/fixtures/compliant_drawbar_states.json').read_text())
    return {'real_cart_'+key:np.asarray([r['values'][key] for r in data['samples']],dtype=np.float64)
            for key in data['samples'][0]['values']}


def test_actual_native_samples_reconstruct_and_candidate_is_rejected_by_old_auditor():
    z=sample_arrays()
    assert max(original_errors(z).values()) < .001
    tensors=[torch.tensor(z['real_cart_'+key]) for key in ('robot_hitch_pre_w','cart_hitch_pre_w',
        'robot_hitch_velocity_pre_w','cart_hitch_velocity_pre_w')]
    active=torch.ones(len(tensors[0]),dtype=torch.bool)
    old=drawbar_force(*tensors,active,DrawbarParameters())
    native=native_force_function()(*tensors,active)
    for a,b in zip(old,native): assert torch.equal(a,b)
    p=DrawbarParameters(500.,75.);force,length,tension=drawbar_force(*tensors,active,p)
    candidate=deepcopy(z);force=force.numpy()
    candidate['real_cart_robot_force_w_N']=-force
    candidate['real_cart_robot_torque_w_Nm']=np.cross(z['real_cart_robot_arm_pre_w'],-force)
    candidate['real_cart_cart_torques_w_Nm'][:,0]=np.cross(z['real_cart_cart_arm_pre_w'],force)
    errors,mechanics=drawbar_errors(candidate,p)
    assert max(errors.values()) < 1.e-10
    assert original_errors(candidate)['spring_force_max_N'] > 100.
    assert mechanics['initial_spring_energy_J'] == pytest.approx(.09805,abs=1.e-5)
    candidate['real_cart_robot_torque_w_Nm'][0,1]+=1.
    assert drawbar_errors(candidate,p)[0]['robot_arm_torque_max_Nm'] == pytest.approx(1.)


class Asset:
    def __init__(self,data,num_bodies): self.data=data;self.num_bodies=num_bodies
    def set_external_force_and_torque(self,force,torque,**kwargs):
        self.applied_force=force.clone();self.applied_torque=torque.clone();self.write_args=kwargs


def quat_apply(q,v):
    qv=q[:,1:]
    t=2.*torch.cross(qv,v,dim=-1)
    return v+q[:,:1]*t+torch.cross(qv,t,dim=-1)


def fake_env():
    n=4;identity=torch.tensor([[1.,0.,0.,0.]]).repeat(n,1)
    def state(pos,bodies):
        return SimpleNamespace(root_pos_w=pos,root_quat_w=identity.clone(),
            body_com_pos_w=pos[:,None,:].repeat(1,bodies,1),root_com_lin_vel_w=torch.zeros(n,3),
            root_ang_vel_w=torch.zeros(n,3),joint_vel=torch.tensor([[1.,-1.,0.,.3]]).repeat(n,1))
    robot=Asset(state(torch.tensor([[1.1,0.,.2],[.8,0.,.2],[3.,0.,.2],[1.1,0.,.2]]),1),1)
    cart=Asset(state(torch.zeros(n,3),5),5)
    robot.data.root_ang_vel_w[:,1]=.3
    cart.data.root_com_lin_vel_w[:,0]=.1
    return SimpleNamespace(num_envs=n,device='cpu',scene={'robot':robot,'trailer':cart,
        'trailer_contacts':SimpleNamespace(data=SimpleNamespace(net_forces_w=torch.tensor([[[0.,0.,80.]]*4]*n)))},
        cfg=SimpleNamespace(rolling_coefficient=.02),r1_towed_kg=torch.tensor([40.,40.,40.,0.]),
        r1_robot_hitch=torch.tensor([[-.1,0.,-.1]]).repeat(n,1),r1_cart_hitch=torch.tensor([[.1,0.,-.1]]).repeat(n,1),
        r1_base_id=0,r1_cart_base_id=0,r1_cart_masses=torch.tensor([[36.,1.,1.,1.,1.]]).repeat(n,1),
        r1_cart_sensor_ids=[0,1,2,3],r1_normal=torch.tensor([[0.,0.,1.]]).repeat(n,1),
        r1_cart_joint_ids=[0,1,2,3],r1_cart_wheel_ids=[1,2,3,4],r1_cart_axis=torch.tensor([[0.,1.,0.]]).repeat(n,1),
        r1_drawbar_guard=torch.zeros(n,dtype=torch.bool),r1_drawbar_peak=torch.zeros(n))


def original_apply():
    tree=ast.parse((NATIVE/'cart.py').read_text())
    fn=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='apply_cart_forces')
    scope=dict(torch=torch,quat_apply=quat_apply,RADIUS=.14,drawbar_force=native_force_function())
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(NATIVE/'cart.py'),'exec'),scope)
    return scope['apply_cart_forces']


def test_full_native_force_application_is_equal_at_default_and_candidate_keeps_gravity_rr_guards():
    old=fake_env();new=deepcopy(old);original_apply()(old);apply_cart_forces(new,DrawbarParameters(),quat_apply)
    for name in ['robot','trailer']:
        for field in ['applied_force','applied_torque']:
            assert torch.equal(getattr(old.scene[name],field),getattr(new.scene[name],field))
    for field in ['r1_drawbar_length','r1_drawbar_force','r1_drawbar_guard','r1_drawbar_peak','r1_rolling_torque']:
        assert torch.equal(getattr(old,field),getattr(new,field))
    candidate=fake_env();before=deepcopy(candidate);p=DrawbarParameters(500.,75.)
    predicted=estimate_robot_force(candidate,p,quat_apply)
    apply_cart_forces(candidate,p,quat_apply)
    assert torch.equal(predicted,candidate.scene['robot'].applied_force[:,0])
    gravity=torch.zeros(4,5,3);gravity[:,:,2]=-9.81*candidate.r1_cart_masses*(candidate.r1_towed_kg>0)[:,None]
    residual=candidate.scene['trailer'].applied_force-gravity
    torch.testing.assert_close(residual.sum(1)+predicted,torch.zeros_like(predicted),atol=1.e-4,rtol=0)
    assert torch.equal(residual[:,1:],torch.zeros_like(residual[:,1:]))
    assert torch.equal(candidate.r1_rolling_torque,old.r1_rolling_torque)
    assert not candidate.scene['robot'].applied_force[-1].any()
    assert not candidate.scene['trailer'].applied_force[-1].any()
    assert not candidate.scene['trailer'].applied_torque[-1].any()
    for name in ['robot','trailer']:
        for key,val in vars(before.scene[name].data).items(): assert torch.equal(val,getattr(candidate.scene[name].data,key))
    extreme=fake_env();extreme.scene['robot'].data.root_pos_w[0,0]=5.
    apply_cart_forces(extreme,p,quat_apply)
    assert extreme.r1_drawbar_guard[0]
    assert torch.linalg.vector_norm(extreme.scene['robot'].applied_force[0]) == pytest.approx(1500.,abs=2.e-4)
    assert extreme.r1_drawbar_force[0] > 1500.


def test_variant_audit_does_not_remove_failed_behavior_or_hold_checks():
    z=sample_arrays();p=DrawbarParameters()
    original=dict(independent_drawbar_errors={},checks={'robot_rolling':False,'no_termination':True},
        hold_2s_transition_5s={'pass':True},analysis_source_sha256='fixture',limitations='fixture')
    r=apply_variant_audit(original,z,p)
    assert r['checks']['independent_spring_and_moment']
    assert not r['checks']['robot_rolling'] and not r['move_brake_hold_pass']
    original['checks']['robot_rolling']=True;original['hold_2s_transition_5s']['pass']=False
    assert not apply_variant_audit(original,z,p)['move_brake_hold_pass']
    assert 'independent_spring_and_moment' not in original['checks']


def test_declared_parameters_cannot_disagree_with_actual_profile_or_change_guards(tmp_path):
    p=DrawbarParameters(500.,75.);metadata=p.metadata()
    req=dict(diagnostic_drawbar=metadata,physics_variant=metadata['variant'],initial_state_mode='ordinary',
        ordinary_live_entry=True,execution_physics={'drawbar':deepcopy(metadata)})
    assert declared_parameters(req)==p
    req['execution_physics']['drawbar']['damping_Nspm']=150.
    with pytest.raises(ValueError):declared_parameters(req)
    hashes=freeze_sources(tmp_path)
    assert len(hashes)==5
    assert all((tmp_path/'source_snapshot'/name).is_file() for name in hashes)
    target=tmp_path/'source_snapshot'/next(iter(hashes));target.write_text('changed')
    with pytest.raises(ValueError):freeze_sources(tmp_path)


def test_unchanged_old_dataset_rejects_changed_force_law_on_full_teacher_trace(tmp_path):
    """Synthetic force-law fixture, not a new physical trajectory or success label."""
    sys.path.insert(0,str(V7))
    from migration_v1.dataset import read_episode,EvidenceError
    from real_cart_audit import analyze
    run=V7/'results/2026-10-23_admission_continuation/ordinary_gain600_gpu'
    if not run.exists():pytest.skip('Full archived GPU fixture not installed')
    for path in run.iterdir():
        if path.name!='seed43_case00_physics_trace.npz':(tmp_path/path.name).symlink_to(path.resolve(),target_is_directory=path.is_dir())
    with np.load(run/'seed43_case00_physics_trace.npz') as data:z=dict(data)
    p=DrawbarParameters(500.,75.)
    args=[torch.tensor(z['real_cart_'+key].astype(float)) for key in ['robot_hitch_pre_w','cart_hitch_pre_w',
        'robot_hitch_velocity_pre_w','cart_hitch_velocity_pre_w']]
    force,_,tension=drawbar_force(*args,torch.ones(len(args[0]),dtype=torch.bool),p);force=force.numpy()
    z['real_cart_robot_force_w_N']=-force
    z['real_cart_robot_torque_w_Nm']=np.cross(z['real_cart_robot_arm_pre_w'],-force)
    z['real_cart_cart_forces_w_N']=z['real_cart_cart_forces_w_N'].copy()
    z['real_cart_cart_forces_w_N'][:,0]=force+np.array([0.,0.,-9.81*36.])
    z['real_cart_cart_torques_w_Nm']=z['real_cart_cart_torques_w_Nm'].copy()
    z['real_cart_cart_torques_w_Nm'][:,0]=np.cross(z['real_cart_cart_arm_pre_w'],force)
    z['drawbar_tension_interval_N']=tension.numpy()
    np.savez(tmp_path/'seed43_case00_physics_trace.npz',**z)
    audit=analyze(tmp_path,write_report=False)
    assert not audit['checks']['independent_spring_and_moment']
    assert all(value for key,value in audit['checks'].items() if key!='independent_spring_and_moment')
    with pytest.raises(EvidenceError,match='independent_spring_and_moment'):read_episode(tmp_path)
    # Exercise the separate audit on a synthetic unit-test fixture only. The
    # source GPU states did not execute this force law and are not new evidence.
    from compliant_cart_audit import analyze as analyze_variant
    request=json.loads((tmp_path/'request.json').read_text())
    (tmp_path/'request.json').unlink()
    (tmp_path/'source_snapshot').unlink()
    sources=freeze_sources(tmp_path)
    request.update(physics_variant=p.metadata()['variant'],diagnostic_drawbar=p.metadata(),
        diagnostic_drawbar_sources=sources,original_teacher_dataset_eligible=False)
    request['execution_physics']['drawbar']=p.metadata()
    (tmp_path/'request.json').write_text(json.dumps(request))
    with np.load(run/'seed43_case00_policy_trace.npz') as policy:frames=len(policy['executed_action'])
    runtime=dict(parameters=p.metadata(),sources=sources,physics_substeps=len(z['joint_pos']),
        live_estimator_frames=frames,initial_load_estimator_calls=1)
    (tmp_path/'drawbar_runtime.json').write_text(json.dumps(runtime))
    result=analyze_variant(tmp_path,write_report=False)
    assert result['legacy_original_force_audit_rejects']
    assert result['checks']['independent_spring_and_moment']
    assert not result['original_teacher_dataset_eligible']
    assert result['behavior_thresholds_changed'] is False
    runtime['live_estimator_frames']-=1
    (tmp_path/'drawbar_runtime.json').write_text(json.dumps(runtime))
    with pytest.raises(ValueError,match='call count'):analyze_variant(tmp_path,write_report=False)
