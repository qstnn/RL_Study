"""Isolated final-layer constrained ridge fit; no PPO or environment connection.

The two nominal teacher first frames are offline equality constraints. The
deployment graph remains the original observation-only frozen-base residual
Actor; no runtime first-frame action replacement exists.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
from scipy.linalg import solve_triangular
import torch

if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from migration_v1.bc import source_signatures as parent_signatures
from migration_v1.contract import CONTRACT, REPO, masked_observations, sha256
from migration_v1.dataset import EvidenceError, read_teacher_bundle
from migration_v1.policy import tensor_fingerprint
from migration_v1.preparation_sampling import prepare_sampling
from migration_v1.student import load_student as load_parent

SCHEMA = 'r1_migration_v1_constrained_head_candidate_v1'
HEAD_KEYS = {'residual_actor.6.weight','residual_actor.6.bias'}


def source_signatures():
    result = parent_signatures()
    for name in ('constrained_refit.py','constrained_student.py','constrained_student_worker.py'):
        path = Path(__file__).with_name(name).resolve()
        result[str(path)] = sha256(path)
    robot = REPO/'model_lighthw/lighthw_urdf/urdf/lighthw.urdf'
    result[str(robot.resolve())] = sha256(robot)
    return result


def _mapped(raw):
    robot = REPO/'model_lighthw/lighthw_urdf/urdf/lighthw.urdf'
    joints = {joint.get('name'):joint for joint in ET.parse(robot).getroot().findall('joint')}
    limits = np.array([[float(joints[name].find('limit').get(key)) for key in ('lower','upper')]
                       for name in CONTRACT['action_order'][:12]],dtype=raw.dtype)
    default = np.array([0.,.6,-1.5,0.,.6,-1.5,0.,-.6,1.5,0.,-.6,1.5],dtype=raw.dtype)
    return (np.clip(default+raw[:,:12]*np.array([.08,.14,.14]*4,dtype=raw.dtype),limits[:,0],limits[:,1]),
            np.clip(raw[:,12:]*5.,-50.,50.))


def _rmse(value):
    return float(np.sqrt(np.mean(np.asarray(value,dtype=float)**2)))


@torch.inference_mode()
def fit_last_layer(parent, arrays, *, ridge=.01):
    """Pure numerical kernel; callers must separately verify real teacher provenance.

    CPU tests may use artificial arrays. No tensor of the parent is modified;
    only the returned copy's residual final linear weight/bias may differ.
    """
    if not np.isfinite(ridge) or ridge <= 0.:
        raise ValueError('Constrained ridge must be finite and positive')
    parent.assert_invariants()
    x = np.asarray(arrays['observation']); y = np.asarray(arrays['action'])
    group = np.asarray(arrays['group_id']); episode = np.asarray(arrays['episode_id'])
    successful = np.asarray(arrays['successful'])
    if (x.ndim != 2 or x.shape[1] != 300 or len(x) < 4 or y.shape != (len(x),16)
            or x.dtype != np.float32 or y.dtype != np.float32
            or not np.isfinite(x).all() or not np.isfinite(y).all()
            or group.shape != (len(x),) or episode.shape != (len(x),)
            or not np.issubdtype(group.dtype,np.integer) or not np.issubdtype(episode.dtype,np.integer)
            or set(np.unique(episode)) != {0,1} or successful.dtype != np.bool_
            or successful.shape != (len(x),) or not successful.all() or not np.all(x[:,299]>0.)):
        raise EvidenceError('Expected two complete finite deployed towing episodes with executed successful labels')
    before = (tensor_fingerprint(parent.base_actor),tensor_fingerprint(parent.residual_actor))
    candidate = deepcopy(parent).cpu().eval()
    visible = torch.from_numpy(masked_observations(x))
    feature = torch.cat([candidate.residual_actor[:-1](row[None]) for row in visible])
    original = torch.cat([candidate(row[None]) for row in torch.from_numpy(x)]).numpy()
    H = np.column_stack((feature.numpy().astype(float),np.ones(len(x))))
    E = y.astype(float)-original.astype(float)
    ids = np.unique(group); weights = np.zeros(len(x))
    for value in ids:
        selected = group == value
        weights[selected] = 1./len(ids)/selected.sum()
    U = H*np.sqrt(weights[:,None]); Y = E*np.sqrt(weights[:,None])
    anchors = np.array([np.flatnonzero(episode == value)[0] for value in (0,1)])
    A,e = H[anchors],E[anchors]
    Q = U.T@U+ridge*np.eye(H.shape[1]); L = np.linalg.cholesky(Q)
    unconstrained = solve_triangular(L.T,solve_triangular(L,U.T@Y,lower=True),lower=False)
    Aw = solve_triangular(L,A.T,lower=True).T
    left,s,right = np.linalg.svd(Aw,full_matrices=False)
    if len(s) != 2 or s[-1] <= s[0]*max(Aw.shape)*np.finfo(np.float32).eps:
        raise EvidenceError('Two first-frame constraints are not independent at float32 precision')
    z = (right.T/s)@(left.T@(e-A@unconstrained))
    delta = unconstrained+solve_triangular(L.T,z,lower=False)
    head = candidate.residual_actor[-1]
    old_head = np.row_stack((head.weight.numpy().T,head.bias.numpy()[None])).astype(float)
    modified = (old_head+delta).astype(np.float32)
    if not np.isfinite(modified).all():
        raise EvidenceError('Nonfinite float32 constrained head')
    head.weight.copy_(torch.from_numpy(modified[:-1].T.copy())); head.bias.copy_(torch.from_numpy(modified[-1].copy()))
    predicted = torch.cat([candidate(row[None]) for row in torch.from_numpy(x)]).numpy()
    old_leg,old_wheel = _mapped(original); new_leg,new_wheel = _mapped(predicted); target_leg,target_wheel = _mapped(y)
    groups = []
    for value in ids:
        selected = group == value
        groups.append(dict(group_id=int(value),frames=int(selected.sum()),
            before_raw_mse=float(np.mean((original[selected]-y[selected])**2)),
            after_raw_mse=float(np.mean((predicted[selected]-y[selected])**2)),
            before_leg_rmse_rad=_rmse(old_leg[selected]-target_leg[selected]),
            after_leg_rmse_rad=_rmse(new_leg[selected]-target_leg[selected]),
            before_wheel_rmse_rad_s=_rmse(old_wheel[selected]-target_wheel[selected]),
            after_wheel_rmse_rad_s=_rmse(new_wheel[selected]-target_wheel[selected])))
    checks = {name:all(g['after_'+metric] <= g['before_'+metric]+1.e-10 for g in groups)
              for name,metric in [('all_group_raw_mse_nonworsening','raw_mse'),
                                  ('all_group_leg_rmse_nonworsening','leg_rmse_rad'),
                                  ('all_group_wheel_rmse_nonworsening','wheel_rmse_rad_s')]}
    anchor_error = float(abs(predicted[anchors]-y[anchors]).max())
    checks['float32_anchor_raw_residual'] = anchor_error <= 1.e-4
    state,parent_state = candidate.state_dict(),parent.state_dict()
    if any(not torch.equal(value.cpu(),parent_state[key].cpu()) for key,value in state.items()
           if key not in HEAD_KEYS and isinstance(value,torch.Tensor)):
        raise AssertionError('Constrained fit changed more than the residual last linear layer')
    if before != (tensor_fingerprint(parent.base_actor),tensor_fingerprint(parent.residual_actor)):
        raise AssertionError('Constrained fit mutated the parent')
    candidate.assert_invariants()
    report = dict(method='equal_case_stage_teacher_mse_plus_parent_head_ridge_with_two_first_frame_equalities',
        objective_normalization='mean over groups; mean over frames within group; sum of squared residual over 16 raw actions',
        ridge=float(ridge),rows=len(x),group_count=len(ids),anchor_indices=anchors.tolist(),
        optimized_parameters=sorted(HEAD_KEYS),gradient_updates=0,optimizer_steps=0,
        correction_norm=float(np.linalg.norm(delta)),correction_to_parent_head_ratio=(
            float(np.linalg.norm(delta)/np.linalg.norm(old_head)) if np.linalg.norm(old_head)>0. else None),
        objective_condition=float(np.linalg.cond(Q)),constraint_singular_values=s.tolist(),
        constraint_condition=float(s[0]/s[-1]),float64_constraint_raw_max=float(abs(A@delta-e).max()),
        float32_anchor_raw_max=anchor_error,float32_anchor_leg_max_rad=float(abs(new_leg[anchors]-target_leg[anchors]).max()),
        float32_anchor_wheel_max_rad_s=float(abs(new_wheel[anchors]-target_wheel[anchors]).max()),
        float32_vs_linear_max=float(abs(predicted-(original.astype(float)+H@delta)).max()),groups=groups,checks=checks,
        offline_fit_checks_pass=all(checks.values()),pointwise_error_nonworsening_not_claimed=True,
        independent_holdout_available=False,teacher_runtime_required=False,autonomous_behavior_accepted=False,
        formal_ppo_authorized=False)
    return candidate,report


def fit_from_checkpoint(parent_checkpoint,output,*,ridge=.01):
    """Explicit offline artifact creation. Root/user invokes this; import does no work."""
    output = Path(output).resolve()
    if output.exists():
        raise EvidenceError('Use a fresh constrained-student output directory')
    signatures = source_signatures()
    parent_checkpoint = Path(parent_checkpoint).resolve(strict=True)
    parent_sha = sha256(parent_checkpoint); manifest_sha = sha256(parent_checkpoint.with_name('manifest.json'))
    interface,parent_manifest = load_parent(parent_checkpoint,device='cpu',inference_mode='row')
    arrays,bundle = read_teacher_bundle([ep['directory'] for ep in parent_manifest['teacher_bundle']['episodes']])
    if bundle != parent_manifest['teacher_bundle'] or len(bundle['episodes']) != 2:
        raise EvidenceError('Constrained nominal fit requires the exact two complete parent teacher episodes')
    grouped,sampling = prepare_sampling(arrays,bundle,enabled=True)
    candidate,fitting = fit_last_layer(interface.actor,grouped,ridge=ridge)
    if not fitting['offline_fit_checks_pass']:
        raise EvidenceError('Constrained fit worsened a protected teacher group or failed float32 first-frame reconstruction')
    if signatures != source_signatures() or sha256(parent_checkpoint) != parent_sha or sha256(parent_checkpoint.with_name('manifest.json')) != manifest_sha:
        raise EvidenceError('Constrained fit source or parent changed while computing')
    for episode in bundle['episodes']:
        for name,digest in episode['source_files'].items():
            if sha256(Path(episode['directory'])/name) != digest:
                raise EvidenceError('Complete teacher evidence changed during constrained fitting')
    metadata = dict(schema=SCHEMA,artifact_kind='offline_constrained_student_candidate',
        parent_checkpoint=dict(path=str(parent_checkpoint),sha256=parent_sha,manifest_sha256=manifest_sha),
        parent_schema=parent_manifest['schema'],architecture=candidate.get_extra_state(),contract=CONTRACT,
        baseline=parent_manifest['baseline'],teacher_bundle=bundle,sampling=sampling,fitting=fitting,sources=signatures,
        policy_dim=300,action_dim=16,critic_in_artifact=False,teacher_runtime_required=False,
        deployment_inputs_only=True,runtime_first_frame_override=False,
        autonomous_behavior_accepted=False,formal_ppo_authorized=False)
    metadata = json.loads(json.dumps(metadata,allow_nan=False))
    output.mkdir(parents=True,exist_ok=False)
    torch.save(dict(schema=SCHEMA,metadata=metadata,actor_state_dict=candidate.state_dict()),output/'student.pt')
    saved = dict(metadata,student_checkpoint_sha256=sha256(output/'student.pt'))
    (output/'manifest.json').write_text(json.dumps(saved,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    return saved


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--ridge',type=float,default=.01)
    args = parser.parse_args(argv)
    try:
        result = fit_from_checkpoint(args.parent_checkpoint,args.output,ridge=args.ridge)
    except (ValueError,KeyError,OSError,TypeError,RuntimeError) as error:
        parser.exit(2,'Constrained fit rejected: '+str(error)+'\n')
    print(json.dumps(dict(candidate=str(args.output/'student.pt'),fitting=result['fitting'],formal_ppo_authorized=False)))


if __name__=='__main__':
    main()
