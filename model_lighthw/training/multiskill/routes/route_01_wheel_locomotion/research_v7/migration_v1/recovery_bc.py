"""New-schema offline recovery BC; original BC and its ten sources stay frozen."""
import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch

if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from migration_v1 import bc
from migration_v1.contract import CONTRACT, sha256
from migration_v1.dataset import EvidenceError
from migration_v1.recovery_dataset import (read_recovery_bundle, source_student_provenance,
    episode_identity, CONSTRAINED_STUDENT_SCHEMA)

SCHEMA = 'r1_migration_v1_executed_recovery_bc_candidate_v1'
ARTIFACT_KIND = 'offline_executed_recovery_fitted_student_candidate'


def source_signatures():
    # No real recovery artifact predates this explicit dual-parent schema support.
    # Constrained signatures recursively contain BC10, its three files and URDF.
    from migration_v1.constrained_refit import source_signatures as constrained_signatures
    result = dict(constrained_signatures())
    for name in ('recovery_dataset.py','recovery_bc.py','recovery_student.py','recovery_student_worker.py','runtime_checks.py'):
        path=Path(__file__).with_name(name);result[str(path.resolve())]=sha256(path)
    return result


def prior_training_identities(parent):
    episodes=parent['teacher_bundle']['episodes']
    if parent['schema'] == CONSTRAINED_STUDENT_SCHEMA:
        # The constrained ridge objective and first-frame equalities consumed
        # both complete nominal episodes, regardless of the older BC split.
        if len(episodes) != 2 or parent['fitting'].get('independent_holdout_available') is not False:
            raise EvidenceError('Invalid complete constrained-parent training provenance')
        trained = list(range(len(episodes)))
    elif parent['schema'] == bc.SCHEMA:
        trained=parent['fitting']['split']['train_episode_ids']
    else:
        raise EvidenceError('Unsupported warm-start ancestor training schema')
    if any(not isinstance(index,int) or index<0 or index>=len(episodes) for index in trained):
        raise EvidenceError('Invalid warm-start ancestor training episode provenance')
    return sorted({episode_identity(episodes[index]) for index in trained})


def validate_holdout_lineage(arrays,bundle,parent,validation_episode_ids):
    train,held,split=bc.split_episodes(arrays,validation_episode_ids)
    previous=set(prior_training_identities(parent))
    for index in split['validation_episode_ids']:
        if episode_identity(bundle['episodes'][index]) in previous:
            raise EvidenceError('Holdout episode was already trained by the warm-start ancestor')
    cumulative=previous|{episode_identity(bundle['episodes'][index]) for index in split['train_episode_ids']}
    return dict(ancestor_training_episode_sha256=sorted(previous),cumulative_training_episode_sha256=sorted(cumulative),
        validation_kind=split['validation_kind'],whole_episode_split=True,neighboring_frames_split=False,
        ancestor_seen_episodes_excluded_from_independent_holdout=True)


def fit_from_recovery(nominal_runs,recovery_runs,warm_start_student,output,*,steps=2000,batch_size=256,
                      learning_rate=3.e-4,seed=43,validation_episode_ids=(),balance_preparation=True,device='cpu'):
    output=Path(output).resolve()
    if output.exists():raise EvidenceError('Use a fresh recovery student output directory')
    signatures=source_signatures()
    arrays,bundle=read_recovery_bundle(nominal_runs,recovery_runs,warm_start_student,balance_preparation=balance_preparation)
    interface,parent,provenance=source_student_provenance(warm_start_student)
    if provenance!=bundle['source_student']:raise EvidenceError('Warm-start student provenance changed while reading')
    lineage=validate_holdout_lineage(arrays,bundle,parent,validation_episode_ids)
    if source_signatures()!=signatures:raise EvidenceError('Recovery fitting source changed before optimization')
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    # The parent residual is retained; only a fresh Adam for residual parameters is made.
    actor,fitting=bc._fit_arrays(interface.actor,arrays,steps=steps,batch_size=batch_size,learning_rate=learning_rate,
        seed=seed,validation_episode_ids=validation_episode_ids,device=device)
    train,_,_=bc.split_episodes(arrays,validation_episode_ids)
    active=np.unique(arrays['group_id'][train])
    fitting.update(sampling_configuration=dict(algorithm=bundle['sampling'],groups=bundle['groups'],
        balance_preparation=bool(balance_preparation),training_group_ids=active.tolist(),probability_per_training_group=1./len(active),
        actual_training_group_counts={str(group):int(np.sum(arrays['group_id'][train]==group)) for group in active},
        actor_inputs_changed=False,action_labels_changed=False,prefix_labels_used=False,shadow_labels_used=False),
        warm_start_student=provenance,optimizer_restored=False,validation_lineage=lineage)
    if source_signatures()!=signatures:raise EvidenceError('Recovery fitting source changed during optimization')
    for episode in bundle['episodes']:
        if any(sha256(Path(episode['directory'])/name)!=digest for name,digest in episode['source_files'].items()):
            raise EvidenceError('Recovery training source changed during optimization')
    if sha256(provenance['path'])!=provenance['checkpoint_sha256'] or sha256(Path(provenance['path']).with_name('manifest.json'))!=provenance['manifest_sha256']:
        raise EvidenceError('Warm-start artifact changed during optimization')
    metadata=dict(schema=SCHEMA,artifact_kind=ARTIFACT_KIND,architecture=actor.get_extra_state(),contract=CONTRACT,
        baseline=parent['baseline'],teacher_bundle=bundle['teacher_bundle'],recovery_bundle=bundle,
        source_student=provenance,fitting=fitting,sources=signatures,policy_dim=300,action_dim=16,
        deployment_inputs_only=True,teacher_runtime_required=False,critic_in_artifact=False,
        autonomous_behavior_accepted=False,formal_ppo_authorized=False)
    metadata=json.loads(json.dumps(metadata,allow_nan=False));actor.cpu().eval().assert_invariants()
    output.mkdir(parents=True,exist_ok=False)
    torch.save(dict(schema=SCHEMA,actor_state_dict=actor.state_dict(),metadata=metadata),output/'student.pt')
    metadata['student_checkpoint_sha256']=sha256(output/'student.pt')
    (output/'manifest.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    return metadata


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nominal-runs',nargs='+',required=True,type=Path)
    parser.add_argument('--recovery-runs',nargs='+',required=True,type=Path)
    parser.add_argument('--warm-start-student',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--steps',type=int,default=2000)
    parser.add_argument('--batch-size',type=int,default=256)
    parser.add_argument('--learning-rate',type=float,default=3.e-4)
    parser.add_argument('--seed',type=int,default=43)
    parser.add_argument('--validation-episode-ids',nargs='*',type=int,default=[])
    parser.add_argument('--no-balance-preparation',action='store_true')
    parser.add_argument('--device',default='cpu')
    args=parser.parse_args(argv)
    try:
        result=fit_from_recovery(args.nominal_runs,args.recovery_runs,args.warm_start_student,args.output,
            steps=args.steps,batch_size=args.batch_size,learning_rate=args.learning_rate,seed=args.seed,
            validation_episode_ids=args.validation_episode_ids,balance_preparation=not args.no_balance_preparation,device=args.device)
    except (ValueError,KeyError,OSError,TypeError,RuntimeError) as error:
        parser.exit(2,'Executed recovery BC rejected: '+str(error)+'\n')
    print(json.dumps(dict(candidate=str(args.output/'student.pt'),schema=SCHEMA,
        final_group_mse=result['fitting']['final_train']['equal_group_mean'],autonomous_behavior_accepted=False,formal_ppo_authorized=False)))


if __name__=='__main__':main()
