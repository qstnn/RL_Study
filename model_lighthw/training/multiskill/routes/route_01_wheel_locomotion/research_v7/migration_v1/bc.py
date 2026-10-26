"""隔离离线BC入口：必须先重审真实双向完整教师；不连接仿真/PPO。

仅拟合冻结base的拖载残差。CLI在证据不足时创建输出前拒绝；CPU单元测试
可以独立调用纯数组优化核，但合成测试不是机器人教师或行为成功证据。
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch import nn

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from migration_v1.contract import CONTRACT, SPEC_PATH, POSITION_SOURCE, _load, sha256
from migration_v1.dataset import EvidenceError, read_teacher_bundle
from migration_v1.policy import FrozenBaseTowingResidualActor, tensor_fingerprint
from migration_v1.preparation_sampling import prepare_sampling


SCHEMA = 'r1_migration_v1_offline_bc_candidate'


def _mlp():
    return nn.Sequential(nn.Linear(300, 512), nn.ELU(), nn.Linear(512, 256), nn.ELU(),
                         nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 16))


def load_baseline(path, expected_sha256):
    """只接受真实教师出处指定SHA，校验具名合同；不恢复旧Adam或Critic。"""
    path = Path(path).resolve(strict=True)
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64 or sha256(path) != expected_sha256:
        raise EvidenceError('Baseline checkpoint SHA differs from verified teacher provenance')
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    declared = checkpoint.get('infos', {}).get('r1_unified_contract', {})
    original = _load(SPEC_PATH, '_r1_migration_bc_base_contract').CONTRACT
    fields = ('policy_dim', 'action_dim', 'history_frames', 'history_layout', 'ignored_policy_columns',
              'action_order', 'leg_scale', 'leg_mapping', 'leg_pd', 'wheel_scale', 'wheel_clip',
              'wheel_joint_velocity_limit_rad_s', 'wheel_damping', 'physics_dt', 'policy_dt',
              'empirical_normalization', 'urdf_sha256')
    for field in fields:
        if declared.get(field) != original[field]:
            raise EvidenceError('Baseline deployed semantic mismatch: '+field)
    if list(declared.get('policy_terms', {}).items()) != list(CONTRACT['policy_terms'].items()):
        raise EvidenceError('Baseline observation term order differs')
    if declared.get('version') not in ('lighthw_r1_unified_v4', 'lighthw_r1_unified_v5',
                                      'lighthw_r1_unified_v6', 'lighthw_r1_direct_v8'):
        raise EvidenceError('Undeclared baseline route/version')
    state = {key.removeprefix('actor.'): value for key, value in checkpoint['model_state_dict'].items() if key.startswith('actor.')}
    base = _mlp()
    base.load_state_dict(state, strict=True)
    actor = FrozenBaseTowingResidualActor(base)
    provenance = dict(path=str(path), checkpoint_sha256=expected_sha256,
        frozen_base_tensor_sha256=tensor_fingerprint(actor.base_actor),
        source_contract=declared, source_iteration=checkpoint.get('iter'), optimizer_restored=False)
    return actor, json.loads(json.dumps(provenance, allow_nan=False))


def split_episodes(arrays, validation_episode_ids=()):
    """只按完整episode拆分；训练保留上下坡，缺独立回合时明确不报告验证损失。"""
    episode = np.asarray(arrays['episode_id'])
    keys = np.asarray(arrays['case_key'])
    ids = set(int(value) for value in np.unique(episode))
    held = set(int(value) for value in validation_episode_ids)
    if not held <= ids:
        raise EvidenceError('Validation requested an unknown episode id')
    train = ~np.isin(episode, list(held))
    validation = ~train
    if not train.any() or set(keys[train]) != set(keys):
        raise EvidenceError('Whole-episode split removed a required uphill/downhill training case')
    return np.flatnonzero(train), np.flatnonzero(validation), dict(
        train_episode_ids=sorted(ids-held), validation_episode_ids=sorted(held),
        validation_kind='whole_independent_episodes' if held else 'unavailable_no_independent_holdout',
        neighboring_frames_split=False)


def balanced_indices(group_ids, eligible, count, generator):
    """先均匀抽case×stage，再均匀抽该组帧；长保持/易工况不淹没过渡段。"""
    groups = np.asarray(group_ids)
    eligible = np.asarray(eligible, dtype=np.int64)
    if count < 1 or not len(eligible):
        raise ValueError('Balanced sampling requires positive count and eligible rows')
    values = np.unique(groups[eligible])
    chosen = generator.choice(values, size=count)
    result = np.empty(count, dtype=np.int64)
    for value in values:
        slots = np.flatnonzero(chosen == value)
        members = eligible[groups[eligible] == value]
        result[slots] = generator.choice(members, size=len(slots))
    return result


@torch.no_grad()
def _loss_report(actor, observation, action, group_ids, indices, scales, batch_size=1024):
    if not len(indices):
        return None
    losses = []
    for begin in range(0, len(indices), batch_size):
        selected = torch.as_tensor(indices[begin:begin+batch_size], device=observation.device)
        losses.append(((actor(observation[selected])-action[selected])/scales).square().mean(-1).cpu().numpy())
    error = np.concatenate(losses)
    group_values = np.asarray(group_ids)[indices]
    means = {str(value): float(error[group_values == value].mean()) for value in np.unique(group_values)}
    return dict(frame_mean=float(error.mean()), equal_group_mean=float(np.mean(list(means.values()))), groups=means)


def _fit_arrays(actor, arrays, *, steps, batch_size, learning_rate, seed, validation_episode_ids=(),
                action_scales=(1.,)*16, device='cpu'):
    """数值核，不作物理准入决定。真实入口必须先调用read_teacher_bundle。

    合成测试允许执行极少优化步骤验证接口；其输出不能标成机器人训练结果。
    """
    if steps < 1 or batch_size < 1 or not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError('Invalid offline BC optimization parameters')
    x, y = np.asarray(arrays['observation']), np.asarray(arrays['action'])
    if x.ndim != 2 or x.shape[1] != 300 or y.shape != (len(x), 16) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise EvidenceError('Expected finite deployed observations and executed action labels')
    successful = np.asarray(arrays['successful'])
    if successful.dtype != np.bool_ or successful.shape != (len(x),) or not successful.all() or not np.all(x[:, 299] > 0):
        raise EvidenceError('BC only accepts successful towing labels')
    for key in ('episode_id', 'group_id', 'case_key', 'stage'):
        if np.asarray(arrays[key]).shape != (len(x),):
            raise EvidenceError('Episode/group metadata shape mismatch: '+key)
    train, held, split = split_episodes(arrays, validation_episode_ids)
    actor = actor.to(device).train()
    observation = torch.as_tensor(x, dtype=torch.float32, device=device)
    action = torch.as_tensor(y, dtype=torch.float32, device=device)
    scales = torch.as_tensor(action_scales, dtype=torch.float32, device=device)
    if scales.shape != (16,) or not torch.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError('Action loss scales must contain 16 positive finite values')
    generator = np.random.default_rng(seed)
    optimizer = torch.optim.Adam(actor.trainable_parameters(), lr=learning_rate)
    actor.assert_invariants(optimizer)
    initial = _loss_report(actor, observation, action, arrays['group_id'], train, scales)
    history = []
    for update in range(steps):
        rows = balanced_indices(arrays['group_id'], train, batch_size, generator)
        selected = torch.as_tensor(rows, device=device)
        prediction = actor(observation[selected])
        loss = ((prediction-action[selected])/scales).square().mean()
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite offline BC loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(list(actor.trainable_parameters()), 1., error_if_nonfinite=True)
        optimizer.step()
        actor.project_constraints(optimizer)
        if update == 0 or (update+1) % max(1, steps//20) == 0 or update+1 == steps:
            history.append(dict(update=update+1, batch_equal_group_mse=float(loss.detach()), gradient_norm=float(gradient)))
    actor.eval()
    return actor, dict(optimizer_steps=steps, batch_size=batch_size, learning_rate=learning_rate,
        action_scales=scales.cpu().tolist(), sampling='uniform_case_stage_then_uniform_frame', seed=seed,
        split=split, initial_train=initial, final_train=_loss_report(actor, observation, action, arrays['group_id'], train, scales),
        held_out=_loss_report(actor, observation, action, arrays['group_id'], held, scales), history=history,
        optimizer_saved=False, base_frozen=True, behavior_accepted=False, formal_ppo_authorized=False)


def source_signatures():
    files = [Path(__file__), Path(__file__).with_name('policy.py'), Path(__file__).with_name('contract.py'),
             Path(__file__).with_name('dataset.py'), Path(__file__).with_name('student.py'),
             Path(__file__).with_name('preparation_sampling.py'),
             Path(__file__).parent.parent/'scripts/real_cart_audit.py',
             Path(__file__).parent.parent/'scripts/rolling_load_audit.py', SPEC_PATH, POSITION_SOURCE]
    return {str(path.resolve()): sha256(path) for path in files}


def fit_from_runs(runs, baseline_checkpoint, output, *, steps=5000, batch_size=256,
                  learning_rate=3.e-4, seed=43, validation_episode_ids=(), device='cpu', balance_preparation=False):
    """唯一真实训练入口：先完整重审双向档案，再加载/拟合；不生成PPO授权。"""
    output = Path(output).resolve()
    if output.exists():
        raise EvidenceError('Use a fresh student output directory')
    signatures = source_signatures()
    arrays, teacher = read_teacher_bundle(runs)
    provenances = [json.loads((Path(ep['directory'])/'request.json').read_text()) for ep in teacher['episodes']]
    expected = {request['checkpoint_sha256'] for request in provenances}
    if len(expected) != 1:
        raise EvidenceError('Teacher provenance refers to different protected baselines')
    # 分组/参数错误在写目录和加载实际模型前拒绝。
    training_rows, _, _ = split_episodes(arrays, validation_episode_ids)
    sampled_arrays, sampling_configuration = prepare_sampling(arrays, teacher, enabled=balance_preparation)
    if balance_preparation:
        selected_groups=np.unique(sampled_arrays['group_id'][training_rows])
        selected_records=[dict(group_id=int(group),count=int(np.sum(sampled_arrays['group_id'][training_rows]==group)),
            case_key=str(arrays['case_key'][training_rows[np.flatnonzero(sampled_arrays['group_id'][training_rows]==group)[0]]]),
            probability=1./len(selected_groups)) for group in selected_groups]
        sampling_configuration['training_groups_after_episode_split']=selected_records
        sampling_configuration['training_case_probability']={case:sum(record['probability'] for record in selected_records if record['case_key']==case)
            for case in sorted(set(arrays['case_key'][training_rows]))}
    if steps < 1 or batch_size < 1 or not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError('Invalid offline BC optimization parameters')
    if source_signatures() != signatures:
        raise EvidenceError('Migration source changed during data reading/grouping')
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    actor, base = load_baseline(baseline_checkpoint, expected.pop())
    actor, fitting = _fit_arrays(actor, sampled_arrays, steps=steps, batch_size=batch_size, learning_rate=learning_rate,
        seed=seed, validation_episode_ids=validation_episode_ids, device=device)
    fitting['sampling_configuration'] = sampling_configuration
    if balance_preparation:
        train, held, _ = split_episodes(arrays, validation_episode_ids)
        observations=torch.as_tensor(arrays['observation'],dtype=torch.float32,device=device)
        actions=torch.as_tensor(arrays['action'],dtype=torch.float32,device=device)
        scales=torch.as_tensor(fitting['action_scales'],dtype=torch.float32,device=device)
        fitting['original_stage_final_train'] = _loss_report(actor, observations, actions, arrays['group_id'], train, scales)
        fitting['original_stage_held_out'] = _loss_report(actor, observations, actions, arrays['group_id'], held, scales)
    if source_signatures() != signatures:
        raise EvidenceError('Migration source changed during fitting; candidate not written')
    # 每份原始档案都保持读取时SHA；不允许训练期间来源被覆盖。
    for episode in teacher['episodes']:
        for name, expected_sha in episode['source_files'].items():
            if sha256(Path(episode['directory'])/name) != expected_sha:
                raise EvidenceError('Teacher source changed during fitting: '+name)
    metadata = dict(schema=SCHEMA, artifact_kind='offline_fitted_student_candidate',
        architecture=actor.get_extra_state(), contract=CONTRACT, baseline=base,
        teacher_bundle=teacher, fitting=fitting, sources=signatures,
        deployment_inputs_only=True, policy_dim=300, action_dim=16, critic_in_artifact=False,
        teacher_runtime_required=False, autonomous_behavior_accepted=False, formal_ppo_authorized=False)
    metadata = json.loads(json.dumps(metadata, allow_nan=False))
    output.mkdir(parents=True, exist_ok=False)
    actor.cpu().assert_invariants()
    torch.save(dict(schema=SCHEMA, actor_state_dict=actor.state_dict(), metadata=metadata), output/'student.pt')
    metadata['student_checkpoint_sha256'] = sha256(output/'student.pt')
    (output/'manifest.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--baseline-checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=5000)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--learning-rate', type=float, default=3.e-4)
    parser.add_argument('--seed', type=int, default=43)
    parser.add_argument('--validation-episode-ids', nargs='*', type=int, default=[])
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--balance-preparation', action='store_true',
        help='Offline-only split of recorded entry/capture/prepared-wait; default retains original sampling')
    args = parser.parse_args()
    try:
        result = fit_from_runs(args.runs, args.baseline_checkpoint, args.output, steps=args.steps,
            batch_size=args.batch_size, learning_rate=args.learning_rate, seed=args.seed,
            validation_episode_ids=args.validation_episode_ids, device=args.device,balance_preparation=args.balance_preparation)
    except (ValueError, KeyError, OSError, TypeError, RuntimeError) as error:
        parser.exit(2, 'Offline student fit rejected: '+str(error)+'\n')
    print(json.dumps(dict(candidate=str(args.output/'student.pt'),
        final_group_mse=result['fitting']['final_train']['equal_group_mean'],
        autonomous_behavior_accepted=False, formal_ppo_authorized=False)))


if __name__ == '__main__':
    main()
