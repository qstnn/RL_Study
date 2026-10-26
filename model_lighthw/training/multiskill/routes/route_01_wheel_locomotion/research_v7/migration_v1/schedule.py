"""纯CPU工况分配：首批铺满实际许可64种，余槽确定性加权；不创建环境。"""
from collections import Counter
import math

import numpy as np

from .contract import SOURCE, V7, _load, sha256


SPEC = SOURCE/'wheel_direct_v8/spec.py'
FULL_PROTOCOL = V7.parent/'direct_rl_v8/scripts/protocol.py'
GROUPS = ('unloaded', 'payload_only', 'towing_only')


def _group(case):
    return 'towing_only' if case[2] > 0 else ('payload_only' if case[1] > 0 else 'unloaded')


def permitted_catalog():
    """交叉读取训练声明与真实full验收列表，不从公共路线catalog猜工况。"""
    spec = _load(SPEC, '_r1_migration_schedule_spec')
    protocol = _load(FULL_PROTOCOL, '_r1_migration_schedule_protocol')
    c = spec.CONTRACT
    if c.get('excluded_conditions') != [{'slope_deg': -25., 'towed_kg_gt': 0.}] or not c.get('payload_towing_mutually_exclusive'):
        raise ValueError('Permitted R1 exclusion/mutual-exclusivity contract changed')
    declared = {(float(s), float(p), float(t)) for s in spec.ANGLES for p in spec.PAYLOADS for t in spec.TOWED_MASSES
                if not (p > 0 and t > 0) and not (s == -25. and t > 0)}
    full = protocol.cases(suite='full')
    observed = Counter((float(row['slope_deg']), float(row['payload_kg']), float(row['towed_kg'])) for row in full)
    if len(declared) != 64 or c.get('training_condition_count') != 64 or set(observed) != declared or set(observed.values()) != {2}:
        raise ValueError('Training declaration and real 128-case protocol do not define the same 64 conditions')
    for case in declared:
        speeds = {float(row['speed']) for row in full if (row['slope_deg'], row['payload_kg'], row['towed_kg']) == case}
        if speeds != {0., 3.1 if _group(case) == 'unloaded' else .3}:
            raise ValueError('Full protocol no longer pairs each condition with static/moving cases')
    weights = {name: float(c['condition_class_weights'][name]) for name in GROUPS}
    if any(not math.isfinite(value) or value <= 0 for value in weights.values()) or not math.isclose(sum(weights.values()), 1.):
        raise ValueError('Condition weights must be finite, positive and sum to one')
    anchors = [(float(row['slope_deg']), 0., float(row['towed_kg'])) for row in c['hard_tow_anchors']]
    if set(anchors) != {(25., 0., 40.), (-20., 0., 40.)} or not set(anchors) <= declared:
        raise ValueError('The existing uphill/downhill heavy towing anchors changed')
    anchor_fraction = float(c['target_condition_fraction'])/weights['towing_only']
    if not 0 <= anchor_fraction <= 1:
        raise ValueError('Invalid remaining towing-slot anchor fraction')
    return dict(cases=sorted(declared), class_weights=weights, towing_anchors=anchors,
        extra_towing_anchor_fraction=anchor_fraction, source_sha256={str(SPEC): sha256(SPEC), str(FULL_PROTOCOL): sha256(FULL_PROTOCOL)})


def _counts(total, weights):
    ideal = np.asarray(weights, dtype=float)*total
    result = np.floor(ideal).astype(int)
    remainder = total-int(result.sum())
    order = sorted(range(len(result)), key=lambda index: (-(ideal[index]-result[index]), index))
    for index in order[:remainder]:
        result[index] += 1
    return result.tolist()


def make_schedule(num_envs, *, mode='full_first_batch'):
    """返回配置表，不是已执行rollout；少于64仅能显式diagnostic模式。"""
    if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs < 1:
        raise ValueError('num_envs must be a positive integer')
    if mode not in ('full_first_batch', 'diagnostic'):
        raise ValueError('Unknown schedule mode')
    catalog = permitted_catalog()
    cases = catalog['cases']
    if mode == 'full_first_batch' and num_envs < len(cases):
        raise ValueError('Full first-batch mode requires at least 64 slots; smaller smoke is diagnostic only')
    selected = list(cases[:num_envs])
    extra = max(0, num_envs-len(cases))
    quotas = _counts(extra, [catalog['class_weights'][name] for name in GROUPS])
    for name, count in zip(GROUPS, quotas):
        group = [case for case in cases if _group(case) == name]
        anchors = round(count*catalog['extra_towing_anchor_fraction']) if name == 'towing_only' else 0
        selected.extend(group[index % len(group)] for index in range(count-anchors))
        selected.extend(catalog['towing_anchors'][index % len(catalog['towing_anchors'])] for index in range(anchors))
    counts = Counter(_group(case) for case in selected)
    return dict(version='r1_migration_v1_all_conditions_before_weighting', mode=mode,
        conditions=[list(case) for case in selected], num_envs=num_envs,
        unique_conditions=len(set(selected)), class_counts={name: counts[name] for name in GROUPS},
        extra_class_quotas=dict(zip(GROUPS, quotas)), catalog_source_sha256=catalog['source_sha256'],
        configured_full_coverage=set(selected) == set(cases), runtime_pre_action_verified=False,
        diagnostic_only=mode == 'diagnostic', behavior_accepted=False, formal_ppo_authorized=False)


def validate_pre_action_conditions(metadata, *, mode='full_first_batch'):
    """未来worker必须传真实pre-action N×4(slope,payload,tow,phase)，不得传配表冒充。

    这里只验证调用者提供的张量；来源真实执行需由未来运行记录/源签名证明。
    即使集合正确，也不授权PPO或宣称行为通过。
    """
    if mode not in ('full_first_batch', 'diagnostic'):
        raise ValueError('Unknown schedule mode')
    if hasattr(metadata, 'detach'):
        metadata = metadata.detach().cpu().numpy()
    values = np.asarray(metadata)
    if values.ndim != 2 or values.shape[1] != 4 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Expected finite pre-action condition/phase metadata with shape [N,4]')
    rounded = np.rint(values)
    if np.max(abs(values-rounded)) > 1.e-4 or ((rounded[:, 3] < 0) | (rounded[:, 3] > 3)).any():
        raise ValueError('Pre-action condition/phase values do not match declared units or phases')
    actual = {tuple(float(value) for value in row) for row in rounded[:, :3]}
    expected = set(permitted_catalog()['cases'])
    extra = actual-expected
    if extra:
        raise ValueError('Pre-action metadata contains forbidden conditions: '+repr(sorted(extra)))
    missing = expected-actual
    if mode == 'full_first_batch' and missing:
        raise ValueError('First pre-action batch is missing permitted conditions: '+repr(sorted(missing)))
    return dict(mode=mode, rows=len(values), unique_conditions=len(actual), missing_conditions=sorted(missing),
        supplied_metadata_covers_all_64=not missing, diagnostic_only=mode == 'diagnostic',
        input_kind='pre_action_metadata_supplied_by_caller', execution_provenance_verified=False,
        behavior_accepted=False, formal_ppo_authorized=False)
