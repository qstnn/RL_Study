"""Route16 有界正式续训的来源/状态保护；纯 CPU 可复核，不改变 PPO 算法。"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import torch

REPO = Path(__file__).resolve().parents[3]
ROUTE = REPO/'model_lighthw/training/routes/route_16_ppo_observation_reference'
LOG_ROOT = ROUTE/'logs/rsl_rl/lighthw_ppo_observation_reference'
SOURCE_RUN = LOG_ROOT/'2026-09-23_16-08-45_lighthw_reference_obs_v1'
SOURCE = SOURCE_RUN/'model_999.pt'
SOURCE_SHA256 = '4336cb0122b3721a2e2113afae201cdd0131af108f4604fe2f44d52256188176'
SOURCE_CONFIG_SHA256 = {
    'env.yaml': '2ed045e78ab258e8bcb999bfb93bee1eb6bffa867f906f63246b7da18f5c2e58',
    'agent.yaml': 'f4c7ff5f8f48babf14baedafb74b9e258d2bf20d0ee59db49dbc266e464b6987',
}
BASELINE_EVALUATION = SOURCE_RUN/'evaluations/2026-09-23_16-39-01_087640'
TASK = 'Flat-Deeprobotics-LightHW-PPOReference-v0'
OBSERVATION_VERSION = 'lighthw_named_joint_positions_v1'
CONTINUATION_VERSION = 'lighthw_reference_continue300_v1'
EVALUATION_VERSION = 'lighthw_phase_start_stop_v2'
INFO_KEY = 'lighthw_reference_continuation'
SOURCE_UPDATES = 1000
ADDITIONAL_UPDATES = 300
SOURCE_ADAM_STEP = 20000
MINIBATCHES_PER_UPDATE = 20


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tree_digest(value):
    """逐张量/标量哈希，包含 Adam step/矩和 LR，而不是仅比较网络权重。"""
    digest = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            digest.update(str((str(item.dtype), tuple(item.shape))).encode())
            digest.update(item.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                digest.update(repr(key).encode()); visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode())
            for child in item: visit(child)
        else: digest.update(repr(item).encode())
    visit(value)
    return digest.hexdigest()


def adam_steps(saved):
    states = saved['optimizer_state_dict']['state']
    if not states: raise ValueError('Full continuation requires nonempty Adam state')
    return sorted({int(state['step']) for state in states.values()})


def validate_source(path=SOURCE):
    """无 validation 后门：短验与正式入口都必须读取同一个原始正式 model999。"""
    path = Path(path).resolve()
    if path != SOURCE.resolve() or sha256(path) != SOURCE_SHA256:
        raise ValueError('Continuation source must be the pinned original formal model_999.pt with its exact SHA256')
    for name, expected in SOURCE_CONFIG_SHA256.items():
        if sha256(SOURCE_RUN/'params'/name) != expected: raise ValueError('Source saved configuration changed: '+name)
    record = json.loads((SOURCE_RUN/'reference_run.json').read_text())
    if any(record.get(key) != value for key, value in dict(task=TASK, role='formal', fresh_start=True,
            status='completed', planned_updates=1000, num_envs=1024, seed=42, save_interval=100,
            observation_version=OBSERVATION_VERSION).items()):
        raise ValueError('Continuation requires the completed original formal run, not a validation run')
    saved = torch.load(path, map_location='cpu', weights_only=False)
    info = saved['infos']['lighthw_observation_reference']
    curriculum = saved['infos']['lighthw_action_curriculum']
    if (saved['iter'] != 999 or curriculum['schedule']['total_updates'] != SOURCE_UPDATES
            or curriculum['control_steps'] != 24000 or adam_steps(saved) != [SOURCE_ADAM_STEP]
            or info['run_role'] != 'formal' or info['origin'] != 'from_scratch'
            or info['resumed_from'] is not None or info['source_checkpoint'] is not None
            or info['observation_contract']['version'] != OBSERVATION_VERSION
            or info['empirical_normalization'] is not False
            or info['actor_normalizer'] != 'Identity' or info['critic_normalizer'] != 'Identity'):
        raise ValueError('Original formal checkpoint state/counters/observation contract do not match')
    return saved


def plan(additional_updates=300, *, short_check=False):
    if (not short_check and additional_updates != 300) or (short_check and not 1 <= additional_updates <= 3):
        raise ValueError('Formal continuation is exactly +300; isolated short checks allow only +1..3')
    return dict(version=CONTINUATION_VERSION, source_checkpoint=str(SOURCE), source_sha256=SOURCE_SHA256,
        role='continuation', short_check=bool(short_check), source_total_updates=1000,
        additional_updates=int(additional_updates), target_total_updates=1000+int(additional_updates),
        source_adam_step=20000, expected_final_adam_step=20000+20*int(additional_updates),
        source_control_steps=24000, target_control_steps=24000+24*int(additional_updates),
        num_envs=1024, seed=42, save_interval=100, evaluation_version=EVALUATION_VERSION,
        changes_to_training_behavior=[], simulator_state_restored=False,
        limitation='Native checkpoint has no simulator/RNG state; new seeded episodes, not bitwise uninterrupted training')


def verify_restoration(runner, saved):
    expected_model = tree_digest(saved['model_state_dict'])
    actual_model = tree_digest(runner.alg.policy.state_dict())
    expected_adam = tree_digest(saved['optimizer_state_dict'])
    actual_adam = tree_digest(runner.alg.optimizer.state_dict())
    if (expected_model != actual_model or expected_adam != actual_adam
            or runner.alg.learning_rate != saved['optimizer_state_dict']['param_groups'][0]['lr']
            or runner._curriculum_term.schedule.total_updates != SOURCE_UPDATES
            or runner.current_learning_iteration != 999):
        raise ValueError('Full restoration mismatch: network/noise/Adam/LR/counters')
    if (runner.empirical_normalization or runner.obs_normalizer.state_dict()
            or runner.privileged_obs_normalizer.state_dict()):
        raise ValueError('Original normalizers are Identity and must remain Identity')
    return dict(model_and_log_std_exact=True, adam_all_states_exact=True, normalizers='Identity, exact',
        model_sha256=actual_model, adam_sha256=actual_adam, adam_steps=adam_steps(saved),
        actual_learning_rate=float(runner.alg.learning_rate),
        raw_std=runner.alg.policy.log_std.detach().exp().cpu().tolist(),
        source_iteration=999, next_iteration=1000, total_updates=1000, control_steps=24000)


def validate_completion(saved, record):
    target = record['target_total_updates']
    state = saved['infos']['lighthw_action_curriculum']
    if (saved['iter'] != target-1 or state['schedule']['total_updates'] != target
            or state['control_steps'] != record['target_control_steps']
            or adam_steps(saved) != [record['expected_final_adam_step']]
            or saved['infos'].get(INFO_KEY, {}).get('source_sha256') != SOURCE_SHA256):
        raise ValueError('Continuation completion/cumulative update/Adam/source mismatch')
    for section in (saved['model_state_dict'], saved['optimizer_state_dict']['state']):
        def finite(value):
            if isinstance(value, torch.Tensor): return bool(torch.isfinite(value).all())
            if isinstance(value, dict): return all(finite(v) for v in value.values())
            return True
        if not finite(section): raise ValueError('Nonfinite continuation state')


def audit_configuration(env_cfg, agent_cfg):
    """与源 run 的全部保存配置比较，仅允许续训元数据/预算变化。"""
    import yaml
    from reference_config_audit import _normalize, differences
    current_env = _normalize(yaml.load(yaml.dump(env_cfg.to_dict()), Loader=yaml.BaseLoader))
    source_env = _normalize(yaml.load((SOURCE_RUN/'params/env.yaml').read_text(), Loader=yaml.BaseLoader))
    current_agent = yaml.load(yaml.dump(agent_cfg.to_dict()), Loader=yaml.BaseLoader)
    source_agent = yaml.load((SOURCE_RUN/'params/agent.yaml').read_text(), Loader=yaml.BaseLoader)
    # terrain's config is populated by scene construction; logical parallelism/spacing
    # must still match scene exactly. This is an alias, not a changed physical value.
    for value in (current_env, source_env):
        value['scene']['terrain']['num_envs'] = value['scene']['num_envs']
        value['scene']['terrain']['env_spacing'] = value['scene']['env_spacing']
    allowed = ('max_iterations','run_name','resume','load_run','load_checkpoint',
               'lighthw_reference_run_role','lighthw_reference_short_check')
    metadata = {k:current_agent.pop(k, None) for k in allowed}
    for k in allowed: source_agent.pop(k, None)
    drift = differences(source_env, current_env, '/env') + differences(source_agent, current_agent, '/agent')
    if drift: raise ValueError('Continuation must preserve the actual formal configuration:\n'+json.dumps(drift, indent=2))
    return dict(status='matched', source_config_sha256=SOURCE_CONFIG_SHA256,
                allowed_metadata=metadata, unexpected_differences=drift,
                preserved='all saved numerical environment and PPO settings including original training gates')
