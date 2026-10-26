"""CPU boundaries only; synthetic imports/data do not establish GPU behavior."""
import ast
from pathlib import Path
import sys
from types import ModuleType

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from migration_v1 import environment_probe as probe, env_adapter
from migration_v1.contract import sha256
from migration_v1.dataset import EvidenceError
from migration_v1.schedule import make_schedule


def test_existing_runtime_type_cache_preserves_class_identity_without_real_isaac(monkeypatch):
    prefix = 'rl_training.tasks.manager_based.locomotion.multiskill.lighthw.wheel_unified_v6'
    package = ModuleType(prefix)
    package.core = object()
    env_module = ModuleType(prefix+'.env')
    terrain_module = ModuleType(prefix+'.terrain')
    env_module.UnifiedEnv = type('SyntheticUnifiedEnv', (), {})
    terrain_module.UnifiedTerrain = type('SyntheticUnifiedTerrain', (), {})
    monkeypatch.setitem(sys.modules, prefix, package)
    monkeypatch.setitem(sys.modules, prefix+'.env', env_module)
    monkeypatch.setitem(sys.modules, prefix+'.terrain', terrain_module)
    env_adapter._runtime_types.cache_clear()
    try:
        first = env_adapter._runtime_types()
        second = env_adapter._runtime_types()
        assert first is second and first[0] is second[0] and first[1] is second[1]
        assert isinstance(first[1](), second[1])
    finally:
        env_adapter._runtime_types.cache_clear()


@pytest.mark.parametrize('steps', [0, 3, 64, True, 1.5])
def test_probe_never_accepts_long_or_optimizer_shaped_requests(monkeypatch, tmp_path, steps):
    monkeypatch.setattr(probe, 'load_student', lambda *a, **k: pytest.fail('Source load must follow step-limit check'))
    with pytest.raises(EvidenceError, match='one or two'):
        probe.prepare(tmp_path/'absent.pt', tmp_path/'never_created', steps)
    assert not (tmp_path/'never_created').exists()


def test_missing_candidate_rejected_before_app_or_output(monkeypatch, tmp_path):
    def missing(*args, **kwargs):
        raise EvidenceError('Missing complete candidate provenance')
    monkeypatch.setattr(probe, 'load_student', missing)
    monkeypatch.setattr(probe, 'run_simulation', lambda _: pytest.fail('Simulation called before strict gate'))
    output = tmp_path/'never_created'
    with pytest.raises(SystemExit) as failure:
        probe.main(['--checkpoint', str(tmp_path/'absent.pt'), '--output', str(output)])
    assert failure.value.code == 2 and not output.exists()


def test_changed_execution_source_rejected_before_app_import_or_output(tmp_path):
    source = tmp_path/'synthetic_source.py'
    source.write_text('original')
    digest = sha256(source)
    source.write_text('changed')
    output = tmp_path/'never_created'
    with pytest.raises(EvidenceError, match='execution source changed'):
        probe.run_simulation(dict(sources={str(source): digest}, output=output))
    assert not output.exists()


def test_probe_source_scope_and_lazy_application_import():
    sources = probe.probe_sources(make_schedule(64))
    names = {Path(name).name for name in sources}
    assert {'environment_probe.py', 'env_adapter.py', 'command_adapter.py', 'reward_adapter.py',
            'schedule.py', 'reward_manager.py', 'yaml.py', 'parse_cfg.py'} <= names
    assert not {'recovery_bc.py', 'recovery_dataset.py', 'recovery_student_worker.py',
                'ppo_adapter.py', 'ppo_policy.py', 'student_prefix.py'} & names
    for module in (probe, env_adapter):
        top = ast.parse(Path(module.__file__).read_text()).body
        imports = [node for node in top if isinstance(node, ast.ImportFrom)]
        assert not any(node.module and node.module.startswith(('isaaclab', 'rl_training')) for node in imports)
