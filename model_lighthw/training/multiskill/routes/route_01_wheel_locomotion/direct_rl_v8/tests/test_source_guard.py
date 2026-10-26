"""通过真实启动入口验证目录变更边界；在运行时发现之前停止，不启动训练。"""
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import launch


class ReachedRuntimeDiscovery(Exception):
    pass


def test_formal_command_reaches_runtime_discovery(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['train.py', '--headless', '--num_envs', '512',
                                   '--max_iterations', '3000', '--eval_interval', '100'])
    def stop(name):
        assert name == 'isaacsim'
        raise ReachedRuntimeDiscovery
    monkeypatch.setattr(launch.importlib.util, 'find_spec', stop)
    with pytest.raises(ReachedRuntimeDiscovery):
        launch.main('train')


@pytest.mark.parametrize('edit,accepted', [
    ('other_route', True),
    ('r1_route', False),
    ('joint_order', False),
    ('physics', False),
    ('outside_code', False),
    ('directory_expression', False),
    ('invalid_syntax', False),
])
def test_formal_command_catalog_change_boundary(monkeypatch, capsys, edit, accepted):
    """模拟磁盘目录变化，通过真实CLI检查；不写共享catalog、不启动Isaac。"""
    import hashlib
    from source_guard import CATALOG_RELATIVE
    path = launch.REPO / CATALOG_RELATIVE
    current = path.read_text()
    changes = {
        'other_route': current.replace('MultiSkill-LightHW-R4a-RearBalance-v4', 'Future-R4a-v9'),
        'r1_route': current.replace('"status": "implemented_runtime_pending"', '"status": "changed"'),
        'joint_order': current.replace('("FR", "FL", "RR", "RL")', '("FL", "FR", "RR", "RL")'),
        'physics': current.replace('"physics_dt": 0.001', '"physics_dt": 0.002'),
        'outside_code': current + '\nCONTRACT["physics_dt"] = .002\n',
        'directory_expression': current.replace('"directory": "route_00_baseline_reference"', '"directory": str("route_00_baseline_reference")'),
        'invalid_syntax': current + '\ninvalid(\n',
    }
    modified = changes[edit]
    assert modified != current
    original_read = Path.read_text
    original_sha = launch.sha256
    monkeypatch.setattr(Path, 'read_text', lambda self, *a, **kw: modified if self == path else original_read(self, *a, **kw))
    monkeypatch.setattr(launch, 'sha256', lambda p: hashlib.sha256(modified.encode()).hexdigest() if Path(p) == path else original_sha(p))
    monkeypatch.setattr(sys, 'argv', ['train.py', '--headless', '--num_envs', '512',
                                   '--max_iterations', '3000', '--eval_interval', '100'])
    def stop(name):
        raise ReachedRuntimeDiscovery
    monkeypatch.setattr(launch.importlib.util, 'find_spec', stop)
    if accepted:
        with pytest.raises(ReachedRuntimeDiscovery):
            launch.main('train')
    else:
        with pytest.raises(SystemExit) as exc:
            launch.main('train')
        assert exc.value.code == 2
        assert 'Validated V8 training source changed: ' + CATALOG_RELATIVE in capsys.readouterr().err


def test_other_training_source_change_still_rejected(monkeypatch, capsys):
    target = ROOT / 'scripts/worker.py'
    original_sha = launch.sha256
    monkeypatch.setattr(launch, 'sha256', lambda p: 'changed' if Path(p) == target else original_sha(p))
    monkeypatch.setattr(sys, 'argv', ['train.py', '--headless', '--num_envs', '512'])
    with pytest.raises(SystemExit) as exc:
        launch.main('train')
    assert exc.value.code == 2
    assert 'Validated V8 training source changed:' in capsys.readouterr().err


def test_catalog_change_without_scoped_certificate_is_rejected():
    import json
    from source_guard import CATALOG_RELATIVE, validate_sources
    record = json.loads((ROOT / 'initialization/readiness.json').read_text())
    record.pop('catalog_r1_sha256')
    record['validated_source_sha256'][CATALOG_RELATIVE] = 'old'
    with pytest.raises(ValueError, match='catalog.py'):
        validate_sources(launch.REPO, record, launch.sha256)
