"""CPU subprocess checks for early R6 worker failure; no simulator is started."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common
sys.path.insert(0, str(common.ROOT / 'routes/route_06_heavy_payload/scripts'))
import launch


def test_launcher_reports_early_exit_and_keeps_child_returncode(tmp_path, monkeypatch, capsys):
    route = tmp_path / 'route'
    (route / 'scripts').mkdir(parents=True)
    (route / 'scripts/worker.py').write_text("print('synthetic early worker exit', flush=True)\n")
    monkeypatch.setattr(launch, 'ROUTE', route)
    monkeypatch.setattr(launch, 'snapshot', lambda run: None)
    original = launch.importlib.util.find_spec
    monkeypatch.setattr(launch.importlib.util, 'find_spec',
        lambda name: object() if name in ('isaacsim','isaaclab','isaaclab_rl','rsl_rl') else original(name))
    assert launch.main('smoke', ['--num_envs','1','--headless']) == 1
    run, = (route / 'validation').iterdir()
    meta = json.loads((run / 'run.json').read_text())
    assert meta['status'] == 'failed'
    assert meta['worker_returncode'] == 0
    assert 'exited before completing runtime validation' in meta['error']
    assert str(run / 'console.log') in capsys.readouterr().err
    assert not (run / 'runtime_check.json').exists()


@pytest.mark.parametrize('exit_kind', ['exception', 'native_exit'])
def test_worker_records_startup_before_app_launcher(tmp_path, exit_kind):
    fake = tmp_path / 'fake_modules/isaaclab'
    fake.mkdir(parents=True)
    (fake / '__init__.py').write_text('')
    operation = "raise RuntimeError('synthetic AppLauncher failure')" if exit_kind == 'exception' else 'os._exit(0)'
    (fake / 'app.py').write_text(f'import os\nclass AppLauncher:\n    def __init__(self, **kwargs):\n        {operation}\n')
    run = tmp_path / 'run'
    run.mkdir()
    common.write_json(run / 'request.json', {'headless':True,'device':'cpu','mode':'smoke'})
    common.write_json(run / 'payload.json', {})
    result = subprocess.run([sys.executable, str(common.ROOT / 'routes/route_06_heavy_payload/scripts/worker.py'), '--run', str(run)],
        env=dict(os.environ, PYTHONPATH=str(fake.parent)), capture_output=True, text=True)
    (run / 'console.log').write_text(result.stdout + result.stderr)
    status = json.loads((run / 'worker_status.json').read_text())
    assert status['stage'] == 'app_start'
    if exit_kind == 'exception':
        assert result.returncode != 0
        error = json.loads((run / 'worker_error.json').read_text())
        assert error['exception_type'] == 'RuntimeError'
        assert 'synthetic AppLauncher failure' in error['error']
        assert 'Traceback' in error['traceback']
    else:
        assert result.returncode == 0
        assert not (run / 'worker_error.json').exists()
    with pytest.raises(RuntimeError) as error:
        launch.verify_completion(run, 'smoke')
    assert 'app_start' in str(error.value)
    assert str(run / 'console.log') in str(error.value)


def test_worker_keeps_original_error_when_app_close_exits_zero(tmp_path):
    fake = tmp_path / 'fake_modules/isaaclab'
    fake.mkdir(parents=True)
    (fake / '__init__.py').write_text('')
    (fake / 'app.py').write_text('import os\nclass AppLauncher:\n    def __init__(self, **kwargs):\n        self.app = self\n    def close(self):\n        os._exit(0)\n')
    (fake.parent / 'gymnasium.py').write_text("raise RuntimeError('synthetic task import failure')\n")
    run = tmp_path / 'run'
    run.mkdir()
    common.write_json(run / 'request.json', {'headless':True,'device':'cpu','mode':'smoke'})
    common.write_json(run / 'payload.json', {})
    result = subprocess.run([sys.executable, str(common.ROOT / 'routes/route_06_heavy_payload/scripts/worker.py'), '--run', str(run)],
        env=dict(os.environ, PYTHONPATH=str(fake.parent)), capture_output=True, text=True)
    assert result.returncode == 0
    error = json.loads((run / 'worker_error.json').read_text())
    assert error['stage'] == 'task_imports'
    assert error['exception_type'] == 'RuntimeError'
    assert error['error'] == 'synthetic task import failure'
    assert 'synthetic task import failure' in result.stderr
    with pytest.raises(RuntimeError, match='synthetic task import failure'):
        launch.verify_completion(run, 'smoke')
