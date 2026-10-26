"""CPU delivery checks: USB corruption, path isolation and real USD plane authoring."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

REPO = Path(__file__).resolve().parents[4]


def test_corrupt_usb_wheel_is_rejected_before_install(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('_r6_offline_manage', REPO/'offline/manage.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    (tmp_path/'wheelhouse').mkdir()
    (tmp_path/'wheelhouse/fixture.whl').write_bytes(b'changed during transfer')
    (tmp_path/'wheels.json').write_text(json.dumps({'sha256':{'fixture.whl':hashlib.sha256(b'original').hexdigest()}}))
    with pytest.raises(RuntimeError, match='checksum mismatch'):
        module.prepare()
    assert not (tmp_path/'envs').exists()


def test_wrong_sim_override_fails_with_path_diagnostic():
    result = subprocess.run(['bash', str(REPO/'r6.sh'), 'doctor'],
        env=dict(os.environ, R6_ISAAC_SIM='/missing-r6-simulator'), capture_output=True, text=True)
    assert result.returncode == 2
    assert 'complete Isaac Sim installation' in result.stderr


def test_doctor_without_conda_site_packages():
    if 'R6_DEPS' not in os.environ:
        pytest.skip('Run through bash r6.sh python to test the USB environment')
    result = subprocess.run([sys.executable, '-S', str(REPO/'offline/doctor.py'), '--brief'], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((REPO/'offline/checks/doctor-last.json').read_text())
    assert report['status'] == 'passed'
    assert not report['gpu_verified']
    assert all(str(REPO) in p for p in report['source_paths'].values())
    assert all('/envs/lab/lib/' not in (info.get('file') or '') for info in report['modules'].values())


def test_offline_plane_has_real_collision_and_no_external_asset():
    sim = os.environ.get('R6_ISAAC_SIM')
    if not sim:
        pytest.skip('Isaac Sim USD libraries are needed; no simulator is started')
    sdk = next(Path(sim).glob('extscache/omni.usd.libs-*'))
    probe = r'''
import ast
from pathlib import Path
from types import SimpleNamespace
from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade
import sys
stage = Usd.Stage.CreateInMemory()
material = SimpleNamespace()
def make_material(path, cfg):
    prim = UsdShade.Material.Define(stage, path).GetPrim()
    api = UsdPhysics.MaterialAPI.Apply(prim)
    api.CreateStaticFrictionAttr(1.0)
    api.CreateDynamicFrictionAttr(1.0)
    api.CreateRestitutionAttr(0.0)
material.func = make_material
def bind(prim_path, material_path):
    UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(prim_path)).Bind(UsdShade.Material(stage.GetPrimAtPath(material_path)), materialPurpose='physics')
utils = SimpleNamespace(get_current_stage=lambda: stage, bind_physics_material=bind, GroundPlaneCfg=lambda **kw:SimpleNamespace(**kw))
scope = dict(Gf=Gf, UsdGeom=UsdGeom, UsdPhysics=UsdPhysics, sim_utils=utils, TerrainImporter=object)
tree = ast.parse(Path(sys.argv[1]).read_text())
tree.body = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
exec(compile(tree, sys.argv[1], 'exec'), scope)
importer = scope['LocalTerrainImporter']()
importer.cfg = SimpleNamespace(prim_path='/World/ground', physics_material=material)
importer.terrain_prim_paths = []
importer.import_ground_plane('terrain')
plane = stage.GetPrimAtPath('/World/ground/terrain/CollisionPlane')
assert plane.GetTypeName() == 'Plane'
assert plane.GetAttribute('axis').Get() == 'Z'
assert plane.GetAttribute('physics:collisionEnabled').Get() is True
assert plane.GetAttribute('width').Get() == 2.0e6
assert plane.GetRelationship('material:binding:physics').GetTargets()
assert importer.terrain_prim_paths == ['/World/ground/terrain']
assert not any(p.HasAuthoredReferences() or p.HasAuthoredPayloads() for p in stage.Traverse())
assert not stage.GetRootLayer().subLayerPaths
try: importer.import_ground_plane('terrain')
except ValueError: pass
else: raise AssertionError('duplicate ground accepted')
print('USD plane, collision, physics material, zero external asset references: passed')
'''
    result = subprocess.run([sys.executable, '-c', probe, str(REPO/'source/rl_training/rl_training/tasks/manager_based/locomotion/multiskill/lighthw/local_ground_plane.py')],
        env=dict(os.environ, PYTHONPATH=str(sdk)+':'+os.environ['PYTHONPATH'], LD_LIBRARY_PATH=str(sdk/'bin')+':'+os.environ['LD_LIBRARY_PATH']), capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_framework_dataset_import_and_hdf5_roundtrip(tmp_path):
    # Exact import used by RecorderManager; no fake package or simulator is needed.
    import torch
    from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler
    episode = EpisodeData()
    episode.add('actions', torch.tensor([1.0, 2.0]))
    handler = HDF5DatasetFileHandler()
    filename = str(tmp_path/'episode.hdf5')
    handler.create(filename, env_name='R6-source-regression')
    handler.write_episode(episode)
    handler.close()
    reader = HDF5DatasetFileHandler()
    reader.open(filename)
    try:
        assert reader.get_env_name() == 'R6-source-regression'
        restored = reader.load_episode(next(iter(reader.get_episode_names())), device='cpu')
        torch.testing.assert_close(restored.data['actions'], episode.data['actions'])
    finally:
        reader.close()


def test_framework_source_inventory_catches_ignored_package(tmp_path):
    import shutil
    spec = importlib.util.spec_from_file_location('_r6_source_integrity', REPO/'offline/source_integrity.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.verify_sources(REPO)['status'] == 'passed'
    shutil.copytree(REPO/'Isaaclab/source', tmp_path/'Isaaclab/source',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.egg-info'))
    shutil.copy2(REPO/'Isaaclab/VERSION', tmp_path/'Isaaclab/VERSION')
    (tmp_path/'offline').mkdir()
    shutil.copy2(REPO/'offline/isaaclab_source_manifest.json', tmp_path/'offline/isaaclab_source_manifest.json')
    assert module.verify_sources(tmp_path)['status'] == 'passed'
    shutil.rmtree(tmp_path/'Isaaclab/source/isaaclab/isaaclab/utils/datasets')
    report = module.verify_sources(tmp_path)
    assert report['status'] == 'failed'
    assert len(report['errors']) == 4
    assert all('/utils/datasets/' in error and 'missing framework source' in error for error in report['errors'])
