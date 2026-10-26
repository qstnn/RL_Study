"""新adapter纯CPU配表/实际数组错位拒绝；不冒充已加载真实env。"""
from copy import deepcopy
import sys

import numpy as np
import pytest

from migration_v1.env_adapter import planned_layout, validate_assignment
from migration_v1.schedule import make_schedule
from migration_v1.dataset import EvidenceError


def data(n=64):
    schedule=make_schedule(n);plan=planned_layout(schedule)
    obs=np.zeros((n,300));obs[:,290:300]=np.tile(plan['conditions'][:,1:]/[15.,40.],(1,5))
    condition=np.column_stack((plan['conditions'],np.zeros(n)))
    return schedule,plan,condition,obs


@pytest.mark.parametrize('n',[64,128,256])
def test_same_table_has_full_coverage_and_original_physical_lane_centers(n):
    schedule,plan,condition,obs=data(n)
    report=validate_assignment(schedule,condition,plan['terrain_types'],plan['env_origins'],obs)
    assert report['supplied_metadata_covers_all_64']
    assert not report['execution_provenance_verified']
    assert np.array_equal(plan['terrain_types'],((condition[:,0]+25)/5).astype(int))
    assert np.array_equal(plan['env_origins'][:,1],condition[:,0]/5*14)


def test_same_set_but_wrong_slot_mass_assignment_is_rejected():
    schedule,plan,condition,obs=data()
    wrong=condition.copy();wrong[[0,1]]=wrong[[1,0]]
    with pytest.raises(EvidenceError,match='per-slot'):
        validate_assignment(schedule,wrong,plan['terrain_types'],plan['env_origins'],obs)


def test_wrong_terrain_or_actor_context_cannot_be_hidden_by_schedule_metadata():
    schedule,plan,condition,obs=data()
    origins=plan['env_origins'].copy();origins[0,1]+=14
    with pytest.raises(EvidenceError,match='terrain'):
        validate_assignment(schedule,condition,plan['terrain_types'],origins,obs)
    obs[0,299]=1
    with pytest.raises(EvidenceError,match='Actor'):
        validate_assignment(schedule,condition,plan['terrain_types'],plan['env_origins'],obs)


def test_modified_schedule_and_small_formal_batch_are_rejected():
    schedule=make_schedule(64);bad=deepcopy(schedule);bad['conditions'][0]=bad['conditions'][1]
    with pytest.raises(EvidenceError,match='schedule'):
        planned_layout(bad)
    with pytest.raises(ValueError,match='64'):
        make_schedule(63)


def test_import_does_not_load_runtime_or_launch_simulator():
    from migration_v1 import env_adapter
    assert env_adapter._runtime_types.cache_info().currsize==0
    assert 'isaaclab.app' not in sys.modules


def test_local_override_installs_assignment_before_manager_and_payload_startup(monkeypatch):
    """用CPU生命周期替身验证动态分派顺序，不宣称PhysX已写入质量。"""
    from types import ModuleType,SimpleNamespace
    import torch
    from migration_v1 import env_adapter as adapter
    schedule,plan,_,_=data()
    package='rl_training.tasks.manager_based.locomotion.multiskill.lighthw.wheel_unified_v6'
    for count in range(1,len(package.split('.'))+1):
        name='.'.join(package.split('.')[:count])
        module=ModuleType(name);module.__path__=[]
        monkeypatch.setitem(sys.modules,name,module)
    env_module=ModuleType(package+'.env');terrain_module=ModuleType(package+'.terrain');core=ModuleType(package+'.core')
    core.plane_basis=lambda a:(torch.stack((a.cos(),torch.zeros_like(a),a.sin()),-1),
        torch.stack((-a.sin(),torch.zeros_like(a),a.cos()),-1))
    class FakeTerrain:
        def __init__(self,cfg):
            self.device='cpu';self.terrain_types=torch.zeros(64,dtype=torch.long)
            self.terrain_origins=torch.tensor([[[0.,(i-5)*14.,0.] for i in range(11)]])
            self.env_origins=torch.zeros((64,3))
    class FakeEnv:
        def __init__(self,cfg):
            self.cfg=cfg
            self.r1_slope_rad=torch.zeros(64);self.r1_payload_kg=torch.zeros(64);self.r1_towed_kg=torch.zeros(64)
            self.r1_tangent=torch.zeros((64,3));self.r1_normal=torch.zeros((64,3));self.r1_robot_hitch=torch.zeros((64,3))
            self.scene=SimpleNamespace(terrain=cfg.scene.terrain.class_type(cfg.scene.terrain))
            self.load_managers()
        def load_managers(self):
            self.conditions_seen_at_manager_start=torch.stack((self.r1_slope_rad*180/torch.pi,self.r1_payload_kg,self.r1_towed_kg),-1).numpy()
            self.r1_unloaded_mass=torch.ones((64,17))
            self.r1_inertia_verified=True
    env_module.UnifiedEnv=FakeEnv;terrain_module.UnifiedTerrain=FakeTerrain
    monkeypatch.setitem(sys.modules,package+'.env',env_module)
    monkeypatch.setitem(sys.modules,package+'.terrain',terrain_module)
    monkeypatch.setitem(sys.modules,package+'.core',core)
    adapter._runtime_types.cache_clear()
    try:
        terrain,env=adapter._runtime_types()
        terrain_cfg=SimpleNamespace(migration_schedule=deepcopy(schedule),num_envs=64,class_type=terrain)
        cfg=SimpleNamespace(migration_schedule=deepcopy(schedule),sim=SimpleNamespace(device='cpu'),
            scene=SimpleNamespace(terrain=terrain_cfg),hitch_height_m=.04)
        actual=env(cfg)
        assert np.allclose(actual.conditions_seen_at_manager_start,plan['conditions'],atol=1.e-5,rtol=0.)
        assert np.array_equal(actual.scene.terrain.terrain_types.numpy(),plan['terrain_types'])
        assert np.array_equal(actual.scene.terrain.env_origins.numpy(),plan['env_origins'])
        assert np.allclose(actual.r1_robot_hitch[:,2].numpy(),.04-.416223+.20)
    finally:
        adapter._runtime_types.cache_clear()
