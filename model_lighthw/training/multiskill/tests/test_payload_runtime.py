"""CPU公式和伪PhysX事务测试；不宣称真实GPU变载或GUI行为已验证。"""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[4]
SCRIPTS = ROOT/'model_lighthw/training/multiskill/routes/route_06_heavy_payload/scripts'
sys.path.insert(0, str(SCRIPTS))
from payload_runtime import RuntimePayload, _ASSET

_spec = importlib.util.spec_from_file_location('_r6_runtime_test_spec',
    ROOT/'source/rl_training/rl_training/tasks/manager_based/locomotion/multiskill/lighthw/payload_spec.py')
spec = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(spec)
URDF = ROOT/'model_lighthw/lighthw_urdf/urdf/lighthw.urdf'


class FakeView:
    """同真实API一样复用getter缓冲，便于暴露忘记clone的回滚缺陷。"""
    def __init__(self, mass):
        tree = ET.parse(URDF)
        self.names = [link.get('name') for link in tree.findall('link')]
        self.body_id = self.names.index('Base_link')
        count = len(self.names)
        self.values = {'masses':torch.tensor([[float(x.find('inertial/mass').get('value')) for x in tree.findall('link')]]),
            'coms':torch.zeros(1,count,7), 'inertias':torch.zeros(1,count,9)}
        self.values['coms'][...,6] = 1
        for index, link in enumerate(tree.findall('link')):
            inertial = link.find('inertial'); item = inertial.find('inertia')
            self.values['coms'][0,index,:3] = torch.tensor([float(v) for v in inertial.find('origin').get('xyz').split()])
            self.values['inertias'][0,index] = torch.tensor([[float(item.get('i'+''.join(sorted((a,b))))) for b in 'xyz'] for a in 'xyz']).reshape(9)
        inertial = tree.find("link[@name='Base_link']/inertial")
        b = self.body_id
        total, com, inertia = _ASSET.composite_inertia(float(self.values['masses'][0,b]),
            np.array([float(v) for v in inertial.find('origin').get('xyz').split()]),
            np.array([[float(inertial.find('inertia').get('i'+''.join(sorted((a,c))))) for c in 'xyz'] for a in 'xyz']),
            mass, spec.CONTRACT['payload_center_m'], spec.CONTRACT['payload_box_size_m'])
        self.values['masses'][0,b] = total
        self.values['coms'][0,b,:3] = torch.tensor(com)
        self.values['inertias'][0,b] = torch.tensor(inertia).reshape(9)
        # 非单位主轴姿态使测试能发现误将完整惯量再旋转。
        self.values['coms'][0,b,3:] = torch.tensor([0.,0.,.1,(.99)**.5])
        self.buffers = {k:v.clone() for k,v in self.values.items()}
        self.root = torch.tensor([[.1,.2,.43,0.,0.,0.,1.]])
        self.joints = torch.arange(16).reshape(1,16).float()/100
        self.calls, self.fail_once, self.fail_always = [], None, False
        self.corrupt_once = None

    def _get(self, name):
        self.buffers[name].copy_(self.values[name])
        return self.buffers[name]

    def get_masses(self): return self._get('masses')
    def get_coms(self): return self._get('coms')
    def get_inertias(self): return self._get('inertias')
    def get_root_transforms(self): return self.root
    def get_dof_positions(self): return self.joints

    def _set(self, name, value, indices):
        assert value.device.type == indices.device.type == 'cpu'
        assert value.shape == self.values[name].shape
        assert indices.tolist() == [0]
        self.calls.append(name)
        self.values[name].copy_(value)
        if self.corrupt_once == name:
            self.corrupt_once = None
            self.values[name][0,self.body_id].reshape(-1)[0] += .05
        if self.fail_once == name or self.fail_always:
            self.fail_once = None
            raise RuntimeError('injected setter failure')

    def set_masses(self, value, indices): self._set('masses',value,indices)
    def set_coms(self, value, indices): self._set('coms',value,indices)
    def set_inertias(self, value, indices): self._set('inertias',value,indices)


def fixture(initial=65., view_factory=None):
    view = (view_factory or PrincipalFrameFakeView)(initial)
    cfg = SimpleNamespace(payload_mass_kg=initial)
    buffers = {name:SimpleNamespace(timestamp=3.) for name in ('_body_com_pose_b','_root_link_vel_w',
        '_root_state_w','_body_link_state_w','_joint_pos')}
    data = SimpleNamespace(**buffers, default_mass=view.get_masses().clone(), default_inertia=view.get_inertias().clone())
    robot = SimpleNamespace(root_physx_view=view, data=data, find_bodies=lambda name:([view.body_id],['Base_link']))
    env = SimpleNamespace(num_envs=1, scene={'robot':robot}, cfg=cfg)
    env.obs_buf = {'policy':torch.ones(1,58), 'critic':torch.ones(1,237)*2}
    env.obs_buf['policy'][:,-1] = initial/65
    env.obs_buf['critic'][:,-1] = initial/65
    histories = []
    def compute(update_history=False):
        histories.append(update_history)
        result = {k:v.clone() for k,v in env.obs_buf.items()}
        for value in result.values(): value[:,-1] = cfg.payload_mass_kg/65
        return result
    env.observation_manager = SimpleNamespace(compute=compute, active_terms={'policy':['base','payload'],'critic':['base','payload']})
    return RuntimePayload(env,URDF,spec.CONTRACT), env, view, histories


@pytest.mark.parametrize('mass',[0.,5.,15.,32.5,65.])
def test_formula_matches_existing_generated_urdf(tmp_path,mass):
    runtime,_,_,_ = fixture(65.)
    expected = _ASSET.write_payload_urdf(URDF,tmp_path/'payload.urdf',mass)
    actual = runtime.properties(mass)
    assert actual == expected
    assert np.linalg.eigvalsh(actual['base_inertia_kg_m2']).min() > 0
    assert runtime.mass_kg == 65.  # 查询公式不修改当前载荷。


def test_65_to_zero_and_arbitrary_values_preserve_pose_other_bodies_and_refresh_both_inputs():
    runtime,env,view,histories = fixture(65.)
    other = [i for i in range(len(view.names)) if i != view.body_id]
    original = {k:v.clone() for k,v in view.values.items()}
    root, joints = view.root.clone(),view.joints.clone()
    obs_reference = env.obs_buf['policy']
    defaults = (env.scene['robot'].data.default_mass.clone(), env.scene['robot'].data.default_inertia.clone())
    for mass in (0.,5.,15.,32.5,65.,0.,65.,0.):
        record = runtime.set_mass(mass)
        assert runtime.mass_kg == env.cfg.payload_mass_kg == mass
        assert runtime.total_mass_kg == pytest.approx(22.0+mass)
        assert env.obs_buf['policy'] is obs_reference
        assert float(obs_reference[0,-1]) == pytest.approx(mass/65)
        assert float(env.obs_buf['critic'][0,-1]) == pytest.approx(mass/65)
        assert runtime.verify_current()['physical_properties_verified']
        assert record['max_abs_errors']['inertia_max_abs_kg_m2'] < 1e-6
        json.dumps(record,allow_nan=False)
        for key in original:
            torch.testing.assert_close(view.values[key][:,other],original[key][:,other],rtol=0,atol=0)
        torch.testing.assert_close(view.root,root,rtol=0,atol=0)
        torch.testing.assert_close(view.joints,joints,rtol=0,atol=0)
    assert histories == [False]*8
    data = env.scene['robot'].data
    assert data._body_com_pose_b.timestamp == data._root_link_vel_w.timestamp == data._root_state_w.timestamp == -1.
    assert data._joint_pos.timestamp == 3.
    torch.testing.assert_close(data.default_mass,defaults[0],rtol=0,atol=0)
    torch.testing.assert_close(data.default_inertia,defaults[1],rtol=0,atol=0)
    assert view.calls == ['masses','coms','inertias']*8


@pytest.mark.parametrize('bad',[-.1,65.1,float('nan'),float('inf'),-float('inf')])
def test_invalid_request_never_reaches_physx(bad):
    runtime,env,view,_ = fixture()
    with pytest.raises(ValueError): runtime.set_mass(bad)
    assert not view.calls and runtime.mass_kg == env.cfg.payload_mass_kg == 65.


@pytest.mark.parametrize('failed_setter',['masses','coms','inertias'])
def test_partial_write_failure_restores_all_physics_and_inputs(failed_setter):
    runtime,env,view,_ = fixture()
    previous = {k:v.clone() for k,v in view.values.items()}
    observations = {k:v.clone() for k,v in env.obs_buf.items()}
    view.fail_once = failed_setter
    with pytest.raises(RuntimeError,match='previous physics/observations were restored'):
        runtime.set_mass(5.)
    assert_restored(view,previous)
    for key in observations: torch.testing.assert_close(env.obs_buf[key],observations[key],rtol=0,atol=0)
    assert runtime.mass_kg == env.cfg.payload_mass_kg == 65.
    assert runtime.verify_current()['physical_properties_verified']


def test_silent_inertia_corruption_is_detected_and_rolled_back():
    runtime,env,view,_ = fixture()
    previous = {k:v.clone() for k,v in view.values.items()}
    view.corrupt_once = 'inertias'
    with pytest.raises(RuntimeError,match='restored'): runtime.set_mass(15.)
    assert_restored(view,previous)
    assert env.cfg.payload_mass_kg == 65.


def test_failed_observation_refresh_rolls_back_physics_and_configuration():
    runtime,env,view,_ = fixture()
    previous = {k:v.clone() for k,v in view.values.items()}
    def fail(**kwargs): raise RuntimeError('injected observation failure')
    env.observation_manager.compute = fail
    with pytest.raises(RuntimeError,match='restored'): runtime.set_mass(0.)
    assert_restored(view,previous)
    assert env.cfg.payload_mass_kg == runtime.mass_kg == 65.
    assert env.obs_buf['policy'][0,-1] == env.obs_buf['critic'][0,-1] == 1.


def test_rollback_failure_is_not_reported_as_restoration():
    runtime,env,view,_ = fixture()
    view.fail_always = True
    with pytest.raises(RuntimeError,match='rollback also failed.*stop playback'): runtime.set_mass(5.)
    assert env.cfg.payload_mass_kg == runtime.mass_kg == 65.


def test_independent_readback_detects_later_physics_or_observation_drift():
    runtime,env,view,_ = fixture()
    runtime.set_mass(15.)
    view.values['masses'][0,view.body_id] += 1.
    with pytest.raises(AssertionError): runtime.verify_current()
    runtime.set_mass(15.)
    env.obs_buf['critic'][0,-1] = 0.
    with pytest.raises(AssertionError): runtime.verify_current()


def test_wrong_source_or_multiple_envs_are_rejected(tmp_path):
    runtime,env,view,_ = fixture()
    changed = tmp_path/'wrong.urdf';changed.write_text(URDF.read_text()+'\n')
    with pytest.raises(ValueError,match='checksum'): RuntimePayload(env,changed,spec.CONTRACT)
    env.num_envs = 2
    with pytest.raises(ValueError,match='single-environment'): RuntimePayload(env,URDF,spec.CONTRACT)
    assert not view.calls


@pytest.mark.parametrize('sign',[1.,-1.])
def test_non_base_com_quaternion_normalization_and_sign_equivalence(sign):
    runtime,env,view,_ = fixture()
    # 模拟真实set_coms把近单位float32四元数归一化，且可能选取等价负号。
    q = torch.tensor([.2,-.3,.4,.71**.5],dtype=torch.float32)*(1.+1.1920928955078125e-7)
    view.values['coms'][0,1,3:] = q
    runtime = RuntimePayload(env,URDF,spec.CONTRACT)
    original_set = view.set_coms
    def normalized_set(values,indices):
        original_set(values,indices)
        rotation = view.values['coms'][0,1,3:]
        rotation.copy_(sign*rotation/rotation.norm())
    view.set_coms = normalized_set
    record = runtime.set_mass(5.)
    check = record['non_base_com_orientation_check']
    assert 0 < check['max_sign_aligned_component_error'] < 1e-6
    assert check['max_normalized_orientation_component_error'] < 1e-6
    assert runtime.verify_current()['physical_properties_verified']
    assert runtime.mass_kg == 5.


def test_non_base_com_real_rotation_is_rejected_and_rolled_back():
    runtime,env,view,_ = fixture()
    old = {key:value.clone() for key,value in view.values.items()}
    original_set = view.set_coms
    corrupt = [True]
    def rotate_once(values,indices):
        original_set(values,indices)
        if corrupt and corrupt.pop():
            view.values['coms'][0,1,3:] = torch.tensor([.01,0.,0.,(1.-.01**2)**.5])
    view.set_coms = rotate_once
    with pytest.raises(RuntimeError,match='restored'): runtime.set_mass(5.)
    assert_restored(view,old)
    assert runtime.mass_kg == env.cfg.payload_mass_kg == 65.


@pytest.mark.parametrize('field',['masses','coms'])
def test_non_base_physical_values_keep_bitwise_check(field):
    runtime,_,view,_ = fixture()
    expected = runtime._capture()
    actual = {key:value.clone() for key,value in expected.items()}
    if field == 'masses':
        actual[field][0,1] = torch.nextafter(actual[field][0,1],torch.tensor(float('inf')))
    else:
        actual[field][0,1,0] = torch.nextafter(actual[field][0,1,0],torch.tensor(float('inf')))
    with pytest.raises(AssertionError): runtime._verify(actual,expected)


@pytest.mark.parametrize('component',[0,2])
def test_non_base_inertia_uses_whole_matrix_roundoff_bound(component):
    runtime,_,view,_ = fixture()
    expected = runtime._capture()
    actual = {key:value.clone() for key,value in expected.items()}
    scale = expected['inertias'][0,1].abs().max()
    bound = 32.*torch.finfo(torch.float32).eps*scale
    actual['inertias'][0,1,component] += bound/4.
    runtime._verify(actual,expected)
    actual['inertias'][0,1,component] = expected['inertias'][0,1,component]+2.*bound
    with pytest.raises(AssertionError,match='reconstruction bound'): runtime._verify(actual,expected)


def test_repeated_changes_write_original_other_inertias_without_cumulative_roundoff():
    runtime,env,view,_ = fixture()
    original = view.values['inertias'].clone()
    original_set = view.set_inertias
    writes = []
    def rounded_set(value,indices):
        writes.append(value.clone())
        original_set(value,indices)
        # 模拟一次主轴分解重构，偏差若每次以当前表为输入就会累积。
        scale = original[0,1].abs().max()
        view.values['inertias'][0,1,0] += 2.*torch.finfo(torch.float32).eps*scale
    view.set_inertias = rounded_set
    for i in range(24):
        record = runtime.set_mass(5. if i%2 else 65.)
        runtime.verify_current()
        torch.testing.assert_close(writes[-1][:,1:],original[:,1:],rtol=0,atol=0)
        check = record['non_base_inertia_reconstruction_check']
        assert check['epsilon_multiplier'] == 32.
        assert check['max_abs_error_kg_m2'] > 0.
        errors = np.asarray(check['per_body_max_abs_error_kg_m2'])
        bounds = np.asarray(check['per_body_abs_bound_kg_m2'])
        assert (errors <= bounds).all()
    assert len(writes) == 24


class PrincipalFrameFakeView(FakeView):
    """模拟真实GPU语义：惯量setter只写主惯量，COM setter独立维护主轴。

    getter重建link轴完整矩阵。其余FakeView仅是字段存储事务测试，不能揭示
    set/get惯量接口的表示差异；此类专门防止把getter的完整矩阵直接回写。
    """
    @staticmethod
    def decompose(inertia):
        from itertools import permutations
        eigenvalues,eigenvectors = np.linalg.eigh(inertia)
        # 模拟Jacobi主轴顺序保持最接近输入坐标轴，而不是强制升序。
        order = max(permutations(range(3)), key=lambda p:sum(eigenvectors[i,p[i]]**2 for i in range(3)))
        eigenvalues,axes = eigenvalues[list(order)],eigenvectors[:,list(order)]
        if np.linalg.det(axes) < 0: axes[:,-1] *= -1
        return eigenvalues,axes

    def __init__(self,mass):
        from scipy.spatial.transform import Rotation
        super().__init__(mass)
        self.principal,self.axes = self.decompose(self.values['inertias'][0,self.body_id].numpy().reshape(3,3))
        self.values['coms'][0,self.body_id,3:] = torch.tensor(Rotation.from_matrix(self.axes).as_quat(),dtype=torch.float32)

    def _get(self,name):
        if name == 'inertias':
            from scipy.spatial.transform import Rotation
            rotation = Rotation.from_quat(self.values['coms'][0,self.body_id,3:].numpy()).as_matrix()
            full = rotation@np.diag(self.principal)@rotation.T
            self.values['inertias'][0,self.body_id] = torch.tensor(full,dtype=torch.float32).reshape(9)
        return super()._get(name)

    def set_inertias(self,value,indices):
        # 模拟只更新主惯量、丢弃分解所得新方向这一已实测的API行为。
        super().set_inertias(value,indices)
        self.principal,_ = self.decompose(self.values['inertias'][0,self.body_id].numpy().reshape(3,3))


def test_runtime_updates_principal_axes_when_payload_changes_from_65_to_zero():
    runtime,env,view,_ = fixture(view_factory=PrincipalFrameFakeView)
    runtime.set_mass(65.)
    for mass in (0.,5.,15.,32.5,65.,0.):
        record = runtime.set_mass(mass)
        assert record['physical_properties_verified']
        assert runtime.verify_current()['physical_properties_verified']
        np.testing.assert_allclose(view.get_inertias()[0,view.body_id].numpy().reshape(3,3),
                                   runtime.properties(mass)['base_inertia_kg_m2'],atol=2e-6,rtol=1e-4)
        assert env.obs_buf['policy'][0,-1] == pytest.approx(mass/65.)



def assert_restored(view,previous):
    # Base主轴四元数可以改变列顺序/符号，完整link惯量必须恢复；不能逐位比表示。
    actual = {'masses':view.get_masses().clone(),'coms':view.get_coms().clone(),'inertias':view.get_inertias().clone()}
    torch.testing.assert_close(actual['masses'],previous['masses'],rtol=0,atol=0)
    torch.testing.assert_close(actual['coms'][...,:3],previous['coms'][...,:3],rtol=0,atol=0)
    other = [i for i in range(actual['masses'].shape[1]) if i != view.body_id]
    torch.testing.assert_close(actual['coms'][:,other,3:],previous['coms'][:,other,3:],rtol=0,atol=0)
    torch.testing.assert_close(actual['inertias'][:,other],previous['inertias'][:,other],rtol=0,atol=0)
    torch.testing.assert_close(actual['inertias'][:,view.body_id],previous['inertias'][:,view.body_id],rtol=1e-4,atol=2e-6)
