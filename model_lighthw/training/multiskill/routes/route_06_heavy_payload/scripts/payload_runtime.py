"""R6单环境回放的实时变载：规范空载惯量合成、PhysX事务写入和观测同步。

只在两个策略步之间调用，不推进仿真、不重置机器人、不改变动作/电机参数。
质量变化表示理想刚性载荷切换，不模拟搬运碰撞、装卸冲量或热过程。
"""
from copy import deepcopy
import hashlib
import importlib.util
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import torch


_SOURCE = Path(__file__).resolve().parents[6]/'source/rl_training/rl_training/tasks/manager_based/locomotion/multiskill/lighthw/payload_asset.py'
_SPEC = importlib.util.spec_from_file_location('_r6_runtime_payload_asset', _SOURCE)
_ASSET = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_ASSET)


class RuntimePayload:
    """单环境play中切换0～65kg，始终从规范空载URDF重新计算。

    set_mass成功后返回可JSON序列化的请求/实测记录。失败时恢复先前物理属性、
    配置和观测，再抛错；调用方应结束当前回放，不可吞掉错误继续控制。
    """
    def __init__(self, base, canonical_urdf, contract):
        if base.num_envs != 1:
            raise ValueError('Runtime payload is restricted to single-environment playback')
        self.base, self.robot = base, base.scene['robot']
        self.view = self.robot.root_physx_view
        ids, _ = self.robot.find_bodies('Base_link')
        if len(ids) != 1:
            raise ValueError('Exactly one Base_link is required')
        self.body_id = int(ids[0])
        self.contract = deepcopy(contract)
        if self.contract.get('payload_mass_range_kg') != [0.0, 65.0]:
            raise ValueError('Expected the R6 0 to 65 kg payload contract')
        self.center = np.asarray(self.contract['payload_center_m'], dtype=np.float64)
        self.size = np.asarray(self.contract['payload_box_size_m'], dtype=np.float64)
        if self.center.shape != (3,) or self.size.shape != (3,):
            raise ValueError('Payload center and box size must each contain three numbers')
        source = Path(canonical_urdf).resolve(strict=True)
        self.canonical_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
        if self.canonical_sha256 != self.contract['urdf_sha256']:
            raise ValueError('Canonical unloaded URDF checksum does not match the R6 contract')
        tree = ET.parse(source)
        inertial = tree.getroot().find("link[@name='Base_link']/inertial")
        origin, element = inertial.find('origin'), inertial.find('inertia')
        self._base_mass = float(inertial.find('mass').get('value'))
        self._base_com = np.array([float(x) for x in origin.get('xyz', '0 0 0').split()])
        rotation = _ASSET.rotation_rpy([float(x) for x in origin.get('rpy', '0 0 0').split()])
        inertia = np.array([[float(element.get('i'+''.join(sorted((a,b))))) for b in 'xyz'] for a in 'xyz'])
        self._base_inertia = rotation @ inertia @ rotation.T
        self._robot_mass = sum(float(m.get('value')) for m in tree.findall('link/inertial/mass'))
        self.mass_kg = self._validate_mass(base.cfg.payload_mass_kg)
        self.properties(self.mass_kg)  # Validate geometry and positive-definite inertia before any setter.
        self._observation_groups()
        self._reference = self._capture()
        self.verify_current()

    @property
    def total_mass_kg(self):
        """当前标称整机质量；实测值同时保留在set_mass/verify_current返回记录中。"""
        return self._robot_mass+self.mass_kg

    @staticmethod
    def _validate_mass(value):
        value = float(value)
        if not math.isfinite(value) or not 0.0 <= value <= 65.0:
            raise ValueError('R6 payload must be finite and between 0 and 65 kg')
        return value

    def properties(self, mass):
        """返回与write_payload_urdf一致的质量/质心/惯量报告，不修改当前状态。"""
        mass = self._validate_mass(mass)
        total, com, inertia = _ASSET.composite_inertia(
            self._base_mass, self._base_com, self._base_inertia, mass, self.center, self.size)
        return {'payload_kg':mass, 'payload_center_m':self.center.tolist(),
                'payload_box_size_m':self.size.tolist(), 'base_mass_kg':total,
                'base_com_m':com.tolist(), 'base_inertia_kg_m2':inertia.tolist(),
                'unloaded_robot_mass_kg':self._robot_mass, 'loaded_robot_mass_kg':self._robot_mass+mass,
                'payload_collision_geometry':False,
                'model':'rigidly attached inertial payload; no thermal model'}

    def _observation_groups(self):
        groups = self.base.obs_buf
        for name, dimension in (('policy',58), ('critic',237)):
            if name not in groups or tuple(groups[name].shape) != (1, dimension):
                raise ValueError(f'Expected one R6 {name} observation with {dimension} channels')
        active = getattr(self.base.observation_manager, 'active_terms', {})
        for name in ('policy','critic'):
            if name in active and list(active[name])[-1:] != ['payload']:
                raise ValueError(f'R6 {name} payload must remain the last observation term')
        return groups

    def _capture(self):
        # PhysX getters复用CPU缓冲；每一份都clone，避免后续get或set覆盖回滚依据。
        return {'masses':self.view.get_masses().clone(), 'coms':self.view.get_coms().clone(),
                'inertias':self.view.get_inertias().clone()}

    def _write(self, values):
        from scipy.spatial.transform import Rotation
        indices = torch.tensor([0], dtype=torch.int32, device='cpu')
        coms, inertias = values['coms'].clone(), values['inertias'].clone()
        # getter返回link轴完整惯量，但Isaac Sim 5.0 setter只采用主惯量，
        # 不同步主轴方向。GPU 65→0实测若保留旧Q，误差达0.0136023 kg m²。
        # 因此显式编码 I_link = R diag(lambda) R^T，COM保存R，惯量表保存lambda。
        # 回滚也走此路径，从旧完整惯量重新编码，不能恢复旧矩阵却遗留新主轴。
        tensor = inertias[0,self.body_id].double().cpu().numpy().reshape(3,3)
        moments, axes = np.linalg.eigh((tensor+tensor.T)*.5)
        if not np.isfinite(moments).all() or np.min(moments) <= 0.:
            raise ValueError('Base inertia must have finite positive principal moments')
        if np.linalg.det(axes) < 0.:
            axes[:,-1] *= -1.  # 四元数仅表示右手旋转；翻转一列不改变完整惯量。
        coms[0,self.body_id,3:] = torch.tensor(Rotation.from_matrix(axes).as_quat(),dtype=coms.dtype)
        inertias[0,self.body_id] = torch.tensor(np.diag(moments),dtype=inertias.dtype).reshape(9)
        self.view.set_masses(values['masses'], indices)
        self.view.set_coms(coms, indices)
        self.view.set_inertias(inertias, indices)

    def _expected(self, properties, reference):
        expected = {key:value.clone() for key,value in reference.items()}
        expected['masses'][0,self.body_id] = properties['base_mass_kg']
        expected['coms'][0,self.body_id,:3] = torch.tensor(properties['base_com_m'], dtype=reference['coms'].dtype)
        expected['inertias'][0,self.body_id] = torch.tensor(properties['base_inertia_kg_m2'], dtype=reference['inertias'].dtype).reshape(9)
        return expected

    def _invalidate_com_cache(self):
        # update(0)不会递增时间戳。清除COM及依赖COM换算的状态/速度缓存即可，
        # 不推进物理、不写姿态/速度，不触碰关节加速度的差分历史。
        for name, buffer in vars(self.robot.data).items():
            if name.startswith(('_root_', '_body_')) and hasattr(buffer, 'timestamp'):
                buffer.timestamp = -1.0

    def _refresh_observations(self, mass):
        groups = self._observation_groups()
        refreshed = self.base.observation_manager.compute(update_history=False)
        with torch.no_grad():
            for name in ('policy','critic'):
                value = refreshed[name]
                if value.shape != groups[name].shape or not torch.isfinite(value).all():
                    raise RuntimeError(f'Invalid refreshed R6 {name} observation')
                expected = torch.full_like(value[:, -1], mass/65.0)
                torch.testing.assert_close(value[:, -1], expected, rtol=1e-6, atol=1e-7)
                # 原位复制，保留worker刚从step/reset拿到的obs引用和Recorder输入一致。
                groups[name].copy_(value)

    @staticmethod
    def _check_com_quaternions(actual, expected):
        """只对COM姿态容许float32归一化舍入和q/-q等价，不容许真实旋转。

        Isaac Sim 5.0实测set_coms会将未改变的其他link四元数重新归一化，
        可产生1.1920928955078125e-7分量差；质量/COM位置/惯量不因此放宽。
        """
        if actual.numel() == 0:
            return {'max_sign_aligned_component_error':0., 'max_unit_norm_error':0.,
                    'max_normalized_orientation_component_error':0., 'tolerance':1e-6}
        actual64, expected64 = actual.double(), expected.double()
        actual_norm = torch.linalg.vector_norm(actual64, dim=-1, keepdim=True)
        expected_norm = torch.linalg.vector_norm(expected64, dim=-1, keepdim=True)
        torch.testing.assert_close(actual_norm, torch.ones_like(actual_norm), rtol=0, atol=1e-6)
        torch.testing.assert_close(expected_norm, torch.ones_like(expected_norm), rtol=0, atol=1e-6)
        sign = torch.where((actual64*expected64).sum(-1,keepdim=True) < 0., -1., 1.)
        aligned = actual64*sign
        normalized, expected_normalized = aligned/actual_norm, expected64/expected_norm
        torch.testing.assert_close(normalized, expected_normalized, rtol=0, atol=1e-6)
        return {'max_sign_aligned_component_error':float((aligned-expected64).abs().max()),
                'max_unit_norm_error':float(torch.maximum((actual_norm-1.).abs(), (expected_norm-1.).abs()).max()),
                'max_normalized_orientation_component_error':float((normalized-expected_normalized).abs().max()),
                'tolerance':1e-6,
                'semantics':'unit quaternion orientation equivalence; q and -q accepted; float32 normalization only'}

    @staticmethod
    def _check_inertia_reconstruction(actual, expected):
        """非Base惯量允许主轴分解/重构的float32舍入，按整块矩阵尺度限定。

        每个link的界限为32*float32_eps*max(abs(reference_3x3))；交叉项可能
        近零，因此不使用单个分量的相对误差。实测GPU重构差约6.98e-10 kg m²。
        """
        error = (actual.double()-expected.double()).abs().amax(dim=-1)
        scale = expected.double().abs().amax(dim=-1)
        bound = 32.*torch.finfo(torch.float32).eps*scale
        if torch.any(error > bound):
            raise AssertionError('Non-base inertia changed beyond float32 reconstruction bound: '
                                 f'max_error={float(error.max()):.12g}, max_bound={float(bound.max()):.12g}')
        return {'epsilon_multiplier':32., 'float32_epsilon':torch.finfo(torch.float32).eps,
                'per_body_max_abs_error_kg_m2':error.cpu().tolist(),
                'per_body_matrix_scale_kg_m2':scale.cpu().tolist(),
                'per_body_abs_bound_kg_m2':bound.cpu().tolist(),
                'max_abs_error_kg_m2':float(error.max()) if error.numel() else 0.,
                'semantics':'per-body max matrix component times 32 float32 epsilon; no near-zero component relative tolerance'}

    def _verify(self, actual, expected, properties=None):
        # 非Base质量/COM位置逐位保留；惯量只允许主轴分解重构的float32舍入。
        # Base_link四元数可能被PhysX重算为新主惯量轴，完整惯量另行核对。
        other = [i for i in range(actual['masses'].shape[1]) if i != self.body_id]
        for name in ('masses','coms','inertias'):
            if not torch.isfinite(actual[name]).all():
                raise RuntimeError('PhysX returned non-finite '+name)
            if name == 'masses':
                torch.testing.assert_close(actual[name][:,other], expected[name][:,other], rtol=0, atol=0)
        torch.testing.assert_close(actual['coms'][:,other,:3], expected['coms'][:,other,:3], rtol=0, atol=0)
        self._check_com_quaternions(actual['coms'][:,other,3:], expected['coms'][:,other,3:])
        self._check_inertia_reconstruction(actual['inertias'][:,other], expected['inertias'][:,other])
        torch.testing.assert_close(actual['masses'], expected['masses'], rtol=1e-5, atol=2e-5)
        torch.testing.assert_close(actual['coms'][...,:3], expected['coms'][...,:3], rtol=1e-5, atol=2e-6)
        torch.testing.assert_close(actual['inertias'], expected['inertias'], rtol=1e-4, atol=2e-6)
        if properties is not None:
            wanted = torch.tensor(properties['loaded_robot_mass_kg'], dtype=actual['masses'].dtype)
            torch.testing.assert_close(actual['masses'].sum(), wanted, rtol=1e-5, atol=2e-4)

    def set_mass(self, mass):
        """在策略步边界同步写三项物理属性，读回成功后再更新已知载荷观测。

        失败会回滚并抛RuntimeError；若回滚也失败，错误明确要求结束回放。
        返回实测质量/COM/惯量与最大误差，供独立键盘事件日志记录。
        """
        mass = self._validate_mass(mass)
        properties = self.properties(mass)
        previous = self.mass_kg
        old = self._capture()
        # 其他body始终从初始只读表写入，避免多次重构把舍入误差逐步累积。
        # Base以link轴完整目标矩阵表达；_write负责一致编码主轴和主惯量。
        expected = self._expected(properties, self._reference)
        old_observations = {k:v.clone() for k,v in self._observation_groups().items() if k in ('policy','critic')}
        # 不调用任何pose setter；额外读回验证后端没有改变当前机器人姿态/关节角。
        poses = {name:getattr(self.view,name)().clone() for name in ('get_root_transforms','get_dof_positions')
                 if callable(getattr(self.view,name,None))}
        try:
            self._write(expected)
            actual = self._capture()
            self._verify(actual, expected, properties)
            for name, before in poses.items():
                torch.testing.assert_close(getattr(self.view,name)(), before, rtol=0, atol=1e-6)
            self._invalidate_com_cache()
            self.base.cfg.payload_mass_kg = mass
            self._refresh_observations(mass)
        except Exception as error:
            rollback_error = None
            try:
                self._write(old)
                self._verify(self._capture(), old)
            except Exception as rollback:
                rollback_error = rollback
            finally:
                self._invalidate_com_cache()
                self.base.cfg.payload_mass_kg = previous
                with torch.no_grad():
                    for name,value in old_observations.items():
                        self.base.obs_buf[name].copy_(value)
            if rollback_error is not None:
                raise RuntimeError(f'Payload update failed ({error}); rollback also failed ({rollback_error}); stop playback') from error
            raise RuntimeError(f'Payload update failed and previous physics/observations were restored: {error}') from error
        self.mass_kg = mass
        return self._record(previous, properties, actual, 'immediate_readback')

    def verify_current(self):
        """独立读回当前物理属性；可在变载后的真实env.step之后再次调用。

        此方法不更新任何物理/观测、不恢复错误，用于确认跨物理步仍生效。
        检查失败必须结束回放，不能把setter成功等同于后续仿真已采用新参数。
        """
        properties = self.properties(self.mass_kg)
        actual = self._capture()
        self._verify(actual, self._expected(properties, self._reference), properties)
        if float(self.base.cfg.payload_mass_kg) != self.mass_kg:
            raise RuntimeError('Runtime payload configuration no longer matches verified physical mass')
        for name, values in self._observation_groups().items():
            if name in ('policy','critic'):
                torch.testing.assert_close(values[:, -1], torch.full_like(values[:, -1], self.mass_kg/65.0),
                                           rtol=1e-6, atol=1e-7)
        return self._record(self.mass_kg, properties, actual, 'independent_readback')

    def _record(self, previous, properties, actual, phase):
        mass = properties['payload_kg']
        actual_base_mass = float(actual['masses'][0,self.body_id])
        actual_com = actual['coms'][0,self.body_id,:3].cpu().numpy()
        actual_inertia = actual['inertias'][0,self.body_id].cpu().numpy().reshape(3,3)
        errors = {'base_mass_kg':abs(actual_base_mass-properties['base_mass_kg']),
                  'total_mass_kg':abs(float(actual['masses'].sum())-properties['loaded_robot_mass_kg']),
                  'com_max_abs_m':float(np.max(np.abs(actual_com-np.asarray(properties['base_com_m'])))),
                  'inertia_max_abs_kg_m2':float(np.max(np.abs(actual_inertia-np.asarray(properties['base_inertia_kg_m2']))))}
        other = [i for i in range(actual['masses'].shape[1]) if i != self.body_id]
        orientation_check = self._check_com_quaternions(actual['coms'][:,other,3:], self._reference['coms'][:,other,3:])
        inertia_check = self._check_inertia_reconstruction(actual['inertias'][:,other], self._reference['inertias'][:,other])
        inertia_check['body_indices'] = other
        return {'previous_payload_kg':previous, 'requested_payload_kg':mass, 'payload_kg':mass,
                'requested':properties, 'actual':{'base_mass_kg':actual_base_mass,
                'loaded_robot_mass_kg':float(actual['masses'].sum()), 'base_com_m':actual_com.tolist(),
                'base_inertia_kg_m2':actual_inertia.tolist(),
                'base_com_quat_xyzw':actual['coms'][0,self.body_id,3:].cpu().tolist()}, 'max_abs_errors':errors,
                'canonical_urdf_sha256':self.canonical_sha256, 'observations_updated':True,
                'physical_properties_verified':True, 'reset_performed':False,
                'verification_phase':phase, 'inertia_write_encoding':'Base COM principal axes quaternion + diagonal principal moments; getter verified in link axes',
                'non_base_com_orientation_check':orientation_check,
                'non_base_inertia_reconstruction_check':inertia_check}
