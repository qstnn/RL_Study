"""局部64工况物理接线；不改旧模块，不实现奖励/准备时钟/PPO，也不自动启动仿真。"""
from copy import deepcopy
from functools import lru_cache
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np

from .contract import CONTRACT, SOURCE, SPEC_PATH, V7, _load, sha256
from .dataset import EvidenceError
from .schedule import make_schedule, validate_pre_action_conditions


def planned_layout(schedule):
    """只返回计划；同表同时供terrain和env消费，不能声明运行覆盖。"""
    expected = make_schedule(schedule['num_envs'],mode=schedule['mode'])
    if schedule != expected:
        raise EvidenceError('Migration schedule/source signature changed')
    spec = _load(SPEC_PATH,'_r1_adapter_layout_spec')
    values = np.asarray(schedule['conditions'],dtype=float)
    types = np.array([spec.ANGLES.index(float(slope)) for slope in values[:,0]],dtype=np.int64)
    origins = np.array([[0.,(lane-5)*14.,0.] for lane in types])
    return dict(conditions=values,terrain_types=types,env_origins=origins,
        condition_tuple=tuple(tuple(float(value) for value in row) for row in values))


def validate_assignment(schedule, conditions, terrain_types, env_origins, observation):
    """纯数组核对；实际来源由verify_pre_action的原生读取负责。"""
    plan = planned_layout(schedule)
    actual,types,origins,obs = map(np.asarray,(conditions,terrain_types,env_origins,observation))
    n = schedule['num_envs']
    if (actual.shape != (n,4) or types.shape != (n,) or origins.shape != (n,3)
            or obs.shape != (n,300) or not all(np.isfinite(x).all() for x in (actual,types,origins,obs))):
        raise EvidenceError('Malformed migration pre-action condition/terrain/observation evidence')
    if not np.allclose(actual[:,:3],plan['conditions'],atol=1.e-4,rtol=0.):
        raise EvidenceError('Actual per-slot conditions differ from the frozen all64 schedule')
    if not np.array_equal(types,plan['terrain_types']) or not np.allclose(origins,plan['env_origins'],atol=1.e-6,rtol=0.):
        raise EvidenceError('Actual terrain origins/types disagree with per-slot conditions')
    expected_context = plan['conditions'][:,1:]/np.array([15.,40.])
    if not np.allclose(obs[:,290:300].reshape(n,5,2),expected_context[:,None,:],atol=1.e-7,rtol=0.):
        raise EvidenceError('Actual Actor task-context history differs from nominal conditions')
    return validate_pre_action_conditions(actual,mode=schedule['mode'])


@lru_cache(maxsize=1)
def _runtime_types():
    """只能在AppLauncher之后调用；创建局部子类，没有修改原类/模块函数。"""
    import torch
    from rl_training.tasks.manager_based.locomotion.multiskill.lighthw.wheel_unified_v6.env import UnifiedEnv
    from rl_training.tasks.manager_based.locomotion.multiskill.lighthw.wheel_unified_v6.terrain import UnifiedTerrain
    from rl_training.tasks.manager_based.locomotion.multiskill.lighthw.wheel_unified_v6 import core

    class MigrationTerrain(UnifiedTerrain):
        def __init__(self,cfg):
            plan = planned_layout(cfg.migration_schedule)
            if len(plan['conditions']) != cfg.num_envs:
                raise EvidenceError('Migration terrain slot count differs')
            super().__init__(cfg)
            types = torch.as_tensor(plan['terrain_types'],device=self.device)
            self.terrain_types.copy_(types)
            self.env_origins.copy_(self.terrain_origins[0,types])
            if not np.allclose(self.env_origins.cpu().numpy(),plan['env_origins'],atol=1.e-6,rtol=0.):
                raise EvidenceError('Original terrain geometry no longer matches migration layout')

    class MigrationEnv(UnifiedEnv):
        def load_managers(self):
            # 原UnifiedEnv已建张量与Scene，但Command/Obs及startup payload尚未执行。
            schedule = self.cfg.migration_schedule
            plan = planned_layout(schedule)
            if schedule != self.cfg.scene.terrain.migration_schedule:
                raise EvidenceError('Environment and terrain do not share the same migration schedule')
            if hasattr(self,'r1_unloaded_mass'):
                raise EvidenceError('Migration assignment must precede the original payload startup')
            values = torch.as_tensor(plan['conditions'],dtype=torch.float32,device=self.cfg.sim.device)
            self.r1_slope_rad.copy_(values[:,0]*torch.pi/180.)
            self.r1_payload_kg.copy_(values[:,1]); self.r1_towed_kg.copy_(values[:,2])
            tangent,normal = core.plane_basis(self.r1_slope_rad)
            self.r1_tangent.copy_(tangent); self.r1_normal.copy_(normal)
            terrain = self.scene.terrain
            if (not np.array_equal(terrain.terrain_types.cpu().numpy(),plan['terrain_types'])
                    or not np.allclose(terrain.env_origins.cpu().numpy(),plan['env_origins'],atol=1.e-6,rtol=0.)):
                raise EvidenceError('Migration scene terrain disagrees before manager/startup loading')
            self.r1_robot_hitch[:,2] = self.cfg.hitch_height_m-.416223+.20
            self.r1_migration_assignment_installed = True
            super().load_managers()
            if not getattr(self,'r1_inertia_verified',False):
                raise EvidenceError('Original payload startup did not verify migration inertias')

    return MigrationTerrain,MigrationEnv


def configure_environment(output, num_envs, seed=43, mode='full_first_batch', fixed_speeds=None):
    """App后创建cfg/资产与计划manifest，不创建env；默认全槽移动，未接重拖8s命令。"""
    schedule = make_schedule(num_envs,mode=mode); plan = planned_layout(schedule)
    out = Path(output).resolve()
    if (out/'assets/cart.urdf').exists():
        raise EvidenceError('Use fresh migration assets; existing cart URDF will not be overwritten')
    speeds = np.where(plan['conditions'][:,1:].sum(1)==0.,3.1,.3) if fixed_speeds is None else np.asarray(fixed_speeds,dtype=float)
    upper = np.where(plan['conditions'][:,1:].sum(1)==0.,3.1,.6)
    if speeds.shape != (num_envs,) or not np.isfinite(speeds).all() or np.any((speeds<0.)|(speeds>upper)):
        raise EvidenceError('Expected per-slot original unloaded/loaded command ranges')
    import torch
    sys.path.insert(0,str(V7/'scripts'))
    import research_common as common
    from rounded_cart import replace_wheels
    terrain_type,_ = _runtime_types()
    out.mkdir(parents=True,exist_ok=True)
    cfg,agent = common.configure(out,num_envs,seed,16,4,noise=False)
    cfg.sim.device = 'cuda:0'
    cfg.fixed_slope_deg = cfg.fixed_payload_kg = cfg.fixed_towed_kg = None
    cfg.fixed_speed_mps = torch.as_tensor(speeds,dtype=torch.float32,device=cfg.sim.device)
    cfg.ground_friction = 3.; cfg.rolling_coefficient = .02; cfg.hitch_height_m = .04
    cfg.unloaded_acceleration = 1.; cfg.loaded_acceleration = .3; cfg.command_profile = 'slew'
    cfg.observations.policy.enable_corruption = False
    for material in (cfg.scene.terrain.physics_material,cfg.sim.physics_material):
        material.static_friction = material.dynamic_friction = 3.
        material.friction_combine_mode = 'min'
    cfg.scene.terrain.class_type = terrain_type
    cfg.migration_schedule = deepcopy(schedule)
    cfg.scene.terrain.migration_schedule = deepcopy(schedule)
    geometry = replace_wheels(out/'assets/cart.urdf')
    source_files = {Path(__file__),Path(__file__).with_name('schedule.py'),Path(__file__).with_name('contract.py'),
        V7/'scripts/research_common.py',V7/'scripts/settings.py',V7/'scripts/rounded_cart.py'}
    source_files |= set((SOURCE/'wheel_unified_v6').glob('*.py'))
    source_files |= {Path(name) for name in schedule['catalog_source_sha256']}
    manifest = dict(artifact_kind='migration64_configuration_only',schedule=schedule,
        fixed_speed_mps=speeds.tolist(),command_mode='fixed_vector_all_moving' if fixed_speeds is None else 'fixed_vector_explicit',
        command_recycling_enabled=False,towing_preparation_8s_installed_by_adapter=False,
        adapter_modifies_rewards=False,default_reset='original ordinary pose and zero velocity',
        observation_noise=False,physical_randomization=False,device='cuda:0',ground_friction=3.,
        robot_solver_requested=[16,4],cart_solver_requested=[16,4],hitch_height_m=.04,robot_hitch_offset_m=.20,
        robot_urdf_sha256=sha256(cfg.scene.robot.spawn.asset_path),cart_urdf_sha256=sha256(out/'assets/cart.urdf'),
        rounded_cart_geometry=geometry,source_sha256={str(path.resolve()):sha256(path) for path in sorted(source_files)},
        runtime_pre_action_verified=False,behavior_accepted=False,formal_ppo_authorized=False)
    cfg.migration_configuration_manifest = deepcopy(manifest)
    (out/'migration_schedule.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    return cfg,agent,manifest


def make_environment(cfg, **kwargs):
    """App后构造局部原生环境；调用方负责reset、Actor及GPU运行调度。"""
    if not hasattr(cfg,'migration_schedule'):
        raise EvidenceError('Use configure_environment before creating the migration environment')
    return _runtime_types()[1](cfg,**kwargs)


def _cpu(value):
    return value.detach().cpu().numpy().copy() if hasattr(value,'detach') else np.asarray(value)


def verify_pre_action(env, observation, output=None):
    """reset后、Actor首个动作前读取实际槽位/质量/COM/惯量/观测；不执行step。"""
    base = env.unwrapped
    if not isinstance(base,_runtime_types()[1]) or not getattr(base,'r1_migration_assignment_installed',False):
        raise EvidenceError('Expected the actual local MigrationEnv instance')
    cfg = base.cfg; schedule = cfg.migration_schedule; plan = planned_layout(schedule)
    if cfg.sim.device != 'cuda:0' or str(base.scene['robot'].data.root_state_w.device) != 'cuda:0':
        raise EvidenceError('Migration first-batch evidence requires actual GPU tensors')
    if cfg.observations.policy.enable_corruption:
        raise EvidenceError('Migration observation noise must remain disabled')
    if any(getattr(cfg,key) is not None for key in ('fixed_slope_deg','fixed_payload_kg','fixed_towed_kg')):
        raise EvidenceError('Migration categorical fixed overrides must remain None')
    if not np.all(_cpu(base.episode_length_buf)==0) or not np.all(_cpu(base.r1_eval_phase)==0):
        raise EvidenceError('First pre-action verification must follow ordinary reset before stepping')
    condition = np.column_stack((_cpu(base.r1_slope_rad)*180./np.pi,_cpu(base.r1_payload_kg),
        _cpu(base.r1_towed_kg),_cpu(base.r1_eval_phase)))
    coverage = validate_assignment(schedule,condition,_cpu(base.scene.terrain.terrain_types),
        _cpu(base.scene.env_origins),_cpu(observation))
    term = base.command_manager.get_term('base_velocity')
    if not np.allclose(_cpu(term.command),0.,atol=1.e-8,rtol=0.):
        raise EvidenceError('Migration reset exposed nonzero applied commands')
    if not np.allclose(_cpu(term.requested_vx),_cpu(cfg.fixed_speed_mps),atol=1.e-7,rtol=0.):
        raise EvidenceError('Migration requested speed vector differs from configuration')
    if (list(base.observation_manager.active_terms['policy']) != list(CONTRACT['policy_terms'])
            or len(base.r1_joint_ids)!=16):
        raise EvidenceError('Migration deployed observation/action layout differs')
    robot,cart = base.scene['robot'],base.scene['trailer']
    if [robot.joint_names[i] for i in base.r1_joint_ids] != CONTRACT['action_order']:
        raise EvidenceError('Migration actual named action order differs')
    masses = _cpu(robot.root_physx_view.get_masses()); coms = _cpu(robot.root_physx_view.get_coms())
    inertias = _cpu(robot.root_physx_view.get_inertias()).reshape(schedule['num_envs'],-1,3,3)
    original_mass = _cpu(base.r1_unloaded_mass); original_com = _cpu(base.r1_unloaded_com)
    original_inertia = _cpu(base.r1_unloaded_inertia).reshape(inertias.shape)
    base_id = robot.body_names.index('Base_link'); unchanged = [i for i in range(len(robot.body_names)) if i!=base_id]
    urdf = Path(cfg.scene.robot.spawn.asset_path)
    declared = {link.get('name'):float(link.find('inertial/mass').get('value'))
        for link in ET.parse(urdf).getroot().findall('link') if link.find('inertial/mass') is not None}
    if not np.allclose(original_mass,np.array([declared[name] for name in robot.body_names])[None,:],atol=1.e-5,rtol=1.e-5):
        raise EvidenceError('Migration original link masses differ from canonical URDF')
    core = _load(SOURCE/'wheel_unified_v6/core.py','_r1_adapter_inertia')
    expected_mass = original_mass.copy(); expected_mass[:,base_id] += plan['conditions'][:,1]
    expected_com = original_com[:,:,:3].copy(); expected_inertia = original_inertia.copy()
    for i,payload in enumerate(plan['conditions'][:,1]):
        _,expected_com[i,base_id],expected_inertia[i,base_id] = core.compose_inertia(
            original_mass[i,base_id],original_com[i,base_id,:3],original_inertia[i,base_id],payload,
            [0.,0.,.2],[.3,.25,.2])
    if (not np.allclose(masses,expected_mass,atol=1.e-4,rtol=1.e-5)
            or not np.allclose(coms[:,:,:3],expected_com,atol=1.e-5,rtol=0.)
            or not np.allclose(inertias,expected_inertia,atol=1.e-5,rtol=1.e-4)
            or not np.allclose(inertias[:,unchanged],original_inertia[:,unchanged],atol=1.e-6,rtol=1.e-4)):
        raise EvidenceError('Migration actual payload mass/COM/inertia differs from per-slot rigid composition')
    cart_masses = _cpu(cart.root_physx_view.get_masses()); cart_coms = _cpu(cart.root_physx_view.get_coms())
    cart_inertias = _cpu(cart.root_physx_view.get_inertias())
    ratios = np.where(plan['conditions'][:,2]>0,plan['conditions'][:,2]/40.,1.)
    if (not np.allclose(cart_masses,_cpu(base.r1_original_cart_mass)*ratios[:,None],atol=1.e-4,rtol=1.e-5)
            or not np.allclose(cart_coms[:,:,:3],_cpu(base.r1_original_cart_com)[:,:,:3],atol=1.e-5,rtol=0.)
            or not np.allclose(cart_inertias,_cpu(base.r1_original_cart_inertia)*ratios[:,None,None],atol=1.e-5,rtol=1.e-3)):
        raise EvidenceError('Migration actual passive cart mass/COM/inertia differs from per-slot towing mass')
    if any(getattr(base,key,False) is not True for key in ('r1_inertia_verified','r1_unloaded_inertia_urdf_verified','r1_nonbase_inertia_unchanged')):
        raise EvidenceError('Original physical mass/inertia startup verification is missing')
    initial_root = _cpu(robot.data.root_state_w); normal = _cpu(base.r1_normal)
    angles = np.deg2rad(plan['conditions'][:,0]); quat = np.zeros((len(angles),4));quat[:,0]=np.cos(angles/2);quat[:,2]=-np.sin(angles/2)
    q = _cpu(robot.data.joint_pos[:,base.r1_joint_ids]); qd = _cpu(robot.data.joint_vel[:,base.r1_joint_ids])
    default = np.array([0.,.6,-1.5,0.,.6,-1.5,0.,-.6,1.5,0.,-.6,1.5,0.,0.,0.,0.])
    if (not np.allclose(initial_root[:,:3],plan['env_origins']+.416223*normal,atol=1.e-5,rtol=0.)
            or not np.allclose(initial_root[:,3:7],quat,atol=1.e-6,rtol=0.)
            or not np.allclose(initial_root[:,7:],0.,atol=1.e-7,rtol=0.)
            or not np.allclose(q,default[None,:],atol=1.e-6,rtol=0.) or not np.allclose(qd,0.,atol=1.e-7,rtol=0.)):
        raise EvidenceError('Migration first action does not start from original ordinary zero-speed reset')
    cart_root = _cpu(cart.data.root_state_w); cart_qd = _cpu(cart.data.joint_vel)
    expected_cart = plan['env_origins']+.30*normal-1.73*_cpu(base.r1_tangent)
    inactive = plan['conditions'][:,2]==0.
    expected_cart[inactive,1]=500.; expected_cart[inactive,2]=1.
    robot_hitch,cart_hitch = _cpu(base.r1_robot_hitch),_cpu(base.r1_cart_hitch)
    if (not np.allclose(cart_root[:,:3],expected_cart,atol=1.e-5,rtol=0.)
            or not np.allclose(cart_root[:,3:7],quat,atol=1.e-6,rtol=0.)
            or not np.allclose(cart_root[:,7:],0.,atol=1.e-7,rtol=0.)
            or not np.allclose(cart_qd,0.,atol=1.e-7,rtol=0.)
            or not np.allclose(robot_hitch,[-.38,0.,.04-.416223+.20],atol=1.e-6,rtol=0.)
            or not np.allclose(cart_hitch,[.35,0.,.04-.30],atol=1.e-6,rtol=0.)):
        raise EvidenceError('Migration cart ordinary reset/parking or physical hitch geometry differs')
    # USD/材质实际读回；仅设置cfg不作为16/4与mu3证据。
    from isaaclab.sim import get_current_stage
    from pxr import PhysxSchema, UsdPhysics
    solver = {}; materials = {}
    for prim in get_current_stage().Traverse():
        path = str(prim.GetPath())
        if prim.HasAPI(PhysxSchema.PhysxSceneAPI) or (path.startswith('/World/envs/env_') and prim.HasAPI(UsdPhysics.ArticulationRootAPI)):
            solver[path] = {str(attr.GetName()):attr.Get() for attr in prim.GetAttributes() if 'Iteration' in str(attr.GetName())}
        if path.startswith(cfg.scene.terrain.prim_path+'/') and prim.HasAPI(UsdPhysics.MaterialAPI):
            material = UsdPhysics.MaterialAPI(prim); px = PhysxSchema.PhysxMaterialAPI(prim)
            materials[path] = dict(static=material.GetStaticFrictionAttr().Get(),dynamic=material.GetDynamicFrictionAttr().Get(),combine=px.GetFrictionCombineModeAttr().Get())
    for i in range(schedule['num_envs']):
        for suffix in ('Robot/Base_link','Trailer/cart_base'):
            values = solver.get(f'/World/envs/env_{i}/{suffix}',{})
            if any(values.get('physxArticulation:solver'+key+'IterationCount')!=value for key,value in [('Position',16),('Velocity',4)]):
                raise EvidenceError('Migration actual per-articulation solver differs')
    if any(solver.get('/physicsScene',{}).get('physxScene:max'+key+'IterationCount')!=value for key,value in [('Position',16),('Velocity',4)]):
        raise EvidenceError('Migration actual scene solver differs')
    if not materials or any(item!=dict(static=3.,dynamic=3.,combine='min') for item in materials.values()):
        raise EvidenceError('Migration actual ground material differs')
    asset_materials = {name:_cpu(base.scene[name].root_physx_view.get_material_properties())[:,:,:2] for name in ('robot','trailer')}
    if any(not np.allclose(value,3.,atol=1.e-6,rtol=0.) for value in asset_materials.values()):
        raise EvidenceError('Migration actual robot/cart friction differs')
    manifest = cfg.migration_configuration_manifest
    for name,key in [(urdf,'robot_urdf_sha256'),(Path(cfg.scene.trailer.spawn.asset_path),'cart_urdf_sha256')]:
        if sha256(name)!=manifest[key]:
            raise EvidenceError('Migration physical asset changed after configuration')
    if any(sha256(name)!=digest for name,digest in manifest['source_sha256'].items()):
        raise EvidenceError('Migration allocation/physics source changed after configuration')
    result = dict(artifact_kind='actual_migration64_pre_action_readback',coverage=coverage,
        actual_conditions_phase=condition.tolist(),actual_terrain_types=_cpu(base.scene.terrain.terrain_types).tolist(),
        actual_env_origins=_cpu(base.scene.env_origins).tolist(),actual_nominal_context=_cpu(observation)[:,298:300].tolist(),
        actual_robot_mass_kg=masses.sum(1).tolist(),actual_cart_mass_kg=cart_masses.sum(1).tolist(),
        disconnected_cart_slots=np.flatnonzero(inactive).tolist(),
        actual_robot_hitch_local_m=robot_hitch.tolist(),actual_cart_hitch_local_m=cart_hitch.tolist(),
        inactive_cart_note='Nominal zero tow keeps a disconnected 40kg cart parked away; it is not an active zero-mass trailer.',
        max_mass_error_kg=float(np.abs(masses-expected_mass).max()),max_com_error_m=float(np.abs(coms[:,:,:3]-expected_com).max()),
        max_inertia_error_kg_m2=float(np.abs(inertias-expected_inertia).max()),solver_runtime_usd=solver,terrain_materials=materials,
        runtime_pre_action_verified=True,payload_mass_com_inertia_verified=True,ordinary_reset_verified=True,
        behavior_accepted=False,formal_ppo_authorized=False)
    if output is not None:
        out = Path(output); out.mkdir(parents=True,exist_ok=True)
        raw = out/'migration_pre_action_physics.npz'
        if raw.exists() or (out/'migration_pre_action_review.json').exists():
            raise EvidenceError('Do not overwrite an earlier migration pre-action readback')
        np.savez_compressed(raw,conditions=condition,observation=_cpu(observation),robot_mass=masses,robot_com=coms,
            robot_inertia=inertias,unloaded_mass=original_mass,unloaded_com=original_com,unloaded_inertia=original_inertia,
            cart_mass=cart_masses,cart_com=cart_coms,cart_inertia=cart_inertias,
            robot_root_state=initial_root,cart_root_state=cart_root,cart_joint_velocity=cart_qd,
            robot_hitch_local=robot_hitch,cart_hitch_local=cart_hitch,joint_pos=q,joint_vel=qd,
            robot_material=asset_materials['robot'],cart_material=asset_materials['trailer'])
        result['raw_readback_sha256'] = sha256(raw)
        (out/'migration_pre_action_review.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    return result
