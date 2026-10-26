"""R1 V8: change four wheel numerical body caps before cloning/physics startup."""
from copy import deepcopy
import isaaclab.sim as sim_utils

WHEEL_NAMES=frozenset(f'{leg}_wheel_link' for leg in ('FR','FL','RR','RL'))
SOURCE_BODY_AUDIT={}


def _body_properties(prim):
    return {str(a.GetName()):str(a.Get()) for a in prim.GetAuthoredAttributes()
            if str(a.GetName()).startswith(('physics:', 'physx'))}


@sim_utils.clone
def spawn_lighthw_wheel_rigid_cap(prim_path,cfg,translation=None,orientation=None,**kwargs):
    """四轮刚体1000→4000deg/s；先改源机器人再复制，关节速度另设60rad/s，力矩仍48Nm。"""
    from pxr import Usd,UsdPhysics,PhysxSchema
    from rl_training.assets.deeprobotics import spawn_lighthw_ironman
    # The outer clone decorator resolves one concrete source path. The original
    # decorated spawner sees only that path, so no clone can retain the old cap.
    prim=spawn_lighthw_ironman(prim_path,cfg,translation=translation,orientation=orientation,**kwargs)
    before={};after={};relative_paths={};changed=[]
    for body in Usd.PrimRange(prim):
        if not body.HasAPI(UsdPhysics.RigidBodyAPI):continue
        name=body.GetName();api=PhysxSchema.PhysxRigidBodyAPI(body)
        if name in before:raise ValueError('Duplicate rigid-body name: '+name)
        before[name]=_body_properties(body)
        relative_paths[name]=str(body.GetPath().MakeRelativePath(prim.GetPath()))
        if api.GetMaxAngularVelocityAttr().Get()!=1000.:raise ValueError('Unexpected original body cap: '+name)
        if name in WHEEL_NAMES:
            api.GetMaxAngularVelocityAttr().Set(4000.)
            changed.append(name)
        after[name]=_body_properties(body)
    if set(changed)!=WHEEL_NAMES or len(before)!=17:raise ValueError('Expected exactly four wheels among17 bodies')
    delta=[]
    for name,properties in before.items():
        for key in properties.keys()|after[name].keys():
            if properties.get(key)!=after[name].get(key):
                delta.append(dict(body=name,attribute=key,before=properties.get(key),after=after[name].get(key)))
    if len(delta)!=4 or any(d['body'] not in WHEEL_NAMES or d['attribute']!='physxRigidBody:maxAngularVelocity'
                           or d['before']!='1000.0' or d['after']!='4000.0' for d in delta):
        raise ValueError('Only four wheel maxAngularVelocity attributes may change: '+repr(delta))
    SOURCE_BODY_AUDIT.clear()
    SOURCE_BODY_AUDIT.update(source_prim=str(prim.GetPath()),before=before,after=after,
        relative_paths=relative_paths,differences=delta,unit='deg/s',changed_before_cloning_and_physics=True)
    return prim


def audit_runtime_caps(env):
    """读取所有环境实际USD刚体属性，防止仅env0改对；不写训练状态。"""
    from pxr import PhysxSchema
    if not SOURCE_BODY_AUDIT:raise ValueError('Expected route25 spawner was not executed')
    stage=sim_utils.get_current_stage();count=0;wheel_count=0
    # All bodies have identical relative paths in the cloned articulations.
    root_suffix=SOURCE_BODY_AUDIT['source_prim'].split('/env_0',1)[1]
    for env_path in env.scene.env_prim_paths:
        for name,relative in SOURCE_BODY_AUDIT['relative_paths'].items():
            path=f'{env_path}{root_suffix}/{relative}'
            prim=stage.GetPrimAtPath(path)
            if not prim.IsValid():raise ValueError('Missing cloned rigid body: '+path)
            value=PhysxSchema.PhysxRigidBodyAPI(prim).GetMaxAngularVelocityAttr().Get()
            expected=4000. if name in WHEEL_NAMES else 1000.
            if value!=expected:raise ValueError(f'Wrong runtime cap: {path}={value}, expected{expected}')
            count+=1;wheel_count+=int(name in WHEEL_NAMES)
    if count!=17*env.num_envs or wheel_count!=4*env.num_envs:raise ValueError('Incomplete clone-cap coverage')
    return dict(status='matched',unit='deg/s',environments=env.num_envs,checked_bodies=count,
        wheel_bodies=wheel_count,wheel_cap=4000.,other_body_cap=1000.,source=deepcopy(SOURCE_BODY_AUDIT))
