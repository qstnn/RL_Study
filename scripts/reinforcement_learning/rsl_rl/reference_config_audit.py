"""Check the instantiated reference against route14's actual saved numeric configuration."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[3]
SNAPSHOT=ROOT/'model_lighthw/training/routes/route_16_ppo_observation_reference/configs/route14_effective_reference.json'


def _normalize(value):
    if isinstance(value,dict):return {k:_normalize(v) for k,v in value.items()}
    if isinstance(value,list):return [_normalize(v) for v in value]
    if isinstance(value,str):return value.replace('{ENV_REGEX_NS}','/World/envs/env_.*')
    return value


def _environment(env,*,reference):
    result=_normalize(deepcopy(env))
    terrain=result['scene']['terrain']
    terrain['num_envs']='<parallelism>'
    terrain['env_spacing']=result['scene']['env_spacing']
    action_class=result['actions']['joint_pos']['class_type']
    expected_class=('ppo_leg_desaturation_curriculum:LightHWLegDesaturationLegAction' if reference
                    else 'ppo_reference_env_cfg:LightHWReferenceLegAction')
    if not action_class.endswith(expected_class):raise ValueError('Unexpected reference leg action identity')
    result['actions']['joint_pos']['class_type']='<versioned identity; physical fields checked>'
    for name in ('policy','critic'):
        term=result['observations'][name]['joint_pos']
        if reference:
            expected='rl_training.tasks.manager_based.locomotion.velocity.mdp.observations:joint_pos_rel_without_wheel'
            if term['func']!=expected:raise ValueError('Unexpected source observation function')
            del term['params']['wheel_asset_cfg']
        else:
            if not term['func'].endswith('reference_observations:joint_positions_v1'):
                raise ValueError('New reference must have corrected observations')
            del term['params']['wheel_joint_names']
        term['func']='<only intended observation-content replacement>'
    return result


def differences(a,b,path=''):
    if isinstance(a,dict) and isinstance(b,dict):
        return [item for key in sorted(a.keys()|b.keys())
                for item in differences(a.get(key),b.get(key),path+'/'+key)]
    if a!=b:return [dict(path=path,source=a,current=b)]
    return []


def audit_reference_configuration(env_cfg,agent_cfg):
    frozen=json.loads(SNAPSHOT.read_text())
    # Same serialization as Isaac Lab dump_yaml, without introducing temporary files.
    current=yaml.load(yaml.dump(env_cfg.to_dict()),Loader=yaml.BaseLoader)
    current_agent=yaml.load(yaml.dump(agent_cfg.to_dict()),Loader=yaml.BaseLoader)
    selected={k:current[k] for k in frozen['environment']}
    selected['scene']={k:current['scene'][k] for k in frozen['environment']['scene']}
    actual_agent={k:current_agent[k] for k in frozen['agent']}
    drift=differences(_environment(frozen['environment'],reference=True),_environment(selected,reference=False))
    drift+=differences(frozen['agent'],actual_agent,'/agent')
    if drift:raise ValueError('Reference numeric configuration drift:\n'+json.dumps(drift,ensure_ascii=False,indent=2))
    return dict(status='matched',source=frozen['source'],source_sha256=frozen['source_sha256'],
        snapshot_sha256=hashlib.sha256(SNAPSHOT.read_bytes()).hexdigest(),unexpected_differences=drift,
        intended_changes=['named position observation v1 in policy and critic','independent action identity/profile',
                          'fresh network/noise/Identity normalizers/Adam','1000-update observation budget'],
        preserved='URDF/default pose/actions/limits/actuators/physics/rewards/noise/PPO/command distribution')
