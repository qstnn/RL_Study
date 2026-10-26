"""Route48：平地纯侧移任务契约和可独立测试的命令采样规则。"""
from copy import deepcopy
import torch
VERSION='lighthw_flat_lateral_commands_v1'
COMMAND_CONTRACT=dict(version=VERSION,terrain='flat_all_tiles',terrain_progression=False,
    vx=[0.,0.],vy=[-1.,1.],wz=[0.,0.],minimum_abs_vy=.2,
    magnitude_rule='signed uniform[-1,1], clamp abs to0.2; same as route46 pure lateral conditional',
    zero_command_probability=.25,resampling_seconds=8.,moving_direction_probabilities=[.5,.5],
    retention_fraction=0.,heading_command=False,all_16_actions_from_actor=True,
    behavior_cloning=False,wheel_feedforward=False)

def lateral_commands(random_signed,stop):
    """将随机标量映射为机身vy命令；零标量取正号，stop仅由独立命令采样决定。

    不读取实际车速、不产生关节目标；返回[N,3]，非零幅值0.2～1m/s。
    """
    if random_signed.ndim!=1 or stop.shape!=random_signed.shape:raise ValueError('Expected matched1D sampling tensors')
    if not torch.isfinite(random_signed).all() or (random_signed.abs()>1).any():raise ValueError('Uniform sample outside[-1,1]')
    sign=torch.where(random_signed>=0,1.,-1.)
    vy=sign*random_signed.abs().clamp_min(.2)
    result=torch.zeros((len(vy),3),device=vy.device,dtype=vy.dtype)
    result[:,1]=torch.where(stop,torch.zeros_like(vy),vy)
    return result

def migrate_state(state,target_mapping):
    """只迁移命令契约；正确观测、腿轮映射、计数、噪声和优化器不在这里改变。"""
    import hashlib,json
    result=deepcopy(state);expected=deepcopy(state['action_mapping'])
    expected['command_distribution']=deepcopy(COMMAND_CONTRACT)
    if expected!=target_mapping:raise ValueError('Unexpected physical action or observation change')
    result['action_mapping']=expected
    result['signature']=hashlib.sha256(json.dumps(expected,sort_keys=True).encode()).hexdigest()
    return result
