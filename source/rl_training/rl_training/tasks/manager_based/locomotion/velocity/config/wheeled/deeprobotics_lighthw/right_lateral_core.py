"""Route51：仅将非零命令的右侧概率从0.5提高到0.75。"""
from copy import deepcopy
import hashlib
import json
import torch
VERSION='lighthw_flat_right_priority_commands_v1'
# 独立纯张量模块允许不启动Isaac Sim就检查真实采样与状态迁移。
COMMAND_CONTRACT=dict(version=VERSION,terrain='flat_all_tiles',terrain_progression=False,
    vx=[0.,0.],vy=[-1.,1.],wz=[0.,0.],minimum_abs_vy=.2,
    magnitude_rule='each direction: uniform[0,1] clamped to0.2; unchanged conditional magnitude distribution',
    zero_command_probability=.25,resampling_seconds=8.,moving_direction_probabilities=[.25,.75],
    direction_order=['left_positive_vy','right_negative_vy'],right_probability=.75,
    retention_fraction=0.,heading_command=False,all_16_actions_from_actor=True,
    behavior_cloning=False,wheel_feedforward=False)

def lateral_commands(random_signed,stop,right_probability=.75):
    """目的：增加右移练习；只在命令重采样调用，返回[N,3]机身速度命令。

    沿用一次stop和一次signed均匀随机抽样，不额外消耗RNG。按CDF的两段
    反变换分别得到左右的均匀幅值，避免单纯移动正负阈值导致速度分布偏斜。
    每方向幅值max(U[0,1],0.2)，包含原有20%的0.2点质量；不读实际车速，
    不生成任何腿轮目标。right_probability=0.5可复现原等概率映射。
    """
    if random_signed.ndim!=1 or stop.shape!=random_signed.shape or stop.dtype!=torch.bool:
        raise ValueError('Expected matching1D signed samples and boolean stop mask')
    if not 0<float(right_probability)<1:raise ValueError('Direction probability must be strictly between0 and1')
    if not torch.isfinite(random_signed).all() or (random_signed.abs()>1).any():
        raise ValueError('Uniform sample outside[-1,1]')
    u=(random_signed+1)*.5
    right=u<right_probability
    magnitude=torch.where(right,1-u/right_probability,(u-right_probability)/(1-right_probability)).clamp_min(.2)
    vy=torch.where(right,-magnitude,magnitude)
    result=torch.zeros((len(vy),3),device=vy.device,dtype=vy.dtype)
    result[:,1]=torch.where(stop,torch.zeros_like(vy),vy)
    return result

def migrate_state(state,target_mapping):
    """只显式迁移命令分布签名；保护物理动作、观测、噪声和全部训练计数。"""
    result=deepcopy(state);expected=deepcopy(state['action_mapping'])
    expected['command_distribution']=deepcopy(COMMAND_CONTRACT)
    if expected!=target_mapping:raise ValueError('Unexpected change beyond command distribution')
    result['action_mapping']=expected
    result['signature']=hashlib.sha256(json.dumps(expected,sort_keys=True).encode()).hexdigest()
    return result
