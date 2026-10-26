"""早期LightHW线性腿动作语义；纯数据，训练/回放/导出/监控共用。"""
from copy import deepcopy
VERSION='lighthw_linear_leg_urdf_v1'
CLASS_PATH='rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_lighthw.linear_leg_action:LightHWLinearURDFLegAction'
SCALE={'.*_hip_joint':.08,'.*_(thigh|calf)_joint':.14}
ORDER=['FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint', 'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint', 'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint', 'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint']
DEFAULT=[0.,.6,-1.5,0.,.6,-1.5,0.,-.6,1.5,0.,-.6,1.5]
LIMITS={'FR_hip_joint': (-0.4, 0.4), 'FR_thigh_joint': (-2.6952, 1.5613), 'FR_calf_joint': (-2.5, 2.5), 'FL_hip_joint': (-0.4, 0.4), 'FL_thigh_joint': (-2.6952, 1.5613), 'FL_calf_joint': (-2.5, 2.5), 'RR_hip_joint': (-0.4, 0.4), 'RR_thigh_joint': (-1.5613, 2.6952), 'RR_calf_joint': (-2.5, 2.5), 'RL_hip_joint': (-0.4, 0.4), 'RL_thigh_joint': (-2.6952, 1.5613), 'RL_calf_joint': (-2.5, 2.5)}
URDF_SHA256='5abb7c07c0ad401f7940018fee739ecc2fb4e0d3e8fc83274e75d1886db19fbb'


def contract():
    return dict(version=VERSION,mapping='clip(default+scale*raw,URDF_lower,URDF_upper)',
        order=list(ORDER),default=list(DEFAULT),scale=deepcopy(SCALE),limits=deepcopy(LIMITS),
        urdf_sha256=URDF_SHA256,tanh=False,all_16_actions_from_actor=True,
        legacy_stage_hip_leg_limits_used_for_targets=False)


def mapped_signature(source):
    result=deepcopy(source)
    result.update(mapping=VERSION,linear_leg_mapping=contract(),absolute_leg_clip=deepcopy(LIMITS))
    result.pop('hip_limits',None);result.pop('leg_limits',None)
    return result
