"""Route16：route14的数值配置＋正确观测，独立从零PPO参考任务。"""
from isaaclab.utils import configclass

from .ppo_leg_desaturation_curriculum import LightHWLegDesaturationLegAction, LightHWLegDesaturationLegActionCfg
from .ppo_leg_desaturation_env_cfg import DeeproboticsLightHWLegDesaturationEnvCfg
from .reference_observations import (
    OBSERVATION_VERSION, REFERENCE_PROFILE, WHEEL_JOINT_NAMES,
    joint_positions_v1, validate_observation_cfg,
)
from .reference_metrics import ReferenceTransitionMonitor, ReferenceWheelMonitor


class LightHWReferenceLegAction(LightHWLegDesaturationLegAction):
    """仅扩展版本签名；保持原tanh映射、范围、奖励和固定1.0上限。"""
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        self.transition_monitor=ReferenceTransitionMonitor(self.num_envs,self.device,env.step_dt)
        self.wheel_monitor=ReferenceWheelMonitor(self.device)
        self._reference_contact_ids=None

    def record_quality(self):
        super().record_quality()
        if not self._training_curriculum:return
        sensor=self._env.scene['contact_forces']
        if self._reference_contact_ids is None:
            wheels=[name.replace('_joint','_link') for name in WHEEL_JOINT_NAMES]
            wheel_ids=sensor.find_bodies(wheels,preserve_order=True)[0]
            other=[i for i,name in enumerate(sensor.body_names) if name not in wheels]
            self._reference_contact_ids=(wheel_ids,other)
        wheel_ids,other=self._reference_contact_ids
        force=sensor.data.net_forces_w
        self.transition_monitor.update(self._env.command_manager.get_command('base_velocity'),
            self._asset.data.root_lin_vel_b,self._asset.data.root_pos_w[:,:2],self._asset.data.projected_gravity_b,
            force[:,wheel_ids,2],force[:,other].norm(dim=-1))
        wheel=self._env.action_manager.get_term('joint_vel')
        self.wheel_monitor.update(self.transition_monitor,self._env.command_manager.get_command('base_velocity'),
            wheel.raw_actions,wheel.processed_actions,self._asset.data.joint_vel[:,wheel._joint_ids],
            self._asset.data.applied_torque[:,wheel._joint_ids],force[:,wheel_ids,2])

    def reset(self,env_ids=None):
        super().reset(env_ids)
        if hasattr(self,'transition_monitor'):self.transition_monitor.reset(env_ids)

    def _signature_payload(self):
        payload = super()._signature_payload()
        payload.update(profile_id=REFERENCE_PROFILE,
                       observation_contract=validate_observation_cfg(self._env.cfg))
        return payload


@configclass
class LightHWReferenceLegActionCfg(LightHWLegDesaturationLegActionCfg):
    class_type: type = LightHWReferenceLegAction


@configclass
class LightHWPPOReferenceEnvCfg(DeeproboticsLightHWLegDesaturationEnvCfg):
    lighthw_observation_version: str = OBSERVATION_VERSION

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1024
        original = self.actions.joint_pos
        self.actions.joint_pos = LightHWReferenceLegActionCfg(
            asset_name=original.asset_name, joint_names=list(original.joint_names),
            preserve_order=original.preserve_order, use_default_offset=original.use_default_offset,
            scale=original.scale, clip=original.clip,
        )
        for group in (self.observations.policy, self.observations.critic):
            group.joint_pos.func = joint_positions_v1
            group.joint_pos.params = {
                "asset_cfg": group.joint_pos.params["asset_cfg"],
                "wheel_joint_names": WHEEL_JOINT_NAMES,
            }
        validate_observation_cfg(self)
        print(f"[LightHW Reference] policy=60, critic=236, action=16; {OBSERVATION_VERSION}; "
              "route14 numerical configuration; all 16 PPO actions; fresh initialization; speed cap=1.0")
