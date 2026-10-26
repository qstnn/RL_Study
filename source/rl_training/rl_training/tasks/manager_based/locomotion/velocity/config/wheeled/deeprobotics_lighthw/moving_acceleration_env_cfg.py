"""Route26独立任务：从route25继承，仅改变前进时两项加速度代价。"""
from isaaclab.utils import configclass
from .wheel_rigid_cap_env_cfg import LightHWWheelRigidCapEnvCfg
from .adapted_rewards import lighthw_joint_acc_l2
from .moving_acceleration_rewards import moving_joint_acc_l2


@configclass
class LightHWMovingAccelerationEnvCfg(LightHWWheelRigidCapEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        for name, weight in (("joint_acc_l2", -2e-7), ("joint_acc_wheel_l2", -1e-7)):
            term = getattr(self.rewards, name)
            if term.func is not lighthw_joint_acc_l2 or term.weight != weight:
                raise ValueError("Route26 acceleration source changed: " + name)
            term.func = moving_joint_acc_l2
            term.params.update(command_name="base_velocity", command_threshold=0.05, moving_multiplier=0.05)
        print("[Route26 effective] commanded vx>0.05: leg/wheel acceleration x0.05; otherwise x1; "
              "normalization/other rewards/physics/PPO unchanged; policy60/critic236/action16")
