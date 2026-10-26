# Copyright (c) 2026 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

import gymnasium as gym

# Independent corrected observation contract; old task registrations stay intact.
gym.register(
    id="Flat-Deeprobotics-LightHW-PPOReference-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ppo_reference_env_cfg:LightHWPPOReferenceEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:LightHWPPOReferenceRunnerCfg",
    },
)

from . import agents

gym.register(
    id="Flat-Deeprobotics-LightHW-RollStop-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.roll_stop_env_cfg:LightHWRollStopEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LightHWRollStopPPORunnerCfg",
    },
)

##
# Register Gym environments.
##

gym.register(
    id="Flat-Deeprobotics-LightHW-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:DeeproboticsLightHWFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsLightHWFlatPPORunnerCfg",
    },
)

gym.register(
    id="Rough-Deeprobotics-LightHW-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:DeeproboticsLightHWRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsLightHWRoughPPORunnerCfg",
    },
)

gym.register(
    id="Rough-Deeprobotics-LightHW-Blind-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.blind_rough_env_cfg:DeeproboticsLightHWBlindRoughEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsLightHWBlindRoughPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Rough-Deeprobotics-LightHW-StableRoll-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.stable_roll_env_cfg:DeeproboticsLightHWStableRollEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "DeeproboticsLightHWStableRollPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Rough-Deeprobotics-LightHW-StableRoll-V3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stable_roll_v3_env_cfg:DeeproboticsLightHWStableRollV3EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsLightHWStableRollV3PPORunnerCfg",
    },
)

gym.register(
    id="Rough-Deeprobotics-LightHW-BalancedHighSpeed-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.balanced_highspeed_rough_env_cfg:"
            "DeeproboticsLightHWBalancedHighSpeedRoughEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "DeeproboticsLightHWBalancedHighSpeedPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Flat-Deeprobotics-LightHW-StraightHighSpeed-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.flat_straight_highspeed_env_cfg:"
            "DeeproboticsLightHWFlatStraightHighSpeedEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "DeeproboticsLightHWFlatStraightHighSpeedPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Flat-Deeprobotics-LightHW-StraightProgressive-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.flat_straight_progressive_env_cfg:"
            "DeeproboticsLightHWFlatStraightProgressiveEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "DeeproboticsLightHWFlatStraightProgressivePPORunnerCfg"
        ),
    },
)

gym.register(
    id="Flat-Deeprobotics-LightHW-StraightRecovery-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.flat_straight_recovery_env_cfg:"
            "DeeproboticsLightHWFlatStraightRecoveryEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "DeeproboticsLightHWFlatStraightRecoveryPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Flat-Deeprobotics-LightHW-StraightPosture-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.flat_straight_posture_env_cfg:"
            "DeeproboticsLightHWFlatStraightPostureEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "DeeproboticsLightHWFlatStraightPosturePPORunnerCfg"
        ),
    },
)

gym.register(
    id="Flat-Deeprobotics-LightHW-StraightStablePhysics-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.flat_straight_stable_physics_env_cfg:"
            "DeeproboticsLightHWFlatStraightStablePhysicsEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "DeeproboticsLightHWFlatStraightStablePhysicsPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Flat-Deeprobotics-LightHW-StraightStablePhysics-Continue-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        # The environment, observations, rewards, actions and reset semantics
        # are identical to the fresh route.  Only the runner/run namespace is
        # separate so model_999 can be warm-started without weakening the
        # fresh-start guard on the original task.
        "env_cfg_entry_point": (
            f"{__name__}.flat_straight_stable_physics_env_cfg:"
            "DeeproboticsLightHWFlatStraightStablePhysicsEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "DeeproboticsLightHWFlatStraightStablePhysicsContinuePPORunnerCfg"
        ),
    },
)

gym.register(
    id="Flat-Deeprobotics-LightHW-StraightStablePhysics-Precision-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        # Keep the validated route-09 environment intact and isolate the
        # reward-precision experiment under a separate task/run namespace.
        "env_cfg_entry_point": (
            f"{__name__}.flat_straight_stable_physics_precision_env_cfg:"
            "DeeproboticsLightHWFlatStraightStablePhysicsPrecisionEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "DeeproboticsLightHWFlatStraightStablePhysicsPrecisionPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Flat-Deeprobotics-LightHW-StraightStablePhysics-Contact-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.flat_straight_stable_physics_contact_env_cfg:"
            "DeeproboticsLightHWFlatStraightStablePhysicsContactEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "DeeproboticsLightHWFlatStraightStablePhysicsContactPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Flat-Deeprobotics-LightHW-StraightStablePhysics-StandGate-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.flat_straight_stable_physics_stand_gate_env_cfg:"
            "DeeproboticsLightHWFlatStraightStablePhysicsStandGateEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "DeeproboticsLightHWFlatStraightStablePhysicsStandGatePPORunnerCfg"
        ),
    },
)

gym.register(
    id="Flat-Deeprobotics-LightHW-StraightStablePhysics-StandHold-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.flat_straight_stable_physics_stand_hold_env_cfg:"
            "DeeproboticsLightHWFlatStraightStablePhysicsStandHoldEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "DeeproboticsLightHWFlatStraightStablePhysicsStandHoldPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Flat-Deeprobotics-LightHW-StraightStablePhysics-StandHoldPrecision-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.flat_straight_stable_physics_stand_hold_precision_env_cfg:"
            "DeeproboticsLightHWFlatStraightStablePhysicsStandHoldPrecisionEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "DeeproboticsLightHWFlatStraightStablePhysicsStandHoldPrecisionPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Flat-Deeprobotics-LightHW-StraightStablePhysics-StandHoldPrecisionVx06-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.flat_straight_stable_physics_stand_hold_precision_vx06_env_cfg:"
            "DeeproboticsLightHWFlatStraightStablePhysicsStandHoldPrecisionVx06EnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "DeeproboticsLightHWFlatStraightStablePhysicsStandHoldPrecisionVx06PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Rough-Deeprobotics-LightHW-HistoricalHighSpeedPlayback-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.historical_highspeed_playback_env_cfg:"
            "DeeproboticsLightHWHistoricalHighSpeedPlaybackEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsLightHWRoughPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Flat-Deeprobotics-LightHW-PPOActionCurriculum-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.ppo_action_curriculum_env_cfg:DeeproboticsLightHWPPOActionCurriculumEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsLightHWPPOActionCurriculumPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Flat-Deeprobotics-LightHW-PPOForwardSpeed-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.ppo_forward_speed_env_cfg:DeeproboticsLightHWPPOForwardSpeedEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsLightHWPPOForwardSpeedPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Flat-Deeprobotics-LightHW-PPOSpeedBridge-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.ppo_speed_bridge_env_cfg:DeeproboticsLightHWPPOSpeedBridgeEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsLightHWPPOSpeedBridgePPORunnerCfg"
        ),
    },
)

gym.register(
    id="Flat-Deeprobotics-LightHW-PPOGroupedNoise-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.ppo_grouped_noise_env_cfg:DeeproboticsLightHWGroupedNoiseEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsLightHWGroupedNoisePPORunnerCfg"
        ),
    },
)

gym.register(
    id="Flat-Deeprobotics-LightHW-PPOLegDesaturation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.ppo_leg_desaturation_env_cfg:DeeproboticsLightHWLegDesaturationEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsLightHWLegDesaturationPPORunnerCfg"
        ),
    },
)

# Route22/23 keep all historical tasks intact; dedicated guarded runners are required.
for _task, _env in (
    ('Flat-Deeprobotics-LightHW-Speed3-v0', 'LightHWFlatSpeed3EnvCfg'),
    ('Rough-Deeprobotics-LightHW-Speed3Transfer-v0', 'LightHWRoughSpeedTransferEnvCfg'),
):
    gym.register(id=_task, entry_point='isaaclab.envs:ManagerBasedRLEnv', disable_env_checker=True,
        kwargs={'env_cfg_entry_point': f'{__name__}.speed3_env_cfg:{_env}',
                'rsl_rl_cfg_entry_point': f'{__name__}.agents.rsl_rl_ppo_cfg:LightHWPPOReferenceRunnerCfg'})
