"""Original LightHW low-speed standing pose; data-only configuration."""

# Height is measured from flat ground to the Base_link origin.  These values
# match the original low-speed checkpoint and the shared LIGHTHW_CFG pose.
LIGHTHW_DEFAULT_BASE_HEIGHT = 0.416
LIGHTHW_DEFAULT_SPAWN_HEIGHT = 0.430

# The original pose uses the model's reference joint angles rather than the
# later 0.545 m near-straight-leg geometry.  All values are in radians.
LIGHTHW_DEFAULT_JOINT_POS = {
    ".*_hip_joint": 0.0,
    "FL_thigh_joint": 0.6,
    "FR_thigh_joint": 0.6,
    "RL_thigh_joint": -0.6,
    "RR_thigh_joint": -0.6,
    "FL_calf_joint": -1.5,
    "FR_calf_joint": -1.5,
    "RL_calf_joint": 1.5,
    "RR_calf_joint": 1.5,
    ".*_wheel_joint": 0.0,
}
