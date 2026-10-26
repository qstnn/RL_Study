"""站立基线的数据约定；不依赖仿真，可用于资产预检。"""

TASK_ID = "Stand-Deeprobotics-LightHW-v0"
VERSION = "lighthw_stand_flat_v1"
LEGS = ("FR", "FL", "RR", "RL")
JOINT_NAMES = tuple(f"{leg}_{joint}_joint" for leg in LEGS for joint in ("hip", "thigh", "calf", "wheel"))
LEG_JOINT_NAMES = tuple(name for name in JOINT_NAMES if "wheel" not in name)
WHEEL_JOINT_NAMES = tuple(f"{leg}_wheel_joint" for leg in LEGS)
WHEEL_BODY_NAMES = tuple(f"{leg}_wheel_link" for leg in LEGS)
BASE_HEIGHT = 0.416
SPAWN_HEIGHT = 0.430
DEFAULT_JOINT_POS = {
    f"{leg}_{joint}_joint": value
    for leg in LEGS
    for joint, value in (
        ("hip", 0.0),
        ("thigh", 0.6 if leg.startswith("F") else -0.6),
        ("calf", -1.5 if leg.startswith("F") else 1.5),
        ("wheel", 0.0),
    )
}
LEG_TARGET_LIMITS = {
    name: (value - (0.08 if "hip" in name else 0.14), value + (0.08 if "hip" in name else 0.14))
    for name, value in DEFAULT_JOINT_POS.items() if "wheel" not in name
}
