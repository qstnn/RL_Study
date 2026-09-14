# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0
import isaaclab.sim as sim_utils
from pathlib import Path

from isaaclab.actuators import DCMotorCfg, DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sim.spawners.from_files import from_files

from rl_training.assets import ISAACLAB_ASSETS_DATA_DIR


@sim_utils.clone
def spawn_lighthw_ironman(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Spawn LightHW and apply per-link Iron-Man-style visual materials."""
    prim = from_files.spawn_from_urdf(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)

    dark_red_path = f"{prim_path}/Looks/ironman_dark_red"
    thigh_gold_path = f"{prim_path}/Looks/ironman_thigh_gold"
    shank_gold_path = f"{prim_path}/Looks/ironman_shank_gold"
    dark_path = f"{prim_path}/Looks/dark_graphite"

    dark_red = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.32, 0.015, 0.018), metallic=0.58, roughness=0.34)
    thigh_gold = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.32, 0.035), metallic=0.72, roughness=0.30)
    shank_gold = sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.48, 0.06), metallic=0.76, roughness=0.28)
    dark = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.025, 0.025, 0.03), metallic=0.35, roughness=0.45)
    dark_red.func(dark_red_path, dark_red)
    thigh_gold.func(thigh_gold_path, thigh_gold)
    shank_gold.func(shank_gold_path, shank_gold)
    dark.func(dark_path, dark)

    material_by_link = {
        "Base_link": dark_red_path,
        "FR_hip_link": thigh_gold_path,
        "FL_hip_link": thigh_gold_path,
        "RR_hip_link": thigh_gold_path,
        "RL_hip_link": thigh_gold_path,
        "FR_thigh_link": thigh_gold_path,
        "FL_thigh_link": thigh_gold_path,
        "RR_thigh_link": thigh_gold_path,
        "RL_thigh_link": thigh_gold_path,
        "FR_calf_link": shank_gold_path,
        "FL_calf_link": shank_gold_path,
        "RR_calf_link": shank_gold_path,
        "RL_calf_link": shank_gold_path,
        "FR_wheel_link": dark_path,
        "FL_wheel_link": dark_path,
        "RR_wheel_link": dark_path,
        "RL_wheel_link": dark_path,
    }

    applied = 0
    for link_name, material_path in material_by_link.items():
        target_path = f"{prim_path}/{link_name}"
        try:
            sim_utils.bind_visual_material(target_path, material_path)
            applied += 1
        except Exception as exc:
            print(f"[LightHWIronmanMaterial] Skip {target_path}: {exc}")
    print(f"[LightHWIronmanMaterial] Applied {applied}/{len(material_by_link)} link material overrides.")
    return prim


@sim_utils.clone
def spawn_m20_ironman(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Spawn the original M20 and apply per-link Iron-Man-style visual materials."""
    prim = from_files.spawn_from_urdf(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)

    dark_red_path = f"{prim_path}/Looks/ironman_dark_red"
    thigh_gold_path = f"{prim_path}/Looks/ironman_thigh_gold"
    shank_gold_path = f"{prim_path}/Looks/ironman_shank_gold"
    dark_path = f"{prim_path}/Looks/dark_graphite"

    dark_red = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.32, 0.015, 0.018), metallic=0.58, roughness=0.34)
    thigh_gold = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.32, 0.035), metallic=0.72, roughness=0.30)
    shank_gold = sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.48, 0.06), metallic=0.76, roughness=0.28)
    dark = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.025, 0.025, 0.03), metallic=0.35, roughness=0.45)
    dark_red.func(dark_red_path, dark_red)
    thigh_gold.func(thigh_gold_path, thigh_gold)
    shank_gold.func(shank_gold_path, shank_gold)
    dark.func(dark_path, dark)

    material_by_link = {
        "base_link": dark_red_path,
        "fr_hipy": thigh_gold_path,
        "fl_hipy": thigh_gold_path,
        "hr_hipy": thigh_gold_path,
        "hl_hipy": thigh_gold_path,
        "fr_hipx": thigh_gold_path,
        "fl_hipx": thigh_gold_path,
        "hr_hipx": thigh_gold_path,
        "hl_hipx": thigh_gold_path,
        "fr_knee": shank_gold_path,
        "fl_knee": shank_gold_path,
        "hr_knee": shank_gold_path,
        "hl_knee": shank_gold_path,
        "fr_wheel": dark_path,
        "fl_wheel": dark_path,
        "hr_wheel": dark_path,
        "hl_wheel": dark_path,
    }

    applied = 0
    for link_name, material_path in material_by_link.items():
        target_path = f"{prim_path}/{link_name}"
        try:
            sim_utils.bind_visual_material(target_path, material_path)
            applied += 1
        except Exception as exc:
            print(f"[M20IronmanMaterial] Skip {target_path}: {exc}")
    print(f"[M20IronmanMaterial] Applied {applied}/{len(material_by_link)} link material overrides.")
    return prim


DEEPROBOTICS_LITE3_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Lite3/Lite3_usd/Lite3.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=1
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.375),
        joint_pos={
            ".*HipX_joint": 0.0,
            ".*HipY_joint": -0.65,
            ".*Knee_joint": 1.3,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.99,
    actuators={
        "Hip": DelayedPDActuatorCfg(
            joint_names_expr=[".*_Hip[X,Y]_joint"],
            effort_limit=24.0,
            velocity_limit=26.2,
            stiffness=30.0,
            damping=1.0,
            friction=0.0,
            armature=0.0,
            min_delay=0,
            max_delay=1,
        ),
        "Knee": DelayedPDActuatorCfg(
            joint_names_expr=[".*_Knee_joint"],
            effort_limit=36.0,
            velocity_limit=17.3,
            stiffness=30.0,
            damping=1.0,
            friction=0.0,
            armature=0.0,
            min_delay=0,
            max_delay=1,
        ),
    },
)

# LightHW 物理模型公共配置。这里的质量、关节限位、PD 增益、力矩/速度上限会改变
# 环境动力学和动作含义，但它们不是 reward。做奖励消融实验时应保持本段不变；若
# 修改了执行器参数，旧 checkpoint 的闭环行为可能在第 0 轮就发生明显变化。
LIGHTHW_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        func=spawn_lighthw_ironman,
        # Keep one canonical LightHW copy in model_lighthw; older duplicate
        # assets under deep_robotics_model are no longer part of the chain.
        asset_path=str(Path(ISAACLAB_ASSETS_DATA_DIR).parent / "model_lighthw" / "lighthw_urdf" / "urdf" / "lighthw.urdf"),
        root_link_name="Base_link",

        # ================== 破局核心参数 ==================
        force_usd_conversion=True,       # 强制重新转换，无视之前残缺的缓存文件！
        collision_from_visuals=False,    # LightHW 已有 collision mesh，避免 visual mesh 凸包造成过大的自/地面接触。
        make_instanceable=True,          # 开启实例优化，多环境必备
        fix_base=False,                  # 不固定底座
        joint_drive=None,                # 禁用默认驱动，防报错
        # ==================================================

        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=1
        ),
    ),
    # 默认四轮起点。当前 calf 默认角为前腿 -1.5、后腿 +1.5；按 URDF 几何估算，
    # 轮心在机身下方约 0.329 m，轮半径约 0.0875 m，因此自然机身高度约 0.416 m。
    init_state=ArticulationCfg.InitialStateCfg(
        # Physical equivalent of the ZXC reference default pose, expressed in the
        # normalized LightHW joint axes: hip is +X; thigh/calf/wheel are +Y.
        pos=(0.0, 0.0, 0.43),
        joint_pos={
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
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    # 腿关节使用位置 PD，轮关节使用速度控制；effort_limit 是仿真执行器安全上限，
    # stiffness/damping 决定闭环响应。改变这些值属于动力学实验，不是奖励调参。
    actuators={
        "joint": DelayedPDActuatorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit=48.0,
            velocity_limit=12.57,
            stiffness=80.0,
            damping=2.0,
            friction=0.0,
            armature=0.0,
            min_delay=0,
            max_delay=1,
        ),
        "wheel": DelayedPDActuatorCfg(
            joint_names_expr=[".*_wheel_joint"],
            effort_limit=24.0,
            velocity_limit=37.7,
            stiffness=0.0,
            damping=0.6,
            friction=0.0,
            armature=0.00243216,
            min_delay=0,
            max_delay=1,
        ),
    },
)

DEEPROBOTICS_M20_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        func=spawn_m20_ironman,
        asset_path=f"{ISAACLAB_ASSETS_DATA_DIR}/M20/M20_urdf/urdf/M20.urdf",
        force_usd_conversion=True,
        collision_from_visuals=True,
        make_instanceable=True,
        fix_base=False,
        joint_drive=None,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=1
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.52),
        joint_pos={
            ".*hipx_joint": 0.0,
            "f[l,r]_hipy_joint": -0.6,
            "h[l,r]_hipy_joint": 0.6,
            "f[l,r]_knee_joint": 1.0,
            "h[l,r]_knee_joint": -1.0,
            ".*wheel_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "joint": DelayedPDActuatorCfg(
            joint_names_expr=[".*hipx_joint", ".*hipy_joint", ".*knee_joint"],
            effort_limit=76.4,
            velocity_limit=22.4,
            stiffness=80.0,
            damping=2.0,
            friction=0.0,
            armature=0.0,
            min_delay=0,
            max_delay=1,
        ),
        "wheel": DelayedPDActuatorCfg(
            joint_names_expr=[".*_wheel_joint"],
            effort_limit=21.6,
            velocity_limit=79.3,
            stiffness=0.0,
            damping=0.6,
            friction=0.0,
            armature=0.00243216,
            min_delay=0,
            max_delay=1,
        ),
    },
)
