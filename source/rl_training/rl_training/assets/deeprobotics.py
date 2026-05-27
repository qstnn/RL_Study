# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0
import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sim.spawners.from_files import from_files

from rl_training.assets import ISAACLAB_ASSETS_DATA_DIR


@sim_utils.clone
def spawn_m20_ironman(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Spawn M20 and apply per-link Iron-Man-style visual materials."""
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

DEEPROBOTICS_M20_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        func=spawn_m20_ironman,
        asset_path=f"{ISAACLAB_ASSETS_DATA_DIR}/M20/M20_urdf/urdf/M20.urdf",

        # ================== 破局核心参数 ==================
        force_usd_conversion=True,       # 强制重新转换，无视之前残缺的缓存文件！
        collision_from_visuals=True,     # 强制生成碰撞刚体，确保接触传感器有地方挂载！
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
    # --- 下面的 init_state 和 actuators 代码继续保持原样，千万别删 ---
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
