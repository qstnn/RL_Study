"""R4独立物理实例：规范URDF、原质量惯量、原腿轮上限；本地静态地面。"""
import isaaclab.sim as sim
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from .spec import LEG_JOINTS, STAND
from isaaclab.sim.converters import UrdfConverter
from isaaclab.sim.spawners.from_files.from_files import _spawn_from_usd_file


class CanonicalConverter(UrdfConverter):
    def _get_urdf_import_config(self):
        config = super()._get_urdf_import_config()
        config.set_import_inertia_tensor(True)
        return config


@sim.clone
def spawn_robot(prim_path, cfg, translation=None, orientation=None, **kwargs):
    converter = CanonicalConverter(cfg)
    prim = _spawn_from_usd_file(prim_path, converter.usd_path, cfg, translation, orientation)
    # 材料在克隆完成后经PhysX张量API统一设定和读回（见env.__init__）。
    return prim


@configclass
class BalanceSceneCfg(InteractiveSceneCfg):
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim.UrdfFileCfg(
            asset_path="", func=spawn_robot, root_link_name="Base_link", fix_base=False,
            force_usd_conversion=True, collision_from_visuals=False, make_instanceable=False,
            joint_drive=None, activate_contact_sensors=True,
            rigid_props=sim.RigidBodyPropertiesCfg(
                disable_gravity=False, linear_damping=0., angular_damping=0.,
                # USD刚体阈值单位deg/s；不可用1000截断电机允许的连杆/轮体世界角速度。
                max_linear_velocity=1000., max_angular_velocity=100000.,
                enable_gyroscopic_forces=True, max_depenetration_velocity=1.),
            articulation_props=sim.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=2),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0., 0., .43), joint_pos={**dict(zip(LEG_JOINTS, STAND)), ".*_wheel_joint": 0.},
            joint_vel={".*": 0.}),
        soft_joint_pos_limit_factor=.95,
        actuators={
            "joint": DelayedPDActuatorCfg(
                joint_names_expr=[".*_(hip|thigh|calf)_joint"], stiffness=80., damping=3.,
                effort_limit=48., effort_limit_sim=48., velocity_limit=12.57, velocity_limit_sim=12.57,
                friction=0., armature=0., min_delay=0, max_delay=0),
            "wheel": DelayedPDActuatorCfg(
                joint_names_expr=[".*_wheel_joint"], stiffness=0., damping=1.,
                effort_limit=24., effort_limit_sim=24., velocity_limit=37.7, velocity_limit_sim=37.7,
                friction=0., armature=.00243216, min_delay=0, max_delay=0),
        },
    )
    ground = AssetBaseCfg(
        prim_path="/World/ground", init_state=AssetBaseCfg.InitialStateCfg(pos=(0., 0., -.05)),
        spawn=sim.CuboidCfg(
            size=(400., 400., .1), collision_props=sim.CollisionPropertiesCfg(),
            physics_material=sim.RigidBodyMaterialCfg(
                static_friction=1., dynamic_friction=1., restitution=0.,
                friction_combine_mode="min", restitution_combine_mode="min"),
            visual_material=sim.PreviewSurfaceCfg(diffuse_color=(.25, .28, .32)),
        ),
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", update_period=.001, history_length=1)
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim.DomeLightCfg(intensity=750.))
