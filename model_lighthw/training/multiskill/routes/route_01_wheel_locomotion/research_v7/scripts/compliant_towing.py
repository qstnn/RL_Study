"""Isolated ordinary-reset drawbar diagnostic; original physics stays the default.

Only spring stiffness/damping vary. The native force law, passive cart gravity,
rolling resistance, equal/opposite forces, arm moments and guards are retained.
"""
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import importlib.util
import math
from pathlib import Path
import shutil

VARIANT = 'ordinary_compliant_drawbar_diagnostic_v1'
ROOT = Path(__file__).resolve().parents[1]
REPO = next(parent for parent in ROOT.parents if (parent/'source/rl_training').is_dir())
NATIVE = REPO/'source/rl_training/rl_training/tasks/manager_based/locomotion/multiskill/lighthw/wheel_unified_v6'


@dataclass(frozen=True)
class DrawbarParameters:
    stiffness_Npm: float = 2000.
    damping_Nspm: float = 150.

    def __post_init__(self):
        if (not math.isfinite(self.stiffness_Npm) or self.stiffness_Npm <= 0.
                or not math.isfinite(self.damping_Nspm) or self.damping_Nspm < 0.):
            raise ValueError('Drawbar stiffness must be positive and damping nonnegative, both finite')

    def metadata(self):
        return dict(variant=VARIANT, stiffness_Npm=float(self.stiffness_Npm), damping_Nspm=float(self.damping_Nspm),
            rest_length_m=1., force_clip_N=1500., force_guard_failure_N=1499., maximum_length_failure_m=1.6,
            hitch_clearance_failure_m=.005, tension_and_compression=True, ordinary_zero_velocity_reset=True,
            gravity_preload_reset=False, original_teacher_dataset_eligible=False,
            baseline=dict(stiffness_Npm=2000., damping_Nspm=150.),
            physical_variant_only=True, robot_mass_inertia_actuator_changed=False)


def diagnostic_parameters(enabled, stiffness, damping, initial_state, live_entry):
    parameters = DrawbarParameters(float(stiffness), float(damping))
    changed = parameters != DrawbarParameters()
    if not enabled:
        if changed:
            raise ValueError('Changed drawbar parameters require --diagnostic-compliant-drawbar')
        return None
    if initial_state != 'ordinary' or not live_entry:
        raise ValueError('Compliant drawbar diagnostic requires ordinary reset and --ordinary-live-entry')
    if not changed:
        raise ValueError('Compliant diagnostic must explicitly differ from the preserved 2000/150 baseline')
    return parameters


@lru_cache(maxsize=1)
def native_force_function():
    # This production source is deliberately simulator independent. Loading it in
    # a private namespace also permits CPU audits without initializing Isaac.
    path = NATIVE/'cart_asset.py'
    spec = importlib.util.spec_from_file_location('_r1_compliant_native_cart_asset', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.drawbar_force


def drawbar_force(robot_point, cart_point, robot_velocity, cart_velocity, active, parameters):
    return native_force_function()(robot_point, cart_point, robot_velocity, cart_velocity, active,
        rest=1., stiffness=parameters.stiffness_Npm, damping=parameters.damping_Nspm)


def endpoint_state(env, quat_apply):
    """Same native root/COM convention as cart.apply_cart_forces and the recorder."""
    import torch
    robot, cart = env.scene['robot'], env.scene['trailer']
    rh = quat_apply(robot.data.root_quat_w, env.r1_robot_hitch)+robot.data.root_pos_w
    ch = quat_apply(cart.data.root_quat_w, env.r1_cart_hitch)+cart.data.root_pos_w
    rr = rh-robot.data.body_com_pos_w[:, env.r1_base_id]
    cr = ch-cart.data.body_com_pos_w[:, env.r1_cart_base_id]
    rv = robot.data.root_com_lin_vel_w+torch.cross(robot.data.root_ang_vel_w, rr, dim=-1)
    cv = cart.data.root_com_lin_vel_w+torch.cross(cart.data.root_ang_vel_w, cr, dim=-1)
    return rh, ch, rr, cr, rv, cv


def estimate_robot_force(env, parameters, quat_apply=None):
    if quat_apply is None:
        from isaaclab.utils.math import quat_apply
    rh, ch, _, _, rv, cv = endpoint_state(env, quat_apply)
    return -drawbar_force(rh, ch, rv, cv, env.r1_towed_kg > 0, parameters)[0]


def apply_cart_forces(env, parameters, quat_apply=None):
    """Local explicit-parameter counterpart of the original 1 kHz cart action.

    No position, velocity, mass, gravity setting, joint target or contact state is
    written. Only the same external-force interface as the original is used.
    """
    import torch
    if quat_apply is None:
        from isaaclab.utils.math import quat_apply
    robot, cart = env.scene['robot'], env.scene['trailer']
    active = env.r1_towed_kg > 0
    rh, ch, rr, cr, rv, cv = endpoint_state(env, quat_apply)
    force, length, tension = drawbar_force(rh, ch, rv, cv, active, parameters)
    robot.set_external_force_and_torque((-force)[:, None, :], torch.cross(rr, -force, dim=-1)[:, None, :],
        body_ids=[env.r1_base_id], is_global=True)
    forces = torch.zeros((env.num_envs, cart.num_bodies, 3), device=env.device)
    forces[:, :, 2] = -9.81*env.r1_cart_masses*active[:, None]
    forces[:, env.r1_cart_base_id] += force
    torques = torch.zeros_like(forces)
    torques[:, env.r1_cart_base_id] = torch.cross(cr, force, dim=-1)
    sensor = env.scene['trailer_contacts']
    normals = (sensor.data.net_forces_w[:, env.r1_cart_sensor_ids]*env.r1_normal[:, None, :]).sum(-1).clamp_min(0.)
    speed = cart.data.joint_vel[:, env.r1_cart_joint_ids]
    resistance = -env.cfg.rolling_coefficient*normals*.14*torch.tanh(speed/.2)*active[:, None]
    axis = quat_apply(cart.data.root_quat_w, env.r1_cart_axis)
    torques[:, env.r1_cart_wheel_ids] += resistance[:, :, None]*axis[:, None, :]
    cart.set_external_force_and_torque(forces, torques, is_global=True)
    env.r1_drawbar_length = length
    env.r1_drawbar_force = tension
    env.r1_drawbar_guard |= tension.abs() >= 1499.
    env.r1_drawbar_peak = torch.maximum(env.r1_drawbar_peak, tension.abs())
    env.r1_rolling_torque = resistance
    env.r1_compliant_drawbar_physics_calls = getattr(env, 'r1_compliant_drawbar_physics_calls', 0)+1
    env.r1_compliant_drawbar_parameters = parameters.metadata()


def source_hashes():
    paths = [Path(__file__), ROOT/'scripts/compliant_cart_audit.py', ROOT/'scripts/loaded_cart_worker.py',
        NATIVE/'cart_asset.py', NATIVE/'cart.py']
    return {str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def freeze_sources(output):
    manifest = source_hashes()
    for relative, digest in manifest.items():
        source = REPO/relative
        target = Path(output)/'source_snapshot'/relative
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise ValueError('Compliant drawbar source snapshot collision: '+relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(source, target)
    return manifest
