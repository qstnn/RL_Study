"""Isolated towing reward changes; original non-towing terms remain unchanged.

No task-acceptance decisions occur here. Contact-point velocity uses a nominal
wheel radius for a smooth reward approximation; final behavior still requires
the independent 1 kHz native-CoP audit. Imports of Isaac modules are deferred
until the application has started.
"""
import torch


TOWING_HEIGHT_BAND = (.26, .44)
REWARD_CONTRACT = dict(
    version='migration_v1_partial_support_low_towing_rewards',
    towing_height_free_band_m=list(TOWING_HEIGHT_BAND),
    robot_wheel_radius_m=.0875,
    slip_reward_kind='normal_load_weighted_nominal_contact_point_velocity',
    slip_reward_is_native_cop_acceptance=False,
    preparation_stand_motion_and_drift_costs_enabled=False,
    other_original_motion_effort_and_action_costs_remain=True,
    illegal_contact_and_orientation_terminations_unchanged=True,
    reward_manager_multiplies_policy_dt=True,
    behavior_accepted=False,
)


def height_error(height, towing):
    lo, hi = TOWING_HEIGHT_BAND
    free_band_error = (lo-height).clamp_min(0.)+(height-hi).clamp_min(0.)
    return torch.where(towing, free_band_error, height-.416)


def support_score(normal_load, nominal_normal_weight):
    """Any distribution of positive support counts; four loaded wheels not needed."""
    total = normal_load.clamp_min(0.).sum(-1)
    return (total/(.1*nominal_normal_weight).clamp_min(1.e-6)).clamp(0., 1.)


def nominal_point_slip(normal_load, center_velocity, angular_velocity, normal, radius=.0875):
    """A rigid wheel's point speed includes leg/body motion as well as wheel spin."""
    offset = -float(radius)*normal[:, None, :].expand_as(center_velocity)
    velocity = center_velocity+torch.cross(angular_velocity, offset, dim=-1)
    planar = velocity-normal[:, None, :]*(velocity*normal[:, None, :]).sum(-1, keepdim=True)
    weight = normal_load.clamp_min(0.)*(normal_load > 1.)
    cost = (planar.square().sum(-1)/.3**2).clamp_max(25.)
    return (cost*weight).sum(-1)/weight.sum(-1).clamp_min(1.e-6)


def _mdp():
    from rl_training.tasks.manager_based.locomotion.multiskill.lighthw.wheel_unified_v6 import mdp
    return mdp


def _towing(env):
    return env.r1_towed_kg > 0


def _operational(env):
    """Only a predeclared command clock; never teacher-ready or contact feedback."""
    term = env.command_manager.get_term('base_velocity')
    return term.towing_preparation_complete


def _select(env, ordinary, towing):
    return torch.where(_towing(env), towing, ordinary)


def _normal_load(env, sensor_cfg):
    forces = env.scene[sensor_cfg.name].data.net_forces_w[:, sensor_cfg.body_ids]
    return (forces*env.r1_normal[:, None, :]).sum(-1)


def _tracking_gate(env):
    mdp = _mdp()
    heading = mdp.heading_error(env).squeeze(-1)
    height = ((env.scene['robot'].data.root_pos_w-env.scene.env_origins)*env.r1_normal).sum(-1)
    error = height_error(height, _towing(env))
    orientation = torch.exp(-heading.square()/.35**2-mdp.posture_cost(env)/.08-error.square()/.1**2)
    clearance = (mdp.hitch_clearance(env)/env.cfg.hitch_margin_m).clamp(0., 1.)
    return .05+.95*orientation*clearance


def tracking(env, fine=False):
    mdp = _mdp()
    original = mdp.tracking(env, fine=fine)
    _, velocity, _ = mdp.state(env)
    command = env.command_manager.get_command('base_velocity')
    blend = mdp.loaded_motion_weight(command[:, 0])
    score = mdp.core.velocity_score(velocity-command[:, :2], command[:, 0], fine)
    towing = score*(1.-blend)*_tracking_gate(env)*_operational(env)
    return _select(env, original, towing)


def surface_height(env):
    height = ((env.scene['robot'].data.root_pos_w-env.scene.env_origins)*env.r1_normal).sum(-1)
    return _select(env, _mdp().surface_height(env), height_error(height, _towing(env)).square())


def stand_support(env, sensor_cfg):
    original = _mdp().stand_support(env, sensor_cfg)
    normal = _normal_load(env, sensor_cfg)
    nominal = (_mdp().spec.CONTRACT["robot_mass_kg"]+env.r1_payload_kg)*9.81*env.r1_normal[:, 2]
    towing = _mdp().standing(env)*_operational(env)*support_score(normal, nominal)
    return _select(env, original, towing)


def towing_support_margin(env, sensor_cfg, load_fraction=.25):
    # Preserve the old term's signature/configuration without per-wheel deficits.
    normal = _normal_load(env, sensor_cfg)
    nominal = (_mdp().spec.CONTRACT["robot_mass_kg"]+env.r1_payload_kg)*9.81*env.r1_normal[:, 2]
    return _towing(env)*(1.-support_score(normal, nominal)).square()


def stand_motion(env):
    original = _mdp().stand_motion(env)
    return _select(env, original, original*_operational(env))


def stand_drift(env):
    original = _mdp().stand_drift(env)
    return _select(env, original, original*_operational(env))


def stand_wheels(env, asset_cfg):
    original = _mdp().stand_wheels(env, asset_cfg)
    # Towing ground-point slip is scored separately, including zero commands.
    return _select(env, original, torch.zeros_like(original))


def stand_pose(env, asset_cfg):
    original = _mdp().stand_pose(env, asset_cfg)
    return _select(env, original, torch.zeros_like(original))


def towing_slip(env):
    robot = env.scene['robot']
    if not hasattr(env, 'r1_migration_slip_ids'):
        from rl_training.tasks.manager_based.locomotion.multiskill.lighthw.catalog import WHEEL_BODIES
        bodies = robot.find_bodies(list(WHEEL_BODIES), preserve_order=True)[0]
        sensors = env.scene['contact_forces'].find_bodies(list(WHEEL_BODIES), preserve_order=True)[0]
        env.r1_migration_slip_ids = (bodies, sensors)
    bodies, sensors = env.r1_migration_slip_ids
    forces = env.scene['contact_forces'].data.net_forces_w[:, sensors]
    normal = (forces*env.r1_normal[:, None, :]).sum(-1)
    # Radius is measured from the wheel link center, not its offset inertial COM.
    return _towing(env)*nominal_point_slip(normal, robot.data.body_link_lin_vel_w[:, bodies],
        robot.data.body_ang_vel_w[:, bodies], env.r1_normal)


def apply_reward_overrides(cfg):
    """After AppLauncher: replace functions only, retain original weights and dt."""
    changes = dict(track_broad=tracking, track_fine=tracking, surface_height=surface_height,
        stand_support=stand_support, towing_support_margin=towing_support_margin,
        stand_motion=stand_motion, stand_drift=stand_drift, stand_wheels=stand_wheels,
        stand_pose=stand_pose, towing_slip=towing_slip)
    for name, function in changes.items():
        getattr(cfg.rewards, name).func = function
    return dict(REWARD_CONTRACT, changed_terms=list(changes),
        term_weights={name: float(getattr(cfg.rewards, name).weight) for name in changes})
