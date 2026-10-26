"""R1 V7 research uses V6 physics, with isolated outputs and no PPO entry point."""
from pathlib import Path
import sys
from settings import GROUND_FRICTION, ROLLING_COEFFICIENT, LOADED_ACCELERATION
ROOT = Path(__file__).resolve().parents[1]
ROUTE = ROOT.parent
REPO = next(p for p in ROOT.parents if (p/'source/rl_training').is_dir())
V6 = ROUTE/'unified_v6'
OLD = ROUTE/'unified_v5/results/2026-10-19_coordinated_towing'
sys.path.insert(0, str(V6/'scripts'))
import launch
SPEC = launch.SPEC
URDF = REPO/'model_lighthw/lighthw_urdf/urdf/lighthw.urdf'
LEGS = ('FR', 'FL', 'RR', 'RL')
NAMES = SPEC.CONTRACT['action_order']
BASELINE = launch.BASELINE

def snapshot(out):
    import shutil
    launch.snapshot(out)
    manifest = {}
    for path in [*ROOT.glob('scripts/*.py'), *ROOT.glob('tests/*.py'), *ROOT.glob('tests/fixtures/*.json'), *ROOT.glob('tests/fixtures/*.npz'), *ROOT.glob('references/*.md'), ROOT/'README.md']:
        if not path.exists():continue
        relative = path.relative_to(REPO)
        target = out/'source_snapshot'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        manifest[str(relative)] = launch.sha256(path)
    for path in [OLD/'coordinated_control.py', OLD/'pose_ik.py']:
        relative = path.relative_to(REPO);target = out/'source_snapshot'/relative
        target.parent.mkdir(parents=True, exist_ok=True);shutil.copy2(path, target)
        manifest[str(relative)] = launch.sha256(path)
    launch.write_json(out/'research_source_manifest.json', manifest)

def configure(out, n, seed, position, velocity, noise=False):
    """Call only after AppLauncher; retain physical torque/mass/friction/action contracts."""
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    SPEC.register_task()
    cfg = load_cfg_from_registry(SPEC.TASK, 'env_cfg_entry_point')
    agent = load_cfg_from_registry(SPEC.TASK, 'rsl_rl_cfg_entry_point')
    cfg.scene.num_envs = n;cfg.seed = agent.seed = seed
    cfg.ground_friction = GROUND_FRICTION;cfg.rolling_coefficient = ROLLING_COEFFICIENT
    cfg.loaded_acceleration = LOADED_ACCELERATION
    for material in (cfg.scene.terrain.physics_material, cfg.sim.physics_material):
        material.static_friction = material.dynamic_friction = GROUND_FRICTION
        material.friction_combine_mode = "min"
    cfg.fixed_speed_mps = 0.;cfg.fixed_slope_deg = 25.
    cfg.fixed_payload_kg = 0.;cfg.fixed_towed_kg = 40.;cfg.episode_length_s = 65.
    cfg.observations.policy.enable_corruption = noise
    for asset in (cfg.scene.robot, cfg.scene.trailer):
        asset.spawn.articulation_props.solver_position_iteration_count = position
        asset.spawn.articulation_props.solver_velocity_iteration_count = velocity
    cfg.sim.physx.max_position_iteration_count = position
    cfg.sim.physx.max_velocity_iteration_count = velocity
    launch.CART.write_cart_urdf(out/'assets/cart.urdf')
    cfg.scene.robot.spawn.usd_dir = str(out/'cache/robot')
    cfg.scene.robot.spawn.usd_file_name = 'robot.usd'
    cfg.scene.trailer.spawn.asset_path = str(out/'assets/cart.urdf')
    cfg.scene.trailer.spawn.usd_dir = str(out/'cache/cart')
    cfg.scene.trailer.spawn.usd_file_name = 'cart.usd'
    return cfg, agent
