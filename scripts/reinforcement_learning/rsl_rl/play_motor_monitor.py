# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
from collections import deque
import math
import os
import sys
import csv
import json

from isaaclab.app import AppLauncher

# local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cli_args
from checkpoint_pose import restore_checkpoint_pose

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--use_checkpoint_pose",
    action="store_true",
    help="Restore default joint angles, spawn pose and height target from the checkpoint's params/env.yaml.",
)
parser.add_argument("--keyboard", action="store_true", default=False, help="Whether to use keyboard.")
parser.add_argument(
    "--fixed_command",
    type=float,
    nargs=3,
    default=None,
    metavar=("VX", "VY", "WZ"),
    help="Use a fixed velocity command [vx, vy, wz] for non-interactive playback/testing.",
)
parser.add_argument(
    "--monitor_leg",
    type=str,
    default="fl",
    choices=["fl", "fr", "hl", "hr", "all"],
    help="Motor group to plot/log: fl, fr, hl, hr, or all. For M20, a leg group includes hipx, hipy, knee, and wheel if present.",
)
parser.add_argument(
    "--monitor_joints",
    type=str,
    default=None,
    help=(
        "Comma-separated joint indices or name keywords to monitor, overriding --monitor_leg. "
        "Examples: '0,1,2,12' or 'fl_hipx,fl_hipy,fl_knee,fl_wheel'."
    ),
)
parser.add_argument(
    "--monitor_all_body_contacts",
    action="store_true",
    default=False,
    help=(
        "Log contact forces for every body tracked by the contact sensor in addition to the four wheels. "
        "This is intended for diagnosing unexpected base/leg contacts."
    ),
)
parser.add_argument(
    "--monitor_history_len",
    type=int,
    default=300,
    help="Number of recent steps shown in the live plot.",
)
parser.add_argument(
    "--monitor_plot_every",
    type=int,
    default=5,
    help="Update matplotlib every N simulation steps. Larger values reduce lag.",
)
parser.add_argument(
    "--physics_substep_log",
    action="store_true",
    default=False,
    help=(
        "Additionally log robot state after every physics substep. This keeps the policy/control rate "
        "unchanged and writes motor_monitor/physics_substep_log.csv at the physics rate."
    ),
)
parser.add_argument(
    "--hip_retraction_test",
    action="store_true",
    help="Playback-only FL hip probe: policy -> 0.35 -> 0.30 rad -> policy, 18 seconds with safety stops.",
)
parser.add_argument(
    "--hip_retraction_hold_others",
    action="store_true",
    help="With --hip_retraction_test, hold the other 15 raw actions during 3-13s, then blend back to policy by 15s.",
)
parser.add_argument(
    "--payload_kg",
    type=float,
    default=0.0,
    help="Initial runtime payload mass in kg. It is applied as a downward external force on --payload_body.",
)
parser.add_argument(
    "--payload_body",
    type=str,
    default="base_link",
    help="Robot body name or regex receiving the runtime payload force.",
)
parser.add_argument(
    "--payload_step_kg",
    type=float,
    default=1.0,
    help="Payload mass increment/decrement in kg for keyboard controls.",
)
parser.add_argument(
    "--payload_max_kg",
    type=float,
    default=100.0,
    help="Maximum payload mass in kg when adjusted from the keyboard.",
)
parser.add_argument(
    "--heavy_payload_threshold_kg",
    type=float,
    default=40.0,
    help="Enable heavy actuator mode when the initial payload is at least this value.",
)
parser.add_argument(
    "--payload_ramp_time",
    type=float,
    default=2.0,
    help="Seconds used to smoothly ramp payload force from 0 to the requested payload.",
)
parser.add_argument(
    "--disable_payload_motor_profile",
    action="store_true",
    default=False,
    help=(
        "Keep the task's native actuator gains/limits and skip playback-only payload action smoothing/scaling. "
        "Use this when auditing a trained policy's real contact and actuator behavior; an explicit "
        "--action_smoothing_override is still applied."
    ),
)
parser.add_argument(
    "--action_smoothing_override",
    type=float,
    default=None,
    help=(
        "Override both standing and moving action low-pass coefficients during playback. "
        "Use 0 to disable smoothing; the diagnostic range is [0, 0.95]."
    ),
)
parser.add_argument(
    "--drag_load_kg",
    type=float,
    default=0.0,
    help="Equivalent dragged load mass in kg. Used with --drag_mu to compute horizontal drag force.",
)
parser.add_argument(
    "--drag_mu",
    type=float,
    default=0.3,
    help="Ground friction coefficient for the dragged load: drag_force = drag_load_kg * drag_mu * 9.81.",
)
parser.add_argument(
    "--drag_force_n",
    type=float,
    default=None,
    help="Initial horizontal drag force in Newtons. Overrides --drag_load_kg/--drag_mu when set.",
)
parser.add_argument(
    "--drag_step_n",
    type=float,
    default=10.0,
    help="Horizontal drag force increment/decrement in Newtons for keyboard controls.",
)
parser.add_argument(
    "--drag_max_n",
    type=float,
    default=500.0,
    help="Maximum horizontal drag force in Newtons when adjusted from the keyboard.",
)
parser.add_argument(
    "--spawn_terrain_type",
    type=str,
    default="pyramid_stairs_inv",
    help=(
        "Sub-terrain type used as the playback spawn tile when running on rough terrain. "
        "Use names like random_rough, boxes, hf_pyramid_slope, pyramid_stairs, or pyramid_stairs_inv."
    ),
)
parser.add_argument(
    "--force_plane",
    action="store_true",
    default=False,
    help=(
        "Force the current task terrain to a flat plane for playback diagnostics while keeping the "
        "task asset, action space, and policy observation terms unchanged."
    ),
)
parser.add_argument(
    "--spawn_terrain_level",
    type=int,
    default=0,
    help="Terrain difficulty row used for playback spawn. 0 is the easiest row.",
)
parser.add_argument(
    "--spawn_offset_x",
    type=float,
    default=None,
    help=(
        "Spawn x offset from the selected terrain origin. Defaults to 0 on a plane and 0.9 on generated terrain."
    ),
)
parser.add_argument(
    "--spawn_offset_y",
    type=float,
    default=0.0,
    help="Spawn y offset from the selected terrain origin.",
)
parser.add_argument(
    "--spawn_jitter_xy",
    type=float,
    default=0.05,
    help="Small random spawn jitter around the selected x/y offset.",
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=0,
    help="Maximum number of simulation steps to run. Use 0 to run until the simulator is closed.",
)
parser.add_argument(
    "--no_plots",
    action="store_true",
    default=False,
    help="Disable live matplotlib windows while keeping CSV logging enabled.",
)
parser.add_argument(
    "--static_check",
    action="store_true",
    default=False,
    help=(
        "Run a deterministic static load-check playback: one env, zero velocity command, "
        "zero policy actions, no terrain/randomization events, and CSV-only logging by default."
    ),
)
parser.add_argument(
    "--zero_policy_actions",
    action="store_true",
    default=False,
    help=(
        "Force raw policy actions to zero while preserving the task reset, reference trajectory, "
        "physics, and other playback settings."
    ),
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
if args_cli.hip_retraction_hold_others and not args_cli.hip_retraction_test:
    parser.error("--hip_retraction_hold_others requires --hip_retraction_test.")
if args_cli.hip_retraction_test:
    # 诊断只允许纯平地零命令、原生电机与真实策略，避免其他播放修正混入因果测试。
    if (args_cli.num_envs != 1 or not args_cli.force_plane or args_cli.fixed_command != [0., 0., 0.]
            or not args_cli.disable_payload_motor_profile or args_cli.keyboard
            or args_cli.static_check or args_cli.zero_policy_actions
            or args_cli.action_smoothing_override not in (None, 0.)
            or args_cli.payload_kg != 0. or args_cli.drag_load_kg != 0.
            or args_cli.drag_force_n not in (None, 0.)):
        parser.error("--hip_retraction_test requires --num_envs=1 --force_plane --fixed_command 0 0 0 "
                     "--disable_payload_motor_profile, with no keyboard/zero-action/payload/smoothing overrides.")
    args_cli.physics_substep_log = True
    args_cli.monitor_all_body_contacts = True
    args_cli.monitor_leg = "all"
    args_cli.monitor_joints = None
if args_cli.action_smoothing_override is not None and not (
    0.0 <= args_cli.action_smoothing_override <= 0.95
):
    parser.error("--action_smoothing_override must be within [0, 0.95].")
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# import after SimulationApp is created to avoid early Omniverse/pxr imports
from rl_utils import camera_follow

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform
from packaging import version

# check minimum supported rsl-rl version
RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import gymnasium as gym
import time
import torch
from hip_retraction_probe import HipRetractionProbe, ProbeSafety
import matplotlib.pyplot as plt

import isaaclab.utils.math as math_utils

try:
    import isaacsim.util.debug_draw._debug_draw as omni_debug_draw
except Exception:
    try:
        import omni.isaac.debug_draw._debug_draw as omni_debug_draw
    except Exception:
        omni_debug_draw = None

from rsl_rl.runners import OnPolicyRunner
from ppo_action_curriculum_runner import LightHWPPOActionCurriculumRunner, restore_curriculum_keyboard, curriculum_command_bounds
from ppo_speed_bridge_runner import LightHWPPOSpeedBridgeRunner
from ppo_grouped_noise_runner import LightHWGroupedNoiseRunner
from reference_runner import LightHWReferenceRunner

from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
    #handle_deprecated_rsl_rl_cfg,
)
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import rl_training.tasks  # noqa: F401
import rl_training.tasks.manager_based.locomotion.velocity.mdp as mdp


def _sync_base_velocity_command(env, command_state: torch.Tensor):
    """同步固定/键盘命令到 command manager，使 Isaac 自带 debug_vis 显示真实播放命令。"""
    try:
        command_term = env.command_manager.get_term("base_velocity")
    except Exception:
        return
    if hasattr(command_term, "vel_command_b"):
        command_term.vel_command_b[: command_state.shape[0], :3] = command_state[:, :3]



def _get_motor_tensors(env):
    """Return joint position, velocity, torque, joint names, and torque source for the robot."""
    robot = env.unwrapped.scene["robot"]
    data = robot.data

    joint_pos = getattr(data, "joint_pos", None)
    if joint_pos is None:
        joint_pos = getattr(env, "dof_pos", None)

    joint_vel = getattr(data, "joint_vel", None)
    if joint_vel is None:
        # Fallback for older/custom environments.
        joint_vel = getattr(env, "dof_vel", None)

    torque_source = None
    joint_tau = None
    for attr_name in ("applied_torque", "computed_torque", "joint_torque", "joint_effort"):
        if hasattr(data, attr_name):
            joint_tau = getattr(data, attr_name)
            torque_source = f"robot.data.{attr_name}"
            break

    if joint_tau is None and hasattr(env, "torques"):
        joint_tau = env.torques
        torque_source = "env.torques"

    joint_names = getattr(robot, "joint_names", None)
    if joint_names is None:
        joint_names = [f"joint_{i}" for i in range(joint_vel.shape[1])]

    return joint_pos, joint_vel, joint_tau, joint_names, torque_source


def _select_monitor_joints(joint_names, monitor_leg="fl", monitor_joints=None):
    """Select which joints to monitor from CLI options.

    Priority:
      1. --monitor_joints: comma-separated indices or name keywords.
      2. --monitor_leg all: every joint.
      3. --monitor_leg fl/fr/hl/hr: all joints whose names start with that leg prefix.

    For M20, leg prefix selection normally returns 4 motors:
    hipx, hipy, knee, and wheel.
    """
    joint_names = list(joint_names)

    if monitor_joints:
        selected = []
        tokens = [t.strip() for t in monitor_joints.split(",") if t.strip()]
        for token in tokens:
            if token.isdigit():
                idx = int(token)
                if idx < 0 or idx >= len(joint_names):
                    raise ValueError(f"Joint index {idx} is out of range 0..{len(joint_names)-1}")
                selected.append(idx)
            else:
                # First try exact match, then substring match for convenience.
                exact = [i for i, name in enumerate(joint_names) if name == token]
                partial = [i for i, name in enumerate(joint_names) if token.lower() in name.lower()]
                matches = exact if exact else partial
                if not matches:
                    raise ValueError(
                        f"Could not match joint token '{token}'. Available joints: {joint_names}"
                    )
                selected.extend(matches)

        # Remove duplicates while preserving user order.
        deduped = []
        for idx in selected:
            if idx not in deduped:
                deduped.append(idx)
        return deduped

    if monitor_leg == "all":
        return list(range(len(joint_names)))

    prefix = f"{monitor_leg.lower()}_"
    selected = [i for i, name in enumerate(joint_names) if name.lower().startswith(prefix)]

    # Fallback for naming styles like FL_HipX_joint.
    if not selected:
        prefix_upper = f"{monitor_leg.upper()}_"
        selected = [i for i, name in enumerate(joint_names) if name.upper().startswith(prefix_upper)]

    if not selected:
        raise ValueError(
            f"No joints matched --monitor_leg={monitor_leg}. "
            f"Use --monitor_joints with indices/names. Available joints: {joint_names}"
        )
    return selected


def _try_find_contact_body(contact_sensor, aliases):
    """Resolve one contact-sensor body from several common M20 wheel names."""
    for alias in aliases:
        try:
            body_ids, body_names = contact_sensor.find_bodies(alias, preserve_order=True)
        except TypeError:
            try:
                body_ids, body_names = contact_sensor.find_bodies(alias)
            except Exception:
                continue
        except Exception:
            continue
        if body_ids:
            body_name = body_names[0] if body_names else alias
            return int(body_ids[0]), str(body_name)
    return None, None


def _contact_body_label(body_name):
    """Build a stable CSV label for one contact-sensor body."""
    normalized = "".join(character.lower() if character.isalnum() else "_" for character in str(body_name))
    normalized = "_".join(part for part in normalized.split("_") if part)
    return f"body_{normalized}"


def _resolve_contact_monitor(env, include_all_bodies=False):
    """Resolve wheel contacts and optionally every tracked robot body."""
    scene = getattr(env.unwrapped, "scene", None)
    sensors = getattr(scene, "sensors", None)
    contact_sensor = None
    if sensors is not None:
        try:
            contact_sensor = sensors["contact_forces"]
        except Exception:
            try:
                contact_sensor = sensors.get("contact_forces", None)
            except Exception:
                contact_sensor = None
    if contact_sensor is None:
        print("[Contact] contact_forces sensor not found; contact force columns will be skipped.")
        return None, [], [], []

    wheel_aliases = (
        ("fl", ("FL_wheel_link", "FL_wheel", "fl_wheel_link", "fl_wheel")),
        ("fr", ("FR_wheel_link", "FR_wheel", "fr_wheel_link", "fr_wheel")),
        ("rl", ("RL_wheel_link", "HL_wheel_link", "rl_wheel_link", "hl_wheel_link", "rl_wheel", "hl_wheel")),
        ("rr", ("RR_wheel_link", "HR_wheel_link", "rr_wheel_link", "hr_wheel_link", "rr_wheel", "hr_wheel")),
    )

    labels = []
    body_ids = []
    body_names = []
    for label, aliases in wheel_aliases:
        body_id, body_name = _try_find_contact_body(contact_sensor, aliases)
        if body_id is None:
            available = getattr(contact_sensor, "body_names", None)
            print(f"[Contact] Could not resolve {label} wheel from aliases={aliases}. Available={available}.")
            return None, [], [], []
        labels.append(label)
        body_ids.append(body_id)
        body_names.append(body_name)

    if include_all_bodies:
        available_body_names = list(getattr(contact_sensor, "body_names", []) or [])
        used_body_ids = set(body_ids)
        used_labels = set(labels)
        for body_id, body_name in enumerate(available_body_names):
            if body_id in used_body_ids:
                continue
            label = _contact_body_label(body_name)
            suffix = 2
            unique_label = label
            while unique_label in used_labels:
                unique_label = f"{label}_{suffix}"
                suffix += 1
            labels.append(unique_label)
            body_ids.append(body_id)
            body_names.append(str(body_name))
            used_labels.add(unique_label)
        print(f"[Contact] Added {len(body_ids) - len(wheel_aliases)} non-wheel bodies to contact logging.")

    mapping = ", ".join(f"{label}:{name}[{body_id}]" for label, name, body_id in zip(labels, body_names, body_ids))
    print(f"[Contact] Monitoring contact forces: {mapping}.")
    return contact_sensor, body_ids, labels, body_names


def _contact_monitor_header(labels):
    if not labels:
        return []
    return (
        [f"contact_{label}_fz" for label in labels]
        + [f"contact_{label}_norm" for label in labels]
        + [f"contact_{label}_hist_max_fz" for label in labels]
        + [
            "contact_fz_mean",
            "contact_fz_min",
            "contact_fz_max",
            "contact_fz_spread",
            "contact_unloaded_count_5n",
        ]
    )


def _contact_monitor_row(contact_sensor, body_ids, labels, env_id=0):
    if contact_sensor is None or not body_ids:
        return []

    try:
        forces_w = contact_sensor.data.net_forces_w
        ids = torch.as_tensor(body_ids, dtype=torch.long, device=forces_w.device)
        force_values = torch.index_select(forces_w[env_id], 0, ids)
        fz = torch.abs(force_values[:, 2]).detach()
        norm = torch.linalg.norm(force_values, dim=-1).detach()
    except Exception:
        return [math.nan for _ in _contact_monitor_header(labels)]

    hist_max_fz = torch.full_like(fz, math.nan)
    history = getattr(contact_sensor.data, "net_forces_w_history", None)
    if history is not None:
        try:
            history_values = torch.index_select(history[env_id], 1, ids)
            hist_max_fz = torch.max(torch.abs(history_values[:, :, 2]), dim=0)[0].detach()
        except Exception:
            pass

    mean_fz = torch.mean(fz)
    min_fz = torch.min(fz)
    max_fz = torch.max(fz)
    spread = (max_fz - min_fz) / (mean_fz + 1e-6)
    unloaded = torch.sum(fz < 5.0)

    return (
        [float(value) for value in fz.detach().cpu().tolist()]
        + [float(value) for value in norm.detach().cpu().tolist()]
        + [float(value) for value in hist_max_fz.detach().cpu().tolist()]
        + [
            float(mean_fz.item()),
            float(min_fz.item()),
            float(max_fz.item()),
            float(spread.item()),
            int(unloaded.item()),
        ]
    )


def _resolve_payload_body_ids(robot, body_name):
    """Resolve payload body ids from an exact body name or regex."""
    try:
        body_ids, body_names = robot.find_bodies(body_name, preserve_order=True)
    except ValueError as exc:
        body_ids = []
        body_names = []
        body_name_lower = str(body_name).lower()
        for body_index, candidate_name in enumerate(robot.body_names):
            if candidate_name.lower() == body_name_lower:
                body_ids = [body_index]
                body_names = [candidate_name]
                print(f"[INFO] Payload body matched case-insensitively: {body_name!r} -> {candidate_name!r}.")
                break
        if not body_ids:
            raise exc
    if not body_ids:
        raise ValueError(f"No robot body matched --payload_body={body_name!r}. Available bodies: {robot.body_names}")
    return body_ids, body_names


def _set_payload_force(robot, body_ids, payload_kg, env_ids, drag_force_w=None):
    """Apply vertical payload and optional horizontal drag as global forces."""
    payload_kg = max(0.0, float(payload_kg))
    forces = torch.zeros((len(env_ids), len(body_ids), 3), device=robot.device)
    torques = torch.zeros_like(forces)
    if payload_kg > 0.0:
        forces[..., 2] = -payload_kg * 9.81
    if drag_force_w is not None:
        drag_force_w = torch.as_tensor(drag_force_w, dtype=forces.dtype, device=robot.device)
        if drag_force_w.ndim == 1:
            drag_force_w = drag_force_w.unsqueeze(0)
        forces += drag_force_w[:, None, :]
    robot.set_external_force_and_torque(
        forces=forces,
        torques=torques,
        env_ids=env_ids,
        body_ids=body_ids,
        is_global=True,
    )


def _compute_drag_force_w(robot, drag_force_n, command_state=None):
    """Compute a world-frame horizontal force opposing command direction or measured velocity."""
    drag_force_n = max(0.0, float(drag_force_n))
    force_w = torch.zeros(3, dtype=torch.float32, device=robot.device)
    if drag_force_n <= 0.0:
        return force_w

    direction_xy_w = None
    if command_state is not None:
        command_xy_b = command_state[0, :2].detach().to(device=robot.device, dtype=torch.float32)
        command_norm = torch.linalg.norm(command_xy_b)
        if command_norm > 1e-4:
            direction_b = torch.zeros((1, 3), dtype=torch.float32, device=robot.device)
            direction_b[0, :2] = command_xy_b / command_norm
            direction_w = math_utils.quat_apply(robot.data.root_quat_w[0:1], direction_b)[0]
            direction_xy_w = direction_w[:2]

    if direction_xy_w is None:
        root_lin_vel_w = getattr(robot.data, "root_lin_vel_w", None)
        if root_lin_vel_w is not None:
            velocity_xy_w = root_lin_vel_w[0, :2].detach().to(dtype=torch.float32)
            velocity_norm = torch.linalg.norm(velocity_xy_w)
            if velocity_norm > 1e-3:
                direction_xy_w = velocity_xy_w / velocity_norm

    if direction_xy_w is None:
        return force_w

    direction_norm = torch.linalg.norm(direction_xy_w)
    if direction_norm <= 1e-6:
        return force_w
    force_w[:2] = -drag_force_n * direction_xy_w / direction_norm
    return force_w


def _payload_command_scale(payload_kg, heavy_threshold_kg):
    """Reduce commanded speed under payload so the old policy does not overdrive the gait."""
    if payload_kg <= heavy_threshold_kg:
        return 1.0
    medium_full_kg = 50.0
    heavy_full_kg = 65.0
    medium_scale = 0.35
    heavy_scale = 0.30
    if payload_kg <= medium_full_kg:
        alpha = min(1.0, max(0.0, (payload_kg - heavy_threshold_kg) / max(1.0, medium_full_kg - heavy_threshold_kg)))
        return 1.0 * (1.0 - alpha) + medium_scale * alpha
    alpha = min(1.0, max(0.0, (payload_kg - medium_full_kg) / max(1.0, heavy_full_kg - medium_full_kg)))
    return medium_scale * (1.0 - alpha) + heavy_scale * alpha


def _payload_motor_profile(payload_kg, heavy_threshold_kg):
    """Select a motor/action profile from the currently applied payload force."""
    if payload_kg <= 0.1:
        return "empty", {
            "joint_effort": 76.4,
            "joint_stiffness": 80.0,
            "joint_damping": 2.0,
            "joint_armature": 0.0,
            "wheel_effort": 21.6,
            "wheel_damping": 0.6,
            "hipx_action_ratio": 1.0,
            "stand_hipx_action_ratio": 1.0,
            "hipy_action_ratio": 1.0,
            "knee_action_ratio": 1.0,
            "wheel_action_ratio": 1.0,
            "action_smoothing": 0.0,
            "front_hipy_action_bias": 0.0,
            "front_knee_action_bias": 0.0,
            "rear_hipy_action_bias": 0.0,
            "rear_knee_action_bias": 0.0,
            "posture_guard_weight": 0.0,
            "hipx_guard_weight": 0.0,
            "stand_wheel_action_ratio": 1.0,
            "move_action_smoothing": 0.0,
            "move_posture_guard_scale": 1.0,
            "move_hipx_guard_scale": 1.0,
        }
    light_profile = {
        "joint_effort": 180.0,
        "joint_stiffness": 100.0,
        "joint_damping": 4.0,
        "joint_armature": 0.005,
        "wheel_effort": 100.0,
        "wheel_damping": 0.8,
        "hipx_action_ratio": 0.10 / 0.125,
        "stand_hipx_action_ratio": 0.10 / 0.125,
        "hipy_action_ratio": 0.20 / 0.25,
        "knee_action_ratio": 0.24 / 0.25,
        "wheel_action_ratio": 5.3 / 5.0,
        "action_smoothing": 0.0,
        "front_hipy_action_bias": 0.0,
        "front_knee_action_bias": 0.0,
        "rear_hipy_action_bias": 0.0,
        "rear_knee_action_bias": 0.0,
        "posture_guard_weight": 0.0,
        "hipx_guard_weight": 0.0,
        "stand_wheel_action_ratio": 1.0,
        "move_action_smoothing": 0.0,
        "move_posture_guard_scale": 1.0,
        "move_hipx_guard_scale": 1.0,
    }
    if payload_kg <= heavy_threshold_kg:
        return "light_stair", light_profile

    # The 50 kg case crouched without torque saturation, so give it a separate medium-load posture profile.
    medium_full_kg = 50.0
    heavy_full_kg = 65.0
    medium_profile = {
        "joint_effort": 250.0,
        "joint_stiffness": 124.0,
        "joint_damping": 6.2,
        "joint_armature": 0.014,
        "wheel_effort": 125.0,
        "wheel_damping": 1.0,
        "hipx_action_ratio": 0.09 / 0.125,
        "stand_hipx_action_ratio": 0.055 / 0.125,
        "hipy_action_ratio": 0.17 / 0.25,
        "knee_action_ratio": 0.20 / 0.25,
        "wheel_action_ratio": 4.85 / 5.0,
        "action_smoothing": 0.18,
        "front_hipy_action_bias": 0.0,
        "front_knee_action_bias": 0.0,
        "rear_hipy_action_bias": 0.0,
        "rear_knee_action_bias": 0.0,
        "posture_guard_weight": 0.50,
        "hipx_guard_weight": 0.85,
        "stand_wheel_action_ratio": 0.20,
        "move_action_smoothing": 0.08,
        "move_posture_guard_scale": 0.05,
        "move_hipx_guard_scale": 0.0,
    }
    heavy_profile = {
        "joint_effort": 290.0,
        "joint_stiffness": 136.0,
        "joint_damping": 6.8,
        "joint_armature": 0.018,
        "wheel_effort": 145.0,
        "wheel_damping": 1.0,
        "hipx_action_ratio": 0.09 / 0.125,
        "stand_hipx_action_ratio": 0.09 / 0.125,
        "hipy_action_ratio": 0.205 / 0.25,
        "knee_action_ratio": 0.245 / 0.25,
        "wheel_action_ratio": 5.35 / 5.0,
        "action_smoothing": 0.10,
        # Keep the trained policy's nominal posture. Bias made the heavy gait fight itself and roll over.
        "front_hipy_action_bias": 0.0,
        "front_knee_action_bias": 0.0,
        "rear_hipy_action_bias": 0.0,
        "rear_knee_action_bias": 0.0,
        "posture_guard_weight": 0.0,
        "hipx_guard_weight": 0.0,
        "stand_wheel_action_ratio": 0.60,
        "move_action_smoothing": 0.08,
        "move_posture_guard_scale": 1.0,
        "move_hipx_guard_scale": 1.0,
    }

    if payload_kg <= medium_full_kg:
        alpha = min(1.0, max(0.0, (payload_kg - heavy_threshold_kg) / max(1.0, medium_full_kg - heavy_threshold_kg)))
        blended_profile = {
            key: light_profile[key] * (1.0 - alpha) + medium_profile[key] * alpha for key in light_profile
        }
        return f"medium_blend_{int(alpha * 100):03d}", blended_profile

    alpha = min(1.0, max(0.0, (payload_kg - medium_full_kg) / max(1.0, heavy_full_kg - medium_full_kg)))
    blended_profile = {
        key: medium_profile[key] * (1.0 - alpha) + heavy_profile[key] * alpha for key in medium_profile
    }
    return f"heavy_blend_{int(alpha * 100):03d}", blended_profile


def _fill_actuator_tensor(actuator, attr_name, value):
    attr = getattr(actuator, attr_name, None)
    if isinstance(attr, torch.Tensor):
        attr.fill_(float(value))
    elif attr is not None:
        setattr(actuator, attr_name, float(value))


def _actuator_scalar(actuator, attr_name, default=math.nan):
    attr = getattr(actuator, attr_name, default)
    if isinstance(attr, torch.Tensor):
        if attr.numel() == 0:
            return float(default)
        return float(attr.flatten()[0].item())
    if attr is None:
        return float(default)
    return float(attr)


def _native_motor_profile(robot):
    """Return an identity playback profile while preserving native task actuator values."""
    joint_actuator = robot.actuators["joint"]
    wheel_actuator = robot.actuators["wheel"]
    return "native", {
        "joint_effort": _actuator_scalar(joint_actuator, "effort_limit"),
        "joint_stiffness": _actuator_scalar(joint_actuator, "stiffness"),
        "joint_damping": _actuator_scalar(joint_actuator, "damping"),
        "joint_armature": _actuator_scalar(joint_actuator, "armature"),
        "wheel_effort": _actuator_scalar(wheel_actuator, "effort_limit"),
        "wheel_damping": _actuator_scalar(wheel_actuator, "damping"),
        "hipx_action_ratio": 1.0,
        "stand_hipx_action_ratio": 1.0,
        "hipy_action_ratio": 1.0,
        "knee_action_ratio": 1.0,
        "wheel_action_ratio": 1.0,
        "action_smoothing": 0.0,
        "front_hipy_action_bias": 0.0,
        "front_knee_action_bias": 0.0,
        "rear_hipy_action_bias": 0.0,
        "rear_knee_action_bias": 0.0,
        "posture_guard_weight": 0.0,
        "hipx_guard_weight": 0.0,
        "stand_wheel_action_ratio": 1.0,
        "move_action_smoothing": 0.0,
        "move_posture_guard_scale": 1.0,
        "move_hipx_guard_scale": 1.0,
    }


def _apply_payload_motor_profile(robot, profile_name, profile, payload_kg):
    """Apply runtime actuator gains/limits for the selected payload profile."""
    joint_actuator = robot.actuators["joint"]
    wheel_actuator = robot.actuators["wheel"]
    _fill_actuator_tensor(joint_actuator, "effort_limit", profile["joint_effort"])
    _fill_actuator_tensor(joint_actuator, "stiffness", profile["joint_stiffness"])
    _fill_actuator_tensor(joint_actuator, "damping", profile["joint_damping"])
    _fill_actuator_tensor(joint_actuator, "armature", profile["joint_armature"])
    _fill_actuator_tensor(wheel_actuator, "effort_limit", profile["wheel_effort"])
    _fill_actuator_tensor(wheel_actuator, "damping", profile["wheel_damping"])
    print(
        f"[Payload] Motor profile -> {profile_name}: payload={payload_kg:.2f}kg "
        f"({payload_kg * 9.81:.1f}N), leg effort={profile['joint_effort']:.0f}Nm, "
        f"leg PD=({profile['joint_stiffness']:.0f}, {profile['joint_damping']:.1f}), "
        f"wheel effort={profile['wheel_effort']:.0f}Nm, smoothing={profile['action_smoothing']:.2f}, "
        f"front stand bias=({profile['front_hipy_action_bias']:.2f}, {profile['front_knee_action_bias']:.2f}), "
        f"posture guard={profile['posture_guard_weight']:.2f}, hipx guard={profile['hipx_guard_weight']:.2f}, "
        f"stand hipx ratio={profile['stand_hipx_action_ratio']:.2f}, "
        f"stand wheel ratio={profile['stand_wheel_action_ratio']:.2f}, "
        f"move smoothing={profile['move_action_smoothing']:.2f}."
    )


def _scale_actions_for_payload(actions, profile):
    """Adjust raw policy actions so the effective action scale follows the active payload profile."""
    if actions.shape[1] < 16:
        return actions
    scaled_actions = actions.clone()
    scaled_actions[:, [0, 3, 6, 9]] *= profile["hipx_action_ratio"]
    scaled_actions[:, [1, 4, 7, 10]] *= profile["hipy_action_ratio"]
    scaled_actions[:, [2, 5, 8, 11]] *= profile["knee_action_ratio"]
    scaled_actions[:, 12:16] *= profile["wheel_action_ratio"]
    scaled_actions[:, [1, 4]] += profile["front_hipy_action_bias"]
    scaled_actions[:, [2, 5]] += profile["front_knee_action_bias"]
    scaled_actions[:, [7, 10]] += profile["rear_hipy_action_bias"]
    scaled_actions[:, [8, 11]] += profile["rear_knee_action_bias"]
    return scaled_actions


def _apply_posture_guard(actions, robot, joint_names, profile, posture_scale=1.0, hipx_scale=1.0):
    """Gently correct crouch and lateral splay from measured joint angles."""
    guard_weight = float(profile.get("posture_guard_weight", 0.0)) * float(posture_scale)
    hipx_guard_weight = float(profile.get("hipx_guard_weight", 0.0)) * float(hipx_scale)
    if (guard_weight <= 0.0 and hipx_guard_weight <= 0.0) or actions.shape[1] < 16:
        return actions

    joint_pos = getattr(robot.data, "joint_pos", None)
    if joint_pos is None:
        return actions

    joint_index = {name: index for index, name in enumerate(joint_names)}
    corrected_actions = actions.clone()

    def add_lower_limit_correction(joint_name, action_index, lower_limit, gain, max_action_delta):
        pos_index = joint_index.get(joint_name)
        if pos_index is None:
            return
        error = lower_limit - joint_pos[:, pos_index]
        correction = torch.clamp(error * gain * guard_weight, min=0.0, max=max_action_delta * guard_weight)
        corrected_actions[:, action_index] += correction

    def add_upper_limit_correction(joint_name, action_index, upper_limit, gain, max_action_delta):
        pos_index = joint_index.get(joint_name)
        if pos_index is None:
            return
        error = joint_pos[:, pos_index] - upper_limit
        correction = torch.clamp(error * gain * guard_weight, min=0.0, max=max_action_delta * guard_weight)
        corrected_actions[:, action_index] -= correction

    for joint_name, action_index in (("fl_hipy_joint", 1), ("fr_hipy_joint", 4)):
        add_lower_limit_correction(joint_name, action_index, lower_limit=-0.72, gain=0.45, max_action_delta=0.18)
    for joint_name, action_index in (("fl_knee_joint", 2), ("fr_knee_joint", 5)):
        add_upper_limit_correction(joint_name, action_index, upper_limit=1.22, gain=0.36, max_action_delta=0.22)
    for joint_name, action_index in (("hl_hipy_joint", 7), ("hr_hipy_joint", 10)):
        add_lower_limit_correction(joint_name, action_index, lower_limit=0.48, gain=0.30, max_action_delta=0.12)
    for joint_name, action_index in (("hl_knee_joint", 8), ("hr_knee_joint", 11)):
        add_lower_limit_correction(joint_name, action_index, lower_limit=-1.18, gain=0.34, max_action_delta=0.18)

    if hipx_guard_weight > 0.0:
        def add_hipx_guard(joint_name, action_index, outward_sign):
            pos_index = joint_index.get(joint_name)
            if pos_index is None:
                return
            outward_pos = joint_pos[:, pos_index] * outward_sign
            error = outward_pos - 0.10
            correction = torch.clamp(error * 2.2 * hipx_guard_weight, min=0.0, max=0.75 * hipx_guard_weight)
            corrected_actions[:, action_index] -= correction * outward_sign

        for joint_name, action_index, outward_sign in (
            ("fl_hipx_joint", 0, -1.0),
            ("fr_hipx_joint", 3, 1.0),
            ("hl_hipx_joint", 6, -1.0),
            ("hr_hipx_joint", 9, 1.0),
        ):
            add_hipx_guard(joint_name, action_index, outward_sign)

    return corrected_actions


LIVE_TUNING_PARAMS = (
    ("joint_effort", 10.0, 0.0, 500.0),
    ("joint_stiffness", 5.0, 0.0, 400.0),
    ("joint_damping", 0.5, 0.0, 40.0),
    ("joint_armature", 0.002, 0.0, 0.100),
    ("wheel_effort", 5.0, 0.0, 300.0),
    ("wheel_damping", 0.1, 0.0, 10.0),
    ("hipx_action_ratio", 0.05, 0.05, 2.0),
    ("stand_hipx_action_ratio", 0.05, 0.0, 2.0),
    ("hipy_action_ratio", 0.05, 0.05, 2.0),
    ("knee_action_ratio", 0.05, 0.05, 2.0),
    ("wheel_action_ratio", 0.05, 0.05, 2.0),
    ("action_smoothing", 0.02, 0.0, 0.95),
    ("move_action_smoothing", 0.02, 0.0, 0.95),
    ("posture_guard_weight", 0.05, 0.0, 2.0),
    ("hipx_guard_weight", 0.05, 0.0, 2.0),
    ("move_posture_guard_scale", 0.05, 0.0, 2.0),
    ("move_hipx_guard_scale", 0.05, 0.0, 2.0),
    ("stand_wheel_action_ratio", 0.05, 0.0, 2.0),
)
LIVE_TUNING_PARAM_NAMES = tuple(param[0] for param in LIVE_TUNING_PARAMS)


def _apply_live_tuning_overrides(profile, overrides):
    """Return a profile copy with keyboard live-tuning values applied."""
    if not overrides:
        return profile
    tuned_profile = dict(profile)
    for name, value in overrides.items():
        if name in tuned_profile:
            tuned_profile[name] = float(value)
    return tuned_profile


def _motor_profile_key(profile_name, profile):
    """Create a stable key so runtime profile edits are pushed immediately."""
    return (
        profile_name,
        tuple((name, round(float(profile[name]), 6)) for name in LIVE_TUNING_PARAM_NAMES if name in profile),
    )


def _terrain_column_for_subterrain(terrain_generator_cfg, subterrain_name):
    """Map a named sub-terrain to the first deterministic curriculum column using the generator proportions."""
    if terrain_generator_cfg is None or subterrain_name is None:
        return None
    sub_terrains = getattr(terrain_generator_cfg, "sub_terrains", None)
    if not sub_terrains:
        return None

    names = list(sub_terrains.keys())
    proportions = [float(getattr(sub_terrains[name], "proportion", 0.0)) for name in names]
    total = sum(proportions)
    if total <= 0.0:
        return None

    normalized = [value / total for value in proportions]
    cumulative = []
    running = 0.0
    for value in normalized:
        running += value
        cumulative.append(running)

    num_cols = int(getattr(terrain_generator_cfg, "num_cols", 1))
    for col in range(num_cols):
        selector = col / max(1, num_cols) + 0.001
        sub_index = 0
        for index, threshold in enumerate(cumulative):
            if selector < threshold:
                sub_index = index
                break
        if names[sub_index] == subterrain_name:
            return col

    available = ", ".join(names)
    print(f"[Spawn] Could not find sub-terrain '{subterrain_name}'. Available: {available}. Falling back to column 0.")
    return 0


def _force_playback_spawn_terrain_cell(env, level, terrain_col, terrain_name=None):
    """Force all playback envs to reset from one selected rough-terrain cell."""
    terrain = getattr(env.unwrapped.scene, "terrain", None)
    terrain_origins = getattr(terrain, "terrain_origins", None)
    if terrain is None or terrain_origins is None:
        print("[Spawn] Terrain origins are not available; using the environment default spawn.")
        return

    num_levels = terrain_origins.shape[0]
    num_cols = terrain_origins.shape[1]
    level = max(0, min(int(level), num_levels - 1))
    terrain_col = max(0, min(int(terrain_col), num_cols - 1))
    env_ids = torch.arange(env.unwrapped.num_envs, device=terrain.env_origins.device)

    if hasattr(terrain, "terrain_levels"):
        terrain.terrain_levels[env_ids] = level
    if hasattr(terrain, "terrain_types"):
        terrain.terrain_types[env_ids] = terrain_col
    terrain.env_origins[env_ids] = terrain_origins[level, terrain_col]

    terrain_label = terrain_name if terrain_name else f"column_{terrain_col}"
    origin = terrain_origins[level, terrain_col].detach().cpu().tolist()
    print(
        f"[Spawn] Playback spawn forced to rough terrain '{terrain_label}' "
        f"(level={level}, col={terrain_col}, origin={origin})."
    )


def _reset_root_state_on_selected_terrain(
    env,
    env_ids,
    pose_range,
    velocity_range,
    terrain_level=0,
    terrain_col=0,
    asset_cfg=SceneEntityCfg("robot"),
):
    """Reset the robot on a selected terrain cell while keeping the original rough terrain."""
    asset = env.scene[asset_cfg.name]
    terrain = getattr(env.scene, "terrain", None)
    terrain_origins = getattr(terrain, "terrain_origins", None)

    if terrain is not None and terrain_origins is not None:
        level = max(0, min(int(terrain_level), terrain_origins.shape[0] - 1))
        col = max(0, min(int(terrain_col), terrain_origins.shape[1] - 1))
        if hasattr(terrain, "terrain_levels"):
            terrain.terrain_levels[env_ids] = level
        if hasattr(terrain, "terrain_types"):
            terrain.terrain_types[env_ids] = col
        terrain.env_origins[env_ids] = terrain_origins[level, col]

    root_states = asset.data.default_root_state[env_ids].clone()

    pose_keys = ["x", "y", "z", "roll", "pitch", "yaw"]
    pose_ranges = torch.tensor([pose_range.get(key, (0.0, 0.0)) for key in pose_keys], device=asset.device)
    pose_samples = math_utils.sample_uniform(
        pose_ranges[:, 0], pose_ranges[:, 1], (len(env_ids), 6), device=asset.device
    )
    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + pose_samples[:, 0:3]
    orientations_delta = math_utils.quat_from_euler_xyz(
        pose_samples[:, 3], pose_samples[:, 4], pose_samples[:, 5]
    )
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)

    velocity_keys = ["x", "y", "z", "roll", "pitch", "yaw"]
    velocity_ranges = torch.tensor(
        [velocity_range.get(key, (0.0, 0.0)) for key in velocity_keys], device=asset.device
    )
    velocity_samples = math_utils.sample_uniform(
        velocity_ranges[:, 0], velocity_ranges[:, 1], (len(env_ids), 6), device=asset.device
    )
    velocities = root_states[:, 7:13] + velocity_samples

    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


def _disable_event_if_present(events, name):
    if hasattr(events, name):
        setattr(events, name, None)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    task_name = args_cli.task.split(":")[-1]
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else 50
    if args_cli.static_check:
        env_cfg.scene.num_envs = 1
        args_cli.fixed_command = [0.0, 0.0, 0.0]
        args_cli.spawn_jitter_xy = 0.0
        args_cli.payload_ramp_time = 0.0
        args_cli.no_plots = True

    # handle deprecated configurations (convert old policy format to new actor/critic format)
    #agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Keep the original rough terrain for playback by default, but optionally force a pure plane for
    # flat-ground diagnostics without switching to another task registration. We intentionally keep
    # height-scan observation terms unchanged so checkpoints trained on rough observations still load.
    spawn_terrain_col = None
    if getattr(env_cfg.scene, "terrain", None) is not None:
        if args_cli.force_plane:
            env_cfg.scene.terrain.terrain_type = "plane"
            env_cfg.scene.terrain.terrain_generator = None
            env_cfg.scene.terrain.max_init_terrain_level = None
            env_cfg.scene.terrain.visual_material = None
        else:
            env_cfg.scene.terrain.max_init_terrain_level = max(0, int(args_cli.spawn_terrain_level))
        if args_cli.static_check and getattr(env_cfg.scene.terrain, "terrain_type", None) == "plane":
            env_cfg.scene.terrain.visual_material = None
        if env_cfg.scene.terrain.terrain_generator is not None:
            env_cfg.scene.terrain.terrain_generator.curriculum = not args_cli.static_check
            spawn_terrain_col = _terrain_column_for_subterrain(
                env_cfg.scene.terrain.terrain_generator, args_cli.spawn_terrain_type
            )
    has_generated_terrain = (
        getattr(env_cfg.scene, "terrain", None) is not None
        and env_cfg.scene.terrain.terrain_generator is not None
    )
    spawn_offset_x = args_cli.spawn_offset_x
    if spawn_offset_x is None:
        spawn_offset_x = 0.9 if has_generated_terrain else 0.0
    env_cfg.curriculum.terrain_levels = None
    if hasattr(env_cfg.events, "randomize_reset_base") and env_cfg.events.randomize_reset_base is not None:
        reset_event = env_cfg.events.randomize_reset_base
        reset_params = reset_event.params
        if "pose_range" in reset_params and "velocity_range" in reset_params:
            reset_event.func = _reset_root_state_on_selected_terrain
            reset_params["terrain_level"] = max(0, int(args_cli.spawn_terrain_level))
            reset_params["terrain_col"] = 0 if spawn_terrain_col is None else spawn_terrain_col
            reset_params["pose_range"].update(
                {
                    "x": (
                        spawn_offset_x - abs(args_cli.spawn_jitter_xy),
                        spawn_offset_x + abs(args_cli.spawn_jitter_xy),
                    ),
                    "y": (
                        args_cli.spawn_offset_y - abs(args_cli.spawn_jitter_xy),
                        args_cli.spawn_offset_y + abs(args_cli.spawn_jitter_xy),
                    ),
                    "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0),
                    "yaw": (0.0, 0.0),
                }
            )
            reset_params["velocity_range"].update(
                {
                    "x": (0.0, 0.0),
                    "y": (0.0, 0.0),
                    "z": (0.0, 0.0),
                    "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0),
                    "yaw": (0.0, 0.0),
                }
            )
    # disable randomization for play
    env_cfg.observations.policy.enable_corruption = False
    # remove random pushing
    env_cfg.events.randomize_apply_external_force_torque = None
    env_cfg.events.randomize_push_robot = None
    env_cfg.events.push_robot = None
    env_cfg.curriculum.command_levels = None
    if args_cli.static_check:
        for event_name in (
            "randomize_reset_base",
            "randomize_rigid_body_mass_base",
            "randomize_rigid_body_mass",
            "randomize_com_positions",
            "randomize_rigid_body_material",
            "randomize_actuator_gains",
            "randomize_rigid_body_inertia",
            "randomize_reset_joints",
            "randomize_apply_external_force_torque",
            "randomize_push_robot",
            "push_robot",
        ):
            _disable_event_if_present(env_cfg.events, event_name)
        if hasattr(env_cfg.terminations, "time_out"):
            env_cfg.terminations.time_out = None
        if hasattr(env_cfg.scene, "height_scanner"):
            env_cfg.scene.height_scanner = None
        if hasattr(env_cfg.scene, "height_scanner_base"):
            env_cfg.scene.height_scanner_base = None
        if hasattr(env_cfg.observations, "policy") and hasattr(env_cfg.observations.policy, "height_scan"):
            env_cfg.observations.policy.height_scan = None
        if hasattr(env_cfg.observations, "critic") and hasattr(env_cfg.observations.critic, "height_scan"):
            env_cfg.observations.critic.height_scan = None
        if (
            hasattr(env_cfg, "rewards")
            and hasattr(env_cfg.rewards, "base_height_l2")
            and env_cfg.rewards.base_height_l2 is not None
        ):
            env_cfg.rewards.base_height_l2.params["sensor_cfg"] = None
        print("[StaticCheck] Enabled: single env, zero command, zero actions, randomization disabled.")
    elif args_cli.zero_policy_actions:
        print(
            "[ZeroPolicyActions] Enabled: policy residuals are zero; task reset, reference, and physics remain active."
        )
    keyboard_command_state = None
    curriculum_keyboard_bounds = None
    payload_state = {
        "enabled": args_cli.payload_kg > 0.0,
        "kg": max(0.0, float(args_cli.payload_kg)),
    }
    initial_drag_force_n = (
        float(args_cli.drag_force_n)
        if args_cli.drag_force_n is not None
        else max(0.0, float(args_cli.drag_load_kg)) * max(0.0, float(args_cli.drag_mu)) * 9.81
    )
    drag_state = {
        "enabled": initial_drag_force_n > 0.0,
        "force_n": min(max(0.0, initial_drag_force_n), max(0.0, float(args_cli.drag_max_n))),
    }
    live_tuning_state = {
        "index": 0,
        "overrides": {},
        "base_profile": None,
        "effective_profile": None,
        "dirty": False,
    }
    if args_cli.keyboard or args_cli.fixed_command is not None:
        env_cfg.scene.num_envs = 1
        env_cfg.terminations.time_out = None
    if args_cli.fixed_command is not None:
        fixed_command = torch.tensor(args_cli.fixed_command, dtype=torch.float32)

        def _fixed_command_obs_term(env):
            nonlocal keyboard_command_state
            payload_for_command = payload_state["kg"] if payload_state["enabled"] else 0.0
            command_scale = _payload_command_scale(payload_for_command, args_cli.heavy_payload_threshold_kg)
            keyboard_command_state = (fixed_command * command_scale).unsqueeze(0).to(env.device)
            _sync_base_velocity_command(env, keyboard_command_state)
            return keyboard_command_state

        if getattr(env_cfg.observations.policy, "velocity_commands", None) is not None:
            env_cfg.observations.policy.velocity_commands = ObsTerm(
                func=_fixed_command_obs_term,
            )
        else:
            print(
                "[INFO] Policy has no velocity-command observation; keeping its "
                "training observation dimensions unchanged."
            )
    elif args_cli.keyboard:
        if getattr(env_cfg.observations.policy, "velocity_commands", None) is None:
            raise ValueError(
                "Keyboard control requires a policy trained with the "
                "'velocity_commands' observation term."
            )
        config = Se2KeyboardCfg(
            v_x_sensitivity=env_cfg.commands.base_velocity.ranges.lin_vel_x[1],
            v_y_sensitivity=env_cfg.commands.base_velocity.ranges.lin_vel_y[1],
            omega_z_sensitivity=env_cfg.commands.base_velocity.ranges.ang_vel_z[1],
        )
        controller = Se2Keyboard(config)

        def _print_payload_state():
            status = "ON" if payload_state["enabled"] and payload_state["kg"] > 0.0 else "OFF"
            print(f"[Payload] {status}: {payload_state['kg']:.2f} kg on {args_cli.payload_body}")
            print("[Payload] Motor profile will switch automatically from the currently applied payload force.")

        def _toggle_payload():
            payload_state["enabled"] = not payload_state["enabled"]
            _print_payload_state()

        def _increase_payload():
            payload_state["kg"] = min(args_cli.payload_max_kg, payload_state["kg"] + args_cli.payload_step_kg)
            payload_state["enabled"] = payload_state["kg"] > 0.0
            _print_payload_state()

        def _decrease_payload():
            payload_state["kg"] = max(0.0, payload_state["kg"] - args_cli.payload_step_kg)
            payload_state["enabled"] = payload_state["kg"] > 0.0
            _print_payload_state()

        def _print_drag_state():
            status = "ON" if drag_state["enabled"] and drag_state["force_n"] > 0.0 else "OFF"
            equivalent_kg = drag_state["force_n"] / max(1e-6, max(0.0, float(args_cli.drag_mu)) * 9.81)
            print(
                f"[Drag] {status}: {drag_state['force_n']:.1f} N horizontal resistance "
                f"(equiv {equivalent_kg:.1f} kg at mu={args_cli.drag_mu:.2f}). "
                "Direction opposes command or current velocity."
            )

        def _toggle_drag():
            drag_state["enabled"] = not drag_state["enabled"]
            _print_drag_state()

        def _increase_drag():
            drag_state["force_n"] = min(
                max(0.0, float(args_cli.drag_max_n)),
                drag_state["force_n"] + max(0.0, float(args_cli.drag_step_n)),
            )
            drag_state["enabled"] = drag_state["force_n"] > 0.0
            _print_drag_state()

        def _decrease_drag():
            drag_state["force_n"] = max(
                0.0,
                drag_state["force_n"] - max(0.0, float(args_cli.drag_step_n)),
            )
            drag_state["enabled"] = drag_state["force_n"] > 0.0
            _print_drag_state()

        def _profile_for_live_tuning():
            profile = live_tuning_state.get("effective_profile") or live_tuning_state.get("base_profile")
            if profile is None:
                payload_for_profile = payload_state["kg"] if payload_state["enabled"] else 0.0
                _, profile = _payload_motor_profile(payload_for_profile, args_cli.heavy_payload_threshold_kg)
            return profile

        def _print_live_tuning_state():
            name, step, low, high = LIVE_TUNING_PARAMS[live_tuning_state["index"]]
            profile = _profile_for_live_tuning()
            current = profile.get(name, 0.0)
            base = live_tuning_state.get("base_profile") or profile
            base_value = base.get(name, current)
            print(
                f"[Tune] Selected {name}: current={current:.4f}, base={base_value:.4f}, "
                f"step={step:g}, range=[{low:g}, {high:g}]"
            )
            if live_tuning_state["overrides"]:
                overrides = ", ".join(
                    f"{key}={value:.4f}" for key, value in sorted(live_tuning_state["overrides"].items())
                )
                print(f"[Tune] Active overrides: {overrides}")
            else:
                print("[Tune] Active overrides: none")

        def _select_live_tuning_param(delta):
            live_tuning_state["index"] = (live_tuning_state["index"] + delta) % len(LIVE_TUNING_PARAMS)
            _print_live_tuning_state()

        def _adjust_live_tuning_param(direction):
            name, step, low, high = LIVE_TUNING_PARAMS[live_tuning_state["index"]]
            profile = _profile_for_live_tuning()
            current = float(profile.get(name, 0.0))
            new_value = min(high, max(low, current + direction * step))
            live_tuning_state["overrides"][name] = new_value
            live_tuning_state["dirty"] = True
            print(f"[Tune] {name} -> {new_value:.4f}")

        def _reset_live_tuning():
            live_tuning_state["overrides"].clear()
            live_tuning_state["dirty"] = True
            print("[Tune] Cleared live motor/profile overrides.")

        controller.add_callback("P", _toggle_payload)
        controller.add_callback("U", _increase_payload)
        controller.add_callback("J", _decrease_payload)
        controller.add_callback("O", _toggle_drag)
        controller.add_callback("I", _increase_drag)
        controller.add_callback("K", _decrease_drag)
        controller.add_callback("H", _print_drag_state)
        controller.add_callback("C", lambda: _select_live_tuning_param(-1))
        controller.add_callback("V", lambda: _select_live_tuning_param(1))
        controller.add_callback("B", lambda: _adjust_live_tuning_param(-1))
        controller.add_callback("N", lambda: _adjust_live_tuning_param(1))
        controller.add_callback("M", _print_live_tuning_state)
        controller.add_callback("R", _reset_live_tuning)

        def _keyboard_obs_term(env):
            nonlocal keyboard_command_state
            payload_for_command = payload_state["kg"] if payload_state["enabled"] else 0.0
            command_scale = _payload_command_scale(payload_for_command, args_cli.heavy_payload_threshold_kg)
            keyboard_command_state = (
                torch.tensor(controller.advance(), dtype=torch.float32) * command_scale
            ).unsqueeze(0).to(env.device)
            if curriculum_keyboard_bounds is not None:
                keyboard_command_state = keyboard_command_state.clamp(
                    curriculum_keyboard_bounds[:, 0], curriculum_keyboard_bounds[:, 1]
                )
            _sync_base_velocity_command(env, keyboard_command_state)
            return keyboard_command_state

        env_cfg.observations.policy.velocity_commands = ObsTerm(
            func=_keyboard_obs_term,
        )

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    if args_cli.use_checkpoint_pose:
        restore_checkpoint_pose(env_cfg, resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during playback.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    if spawn_terrain_col is not None:
        print(
            f"[Spawn] Playback reset target: rough terrain '{args_cli.spawn_terrain_type}' "
            f"(level={args_cli.spawn_terrain_level}, col={spawn_terrain_col})."
        )

    policy = None
    if args_cli.static_check:
        print("[StaticCheck] Skipping RSL-RL checkpoint loading; using zero actions for static load check.")
    else:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        # convert config to dict and create runner
        train_cfg = agent_cfg.to_dict()
        if getattr(agent_cfg, "lighthw_ppo_action_curriculum", False):
            runner_type = (LightHWReferenceRunner if getattr(agent_cfg, "lighthw_reference", False)
                           else LightHWGroupedNoiseRunner if getattr(agent_cfg, "lighthw_grouped_noise", False)
                           else LightHWPPOSpeedBridgeRunner if getattr(agent_cfg, "lighthw_speed_bridge", False)
                           else LightHWPPOActionCurriculumRunner)
            ppo_runner = runner_type(
                env, train_cfg, log_dir=None, device=agent_cfg.device, curriculum_training=False
            )
        else:
            ppo_runner = OnPolicyRunner(env, train_cfg, log_dir=None, device=agent_cfg.device)
        ppo_runner.load(resume_path)
        if getattr(agent_cfg, "lighthw_forward_speed", False) and args_cli.fixed_command is not None:
            bounds = curriculum_command_bounds(env)
            requested = torch.as_tensor(args_cli.fixed_command, device=bounds.device, dtype=bounds.dtype)
            if not torch.isfinite(requested).all() or torch.any((requested < bounds[:, 0]) | (requested > bounds[:, 1])):
                raise ValueError(f"Fixed command {args_cli.fixed_command} exceeds checkpoint bounds {bounds.tolist()}.")
        if getattr(agent_cfg, "lighthw_ppo_action_curriculum", False) and args_cli.keyboard and args_cli.fixed_command is None:
            curriculum_keyboard_bounds = restore_curriculum_keyboard(controller, env)

        # obtain the trained policy for inference
        policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
        if getattr(agent_cfg, "lighthw_ppo_action_curriculum", False):
            ppo_runner.export_action_mapping(export_model_dir, checkpoint_path=resume_path)

        if version.parse(installed_version) >= version.parse("4.0.0"):
            # Use runner-native exporters for rsl-rl >= 4.0.0
            ppo_runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
            ppo_runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")
            policy_nn = None
        else:
            # Fallback for rsl-rl < 4.0.0
            if version.parse(installed_version) >= version.parse("2.3.0"):
                policy_nn = ppo_runner.alg.policy
            else:
                policy_nn = ppo_runner.alg.actor_critic

            if hasattr(policy_nn, "actor_obs_normalizer"):
                normalizer = policy_nn.actor_obs_normalizer
            else:
                normalizer = None

            export_policy_as_onnx(
                policy=policy_nn,
                normalizer=normalizer,
                path=export_model_dir,
                filename="policy.onnx",
            )
            export_policy_as_jit(
                policy=policy_nn,
                normalizer=normalizer,
                path=export_model_dir,
                filename="policy.pt",
            )

    dt = env.unwrapped.step_dt
    # reset environment
    obs, _ = env.reset()
    robot = env.unwrapped.scene["robot"]
    payload_env_ids = torch.tensor([0], dtype=torch.long, device=robot.device)
    payload_body_ids, payload_body_names = _resolve_payload_body_ids(robot, args_cli.payload_body)
    applied_payload_kg = 0.0
    print(
        f"[INFO] Runtime payload body ids: {payload_body_ids}, names: {payload_body_names}. "
        "Keyboard: U/J/P payload, I/K/O/H drag, C/V select tune param, B/N decrease/increase, M print, R reset."
    )
    if payload_state["kg"] > 0.0:
        print(f"[Payload] ON: {payload_state['kg']:.2f} kg on {args_cli.payload_body}")
    if drag_state["force_n"] > 0.0:
        print(
            f"[Drag] ON: {drag_state['force_n']:.1f} N horizontal resistance "
            f"(from load={args_cli.drag_load_kg:.1f} kg, mu={args_cli.drag_mu:.2f})."
        )
    active_motor_profile = None
    filtered_actions = None

    # -------------------------------------------------------------------------
    # Motor monitor: real-time joint position/velocity/torque curves + CSV logging.
    # Keyboard mode forces num_envs=1 above, so env_id=0 is the controlled robot.
    # -------------------------------------------------------------------------
    monitor_env_id = 0
    monitor_history_len = args_cli.monitor_history_len
    monitor_plot_every = max(1, args_cli.monitor_plot_every)  # update matplotlib every N simulation steps
    monitor_csv_flush_every = 100
    motor_csv_file = None
    motor_writer = None

    joint_pos_tensor, joint_vel_tensor, joint_tau_tensor, joint_names, torque_source = _get_motor_tensors(env)
    if joint_pos_tensor is None:
        raise RuntimeError("Could not find joint position tensor. Tried robot.data.joint_pos and env.dof_pos.")
    if joint_vel_tensor is None:
        raise RuntimeError("Could not find joint velocity tensor. Tried robot.data.joint_vel and env.dof_vel.")
    if joint_tau_tensor is None:
        raise RuntimeError(
            "Could not find joint torque tensor. Tried robot.data.applied_torque, computed_torque, "
            "joint_torque, joint_effort, and env.torques."
        )

    num_joints = joint_vel_tensor.shape[1]
    joint_names = list(joint_names)[:num_joints]
    if len(joint_names) < num_joints:
        joint_names += [f"joint_{i}" for i in range(len(joint_names), num_joints)]

    monitor_indices = _select_monitor_joints(
        joint_names, monitor_leg=args_cli.monitor_leg, monitor_joints=args_cli.monitor_joints
    )
    monitor_names = [joint_names[i] for i in monitor_indices]
    contact_sensor, contact_body_ids, contact_labels, contact_body_names = _resolve_contact_monitor(
        env, include_all_bodies=args_cli.monitor_all_body_contacts
    )
    contact_headers = _contact_monitor_header(contact_labels)
    hip_probe = None
    probe_safety = None
    if args_cli.hip_retraction_test:
        from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction

        if type(env.unwrapped.action_manager.get_term("joint_pos")) is not JointPositionAction:
            raise ValueError("Hip probe supports only the standard affine JointPositionAction.")
        if env.unwrapped.action_manager.get_term("joint_pos")._asset is not robot:
            raise ValueError("Hip probe position action must address the monitored robot.")
        if contact_sensor is None or contact_labels[:4] != ["fl", "fr", "rl", "rr"]:
            raise ValueError("Hip probe requires all four wheel contact sensors.")
        hip_probe = HipRetractionProbe(
            env.unwrapped.action_manager, robot, agent_cfg.clip_actions,
            hold_other_actions=args_cli.hip_retraction_hold_others,
        )
        probe_safety = ProbeSafety()
        print(f"[HipProbe] FL action index={hip_probe.action_id}, scale={hip_probe.scale}, "
              f"offset={hip_probe.offset}; native actuators, "
              f"other actions={'held during probe' if hip_probe.hold_other_actions else 'policy-controlled'}.")
    print(f"[INFO] Motor monitor available joints: {num_joints}")
    print(f"[INFO] Torque source: {torque_source}")
    print(f"[INFO] All joint names: {joint_names}")
    print(f"[INFO] Monitoring joint indices: {monitor_indices}")
    print(f"[INFO] Monitoring joint names: {monitor_names}")

    pos_history = [deque(maxlen=monitor_history_len) for _ in monitor_indices]
    vel_history = [deque(maxlen=monitor_history_len) for _ in monitor_indices]
    tau_history = [deque(maxlen=monitor_history_len) for _ in monitor_indices]

    if not args_cli.no_plots:
        plt.ion()
        fig_pos, ax_pos = plt.subplots(num="Joint Position")
        fig_vel, ax_vel = plt.subplots(num="Joint Velocity")
        fig_tau, ax_tau = plt.subplots(num="Joint Torque")
        pos_lines = [ax_pos.plot([], [], label=name)[0] for name in monitor_names]
        vel_lines = [ax_vel.plot([], [], label=name)[0] for name in monitor_names]
        tau_lines = [ax_tau.plot([], [], label=name)[0] for name in monitor_names]

        ax_pos.set_title("Joint Position")
        ax_pos.set_xlabel("step")
        ax_pos.set_ylabel("rad")
        ax_pos.grid(True)
        ax_pos.legend(fontsize=7, ncol=2)

        ax_vel.set_title("Joint Velocity")
        ax_vel.set_xlabel("step")
        ax_vel.set_ylabel("rad/s")
        ax_vel.grid(True)
        ax_vel.legend(fontsize=7, ncol=2)

        ax_tau.set_title("Joint Torque")
        ax_tau.set_xlabel("step")
        ax_tau.set_ylabel("Nm")
        ax_tau.grid(True)
        ax_tau.legend(fontsize=7, ncol=2)

        fig_pos.tight_layout()
        fig_vel.tight_layout()
        fig_tau.tight_layout()
        plt.show(block=False)
    else:
        fig_pos = fig_vel = fig_tau = None
        ax_pos = ax_vel = ax_tau = None
        pos_lines = vel_lines = tau_lines = []

    motor_log_dir = os.path.join(log_dir, "motor_monitor")
    os.makedirs(motor_log_dir, exist_ok=True)
    motor_csv_path = os.path.join(motor_log_dir, "motor_log.csv")
    motor_csv_file = open(motor_csv_path, "w", newline="")
    motor_writer = csv.writer(motor_csv_file)
    physics_substep_file = None
    physics_substep_writer = None
    original_scene_update = None
    root_headers = [
        "episode_step",
        "root_x_env",
        "root_y_env",
        "root_z",
        "root_roll",
        "root_pitch",
        "root_yaw",
        "root_lin_vel_x",
        "root_lin_vel_y",
        "root_lin_vel_z",
        "root_ang_vel_x_b",
        "root_ang_vel_y_b",
        "root_ang_vel_z_b",
        "projected_gravity_x",
        "projected_gravity_y",
        "projected_gravity_z",
        "skill_phase_sin",
        "skill_phase_cos",
    ]
    motor_writer.writerow(
        ["step"]
        + root_headers
        + [f"{name}_pos" for name in monitor_names]
        + [f"{name}_vel" for name in monitor_names]
        + [f"{name}_torque" for name in monitor_names]
        + [f"{name}_computed_torque" for name in monitor_names]
        + [f"{name}_pos_target" for name in monitor_names]
        + [f"{name}_vel_target" for name in monitor_names]
        + [f"{name}_effort_target" for name in monitor_names]
        + ["payload_kg", "drag_force_n", "drag_force_x_w", "drag_force_y_w", "motor_profile"]
        + [f"tune_{name}" for name in LIVE_TUNING_PARAM_NAMES]
        + contact_headers
        + (list(hip_probe.headers) + ["probe_stop_reason"] if hip_probe is not None else [])
    )
    print(f"[INFO] Motor CSV logging to: {motor_csv_path}")

    physics_substep_state = {"physics_step": 0, "control_step": -1, "substep": 0}
    if args_cli.physics_substep_log:
        physics_substep_path = os.path.join(motor_log_dir, "physics_substep_log.csv")
        physics_substep_file = open(physics_substep_path, "w", newline="")
        physics_substep_writer = csv.writer(physics_substep_file)
        physics_substep_writer.writerow(
            ["physics_step", "control_step", "substep", "sim_time_s"]
            + root_headers
            + [f"{name}_pos" for name in monitor_names]
            + [f"{name}_vel" for name in monitor_names]
            + [f"{name}_torque" for name in monitor_names]
            + [f"{name}_computed_torque" for name in monitor_names]
            + [f"{name}_pos_target" for name in monitor_names]
            + [f"{name}_vel_target" for name in monitor_names]
            + contact_headers
            + (list(hip_probe.headers) + ["probe_stop_reason"] if hip_probe is not None else [])
        )

        def _substep_tensor_values(tensor):
            if tensor is None:
                return [math.nan for _ in monitor_indices]
            values = tensor[monitor_env_id].detach().cpu().numpy()
            return [float(values[index]) for index in monitor_indices]

        def _write_physics_substep_row():
            robot_data = robot.data
            joint_pos_values, joint_vel_values, joint_tau_values, _, _ = _get_motor_tensors(env)
            selected_substep_pos = _substep_tensor_values(joint_pos_values)
            selected_substep_vel = _substep_tensor_values(joint_vel_values)
            selected_substep_tau = _substep_tensor_values(joint_tau_values)
            selected_substep_computed_tau = _substep_tensor_values(
                getattr(robot_data, "computed_torque", None)
            )
            selected_substep_pos_target = _substep_tensor_values(
                getattr(robot_data, "joint_pos_target", None)
            )
            selected_substep_vel_target = _substep_tensor_values(
                getattr(robot_data, "joint_vel_target", None)
            )
            substep_contact_values = _contact_monitor_row(
                contact_sensor, contact_body_ids, contact_labels, env_id=monitor_env_id
            )

            root_pos = robot_data.root_pos_w[monitor_env_id]
            root_pos_env = root_pos - env.unwrapped.scene.env_origins[monitor_env_id]
            root_roll, root_pitch, root_yaw = math_utils.euler_xyz_from_quat(
                robot_data.root_quat_w[monitor_env_id].unsqueeze(0)
            )
            root_lin_vel = robot_data.root_lin_vel_w[monitor_env_id]
            root_ang_vel = robot_data.root_ang_vel_b[monitor_env_id]
            projected_gravity = robot_data.projected_gravity_b[monitor_env_id]
            try:
                skill_command = env.unwrapped.command_manager.get_command("skill")[monitor_env_id]
                skill_phase_sin = float(skill_command[-6].item())
                skill_phase_cos = float(skill_command[-5].item())
            except KeyError:
                skill_phase_sin = math.nan
                skill_phase_cos = math.nan

            root_values = [
                int(env.unwrapped.episode_length_buf[monitor_env_id].item()),
                float(root_pos_env[0].item()),
                float(root_pos_env[1].item()),
                float(root_pos[2].item()),
                float(root_roll[0].item()),
                float(root_pitch[0].item()),
                float(root_yaw[0].item()),
                float(root_lin_vel[0].item()),
                float(root_lin_vel[1].item()),
                float(root_lin_vel[2].item()),
                float(root_ang_vel[0].item()),
                float(root_ang_vel[1].item()),
                float(root_ang_vel[2].item()),
                float(projected_gravity[0].item()),
                float(projected_gravity[1].item()),
                float(projected_gravity[2].item()),
                skill_phase_sin,
                skill_phase_cos,
            ]
            physics_step = physics_substep_state["physics_step"]
            if hip_probe is not None:
                probe_safety.update(
                    (physics_step + 1) * env.unwrapped.physics_dt,
                    env.unwrapped.physics_dt,
                    float(projected_gravity[2].item()), float(root_pos[2].item()),
                    contact_sensor.data.net_forces_w[monitor_env_id, contact_body_ids[:4], 2].tolist(),
                )
            physics_substep_writer.writerow(
                [
                    physics_step,
                    physics_substep_state["control_step"],
                    physics_substep_state["substep"],
                    (physics_step + 1) * env.unwrapped.physics_dt,
                ]
                + root_values
                + selected_substep_pos
                + selected_substep_vel
                + selected_substep_tau
                + selected_substep_computed_tau
                + selected_substep_pos_target
                + selected_substep_vel_target
                + substep_contact_values
                + (hip_probe.values + [probe_safety.reason] if hip_probe is not None else [])
            )
            physics_substep_state["physics_step"] += 1
            physics_substep_state["substep"] += 1
            if physics_step % 400 == 0:
                physics_substep_file.flush()

        original_scene_update = env.unwrapped.scene.update

        def _scene_update_with_substep_log(dt):
            original_scene_update(dt)
            _write_physics_substep_row()

        env.unwrapped.scene.update = _scene_update_with_substep_log
        print(
            f"[INFO] Physics-substep CSV logging to: {physics_substep_path} "
            f"at {1.0 / env.unwrapped.physics_dt:.1f} Hz; policy remains at {1.0 / dt:.1f} Hz."
        )
    
    timestep = 0
    recorded_steps = 0
    current_drag_force_w = torch.zeros(3, dtype=torch.float32, device=robot.device)
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        physics_substep_state["control_step"] = timestep
        physics_substep_state["substep"] = 0
        # run everything in inference mode
        with torch.inference_mode():
            target_payload_kg = payload_state["kg"] if payload_state["enabled"] else 0.0
            if args_cli.payload_ramp_time > 0.0:
                max_payload_delta = args_cli.payload_max_kg * dt / args_cli.payload_ramp_time
                if applied_payload_kg < target_payload_kg:
                    applied_payload_kg = min(target_payload_kg, applied_payload_kg + max_payload_delta)
                else:
                    applied_payload_kg = max(target_payload_kg, applied_payload_kg - max_payload_delta)
            else:
                applied_payload_kg = target_payload_kg
            if args_cli.disable_payload_motor_profile:
                motor_profile_name, base_motor_profile = _native_motor_profile(robot)
            else:
                motor_profile_name, base_motor_profile = _payload_motor_profile(
                    applied_payload_kg, args_cli.heavy_payload_threshold_kg
                )
            live_tuning_state["base_profile"] = base_motor_profile
            motor_profile = _apply_live_tuning_overrides(
                base_motor_profile, live_tuning_state["overrides"]
            )
            if args_cli.action_smoothing_override is not None:
                motor_profile["action_smoothing"] = args_cli.action_smoothing_override
                motor_profile["move_action_smoothing"] = args_cli.action_smoothing_override
            live_tuning_state["effective_profile"] = motor_profile
            motor_profile_key = _motor_profile_key(motor_profile_name, motor_profile)
            if motor_profile_key != active_motor_profile or live_tuning_state["dirty"]:
                if args_cli.disable_payload_motor_profile:
                    print(
                        f"[Payload] Motor profile override disabled -> native: "
                        f"leg effort={motor_profile['joint_effort']:.0f}Nm, "
                        f"leg PD=({motor_profile['joint_stiffness']:.0f}, {motor_profile['joint_damping']:.1f}), "
                        f"wheel effort={motor_profile['wheel_effort']:.0f}Nm, "
                        f"smoothing={motor_profile['action_smoothing']:.2f}."
                    )
                else:
                    _apply_payload_motor_profile(robot, motor_profile_name, motor_profile, applied_payload_kg)
                active_motor_profile = motor_profile_key
                live_tuning_state["dirty"] = False

            # agent stepping
            if args_cli.static_check or args_cli.zero_policy_actions:
                action_dim = env.unwrapped.action_manager.total_action_dim
                actions = torch.zeros((env.unwrapped.num_envs, action_dim), device=env.unwrapped.device)
            else:
                actions = policy(obs)
                actions = _scale_actions_for_payload(actions, motor_profile)
            is_keyboard_stand = False
            if args_cli.keyboard and keyboard_command_state is not None:
                command_norm = torch.linalg.norm(keyboard_command_state[:, :3], dim=1)
                stand_env_ids = command_norm < 0.05
                is_keyboard_stand = bool(torch.all(stand_env_ids).item())
            if is_keyboard_stand:
                posture_guard_scale = 1.0
                hipx_guard_scale = 1.0
                action_smoothing = motor_profile["action_smoothing"]
            else:
                posture_guard_scale = motor_profile["move_posture_guard_scale"]
                hipx_guard_scale = motor_profile["move_hipx_guard_scale"]
                action_smoothing = motor_profile["move_action_smoothing"]
            if not (args_cli.static_check or args_cli.zero_policy_actions):
                actions = _apply_posture_guard(
                    actions,
                    robot,
                    joint_names,
                    motor_profile,
                    posture_scale=posture_guard_scale,
                    hipx_scale=hipx_guard_scale,
                )
            if (
                not (args_cli.static_check or args_cli.zero_policy_actions)
                and args_cli.keyboard
                and keyboard_command_state is not None
            ):
                if torch.any(stand_env_ids):
                    stand_hipx_ratio = motor_profile["stand_hipx_action_ratio"] / max(
                        motor_profile["hipx_action_ratio"], 1e-6
                    )
                    actions[stand_env_ids, 0] *= stand_hipx_ratio
                    actions[stand_env_ids, 3] *= stand_hipx_ratio
                    actions[stand_env_ids, 6] *= stand_hipx_ratio
                    actions[stand_env_ids, 9] *= stand_hipx_ratio
                    actions[stand_env_ids, 12:16] *= motor_profile["stand_wheel_action_ratio"]
            if action_smoothing > 0.0:
                if filtered_actions is None:
                    filtered_actions = actions.clone()
                else:
                    filtered_actions = action_smoothing * filtered_actions + (1.0 - action_smoothing) * actions
                actions = filtered_actions
            else:
                filtered_actions = None
            applied_drag_force_n = drag_state["force_n"] if drag_state["enabled"] else 0.0
            current_drag_force_w = _compute_drag_force_w(
                robot, applied_drag_force_n, command_state=keyboard_command_state
            )
            _set_payload_force(
                robot,
                payload_body_ids,
                applied_payload_kg,
                payload_env_ids,
                drag_force_w=current_drag_force_w.unsqueeze(0),
            )

            # 仅诊断模式替换 FL hip 单列动作；奖励、动作项、限位和执行器全部保持原样。
            if hip_probe is not None:
                previous_phase = hip_probe.phase
                try:
                    actions = hip_probe.apply(actions, timestep * dt)
                except ValueError as error:
                    probe_safety.reason = str(error)
                    print(f"[HipProbe] STOP before step: {probe_safety.reason}")
                    break
                if hip_probe.phase != previous_phase:
                    print(f"[HipProbe] t={timestep * dt:.2f}s phase={hip_probe.phase}")

            # env stepping
            obs, _, step_dones, _ = env.step(actions)
            recorded_steps += 1
            if hip_probe is not None and bool(step_dones[0].item()):
                probe_safety.reason = probe_safety.reason or "episode_reset"

        # -----------------------------------------------------------------
        # Motor monitor update. We read the controlled robot (env 0), append
        # values to the rolling buffers, update plots, and save every step.
        # -----------------------------------------------------------------
        joint_pos_tensor, joint_vel_tensor, joint_tau_tensor, _, _ = _get_motor_tensors(env)
        joint_pos = joint_pos_tensor[monitor_env_id].detach().cpu().numpy()
        joint_vel = joint_vel_tensor[monitor_env_id].detach().cpu().numpy()
        joint_tau = joint_tau_tensor[monitor_env_id].detach().cpu().numpy()

        selected_pos = [float(joint_pos[idx]) for idx in monitor_indices]
        selected_vel = [float(joint_vel[idx]) for idx in monitor_indices]
        selected_tau = [float(joint_tau[idx]) for idx in monitor_indices]

        def _selected_tensor_values_or_nan(tensor):
            if tensor is None:
                return [math.nan for _ in monitor_indices]
            values = tensor[monitor_env_id].detach().cpu().numpy()
            return [float(values[idx]) for idx in monitor_indices]

        robot_data = robot.data
        selected_computed_tau = _selected_tensor_values_or_nan(getattr(robot_data, "computed_torque", None))
        selected_pos_target = _selected_tensor_values_or_nan(getattr(robot_data, "joint_pos_target", None))
        selected_vel_target = _selected_tensor_values_or_nan(getattr(robot_data, "joint_vel_target", None))
        selected_effort_target = _selected_tensor_values_or_nan(getattr(robot_data, "joint_effort_target", None))
        contact_values = _contact_monitor_row(contact_sensor, contact_body_ids, contact_labels, env_id=monitor_env_id)

        root_pos = robot_data.root_pos_w[monitor_env_id]
        root_pos_env = root_pos - env.unwrapped.scene.env_origins[monitor_env_id]
        root_roll, root_pitch, root_yaw = math_utils.euler_xyz_from_quat(
            robot_data.root_quat_w[monitor_env_id].unsqueeze(0)
        )
        root_lin_vel = robot_data.root_lin_vel_w[monitor_env_id]
        root_ang_vel = robot_data.root_ang_vel_b[monitor_env_id]
        projected_gravity = robot_data.projected_gravity_b[monitor_env_id]
        try:
            skill_command = env.unwrapped.command_manager.get_command("skill")[monitor_env_id]
            skill_phase_sin = float(skill_command[-6].item())
            skill_phase_cos = float(skill_command[-5].item())
        except KeyError:
            skill_phase_sin = math.nan
            skill_phase_cos = math.nan
        root_values = [
            int(env.unwrapped.episode_length_buf[monitor_env_id].item()),
            float(root_pos_env[0].item()),
            float(root_pos_env[1].item()),
            float(root_pos[2].item()),
            float(root_roll[0].item()),
            float(root_pitch[0].item()),
            float(root_yaw[0].item()),
            float(root_lin_vel[0].item()),
            float(root_lin_vel[1].item()),
            float(root_lin_vel[2].item()),
            float(root_ang_vel[0].item()),
            float(root_ang_vel[1].item()),
            float(root_ang_vel[2].item()),
            float(projected_gravity[0].item()),
            float(projected_gravity[1].item()),
            float(projected_gravity[2].item()),
            skill_phase_sin,
            skill_phase_cos,
        ]

        for k, (pos_value, vel_value, tau_value) in enumerate(zip(selected_pos, selected_vel, selected_tau)):
            pos_history[k].append(pos_value)
            vel_history[k].append(vel_value)
            tau_history[k].append(tau_value)

        if motor_writer is not None:
            motor_writer.writerow(
                [timestep]
                + root_values
                + selected_pos
                + selected_vel
                + selected_tau
                + selected_computed_tau
                + selected_pos_target
                + selected_vel_target
                + selected_effort_target
                + [
                    float(applied_payload_kg),
                    float(torch.linalg.norm(current_drag_force_w[:2]).item()),
                    float(current_drag_force_w[0].item()),
                    float(current_drag_force_w[1].item()),
                    motor_profile_name,
                ]
                + [float(motor_profile.get(name, math.nan)) for name in LIVE_TUNING_PARAM_NAMES]
                + contact_values
                + (hip_probe.values + [probe_safety.reason] if hip_probe is not None else [])
            )
            if timestep % monitor_csv_flush_every == 0:
                motor_csv_file.flush()

        if not args_cli.no_plots and timestep % monitor_plot_every == 0:
            x = range(len(vel_history[0]))
            for k in range(len(monitor_indices)):
                pos_lines[k].set_data(x, list(pos_history[k]))
                vel_lines[k].set_data(x, list(vel_history[k]))
                tau_lines[k].set_data(x, list(tau_history[k]))

            ax_pos.relim()
            ax_pos.autoscale_view()
            ax_vel.relim()
            ax_vel.autoscale_view()
            ax_tau.relim()
            ax_tau.autoscale_view()
            fig_pos.canvas.draw_idle()
            fig_vel.canvas.draw_idle()
            fig_tau.canvas.draw_idle()
            plt.pause(0.001)

        if (
            True
            and False
            and foot_ids is not None
            and phase_offsets is not None
            and cycle_time is not None
            and gait_span is not None
            and gait_psi is not None
            and gait_delta is not None
            and x_offset is not None
            and stance_span is not None
            and cmd_threshold is not None
            and stand_ref_z_offset is not None
            and cmd_hist is not None
            and act_hist is not None
        ):
            local_foot_ids = foot_ids
            robot = env.unwrapped.scene["robot"]
            root_pos = robot.data.root_pos_w[0]
            root_quat = robot.data.root_quat_w[0].unsqueeze(0)

            # Initialize base-fixed stand reference once from current posture.
            if stand_ref_body is None:
                rel_init = robot.data.body_pos_w[0, local_foot_ids, :] - root_pos.unsqueeze(0)
                stand_ref_body = math_utils.quat_apply_inverse(root_quat.expand(len(local_foot_ids), -1), rel_init)
                stand_ref_body[:, 2] += stand_ref_z_offset

            elapsed_t = float(env.unwrapped.common_step_counter) * dt
            phase_s = torch.remainder((2.0 * elapsed_t / max(cycle_time, 1e-6)) + phase_offsets, 2.0)
            cmd_local = _mujoco_phase_traj_body(
                phase_s=phase_s,
                gait_span=gait_span,
                gait_psi=gait_psi,
                gait_delta=gait_delta,
                x_offset=x_offset,
                stance_span=stance_span,
            )
            ref_body = stand_ref_body + cmd_local
            ref_world = root_pos.unsqueeze(0) + math_utils.quat_apply(root_quat.expand(len(local_foot_ids), -1), ref_body)

            actual_world = robot.data.body_pos_w[0, local_foot_ids, :]
            for i in range(4):
                cmd_hist[i].append(ref_world[i].detach().cpu().tolist())
                act_hist[i].append(actual_world[i].detach().cpu().tolist())

            if args_cli.keyboard and keyboard_command_state is not None:
                cmd_vec = keyboard_command_state[0, :3]
            else:
                cmd_vec = env.unwrapped.command_manager.get_command("base_velocity")[0, :3]

            cmd_norm = torch.linalg.norm(cmd_vec).item()
            gate_on = cmd_norm > cmd_threshold

            if args_cli.keyboard:
                ref_gate_on = cmd_norm > 0.1
            else:
                ref_gate_on = gate_on

            draw_interface.clear_lines()
            starts = []
            ends = []
            colors = []
            widths = []

            ref_alpha = 0.95 if gate_on else 0.35
            act_alpha = 0.35 if gate_on else 0.20

            if not phase_vis_z_printed:
                print(
                    "[INFO] phase_foot_trajectory_exp z check: "
                    f"ref_z_mean={ref_world[:, 2].mean().item():.4f}, "
                    f"act_z_mean={actual_world[:, 2].mean().item():.4f}, "
                    f"stand_ref_z_offset={stand_ref_z_offset:.4f}"
                )
                phase_vis_z_printed = True

            for i in range(4):
                act_pts = list(act_hist[i])
                for j in range(1, len(act_pts)):
                    starts.append(act_pts[j - 1])
                    ends.append(act_pts[j])
                    colors.append([0.0, 0.0, 0.0, act_alpha])
                    widths.append(1.5)

                cmd_pts = list(cmd_hist[i])
                if VIS_REF_ENABLE and ref_gate_on:
                    for j in range(1, len(cmd_pts)):
                        starts.append(cmd_pts[j - 1])
                        ends.append(cmd_pts[j])
                        color = color_palette[i].copy()
                        color[3] = ref_alpha
                        colors.append(color)
                        widths.append(2.8)

            if starts:
                draw_interface.draw_lines(starts, ends, colors, widths)
        if hip_probe is not None and (probe_safety.reason or (timestep + 1) * dt >= hip_probe.duration_s):
            print(f"[HipProbe] Finished after {(timestep + 1) * dt:.2f}s: "
                  f"{probe_safety.reason or 'completed'}")
            break

        if args_cli.video:
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        if args_cli.keyboard:
            camera_follow(env)

        timestep += 1
        if args_cli.max_steps > 0 and timestep >= args_cli.max_steps:
            print(f"[INFO] Reached --max_steps={args_cli.max_steps}; exiting playback loop.")
            break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    if hip_probe is not None:
        # 每次测试都记录真实映射及结束原因；终止步的普通CSV可能已重置，子步CSV保留重置前状态。
        with open(os.path.join(motor_log_dir, "hip_retraction_test.json"), "w", encoding="utf-8") as file:
            json.dump({
                "checkpoint": resume_path, "task": args_cli.task, "seed": env_cfg.seed,
                "joint": hip_probe.joint_name, "action_index": hip_probe.action_id,
                "scale": hip_probe.scale, "offset": hip_probe.offset,
                "action_limits": [hip_probe.lower, hip_probe.upper],
                "wrapper_clip": hip_probe.wrapper_clip, "control_dt": dt,
                "physics_dt": env.unwrapped.physics_dt,
                "control_steps": recorded_steps, "duration_s": recorded_steps * dt,
                "status": probe_safety.reason or (
                    "completed" if recorded_steps * dt >= hip_probe.duration_s else "interrupted"
                ),
                "last_phase": hip_probe.phase,
                "hold_other_actions": hip_probe.hold_other_actions,
                "held_raw_actions": None if hip_probe.held_actions is None else hip_probe.held_actions.cpu().tolist(),
                "other_action_indices": hip_probe.other_action_ids,
                "note": "Diagnostic only, not a new trained policy. When hold_other_actions=true, "
                        "other raw actions are captured at 3s, held through 13s, and blended back by 15s; "
                        "actual joints are never locked. On episode_reset use physics rows for pre-reset state.",
            }, file, indent=2, ensure_ascii=False)

    # close logger and simulator
    if motor_csv_file is not None:
        motor_csv_file.flush()
        motor_csv_file.close()
    if physics_substep_file is not None:
        physics_substep_file.flush()
        physics_substep_file.close()
    if original_scene_update is not None:
        env.unwrapped.scene.update = original_scene_update
    if not args_cli.no_plots:
        plt.ioff()
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
