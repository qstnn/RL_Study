# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import json
import sys
import os

from isaaclab.app import AppLauncher

# local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cli_args

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--transfer_checkpoint",
    type=str,
    default=None,
    help="Initialize compatible actor weights from a checkpoint without resuming its optimizer or iteration.",
)
parser.add_argument(
    "--warm_start_checkpoint",
    type=str,
    default=None,
    help="完整策略热启动；通常重置优化器与轮次，LightHW SpeedBridge 专用入口保留 Adam 并重置新课程计数。",
)
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# 在启动仿真前拒绝冲突参数，避免误把塑形热启动当成普通断点续训或仅迁移 Actor。
if args_cli.warm_start_checkpoint is not None:
    if args_cli.resume or args_cli.transfer_checkpoint is not None:
        parser.error("--warm_start_checkpoint 不能与 --resume 或 --transfer_checkpoint 同时使用。")
    args_cli.warm_start_checkpoint = os.path.realpath(os.path.expanduser(args_cli.warm_start_checkpoint))
    if not os.path.isfile(args_cli.warm_start_checkpoint):
        parser.error(f"Warm-start checkpoint does not exist: {args_cli.warm_start_checkpoint}")

if args_cli.task == "Rough-Deeprobotics-LightHW-Blind-v0" and (
    args_cli.resume
    or args_cli.load_run is not None
    or args_cli.checkpoint is not None
    or getattr(args_cli, "transfer_checkpoint", None) is not None
    or getattr(args_cli, "warm_start_checkpoint", None) is not None
):
    parser.error(
        "Rough-Deeprobotics-LightHW-Blind-v0 requires a fresh start; "
        "do not pass resume, load_run, checkpoint, transfer_checkpoint, or warm_start_checkpoint."
    )

if args_cli.task in (
    "Flat-Deeprobotics-LightHW-StraightStablePhysics-Continue-v0",
    "Flat-Deeprobotics-LightHW-StraightStablePhysics-Precision-v0",
    "Flat-Deeprobotics-LightHW-StraightStablePhysics-Contact-v0",
    "Flat-Deeprobotics-LightHW-StraightStablePhysics-StandGate-v0",
    "Flat-Deeprobotics-LightHW-StraightStablePhysics-StandHold-v0",
    "Flat-Deeprobotics-LightHW-StraightStablePhysics-StandHoldPrecision-v0",
) and args_cli.warm_start_checkpoint is None:
    parser.error(
        f"{args_cli.task} requires --warm_start_checkpoint pointing to a "
        "validated route-09 checkpoint."
    )


# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# suppress noisy omni.usd warnings (e.g. unresolved visual prim references)
import carb
carb.logging.acquire_logging().set_level_threshold_for_source(
    "omni.usd", carb.logging.LogSettingBehavior.OVERRIDE, carb.logging.LEVEL_ERROR
)

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
import torch
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner

from episode_metrics import CompleteEpisodeMetricsRunner
from ppo_action_curriculum_runner import LightHWPPOActionCurriculumRunner
from ppo_speed_bridge_runner import LightHWPPOSpeedBridgeRunner
from ppo_grouped_noise_runner import LightHWGroupedNoiseRunner
from reference_runner import LightHWReferenceRunner
from roll_stop_runner import LightHWRollStopRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import rl_training.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def transfer_actor_weights(runner: OnPolicyRunner, checkpoint_path: str):
    """Transfer compatible actor weights while preserving new observations and exploration settings."""
    checkpoint_path = os.path.abspath(checkpoint_path)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Transfer checkpoint does not exist: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=runner.device, weights_only=False)
    source_state = checkpoint["model_state_dict"]
    target_state = runner.alg.policy.state_dict()

    transferred = []
    for key, source_tensor in source_state.items():
        if not key.startswith("actor."):
            continue
        if key not in target_state:
            continue

        target_tensor = target_state[key]
        if source_tensor.shape == target_tensor.shape:
            target_state[key] = source_tensor.to(device=target_tensor.device, dtype=target_tensor.dtype)
            transferred.append(f"{key}: exact {tuple(source_tensor.shape)}")
        elif source_tensor.ndim == target_tensor.ndim:
            copied_tensor = target_tensor.clone()
            if (
                source_tensor.ndim == 2
                and source_tensor.shape[0] == target_tensor.shape[0]
                and source_tensor.shape[1] < target_tensor.shape[1]
            ):
                # Appended observations must not perturb the transferred policy
                # before the new fine-tuning run has learned how to use them.
                copied_tensor.zero_()
            copy_slices = tuple(slice(0, min(src, dst)) for src, dst in zip(source_tensor.shape, target_tensor.shape))
            copied_tensor[copy_slices] = source_tensor[copy_slices].to(
                device=target_tensor.device, dtype=target_tensor.dtype
            )
            target_state[key] = copied_tensor
            transferred.append(f"{key}: overlap {tuple(source_tensor.shape)} -> {tuple(target_tensor.shape)}")

    runner.alg.policy.load_state_dict(target_state)
    print(f"[INFO]: Transferred {len(transferred)} actor tensors from: {checkpoint_path}")
    for item in transferred:
        print(f"  - {item}")


def warm_start_training(
    runner: OnPolicyRunner,
    checkpoint_path: str,
    learning_rate: float,
    reset_noise_std: float | list[float] | tuple[float, ...] | None = None,
) -> dict:
    """受检热启动；route 12保留Adam，route 13仅重设分组噪声及其Adam。"""
    checkpoint_path = os.path.realpath(os.path.expanduser(checkpoint_path))
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Warm-start checkpoint does not exist: {checkpoint_path}")
    if getattr(runner, "_lighthw_grouped_noise", False):
        if reset_noise_std is not None:
            raise ValueError("Grouped-noise transfer owns its conditional exploration profile.")
        runner.load_route12_checkpoint(checkpoint_path)
        return {
            **runner._grouped_noise_transfer_source,
            "initial_iteration": 0, "optimizer_reset": False,
            "empirical_normalization": runner.empirical_normalization,
            "exploration_profile": runner.alg.policy.noise_profile(),
        }
    # route 12 只细化课程，完整保留已有 Adam/adaptive LR；必须在通用热启动
    # 清空优化器之前分流，不能先 load 后再误用下方塑形重置逻辑。
    if getattr(runner, "_lighthw_speed_bridge", False):
        if reset_noise_std is not None:
            raise ValueError("LightHW speed-bridge transfer must preserve the learned action noise.")
        runner.load_route11_checkpoint(checkpoint_path)
        return {
            **runner._route11_transfer_source,
            "initial_iteration": 0,
            "optimizer_reset": False,
            "empirical_normalization": runner.empirical_normalization,
            "reset_noise_std": None,
        }
    checkpoint = torch.load(checkpoint_path, map_location=runner.device, weights_only=False)

    def validate_state(name: str, source_state: dict, target_state: dict):
        # 热启动要求完整兼容；禁止仅拷贝重叠维度而悄悄改变观测或动作接口。
        missing = sorted(set(target_state) - set(source_state))
        unexpected = sorted(set(source_state) - set(target_state))
        mismatched = [
            f"{key}: {tuple(source_state[key].shape)} -> {tuple(target_state[key].shape)}"
            for key in source_state.keys() & target_state.keys()
            if source_state[key].shape != target_state[key].shape
        ]
        if missing or unexpected or mismatched:
            raise ValueError(
                f"Warm-start {name} is incompatible: missing={missing}, unexpected={unexpected}, "
                f"shape_mismatches={mismatched}"
            )
        if any(not torch.isfinite(tensor).all().item() for tensor in source_state.values()):
            raise ValueError(f"Warm-start {name} contains NaN or Inf.")

    validate_state("model", checkpoint["model_state_dict"], runner.alg.policy.state_dict())
    normalization_keys = ("obs_norm_state_dict", "privileged_obs_norm_state_dict")
    source_has_normalization = any(key in checkpoint for key in normalization_keys)
    if source_has_normalization != runner.empirical_normalization:
        raise ValueError("Warm-start requires the same empirical_normalization setting as the source checkpoint.")
    if runner.empirical_normalization:
        # 均值、方差和累计样本数必须一起保留，否则同一权重会产生不同动作。
        validate_state("actor normalizer", checkpoint[normalization_keys[0]], runner.obs_normalizer.state_dict())
        validate_state(
            "critic normalizer", checkpoint[normalization_keys[1]], runner.privileged_obs_normalizer.state_dict()
        )

    source_iteration = int(checkpoint["iter"])
    # route 11 是唯一允许从 route 10 迁移的课程：网络完整继承，但动作语义
    # 必须先经过专用来源检查。普通 load 仍只接受相同课程的 checkpoint。
    if getattr(runner, "_lighthw_forward_speed", False):
        if reset_noise_std is not None:
            raise ValueError("LightHW forward-speed transfer must preserve the learned action noise.")
        runner.load_route10_checkpoint(checkpoint_path)
    else:
        # 官方加载入口同时恢复 Actor/Critic/std 和 normalizer；不加载旧 Adam 动量。
        runner.load(checkpoint_path, load_optimizer=False)
    runner.current_learning_iteration = 0
    runner.alg.optimizer.state.clear()
    runner.alg.learning_rate = float(learning_rate)
    for param_group in runner.alg.optimizer.param_groups:
        param_group["lr"] = runner.alg.learning_rate
    if runner.alg.rnd_optimizer is not None:
        runner.alg.rnd_optimizer.state.clear()

    if reset_noise_std is not None:
        noise = torch.as_tensor(reset_noise_std, device=runner.device, dtype=runner.alg.policy.log_std.dtype)
        if noise.ndim == 0:
            noise = noise.expand_as(runner.alg.policy.log_std)
        if tuple(noise.shape) != tuple(runner.alg.policy.log_std.shape):
            raise ValueError(
                "warm_start_noise_std must be a scalar or match the action dimension: "
                f"got {tuple(noise.shape)}, expected {tuple(runner.alg.policy.log_std.shape)}"
            )
        if torch.any(noise <= 0.0) or not torch.isfinite(noise).all():
            raise ValueError("warm_start_noise_std must contain finite positive values")
        with torch.no_grad():
            runner.alg.policy.log_std.copy_(torch.log(noise))
        print(
            "[INFO]: Warm-start exploration reset; "
            f"noise_std mean={noise.mean().item():g}, min={noise.min().item():g}, max={noise.max().item():g}"
        )

    print(
        f"[INFO]: Warm-started full policy and critic from: {checkpoint_path}\n"
        f"[INFO]: Source iteration={source_iteration}; new iteration=0; "
        f"optimizer reset; learning_rate={runner.alg.learning_rate:g}"
    )
    return {
        "checkpoint": checkpoint_path,
        "source_iteration": source_iteration,
        "initial_iteration": 0,
        "optimizer_reset": True,
        "initial_learning_rate": runner.alg.learning_rate,
        "empirical_normalization": runner.empirical_normalization,
        "reset_noise_std": None if reset_noise_std is None else [float(value) for value in noise.detach().cpu()],
    }


def validate_lighthw_curriculum_training(agent_cfg, args):
    """区分旧课程与前进高速迁移，防止错误入口悄悄重练或只迁移 Actor。"""
    if getattr(agent_cfg, "lighthw_reference", False):
        if (args.warm_start_checkpoint or args.transfer_checkpoint or args.distributed
                or agent_cfg.algorithm.class_name != "PPO"):
            raise ValueError("Reference task uses only fresh PPO or strict same-task resume; no old-policy transfer")
        role = agent_cfg.lighthw_reference_run_role
        if role not in ("formal", "validation", "continuation"):
            raise ValueError("Reference run role must be formal, validation or continuation")
        if role == "continuation" and not agent_cfg.resume:
            raise ValueError("Formal continuation requires explicit full --resume")
        if role == "formal" and agent_cfg.resume:
            raise ValueError("The first formal reference run must start fresh, never from a startup checkpoint")
        if not agent_cfg.resume and (args.checkpoint is not None or args.load_run is not None):
            raise ValueError("Reference checkpoint/load_run options require explicit same-task --resume")
        return
    if getattr(agent_cfg, "lighthw_roll_stop", False):
        if (args.warm_start_checkpoint or args.transfer_checkpoint or args.distributed
                or agent_cfg.algorithm.class_name != "PPO"):
            raise ValueError("RollStop uses PPO from scratch or same-route resume; no transfer/distillation/distributed")
        if not agent_cfg.resume and (args.checkpoint is not None or args.load_run is not None):
            raise ValueError("RollStop checkpoint/load_run requires --resume from this same new task")
        return
    action_curriculum = getattr(agent_cfg, "lighthw_ppo_action_curriculum", False)
    forward_speed = getattr(agent_cfg, "lighthw_forward_speed", False)
    speed_bridge = getattr(agent_cfg, "lighthw_speed_bridge", False)
    grouped_noise = getattr(agent_cfg, "lighthw_grouped_noise", False)
    if grouped_noise and (not forward_speed or speed_bridge):
        raise ValueError("Grouped-noise task requires forward-speed monitoring and its own transfer entry.")
    if speed_bridge and not forward_speed:
        raise ValueError("LightHW speed bridge requires the forward-speed monitoring/action interface.")
    if forward_speed and not action_curriculum:
        raise ValueError("LightHW forward-speed training requires the action curriculum runner.")
    if not action_curriculum:
        return
    if (
        args.transfer_checkpoint is not None
        or agent_cfg.algorithm.class_name != "PPO"
        or args.distributed
    ):
        raise ValueError(
            "LightHW action curricula require single-process PPO; "
            "actor-only transfer, distillation and distributed training are not supported."
        )
    if not agent_cfg.resume and (args.checkpoint is not None or args.load_run is not None):
        raise ValueError("LightHW curriculum checkpoint/load_run options require --resume.")
    if forward_speed:
        if not agent_cfg.resume and args.warm_start_checkpoint is None:
            if grouped_noise:
                raise ValueError(
                    "LightHW grouped-noise training requires --warm_start_checkpoint from route12 at 1.0 m/s "
                    "or --resume from the same grouped-noise task; a fresh random policy is not allowed."
                )
            if speed_bridge:
                raise ValueError(
                    "LightHW speed-bridge training requires --warm_start_checkpoint from route 11 at 0.8 m/s "
                    "or --resume from the same speed-bridge task; a fresh random policy is not allowed."
                )
            raise ValueError(
                "LightHW forward-speed training requires --warm_start_checkpoint from route 10 stage 2 "
                "or --resume from the same forward-speed task; a fresh random policy is not allowed."
            )
        if agent_cfg.resume and args.warm_start_checkpoint is not None:
            raise ValueError("LightHW forward-speed --resume cannot be combined with --warm_start_checkpoint.")
    elif args.warm_start_checkpoint is not None:
        raise ValueError(
            "LightHW PPO action curriculum uses PPO from scratch or same-task --resume; "
            "warm start is not supported for the original route 10 task."
        )


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    validate_lighthw_curriculum_training(agent_cfg, args_cli)
    if getattr(agent_cfg, "fresh_start_only", False):
        checkpoint_options = (
            agent_cfg.resume,
            args_cli.checkpoint is not None,
            args_cli.load_run is not None,
            getattr(args_cli, "transfer_checkpoint", None) is not None,
            getattr(args_cli, "warm_start_checkpoint", None) is not None,
            agent_cfg.algorithm.class_name == "Distillation",
        )
        if any(checkpoint_options):
            raise ValueError(
                f"{args_cli.task} requires a fresh start; "
                "do not pass resume, load_run, checkpoint, transfer_checkpoint, or warm_start_checkpoint."
            )
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )
    if args_cli.warm_start_checkpoint is not None and (
        agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation"
    ):
        raise ValueError("--warm_start_checkpoint 仅用于完整 PPO 策略热启动，不能同时恢复旧训练或执行 Distillation。")

    # Current Isaac Lab/RSL-RL configs already use the new actor/critic format.

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    if getattr(agent_cfg, "lighthw_reference", False):
        from reference_config_audit import audit_reference_configuration
        reference_audit = audit_reference_configuration(env_cfg, agent_cfg)
        os.makedirs(log_dir, exist_ok=False)
        with open(os.path.join(log_dir, "reference_configuration_check.json"), "w", encoding="utf-8") as stream:
            json.dump(reference_audit, stream, ensure_ascii=False, indent=2)
        if agent_cfg.lighthw_reference_run_role == "continuation":
            from reference_continuation import validate_source, plan, audit_configuration, SOURCE
            validate_source()
            plan(agent_cfg.max_iterations,short_check=agent_cfg.lighthw_reference_short_check)
            if (env_cfg.scene.num_envs != 1024 or agent_cfg.seed != 42 or agent_cfg.save_interval != 100
                    or agent_cfg.load_run != SOURCE.parent.name or agent_cfg.load_checkpoint != SOURCE.name):
                raise ValueError("Continuation requires pinned original formal source, 1024 envs, seed42, save100")
            continuation_audit = audit_configuration(env_cfg, agent_cfg)
            with open(os.path.join(log_dir, "continuation_configuration_check.json"), "w") as stream:
                json.dump(continuation_audit, stream, indent=2)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # convert config to dict and create runner
    train_cfg = agent_cfg.to_dict()
    # 新任务可选择完整的条件指标汇总；只影响日志，不改变 PPO 采样、损失或旧任务行为。
    if getattr(agent_cfg, "lighthw_reference", False):
        runner_type = LightHWReferenceRunner
    elif getattr(agent_cfg, "lighthw_roll_stop", False):
        runner_type = LightHWRollStopRunner
    elif getattr(agent_cfg, "lighthw_grouped_noise", False):
        runner_type = LightHWGroupedNoiseRunner
    elif getattr(agent_cfg, "lighthw_speed_bridge", False):
        runner_type = LightHWPPOSpeedBridgeRunner
    elif getattr(agent_cfg, "lighthw_ppo_action_curriculum", False):
        runner_type = LightHWPPOActionCurriculumRunner
    else:
        runner_type = (
            CompleteEpisodeMetricsRunner
            if getattr(agent_cfg, "log_all_episode_metric_keys", False)
            else OnPolicyRunner
        )
    runner = runner_type(env, train_cfg, log_dir=log_dir, device=agent_cfg.device)
    
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    warm_start_metadata = None
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)
    elif args_cli.warm_start_checkpoint is not None:
        warm_start_metadata = warm_start_training(
            runner,
            args_cli.warm_start_checkpoint,
            agent_cfg.algorithm.learning_rate,
            reset_noise_std=getattr(agent_cfg, "warm_start_noise_std", None),
        )
    elif args_cli.transfer_checkpoint is not None:
        transfer_actor_weights(runner, args_cli.transfer_checkpoint)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    with open(os.path.join(log_dir, "run_metadata.json"), "w", encoding="utf-8") as metadata_file:
        json.dump(
            {
                "task": args_cli.task,
                "max_iterations": agent_cfg.max_iterations,
                "transfer_checkpoint": args_cli.transfer_checkpoint,
                "warm_start": warm_start_metadata,
                "run_name": agent_cfg.run_name,
                "warm_start_noise_std": getattr(agent_cfg, "warm_start_noise_std", None),
            },
            metadata_file,
            ensure_ascii=False,
            indent=2,
        )
    
    reference_run_record = None
    if getattr(agent_cfg, "lighthw_reference", False):
        from reference_launch import write_reference_run
        reference_run_record = write_reference_run(log_dir, log_root_path, agent_cfg, env_cfg,
                                                   task=args_cli.task, status="started", runner=runner)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    if reference_run_record is not None:
        write_reference_run(log_dir, log_root_path, agent_cfg, env_cfg, task=args_cli.task, status="completed",runner=runner)

    # close the simulator
    env.close()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
