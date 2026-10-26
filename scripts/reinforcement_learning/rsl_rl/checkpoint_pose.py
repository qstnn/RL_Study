"""Restore a saved robot reference pose without importing Isaac Sim or YAML objects."""

from __future__ import annotations

import math
from pathlib import Path

import yaml


def restore_checkpoint_pose(env_cfg, checkpoint_path: str) -> None:
    """恢复默认角、出生位姿和站高目标，使旧策略的相对关节观测及动作偏置保持一致。

    仅在显式传入 --use_checkpoint_pose 时调用。不恢复执行器、动作缩放、地形或
    整个环境；这些仍由选定任务负责。BaseLoader 将 YAML 作为数据读取，不构造对象。
    """
    source = Path(checkpoint_path).resolve().parent / "params" / "env.yaml"
    if not source.is_file():
        raise FileNotFoundError(f"--use_checkpoint_pose requires the checkpoint's saved config: {source}")
    saved = yaml.load(source.read_text(), Loader=yaml.BaseLoader)
    try:
        initial = saved["scene"]["robot"]["init_state"]

        def vector(key: str, size: int) -> tuple[float, ...]:
            value = tuple(float(x) for x in initial[key])
            if len(value) != size or not all(math.isfinite(x) for x in value):
                raise ValueError(f"invalid init_state.{key}")
            return value

        position = vector("pos", 3)
        rotation = vector("rot", 4)
        if not math.isclose(sum(x * x for x in rotation), 1.0, abs_tol=1e-5):
            raise ValueError("init_state.rot must be a unit quaternion")
        joint_positions = {name: float(value) for name, value in initial["joint_pos"].items()}
        if not joint_positions or not all(math.isfinite(x) for x in joint_positions.values()):
            raise ValueError("invalid init_state.joint_pos")
        height = None
        reward = saved.get("rewards", {}).get("base_height_l2")
        if isinstance(reward, dict) and "target_height" in reward.get("params", {}):
            height = float(reward["params"]["target_height"])
            if not math.isfinite(height):
                raise ValueError("invalid base_height_l2 target_height")
    except (KeyError, TypeError, AttributeError, ValueError) as exc:
        raise ValueError(f"Cannot restore reference pose from {source}: {exc}") from exc

    # Validate all saved data before changing the environment configuration.
    env_cfg.scene.robot.init_state.pos = position
    env_cfg.scene.robot.init_state.rot = rotation
    env_cfg.scene.robot.init_state.joint_pos = joint_positions
    height_reward = getattr(env_cfg.rewards, "base_height_l2", None)
    if height is not None and height_reward is not None:
        height_reward.params["target_height"] = height
    print(f"[INFO] Restored checkpoint reference pose from: {source}")
    print(f"[INFO] Checkpoint spawn z={position[2]:.3f} m, saved base-height target={height}")
