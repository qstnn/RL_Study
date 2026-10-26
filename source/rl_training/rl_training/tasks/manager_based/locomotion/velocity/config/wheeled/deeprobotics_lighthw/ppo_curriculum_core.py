"""Pure-Python promotion rules for the isolated LightHW PPO curriculum.

The schedule is performance gated, not a timer that opens the legs regardless
of behavior.  This module has no Isaac Sim dependency so failure/persistence
cases can be checked without launching a simulator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class LegCurriculumSchedule:
    hip_limits: tuple[float, ...] = (0.01, 0.02, 0.04, 0.08, 0.08, 0.08)
    leg_limits: tuple[float, ...] = (0.025, 0.05, 0.10, 0.20, 0.20, 0.20)
    max_speeds: tuple[float, ...] = (0.20, 0.20, 0.20, 0.20, 0.40, 0.60)
    min_stage_updates: int = 300
    evaluation_interval: int = 100
    required_passes: int = 3
    stage: int = 0
    stage_updates: int = 0
    total_updates: int = 0
    consecutive_passes: int = 0
    window_updates: int = 0

    def __post_init__(self):
        if not self.hip_limits or not (len(self.hip_limits) == len(self.leg_limits) == len(self.max_speeds)):
            raise ValueError("All curriculum stage arrays must have the same nonzero length")
        for values in (self.hip_limits, self.leg_limits, self.max_speeds):
            if any(not math.isfinite(x) or x <= 0 for x in values) or any(a > b for a, b in zip(values, values[1:])):
                raise ValueError("Curriculum limits must be finite, positive and nondecreasing")
        if min(self.min_stage_updates, self.evaluation_interval, self.required_passes) < 1:
            raise ValueError("Curriculum update intervals must be positive")

    def update(self, passed: bool | None = None) -> bool:
        """Advance exactly once after a PPO update; return whether stage changed.

        ``passed`` must be supplied at evaluation boundaries, otherwise None.
        A failing window clears the consecutive-pass count.  The final stage
        remains bounded and continues recording quality, without overflow.
        """
        due = self.window_updates + 1 >= self.evaluation_interval
        if not due and passed is not None:
            raise ValueError("Quality assessment supplied before evaluation boundary")
        if due and passed is None:
            raise ValueError("Quality assessment required at evaluation boundary")
        self.total_updates += 1
        self.stage_updates += 1
        self.window_updates += 1
        if not due:
            return False
        self.window_updates = 0
        self.consecutive_passes = self.consecutive_passes + 1 if passed else 0
        if (
            self.stage < len(self.hip_limits) - 1
            and self.stage_updates >= self.min_stage_updates
            and self.consecutive_passes >= self.required_passes
        ):
            self.stage += 1
            self.stage_updates = 0
            self.consecutive_passes = 0
            return True
        return False

    def state_dict(self) -> dict:
        return {key: getattr(self, key) for key in (
            "stage", "stage_updates", "total_updates", "consecutive_passes", "window_updates"
        )}

    def load_state_dict(self, state: dict, *, reset_window: bool):
        for key in self.state_dict():
            value = state[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"Invalid curriculum counter {key}: {value!r}")
        if state["stage"] >= len(self.hip_limits) or state["window_updates"] >= self.evaluation_interval:
            raise ValueError("Checkpoint curriculum stage/window exceeds configured schedule")
        if state["stage_updates"] > state["total_updates"]:
            raise ValueError("Checkpoint stage updates exceed total updates")
        if state["window_updates"] > state["stage_updates"]:
            raise ValueError("Checkpoint window updates exceed stage updates")
        minimum_per_stage = max(
            math.ceil(self.min_stage_updates / self.evaluation_interval) * self.evaluation_interval,
            self.required_passes * self.evaluation_interval,
        )
        if state["total_updates"] < state["stage"] * minimum_per_stage + state["stage_updates"]:
            raise ValueError("Checkpoint could not have reached its declared curriculum stage")
        if state["consecutive_passes"] * self.evaluation_interval > state["stage_updates"]:
            raise ValueError("Checkpoint pass count exceeds completed stage evaluation windows")
        for key in self.state_dict():
            setattr(self, key, state[key])
        if reset_window:
            # Robot episode states are not restored by RSL-RL.  Require new
            # complete windows after resume, never promote using stale samples.
            self.window_updates = 0
            self.consecutive_passes = 0


def quality_gate(metrics: dict[str, float], *, min_samples: int = 100) -> bool:
    """Require separate forward/backward/stop evidence, not average reward.

    These are noisy-training, 50 Hz promotion gates.  Final acceptance still
    needs deterministic playback with 1 kHz contact measurements and starts/
    stops; passing here does not establish a validated locomotion policy.
    """
    required = ["completed_episodes", "failed_episode_fraction", "nonfinite_samples"]
    for group in ("forward", "reverse", "stop"):
        required.extend(f"{group}/{key}" for key in (
            "samples", "speed_error", "four_wheel_support", "tilt_bad_fraction",
            "height_bad_fraction", "lateral_speed", "yaw_rate", "body_contact_fraction"
        ))
    if any(key not in metrics or not math.isfinite(metrics[key]) for key in required):
        return False
    if metrics["nonfinite_samples"] != 0 or metrics["completed_episodes"] < 8:
        return False
    if metrics["failed_episode_fraction"] > 0.01:
        return False
    for group in ("forward", "reverse", "stop"):
        if metrics[f"{group}/samples"] < min_samples:
            return False
        if metrics[f"{group}/speed_error"] > (0.025 if group == "stop" else 0.05):
            return False
        if metrics[f"{group}/four_wheel_support"] < 0.98:
            return False
        if metrics[f"{group}/tilt_bad_fraction"] > 0.01 or metrics[f"{group}/height_bad_fraction"] > 0.01:
            return False
        if metrics[f"{group}/lateral_speed"] > 0.025 or metrics[f"{group}/yaw_rate"] > 0.035:
            return False
        if metrics[f"{group}/body_contact_fraction"] > 0.001:
            return False
    return True
