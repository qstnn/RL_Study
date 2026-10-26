"""Route 15 的纯数据协议；不依赖 Isaac Sim，可单独测试。"""

import math

PROFILE = "lighthw_roll_stop_v1"
SPEEDS = (0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8, 3.2)
CYCLE_SECONDS = 20.0
STOP_SECONDS = 5.0
ACCELERATION = 0.8
WHEEL_RADIUS = 0.0875
WHEEL_SCALE = 5.0
WHEEL_LIMIT = 37.0
LEG_LIMITS = (0.04, 0.10, 0.10) * 4


def phase_spec(stage):
    """每个速度先纯斜坡，联合通过后才加入25%阶跃周期，再考虑升速。"""
    if type(stage) is not int or not 0 <= stage < 2 * len(SPEEDS):
        raise ValueError(f"Invalid roll/stop stage: {stage}")
    return SPEEDS[stage // 2], (0.0 if stage % 2 == 0 else 0.25)


def thresholds(speed):
    return dict(speed_mae=max(0.08, 0.12 * speed), support=0.90,
                stop_speed=0.035, stop_drift=0.04, tilt_bad=0.01,
                nonwheel=0.0001, cycle_success=0.90, failure=0.001)


def gate_results(metrics, stage, num_envs):
    """缺样本/非有限值必须失败；同时检验稳速、停车和完整起停周期。"""
    speed, step_probability = phase_spec(stage)
    limits = thresholds(speed)
    checks = {}
    for key, limit in limits.items():
        value = metrics.get(key)
        valid = value is not None and math.isfinite(value)
        checks[key] = valid and (value >= limit if key in ("support", "cycle_success") else value <= limit)
    checks["fast_samples"] = metrics.get("fast_samples", 0) >= max(100, num_envs * 10)
    checks["stop_samples"] = metrics.get("stop_samples", 0) >= max(100, num_envs * 10)
    checks["cycles"] = metrics.get("cycles", 0) >= max(8, num_envs // 4)
    if step_probability:
        checks["step_cycles"] = metrics.get("step_cycles", 0) >= max(4, num_envs // 20)
        checks["step_success"] = (metrics.get("step_success") is not None
                                  and metrics["step_success"] >= limits["cycle_success"])
    return checks


def next_stage(stage, passes, stage_updates, checks):
    passes = passes + 1 if all(checks.values()) else 0
    advance = passes >= 2 and stage_updates >= 200 and stage < 2 * len(SPEEDS) - 1
    return (stage + 1, 0, True) if advance else (stage, passes, False)
