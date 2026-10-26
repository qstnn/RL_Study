"""悬空 hip 扫角时序与显式执行器只读记录器；不依赖 Isaac Sim 的导入。"""

import math

import torch


DURATION_S = 18.0


def validate_gravity_mode(body_flags, gravity_vector, with_gravity):
    """核验场景重力与每个刚体开关，防止只改全局重力而机器人仍禁用重力。"""
    expected_disabled = not with_gravity
    if not body_flags or any(value is not expected_disabled for value in body_flags.values()):
        raise ValueError("Robot body gravity flags do not match the requested diagnostic mode.")
    expected = (0., 0., -9.81 if with_gravity else 0.)
    if len(gravity_vector) != 3 or any(
        not math.isfinite(value) or not math.isclose(value, target, abs_tol=1e-5)
        for value, target in zip(gravity_vector, expected)
    ):
        raise ValueError("Physics scene gravity does not match the requested diagnostic mode.")


def target_at(time_s):
    """返回（阶段名、目标rad）；每段2秒，先限位内跟踪，再触及上限并退出。"""
    if not math.isfinite(time_s) or time_s < 0:
        raise ValueError("Probe time must be finite and nonnegative.")
    phases = (
        ("hold_030_initial", .30, .30),
        ("ramp_035_out", .30, .35),
        ("hold_035_out", .35, .35),
        ("ramp_040", .35, .40),
        ("hold_040", .40, .40),
        ("ramp_035_back", .40, .35),
        ("hold_035_back", .35, .35),
        ("ramp_030_back", .35, .30),
        ("hold_030_final", .30, .30),
    )
    index = min(int(time_s // 2), len(phases) - 1)
    phase, start, end = phases[index]
    ratio = min(1., (time_s - index * 2) / 2)
    return phase, start + (end - start) * ratio


class ActuatorAudit:
    """记录步前状态及真实延迟缓冲输出，不重复调用compute或改变执行器返回值。

    IdealPD.compute返回前会清空位置/速度目标，因此在DelayBuffer出口采集延迟后目标。
    每个物理步仍只执行一次原始执行器计算；P/D重构仅用于核验，不写回力矩。
    """

    def __init__(self, robot):
        self.robot = robot
        self.values = {}
        self.restore = []
        shape = robot.data.joint_pos.shape
        if shape[0] != 1:
            raise ValueError("Unloaded audit requires one robot.")
        for field in ("pos_before", "vel_before", "pos_target_delayed", "vel_target_delayed",
                      "effort_feedforward", "torque_p", "torque_d", "computed_torque", "torque"):
            self.values[field] = torch.zeros_like(robot.data.joint_pos)
        covered = []
        try:
            for actuator in robot.actuators.values():
                ids = [list(robot.joint_names).index(name) for name in actuator.joint_names]
                covered.extend(ids)
                self._wrap(actuator, ids)
            if sorted(covered) != list(range(shape[1])):
                raise ValueError("Each monitored joint must belong to exactly one actuator.")
        except Exception:
            self.close()
            raise

    def _wrap(self, actuator, ids):
        delayed = {}
        for attribute, key in (("positions_delay_buffer", "pos"),
                               ("velocities_delay_buffer", "vel"),
                               ("efforts_delay_buffer", "effort")):
            buffer = getattr(actuator, attribute)
            original = buffer.compute

            def capture(data, original=original, key=key):
                result = original(data)
                delayed[key] = result.detach().clone()
                return result

            buffer.compute = capture
            self.restore.append((buffer, "compute", original))
        original_compute = actuator.compute

        def compute(control_action, joint_pos, joint_vel):
            before_pos, before_vel = joint_pos.clone(), joint_vel.clone()
            result = original_compute(control_action, joint_pos, joint_vel)
            values = {
                "pos_before": before_pos, "vel_before": before_vel,
                "pos_target_delayed": delayed["pos"], "vel_target_delayed": delayed["vel"],
                "effort_feedforward": delayed["effort"],
                "torque_p": actuator.stiffness * (delayed["pos"] - before_pos),
                "torque_d": actuator.damping * (delayed["vel"] - before_vel),
                "computed_torque": actuator.computed_effort, "torque": actuator.applied_effort,
            }
            for name, value in values.items():
                self.values[name][:, ids] = value
            return result

        actuator.compute = compute
        self.restore.append((actuator, "compute", original_compute))

    def close(self):
        """恢复原方法；允许重复调用，异常路径也不遗留包装。"""
        for obj, name, original in reversed(self.restore):
            setattr(obj, name, original)
        self.restore.clear()
