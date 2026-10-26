"""仅用于仿真播放的 FL hip 回收响应探针，不改变奖励、模型和执行器。"""

import math

import torch


class HipRetractionProbe:
    """用真实动作仿射映射替换FL动作，默认其他关节仍由闭环策略控制。

    时间表（秒）：0–3 原策略；3–5 渐变到 .35；5–8 保持；8–10 渐变到 .30；
    10–13 保持；13–15 渐变交还策略；15–18 观察恢复。干预后的动作会进入
    last_action 观测，因此这不是固定其他关节的开环机械标定。
    hold_other_actions=True时，3–13秒保持其余15维动作，13–15秒渐变交还策略；
    这只保持目标，实际关节仍自由响应接触/重力，不是物理锁定。
    """

    headers = (
        "probe_phase", "probe_active", "probe_policy_raw", "probe_policy_preclip_rad",
        "probe_policy_clipped_rad", "probe_command_raw", "probe_command_target_rad",
    )
    duration_s = 18.0

    def __init__(self, manager, robot, wrapper_clip=None, hold_other_actions=False):
        # 可选落地对照：保持其他关节的目标，不是锁住它们的实际位置。
        self.hold_other_actions = bool(hold_other_actions)
        self.held_actions = None
        self.hold_mix = 0.
        if self.hold_other_actions:
            self.headers = self.headers + ("probe_hold_others_mix", "probe_other_raw_delta_from_hold")
        self.term = manager.get_term("joint_pos")
        self.joint_name = "FL_hip_joint"
        if (list(self.term._joint_names).count(self.joint_name) != 1
                or list(robot.joint_names).count(self.joint_name) != 1):
            raise ValueError("Hip probe requires one unique FL_hip_joint in robot and position actions.")
        local_id = list(self.term._joint_names).index(self.joint_name)
        self.robot_joint_id = list(robot.joint_names).index(self.joint_name)
        start = 0
        for name in manager.active_terms:
            if name == "joint_pos":
                break
            start += manager.get_term(name).action_dim
        else:
            raise ValueError("Hip probe requires the joint_pos action term.")
        self.action_id = start + local_id
        self.action_dim = manager.total_action_dim
        self.other_action_ids = [i for i in range(self.action_dim) if i != self.action_id]
        if self.term.raw_actions.shape[0] != 1:
            raise ValueError("Hip probe requires exactly one environment.")

        def scalar(value):
            value = torch.as_tensor(value, device=self.term.raw_actions.device)
            return float(torch.broadcast_to(value, self.term.raw_actions.shape)[0, local_id].item())

        self.scale = scalar(self.term._scale)
        self.offset = scalar(self.term._offset)
        if not math.isfinite(self.scale) or abs(self.scale) < 1e-12 or not math.isfinite(self.offset):
            raise ValueError("Hip probe needs a finite, invertible action mapping.")
        if self.term.cfg.clip is None:
            raise ValueError("Hip probe requires explicit position-action limits.")
        self.lower, self.upper = self.term._clip[0, local_id].tolist()
        hard_lower, hard_upper = robot.data.joint_pos_limits[0, self.robot_joint_id].tolist()
        if not all(math.isfinite(x) for x in (self.lower, self.upper, hard_lower, hard_upper)):
            raise ValueError("Hip probe limits must be finite.")
        # 本探针针对当前 ±.4 hip；换资产/坐标时拒绝运行，不能悄悄把错误映射当成机械问题。
        if not (math.isclose(self.lower, -.4, abs_tol=1e-5)
                and math.isclose(self.upper, .4, abs_tol=1e-5)
                and hard_lower <= self.lower + 1e-5 and hard_upper >= self.upper - 1e-5):
            raise ValueError("Hip probe expects FL position clip +/-0.4 inside the physical hard limits.")
        self.wrapper_clip = None if wrapper_clip is None else float(wrapper_clip)
        if self.wrapper_clip is not None:
            if not math.isfinite(self.wrapper_clip) or self.wrapper_clip <= 0:
                raise ValueError("Invalid wrapper action clip.")
            for target in (self.lower, self.upper, .35, .30):
                if abs(self.inverse(target)) > self.wrapper_clip + 1e-6:
                    raise ValueError("Wrapper action clipping prevents this hip target test.")
        self.start_target = None
        self.phase = None
        self.values = ["not_started", 0, 0., 0., 0., 0., 0.]
        if self.hold_other_actions:
            self.values += [0., 0.]

    def inverse(self, target):
        """目标角（rad）反算 raw action，不绕过原有 action clip 或硬限位。"""
        return (target - self.offset) / self.scale

    def apply(self, actions, time_s):
        """返回送入 env.step 的动作及可追溯日志；不原地修改策略输出。"""
        if not math.isfinite(time_s) or time_s < 0.:
            raise ValueError("Hip probe time must be finite and nonnegative.")
        if actions.shape != (1, self.action_dim) or not torch.isfinite(actions).all():
            raise ValueError("Hip probe received invalid policy actions.")
        raw = float(actions[0, self.action_id].item())
        effective_raw = raw if self.wrapper_clip is None else max(-self.wrapper_clip, min(self.wrapper_clip, raw))
        preclip = effective_raw * self.scale + self.offset
        policy_target = max(self.lower, min(self.upper, preclip))
        active = 3. <= time_s < 15.
        if time_s < 3.:
            phase, target = "baseline", policy_target
        elif time_s < 5.:
            if self.start_target is None:
                self.start_target = policy_target
                if not .38 <= self.start_target <= .40001:
                    raise ValueError("FL policy is not near +0.4 at probe start; abort instead of pushing it outward.")
            phase = "ramp_035"
            target = self.start_target + (.35 - self.start_target) * ((time_s - 3.) / 2.)
        elif time_s < 8.:
            phase, target = "hold_035", .35
        elif time_s < 10.:
            phase, target = "ramp_030", .35 + (.30 - .35) * ((time_s - 8.) / 2.)
        elif time_s < 13.:
            phase, target = "hold_030", .30
        elif time_s < 15.:
            # 向策略当步的已裁剪目标渐变，不能向可能严重越界的 preclip 目标插值。
            phase = "release"
            target = .30 + (policy_target - .30) * ((time_s - 13.) / 2.)
        else:
            phase, target = "policy_restored", policy_target
        result = actions
        self.hold_mix = 0.
        if active:
            if self.hold_other_actions:
                if self.held_actions is None:
                    self.held_actions = actions.detach().clone()
                # 3–13秒保持捕获的其他15维动作；13–15秒渐变交还当前闭环策略。
                # 轮子保持捕获时的速度目标，不额外归零或限速。
                self.hold_mix = 1. if time_s < 13. else (15. - time_s) / 2.
                result = self.held_actions.clone() if self.hold_mix == 1. else (
                    self.hold_mix * self.held_actions + (1. - self.hold_mix) * actions
                )
            else:
                result = actions.clone()
            result[0, self.action_id] = self.inverse(target)
        command_raw = float(result[0, self.action_id].item())
        self.phase = phase
        self.values = [phase, int(active), raw, preclip, policy_target, command_raw, target]
        if self.hold_other_actions:
            delta = 0. if self.held_actions is None else float(
                (result[:, self.other_action_ids] - self.held_actions[:, self.other_action_ids]).abs().max().item()
            )
            self.values += [self.hold_mix, delta]
        return result


class ProbeSafety:
    """按物理子步监测；任一轮连续 .2 秒低于5 N、倾斜>20°或机身<.25 m则停止。

    前2秒仅允许初始落地瞬态；非有限状态始终停止。停止标记锁存，当前控制步
    剩余子步结束后退出播放，不继续运行自动重置后的新回合。
    """

    def __init__(self):
        self.unloaded_s = [0.] * 4
        self.reason = ""

    def update(self, time_s, dt, gravity_z, height, wheel_fz):
        if self.reason:
            return
        if not math.isfinite(time_s) or time_s < 0. or not math.isfinite(dt) or dt <= 0.:
            self.reason = "invalid_physics_time"
            return
        if len(wheel_fz) != 4 or not all(math.isfinite(v) for v in (gravity_z, height, *wheel_fz)):
            self.reason = "nonfinite_or_missing_state"
            return
        if time_s < 2.:
            return
        tilt = math.degrees(math.acos(max(-1., min(1., -gravity_z))))
        if tilt > 20.:
            self.reason = "tilt_over_20deg"
        elif height < .25:
            self.reason = "base_below_025m"
        for i, force in enumerate(wheel_fz):
            self.unloaded_s[i] = self.unloaded_s[i] + dt if force < 5. else 0.
        if not self.reason and max(self.unloaded_s) >= .2 - 1e-9:
            self.reason = "wheel_unloaded_020s"
