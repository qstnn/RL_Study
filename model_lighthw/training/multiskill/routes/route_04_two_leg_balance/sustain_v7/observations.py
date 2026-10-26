"""仅诊断旧Actor前缀：明确保留其旧输入，绝不作为持续任务策略观测。"""
import torch
from two_leg_balance_smooth_v6.core import schedule, desired_gravity
from two_leg_balance_split_v4.spec import STAND


class LegacyPrefixObservation:
    def __init__(self):
        self.history = None
        self.step = None

    def __call__(self, b):
        d = b.robot.data
        t = b.physics_steps * b.physics_dt
        phase, amp = schedule(b.state.clock(t), 2.)
        target = desired_gravity(b.actions.new_tensor(b.cfg.target_pitches)[b.mode] * amp)
        frame = torch.cat((d.root_ang_vel_b * .2, d.projected_gravity_b,
            d.joint_pos[:, b.leg_ids] - b.actions.new_tensor(STAND),
            d.joint_vel[:, b.joint_ids] * .05, b.actions,
            torch.nn.functional.one_hot(b.mode, 2).float(), target,
            torch.nn.functional.one_hot(phase, 5).float(), (t / 14.)[:, None]), -1)
        if self.history is None:
            self.history = frame[:, None].repeat(1, 5, 1)
        elif self.step != b.common_step_counter:
            self.history[:, :-1] = self.history[:, 1:].clone()
            self.history[:, -1] = frame
        self.step = b.common_step_counter
        return self.history.flatten(1).clone()
