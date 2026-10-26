"""调姿后的顺坡速度捕获，仅通过原轮速伺服请求有界驱动/制动力。"""
import numpy as np


def quintic(value):
    x=float(np.clip(value,0.,1.))
    return x**3*(10.-15.*x+6.*x*x)


class TorqueCapture:
    def __init__(self,entry_duration,capture_duration,velocity_reference='wheel',capture_acceleration=.1):
        if not np.isfinite([entry_duration,capture_duration]).all() or entry_duration<0 or capture_duration<=0:
            raise ValueError('Capture start must be finite and nonnegative; duration must be positive')
        self.entry_duration=entry_duration;self.capture_duration=capture_duration
        if velocity_reference not in ('wheel','body','ramp'):raise ValueError('Unknown capture velocity reference')
        if not np.isfinite(capture_acceleration) or not .05<=capture_acceleration<=.5:raise ValueError('Invalid capture acceleration')
        self.velocity_reference=velocity_reference
        self.capture_acceleration=capture_acceleration;self.anchor_speed=None;self.anchor_at=None
        self.ready_time=0.;self.handoff_at=None;self.handoff=0.;self.finished=False

    def step(self,time,wheel_velocity,torque,nominal_wheel,velocity,height,goal_height,angles,goal_angles,omega,joint_error,normal_forces):
        """返回4维轮速目标和诊断；过渡前跟随实测轮速，稳定后才交回原控制。

        tau=2*(target-current)，不改变原速度伺服或力矩限幅。
        请求力矩由调用方按真实轮载/摩擦分配；它不是对实际接触力的保证。
        """
        gain=quintic((time-self.entry_duration)/self.capture_duration)
        # body模式以整机滚动速度锚定，避免离地轮/滑移轮的转速被递归写回目标。
        # 仅改目标速度，实际力矩仍由原2Nm/(rad/s)伺服和48Nm限矩决定。
        anchor=np.asarray(wheel_velocity) if self.velocity_reference=='wheel' else np.full(4,velocity/.0875)
        if self.velocity_reference=='ramp':
            # 捕获开始时建立连续速度参考，此后不能随实际倒滑速度一同逃逸。
            if self.anchor_speed is None or time<self.entry_duration:
                self.anchor_speed=float(velocity);self.anchor_at=float(time)
            speed=np.sign(self.anchor_speed)*max(0.,abs(self.anchor_speed)-self.capture_acceleration*max(0.,time-self.anchor_at))
            anchor=np.full(4,speed/.0875)
        wanted=anchor+gain*np.asarray(torque)/2.
        supported=np.asarray(normal_forces)>5.
        rolling_ready=bool(supported.any() and np.max(abs(np.asarray(wheel_velocity)[supported]*.0875-velocity))<.1)
        ready=(gain>=1. and abs(velocity)<.08 and abs(height-goal_height)<.04
            and np.max(np.abs(np.asarray(angles)-goal_angles))<.12
            and np.max(np.abs(omega))<.5 and joint_error<.22 and rolling_ready)
        self.ready_time=self.ready_time+.02 if ready else 0.
        if self.ready_time>=.4 and self.handoff_at is None:self.handoff_at=time
        self.handoff=0. if self.handoff_at is None else quintic(time-self.handoff_at)
        self.finished=self.handoff>=1.
        wheel=np.clip((1-self.handoff)*wanted+self.handoff*np.asarray(nominal_wheel),-50.,50.)
        return wheel,dict(entry_capture_fraction=gain,entry_capture_handoff=self.handoff,
            entry_capture_ready_s=self.ready_time,entry_capture_requested_torque_Nm=gain*np.asarray(torque),
            entry_capture_joint_error_rad=joint_error,entry_capture_velocity_mps=velocity,
            entry_capture_rolling_ready=float(rolling_ready))
