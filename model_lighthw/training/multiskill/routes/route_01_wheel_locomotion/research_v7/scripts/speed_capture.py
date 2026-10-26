"""倒滑捕获的速度指令整形，继续使用原牵引分配和原轮速伺服。"""
import numpy as np


class SpeedCapture:
    def __init__(self,start_s,speed_step,deceleration=.1):
        if not np.isfinite([start_s,speed_step,deceleration]).all() or start_s<0 or not 0<speed_step<=.5 or not .05<=deceleration<=.5:
            raise ValueError('Invalid speed capture schedule')
        self.start_s=start_s;self.speed_step=speed_step;self.deceleration=deceleration
        self.initial=None;self.at=None;self.ready_s=0.;self.finished=False

    def command(self,time,velocity):
        if self.initial is None or time<self.start_s:
            self.initial=min(0.,float(velocity)+self.speed_step);self.at=float(time)
        # 起点仅在捕获前更新；开始后指令只能向零推进，不能跟随持续倒滑。
        return min(0.,self.initial+self.deceleration*max(0.,time-self.at))

    def observe(self,command,velocity,height_error,angles_error,omega,joint_error,wheel_velocity,normal):
        supported=np.asarray(normal)>5.
        ready=(command==0. and abs(velocity)<.08 and abs(height_error)<.04
            and np.max(abs(angles_error))<.12 and np.max(abs(omega))<.5 and joint_error<.22
            and supported.any() and np.max(abs(np.asarray(wheel_velocity)[supported]*.0875-velocity))<.1)
        self.ready_s=self.ready_s+.02 if ready else 0.
        self.finished=self.finished or self.ready_s>=.4
        return dict(entry_speed_reference_mps=command,entry_speed_ready_s=self.ready_s,entry_speed_finished=float(self.finished))
