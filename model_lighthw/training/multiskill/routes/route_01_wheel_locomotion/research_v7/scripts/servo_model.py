"""原50Hz位置/速度接口的恒加速度区间近似；不修改原生PD或物理步长。"""
import numpy as np


def correction(velocity,horizon):
    """返回初始力矩相对周期均值的偏置和qdd系数，单位Nm、Nm/(rad/s²)。

    目标保持h秒，腿τ=Kp(qref-q)-Kd*qd，轮τ=Kd(vref-qd)。
    假设q=q0+qd0*t+qdd*t²/2，仅作为可验证的前馈近似。
    """
    qd=np.asarray(velocity,dtype=float)
    if qd.shape!=(16,) or not np.isfinite(qd).all() or not np.isfinite(horizon) or not 0<=horizon<=.02:
        raise ValueError('Invalid servo velocity or horizon')
    h=float(horizon)
    bias=np.r_[80.*h/2*qd[:12],np.zeros(4)]
    coefficient=np.r_[np.full(12,80.*h*h/6+3.*h/2),np.full(4,2.*h/2)]
    return bias,coefficient


def initial_torque(mean_torque,velocity,acceleration,horizon):
    """给位置/轮速反解的初始力矩；真实执行仍需1kHz核对。"""
    bias,coefficient=correction(velocity,horizon)
    return np.asarray(mean_torque)+bias+coefficient*np.asarray(acceleration)
