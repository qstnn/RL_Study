"""研究用轮速前馈：按上一周期实测轮载分配牵引，不向机身施加助力。"""
import numpy as np


def allocated_targets(command,velocity,load_N,normal_N,mass_kg,slope_rad,mu,uniform=False,front_share=None,signed_load=False):
    """返回四轮速度目标与力分配；用于运动阶段，使用仿真特权轮载/已知负载。

    轮伺服原公式为 tau=2*(omega_target-omega)。用滚动速度加力矩前馈，
    按真实平均法向力分配牵引；保留15%摩擦余量，不宣称预测力就是实际力。
    """
    normal=np.maximum(np.asarray(normal_N,dtype=float),0.)
    values=np.r_[command,velocity,load_N,normal,mass_kg,slope_rad,mu]
    if normal.shape!=(4,) or not np.isfinite(values).all() or mass_kg<=0 or mu<=0 or (load_N<0 and not signed_load):
        raise ValueError('Invalid wheel allocation state')
    if normal.sum()<.25*mass_kg*9.81*np.cos(slope_rad):
        raise ValueError('Insufficient total measured support for allocation')
    # 同速对照把计划牵引平均分配，实际轮力仍由接触/电机决定；这不保证
    # 每轮摩擦约束，失载轮的空转必须由原生滚动审查判定。
    weights=np.full(4,.25) if uniform else normal/normal.sum()
    if front_share is not None:
        if not np.isfinite(front_share) or not 0<front_share<1:raise ValueError('Invalid front axle force share')
        weights=np.array([front_share,front_share,1-front_share,1-front_share])/2.
    # 下坡显式允许负阻力：被动车推动机器人，轮电机应输出制动力矩。
    requested=mass_kg*9.81*np.sin(slope_rad)+load_N+mass_kg*3.*(command-velocity)
    available=.85*mu*normal.sum()
    force=np.clip(requested,-available,available)*weights
    torque=force*.0875
    target=np.clip(command/.0875+torque/2.,-50.,50.)
    return dict(target_rad_s=target,force_N=force,normal_N=normal,weights=weights,
                torque_feedforward_Nm=torque,requested_total_force_N=float(requested))


def heading_velocity_offset(heading,yaw_rate,lateral_positions,gain,rate_gain=.5):
    """向坡道中心方向纠偏的差速轮速；不施加外力，正偏航时右轮减速。

    激活条件由入口明确指定；返回四轮rad/s，转向角速度限制0.5rad/s。
    实际侧滑与是否恢复仍须由原生物理轨迹验证。
    """
    y=np.asarray(lateral_positions,dtype=float)
    if y.shape!=(4,) or not np.isfinite(np.r_[heading,yaw_rate,y,gain,rate_gain]).all() or gain<0 or rate_gain<0:
        raise ValueError('Invalid heading feedback state')
    turn=np.clip(-gain*heading-rate_gain*yaw_rate,-.5,.5)
    return -turn*y/.0875
