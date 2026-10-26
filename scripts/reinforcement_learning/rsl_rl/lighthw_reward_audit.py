"""CPU离线奖励核对；读取实际保存权重/归一化，不改变训练奖励。"""
import math


def joint_acceleration_cost(squared_sum, term, dt):
    """由原始Σqdd²重构权重后贡献：返回每秒项及乘控制dt后的每步项。

    M/E仍为未加权物理量；normalization来自该run的保存配置，不能漏乘或重复乘。
    """
    squared_sum=float(squared_sum);weight=float(term['weight'])
    normalization=float(term['params'].get('normalization',1.0));dt=float(dt)
    if not all(math.isfinite(v)for v in (squared_sum,weight,normalization,dt)) or squared_sum<0 or normalization<0 or dt<=0:
        raise ValueError('Invalid squared acceleration/reward normalization/control period')
    rate=weight*normalization*squared_sum
    return dict(raw_squared_sum=squared_sum,weight=weight,normalization=normalization,dt=dt,per_second=rate,per_control_step=rate*dt)
