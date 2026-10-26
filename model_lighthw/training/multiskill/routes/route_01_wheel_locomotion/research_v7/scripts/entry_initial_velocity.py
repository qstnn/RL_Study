"""隔离捕获诊断的共同顺坡初速；仅重置使用，不是普通初态。"""
import numpy as np
from model import Model


def rolling_joint_velocity(reference,slope_deg,speed):
    """保持腿角速度零，按轮心接触雅可比求与共同机身初速匹配的滚动轮速。"""
    if not np.isfinite(speed) or not -2<=speed<=0:raise ValueError('Diagnostic initial speed must be within [-2,0]m/s')
    angle=np.deg2rad(slope_deg);normal=np.array([-np.sin(angle),0.,np.cos(angle)]);tangent=np.array([np.cos(angle),0.,np.sin(angle)])
    model=Model();A,_,_=model.contacts(model.state(normal*reference['height_m'],reference['quat'],reference['q']),normal)
    v=np.r_[tangent*speed,np.zeros(19)]
    for i in range(4):
        row=tangent@A[3*i:3*i+3];v[18+i]=-float(row@v)/row[18+i]
    return v[6:]
