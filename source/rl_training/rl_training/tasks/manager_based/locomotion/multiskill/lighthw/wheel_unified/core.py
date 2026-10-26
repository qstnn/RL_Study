"""R1 纯数学：固定混合配额、坡面网格、质量惯量及坡面奖励计算。"""
import math
import numpy as np
import torch


def assignments(num_envs, slopes, payloads, towed=(0.,), fixed_slope=None, fixed_payload=None, fixed_towed=None):
    """循环铺满笛卡尔工况；无迭代课程，最难工况从第一批即出现。"""
    if num_envs < 1:
        raise ValueError("num_envs must be positive")
    pairs = [(slope, mass, tow) for slope in slopes for mass in payloads for tow in towed]
    # 短验也优先覆盖最大正/负坡，避免只取列表开头的平地。
    unique = sorted(set(pairs), key=lambda p: (-abs(p[0]), -p[1], -p[2], p[0]))
    remaining = list(pairs)
    for pair in unique:
        remaining.remove(pair)
    order = unique + remaining
    # 80%保留全笛卡尔覆盖，20%固定在15kg+40kg的正/负25度目标。
    # 全组合仍从第一轮出现，提高目标工况占比也不构成随时间晋级。
    base_count=max(1,int(num_envs*.8))
    selected=[order[i % len(order)] for i in range(base_count)]
    anchors=[(25.,15.,40.),(-25.,15.,40.)]
    selected.extend(anchors[i%2] for i in range(num_envs-base_count))
    angle = np.array([p[0] if fixed_slope is None else fixed_slope for p in selected])
    mass = np.array([p[1] if fixed_payload is None else fixed_payload for p in selected])
    if not np.isfinite(angle).all() or np.max(np.abs(angle)) > 25:
        raise ValueError("Slope must be finite and within +/-25 degrees")
    if not np.isfinite(mass).all() or np.min(mass) < 0 or np.max(mass) > 15:
        raise ValueError("Payload must be finite and within 0..15 kg")
    tow = np.array([p[2] if fixed_towed is None else fixed_towed for p in selected])
    if not np.isfinite(tow).all() or np.min(tow) < 0 or np.max(tow) > 40:
        raise ValueError("Towed mass must be finite and within 0..40 kg")
    return angle, mass, tow


def slope_mesh(angle_deg, y_center, x_min=-20., x_max=100., width=12.):
    """生成有限厚度长坡；angle为度，斜率为tan(angle)，出生点地面z=0。"""
    gradient = math.tan(math.radians(angle_deg))
    top = np.array([[x_min,y_center-width/2,x_min*gradient], [x_max,y_center-width/2,x_max*gradient],
                    [x_max,y_center+width/2,x_max*gradient], [x_min,y_center+width/2,x_min*gradient]])
    vertices = np.concatenate((top, top-[0.,0.,1.]))
    faces = np.array([[0,1,2],[0,2,3],[4,6,5],[4,7,6],
                      [0,4,5],[0,5,1],[1,5,6],[1,6,2],[2,6,7],[2,7,3],[3,7,4],[3,4,0]])
    return vertices, faces


def compose_inertia(base_mass, base_com, base_inertia, mass, center, size):
    """刚性载荷合成；输入惯量为Base_link轴系、各自质心处的惯量，返回合成质量/质心/惯量。"""
    center, size, base_com = map(np.asarray, (center, size, base_com))
    if mass < 0 or base_mass <= 0 or min(size) <= 0:
        raise ValueError("Invalid inertial payload")
    total = base_mass+mass
    com = (base_mass*base_com+mass*center)/total
    def shift(m, d):
        return m*((d@d)*np.eye(3)-np.outer(d,d))
    box = mass/12*np.diag([size[1]**2+size[2]**2,size[0]**2+size[2]**2,size[0]**2+size[1]**2])
    inertia = base_inertia+box+shift(base_mass,base_com-com)+shift(mass,center-com)
    if not np.isfinite(inertia).all() or np.linalg.eigvalsh(inertia).min() <= 0:
        raise ValueError("Invalid composite inertia")
    return total, com, inertia


def plane_basis(angle_rad):
    """沿坡正向切线、法线；正坡为沿世界+x上坡，负坡为沿+x下坡。"""
    zero = torch.zeros_like(angle_rad)
    tangent = torch.stack((angle_rad.cos(), zero, angle_rad.sin()),dim=-1)
    normal = torch.stack((-angle_rad.sin(), zero, angle_rad.cos()),dim=-1)
    return tangent, normal


def tangent_velocity(velocity, forward, normal):
    """将机身前轴投影到坡面，用该切向坐标比较速度；不会惩罚正常爬坡的世界竖直速度。"""
    tangent = forward-(forward*normal).sum(-1,keepdim=True)*normal
    tangent = tangent/tangent.norm(dim=-1,keepdim=True).clamp_min(1e-6)
    lateral = torch.cross(normal,tangent,dim=-1)
    return torch.stack(((velocity*tangent).sum(-1),(velocity*lateral).sum(-1)),dim=-1)


def velocity_score(error, command_speed, fine=False):
    """全工况跟踪奖励；粗核在3m/s初始欠速时仍有信号，细核用于精确巡航，返回[0,1]。"""
    std = torch.full_like(command_speed,.35) if fine else 1.+.2*command_speed.abs()
    return torch.exp(-error.square().sum(-1)/std.square())
