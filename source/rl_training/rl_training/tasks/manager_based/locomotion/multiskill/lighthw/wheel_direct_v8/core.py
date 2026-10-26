"""R1 纯数学：固定混合配额、坡面网格、质量惯量及坡面奖励计算。"""
import math
import numpy as np
import torch


def assignments(num_envs, slopes, payloads, towed=(0.,), fixed_slope=None, fixed_payload=None, fixed_towed=None):
    """空载/仅载荷/仅牵引约25/25/50；拖载类60%锚定+25/-20度40kg，首批包含全部许可工况；-25度不牵引。"""
    if num_envs < 1:
        raise ValueError("num_envs must be positive")
    for value, upper, name in ((fixed_payload,15.,"payload"),(fixed_towed,40.,"tow")):
        if value is not None and (not math.isfinite(value) or not 0 <= value <= upper):
            raise ValueError(f"Invalid {name}")
    if fixed_payload and fixed_towed:
        raise ValueError("Payload and towing are mutually exclusive in V8")
    if fixed_slope is not None and (not math.isfinite(fixed_slope) or fixed_slope not in slopes):
        raise ValueError("Fixed slope must be in the terrain angle list")
    groups = [([(s,0.,0.) for s in slopes], .25),
              ([(s,m,0.) for s in slopes for m in payloads if m > 0], .25),
              ([(s,0.,t) for s in slopes for t in towed if t > 0 and s != -25.], .5)]
    selected = []
    fixed = any(v is not None for v in (fixed_slope,fixed_payload,fixed_towed))
    if fixed:
        eligible = [c for group,_ in groups for c in group
                    if (fixed_slope is None or c[0] == fixed_slope)
                    and (fixed_payload is None or c[1] == fixed_payload)
                    and (fixed_towed is None or c[2] == fixed_towed)]
        if not eligible:
            raise ValueError("No permitted mutually exclusive condition matches the requested fixed case")
        selected = [eligible[i % len(eligible)] for i in range(num_envs)]
    else:
        counts = [round(num_envs*.25), round(num_envs*.25)]
        counts.append(num_envs-sum(counts))
        for (group,_),count in zip(groups,counts):
            unique = sorted(set(group), key=lambda c:(-abs(c[0]),-c[1],-c[2],c[0]))
            remaining = list(group)
            for item in unique:
                remaining.remove(item)
            order = unique+remaining
            # 拖载配额50%中20%覆盖全拖载，30%分给上坡/下坡困难锚点。
            is_tow = max(c[2] for c in group) > 0
            base = max(1,round(count*(.4 if is_tow else .8))) if count else 0
            selected.extend(order[i % len(order)] for i in range(base))
            max_mass=max(c[1] for c in group);max_tow=max(c[2] for c in group)
            max_slope=max(abs(c[0]) for c in group)
            anchors=([(25.,0.,40.),(-20.,0.,40.)] if is_tow else
                     [c for c in unique if abs(c[0])==max_slope and c[1]==max_mass and c[2]==max_tow])
            selected.extend(anchors[i % len(anchors)] for i in range(count-base))
    values=np.asarray(selected,dtype=float)
    if (not np.isfinite(values).all() or np.any((values[:,1]>0)&(values[:,2]>0))
            or np.any((values[:,0]==-25.)&(values[:,2]>0))):
        raise ValueError("Invalid V8 condition assignment")
    return values[:,0],values[:,1],values[:,2]


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


def route_velocity(velocity, tangent, normal):
    """所有工况使用固定目标路线坐标；返回沿路线速度和横向速度，不随机身跑偏旋转。"""
    lateral=torch.cross(normal,tangent,dim=-1)
    return torch.stack(((velocity*tangent).sum(-1),(velocity*lateral).sum(-1)),dim=-1)


def slew(current, target, rate, dt):
    """按每环境物理加速度限制平滑指令；不越过目标，返回实际送入Actor/奖励的速度。"""
    delta=(target-current).clamp(-rate*dt,rate*dt)
    result=current+delta
    return torch.where((target-result).abs()<1e-6,target,result)
