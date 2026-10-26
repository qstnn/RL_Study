"""R4b 本体反馈：URDF 几何和前轮无滑滚动假设，不读取仿真特权状态。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import xml.etree.ElementTree as ET

import numpy as np
import torch


JOINT_NAMES = tuple(f"{leg}_{part}_joint" for leg in ("FR", "FL", "RR", "RL") for part in ("hip", "thigh", "calf")) + tuple(
    f"{leg}_wheel_joint" for leg in ("FR", "FL", "RR", "RL")
)
STAND = np.array((0., .6, -1.5, 0., .6, -1.5, 0., -.6, 1.5, 0., -.6, 1.5), dtype=np.float64)
SUPPORT = ("FR", "FL")
RADIUS_M = .0875  # 原碰撞轮外半径；滚动近似只在轮胎确实接地且不滑时成立。
CANONICAL_URDF_SHA256 = "781d9998885ed5962eacaf6363271480a3b98f346879ce8101a311e0314242a9"


def _rotation(axis: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """批量 Rodrigues 旋转；angle 的首维是独立观测帧。"""
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array(((0., -z, y), (z, 0., -x), (-y, x, 0.)))
    return np.eye(3)[None] + np.sin(angle)[:, None, None] * skew + (1. - np.cos(angle))[:, None, None] * (skew @ skew)


def _origin(element: ET.Element | None) -> np.ndarray:
    t = np.eye(4)
    if element is None:
        return t
    t[:3, 3] = np.fromstring(element.get("xyz", "0 0 0"), sep=" ")
    rpy = np.fromstring(element.get("rpy", "0 0 0"), sep=" ")
    # 与规范 URDF FK 的 Rz(yaw) Ry(pitch) Rx(roll) 约定相同。
    rx = _rotation(np.array((1., 0., 0.)), np.array((rpy[0],)))[0]
    ry = _rotation(np.array((0., 1., 0.)), np.array((rpy[1],)))[0]
    rz = _rotation(np.array((0., 0., 1.)), np.array((rpy[2],)))[0]
    t[:3, :3] = rz @ ry @ rx
    return t


@dataclass(frozen=True)
class Joint:
    name: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray
    kind: str


class BodyEstimator:
    """以当前 raw305 最后一帧为唯一运行输入，输出有符号 gap 和 COM 前向速度。

    gap 使用全身 URDF 质量/惯性原点 FK，并严格复制教师的 FR→FL 水平
    支撑线法向定义。速度以两只前轮各自无滑纯滚动推导的 COM 速度取平均；
    无接触或打滑信息时不能判别该假设是否成立。
    """

    def __init__(self, urdf: str | Path):
        path = Path(urdf)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != CANONICAL_URDF_SHA256:
            raise ValueError(f"unexpected LightHW URDF sha256: {actual}")
        root = ET.parse(path).getroot()
        self.masses = {}
        self.inertial = {}
        for link in root.findall("link"):
            name = link.get("name")
            inertia = link.find("inertial")
            self.masses[name] = float(inertia.find("mass").get("value"))
            self.inertial[name] = _origin(inertia.find("origin"))[:3, 3]
        unsorted = {}
        for item in root.findall("joint"):
            name = item.get("name")
            axis = np.fromstring(item.find("axis").get("xyz"), sep=" ") if item.find("axis") is not None else np.zeros(3)
            unsorted[name] = Joint(name, item.find("parent").get("link"), item.find("child").get("link"),
                                   _origin(item.find("origin")), axis, item.get("type"))
        if set(unsorted) != set(JOINT_NAMES):
            raise ValueError("URDF joint set differs from the R4b policy contract")
        order = []
        known = {"Base_link"}
        remaining = list(unsorted.values())
        while remaining:
            ready = [j for j in remaining if j.parent in known]
            if not ready:
                raise ValueError("URDF joint tree is disconnected")
            for j in ready:
                order.append(j)
                known.add(j.child)
                remaining.remove(j)
        if known != set(self.masses):
            raise ValueError("URDF mass/link tree differs")
        self.joints = order
        self.index = {name: i for i, name in enumerate(JOINT_NAMES)}
        self.ancestors = {"Base_link": ()}
        for j in order:
            self.ancestors[j.child] = self.ancestors[j.parent] + (j.name,)
        # raw305 没有轮角；规范轮质心恰在旋转轴上，轮角不影响轮中心/COM。
        for leg in ("FR", "FL", "RR", "RL"):
            offset = self.inertial[f"{leg}_wheel_link"]
            if abs(offset[0]) > 1e-5 or abs(offset[2]) > 1e-5:
                raise ValueError("wheel inertial COM is off-axis; wheel angle would be required")
            if not np.allclose(unsorted[f"{leg}_wheel_joint"].axis, (0., 1., 0.)):
                raise ValueError("expected +Y wheel axis")
        self.mass = sum(self.masses.values())

    def geometry(self, leg_q: np.ndarray):
        """返回 base 坐标下的全身 COM、轮心、解析雅可比与轮轴。"""
        q = np.asarray(leg_q, dtype=np.float64)
        if q.ndim != 2 or q.shape[1] != 12 or not np.isfinite(q).all():
            raise ValueError("leg_q must be finite [N,12] in R4b action order")
        n = len(q)
        frames = {"Base_link": np.broadcast_to(np.eye(4), (n, 4, 4)).copy()}
        axes, pivots = {}, {}
        for j in self.joints:
            joint_frame = frames[j.parent] @ j.origin
            axes[j.name] = joint_frame[:, :3, :3] @ j.axis
            pivots[j.name] = joint_frame[:, :3, 3]
            move = np.broadcast_to(np.eye(4), (n, 4, 4)).copy()
            # 轮角不在政策输入中；轮中心和质心均不受其旋转影响。
            angle = q[:, self.index[j.name]] if self.index[j.name] < 12 else np.zeros(n)
            move[:, :3, :3] = _rotation(j.axis, angle)
            frames[j.child] = joint_frame @ move
        com = np.zeros((n, 3))
        jac = np.zeros((n, 3, 16))
        for link, mass in self.masses.items():
            point = (frames[link] @ np.append(self.inertial[link], 1.))[:, :3]
            com += mass * point
            for name in self.ancestors[link]:
                jac[:, :, self.index[name]] += mass * np.cross(axes[name], point - pivots[name])
        com /= self.mass
        jac /= self.mass
        wheels = np.stack([frames[f"{leg}_wheel_link"][:, :3, 3] for leg in SUPPORT], axis=1)
        wheel_jac = np.zeros((n, 2, 3, 16))
        for k, leg in enumerate(SUPPORT):
            point = wheels[:, k]
            for name in self.ancestors[f"{leg}_wheel_link"]:
                wheel_jac[:, k, :, self.index[name]] = np.cross(axes[name], point - pivots[name])
        wheel_axis = np.stack([axes[f"{leg}_wheel_joint"] for leg in SUPPORT], axis=1)
        wheel_angular_jac = np.zeros((n, 2, 3, 16))
        for k, leg in enumerate(SUPPORT):
            for name in self.ancestors[f"{leg}_wheel_link"]:
                wheel_angular_jac[:, k, :, self.index[name]] = axes[name]
        return com, wheels, jac, wheel_jac, wheel_axis, wheel_angular_jac

    def estimate(self, raw305: np.ndarray):
        """输入 [N,305] 原策略观测；不读取 teacher COM/root 速度/接触。"""
        raw = np.asarray(raw305)
        if raw.ndim != 2 or raw.shape[1] != 305 or not np.isfinite(raw).all():
            raise ValueError("raw305 must be finite [N,305]")
        frame = raw[:, -61:].astype(np.float64)
        gravity = frame[:, 3:6]
        up = -gravity / np.linalg.norm(gravity, axis=1, keepdims=True)
        leg_q = frame[:, 6:18] + STAND
        qdot = frame[:, 18:34] / .05
        omega = frame[:, 0:3] / .2
        com, wheels, jac, wheel_jac, wheel_axis, wheel_angular_jac = self.geometry(leg_q)
        lateral = wheels[:, 1] - wheels[:, 0]
        forward = np.cross(lateral, up)
        forward /= np.linalg.norm(forward, axis=1, keepdims=True)
        front_center = wheels.mean(1)
        gap = np.einsum("ni,ni->n", com - front_center, forward)
        # 对每只前轮应用 0 = v_contact。绝对轮角速度包括本体 IMU 角速度
        # 与髋/大腿/小腿/轮关节速度；仅用轮编码器转速会遗漏起身时的大项。
        # 圆形轮接地点的径向方向是 world-up 在轮轴垂面内的投影。
        # 再加上 COM 相对于轮心的角速度与关节速度差。
        relative = com[:, None, :] - wheels
        relative_jac = jac[:, None, :, :] - wheel_jac
        relative_velocity = np.cross(omega[:, None, :], relative) + np.einsum("nrcj,nj->nrc", relative_jac, qdot)
        wheel_spin = qdot[:, 12:14]
        wheel_omega = omega[:, None, :] + np.einsum("nrcj,nj->nrc", wheel_angular_jac, qdot)
        radial_up = up[:, None, :] - np.sum(up[:, None, :] * wheel_axis, axis=-1, keepdims=True) * wheel_axis
        radial_up /= np.linalg.norm(radial_up, axis=-1, keepdims=True).clip(min=1e-6)
        rolling = RADIUS_M * np.einsum("nri,ni->nr", np.cross(wheel_omega, radial_up), forward)
        full_vertical_rolling = RADIUS_M * np.einsum("nri,ni->nr", np.cross(wheel_omega, up[:, None, :]), forward)
        spin_only_rolling = RADIUS_M * wheel_spin * np.einsum("nri,ni->nr", np.cross(wheel_axis, up[:, None, :]), forward)
        per_front = rolling + np.einsum("nri,ni->nr", relative_velocity, forward)
        offset = np.einsum("nri,ni->nr", relative_velocity, forward)
        return dict(gap_m=gap, velocity_m_s=per_front.mean(1), front_velocity_m_s=per_front,
                    velocity_full_vertical_m_s=(full_vertical_rolling + offset).mean(1),
                    velocity_spin_only_m_s=(spin_only_rolling + offset).mean(1),
                    front_wheel_rate_rad_s=wheel_spin, forward_body=forward,
                    com_body_m=com, front_wheel_body_m=wheels)


def _torch_rotation(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    axis = axis / torch.linalg.vector_norm(axis)
    x, y, z = axis.unbind()
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero)).reshape(3, 3)
    eye = torch.eye(3, device=angle.device, dtype=angle.dtype)
    return eye + torch.sin(angle)[:, None, None] * skew + (1. - torch.cos(angle))[:, None, None] * (skew @ skew)


class TorchBodyEstimator(torch.nn.Module):
    """同一纯本体估计的可导 PyTorch 接口；常量从规范 URDF 构造。

    输出相对 raw305 关节角/速度/重力/角速度可微。输入可以放在 CPU/GPU，
    但本模块本身不调用 Isaac Sim，也不读取仿真根速度/COM/接触。
    """

    def __init__(self, urdf: str | Path):
        super().__init__()
        source = BodyEstimator(urdf)
        self.joints = source.joints
        self.joint_names = JOINT_NAMES
        self.index = source.index
        self.ancestors = source.ancestors
        self.link_names = tuple(source.masses)
        self.register_buffer("stand", torch.tensor(STAND, dtype=torch.float64))
        self.register_buffer("origins", torch.tensor(np.stack([j.origin for j in source.joints]), dtype=torch.float64))
        self.register_buffer("axes", torch.tensor(np.stack([j.axis for j in source.joints]), dtype=torch.float64))
        self.register_buffer("inertial", torch.tensor(np.stack([source.inertial[name] for name in self.link_names]), dtype=torch.float64))
        self.register_buffer("masses", torch.tensor([source.masses[name] for name in self.link_names], dtype=torch.float64))

    def forward(self, raw305: torch.Tensor) -> dict[str, torch.Tensor]:
        if raw305.ndim != 2 or raw305.shape[-1] != 305:
            raise ValueError("raw305 must be [N,305]")
        x = raw305.to(dtype=self.stand.dtype)
        n = x.shape[0]
        frame = x[:, -61:]
        up = -torch.nn.functional.normalize(frame[:, 3:6], dim=-1)
        leg_q = frame[:, 6:18] + self.stand
        qdot = frame[:, 18:34] / .05
        omega = frame[:, :3] / .2
        eye = torch.eye(4, device=x.device, dtype=x.dtype).expand(n, -1, -1)
        frames = {"Base_link": eye}
        pivots = {}
        axes = {}
        for i, j in enumerate(self.joints):
            joint_frame = frames[j.parent] @ self.origins[i]
            axes[j.name] = joint_frame[:, :3, :3] @ self.axes[i]
            pivots[j.name] = joint_frame[:, :3, 3]
            if self.index[j.name] < 12:
                angle = leg_q[:, self.index[j.name]]
                upper = torch.cat((_torch_rotation(self.axes[i], angle), torch.zeros(n, 3, 1, device=x.device, dtype=x.dtype)), dim=-1)
                bottom = torch.tensor((0., 0., 0., 1.), device=x.device, dtype=x.dtype).expand(n, 1, 4)
                move = torch.cat((upper, bottom), dim=-2)
                frames[j.child] = joint_frame @ move
            else:
                # wheel link origin 与沿轴惯性质心不依赖轮角。
                frames[j.child] = joint_frame
        com_sum = torch.zeros(n, 3, device=x.device, dtype=x.dtype)
        jac_terms = {name: torch.zeros_like(com_sum) for name in JOINT_NAMES}
        for i, link in enumerate(self.link_names):
            point = frames[link][:, :3, :3] @ self.inertial[i] + frames[link][:, :3, 3]
            mass = self.masses[i]
            com_sum = com_sum + mass * point
            for name in self.ancestors[link]:
                jac_terms[name] = jac_terms[name] + mass * torch.linalg.cross(axes[name], point - pivots[name])
        total_mass = self.masses.sum()
        com = com_sum / total_mass
        jac = torch.stack([jac_terms[name] / total_mass for name in JOINT_NAMES], dim=-1)
        wheels = torch.stack([frames[f"{leg}_wheel_link"][:, :3, 3] for leg in SUPPORT], dim=1)
        wheel_jacs = []
        for k, leg in enumerate(SUPPORT):
            ancestors = set(self.ancestors[f"{leg}_wheel_link"])
            terms = [torch.linalg.cross(axes[name], wheels[:, k] - pivots[name]) if name in ancestors
                     else torch.zeros_like(com) for name in JOINT_NAMES]
            wheel_jacs.append(torch.stack(terms, dim=-1))
        wheel_jac = torch.stack(wheel_jacs, dim=1)
        wheel_axis = torch.stack([axes[f"{leg}_wheel_joint"] for leg in SUPPORT], dim=1)
        wheel_angular_jacs = []
        for leg in SUPPORT:
            ancestors = set(self.ancestors[f"{leg}_wheel_link"])
            terms = [axes[name] if name in ancestors else torch.zeros_like(com) for name in JOINT_NAMES]
            wheel_angular_jacs.append(torch.stack(terms, dim=-1))
        wheel_angular_jac = torch.stack(wheel_angular_jacs, dim=1)
        lateral = wheels[:, 1] - wheels[:, 0]
        forward = torch.nn.functional.normalize(torch.linalg.cross(lateral, up), dim=-1)
        gap = ((com - wheels.mean(1)) * forward).sum(-1)
        relative = com[:, None] - wheels
        relative_jac = jac[:, None] - wheel_jac
        relative_velocity = torch.linalg.cross(omega[:, None], relative) + torch.einsum("nrcj,nj->nrc", relative_jac, qdot)
        front_rate = qdot[:, 12:14]
        wheel_omega = omega[:, None] + torch.einsum("nrcj,nj->nrc", wheel_angular_jac, qdot)
        radial_up = up[:, None] - (up[:, None] * wheel_axis).sum(-1, keepdim=True) * wheel_axis
        radial_up = torch.nn.functional.normalize(radial_up, dim=-1)
        rolling = RADIUS_M * (torch.linalg.cross(wheel_omega, radial_up) * forward[:, None]).sum(-1)
        full_vertical_rolling = RADIUS_M * (torch.linalg.cross(wheel_omega, up[:, None]) * forward[:, None]).sum(-1)
        spin_only_rolling = RADIUS_M * front_rate * (torch.linalg.cross(wheel_axis, up[:, None]) * forward[:, None]).sum(-1)
        per_front = rolling + (relative_velocity * forward[:, None]).sum(-1)
        offset = (relative_velocity * forward[:, None]).sum(-1)
        return dict(gap_m=gap, velocity_m_s=per_front.mean(1), front_velocity_m_s=per_front,
                    velocity_full_vertical_m_s=(full_vertical_rolling + offset).mean(1),
                    velocity_spin_only_m_s=(spin_only_rolling + offset).mean(1),
                    front_wheel_rate_rad_s=front_rate, forward_body=forward,
                    com_body_m=com, front_wheel_body_m=wheels)
