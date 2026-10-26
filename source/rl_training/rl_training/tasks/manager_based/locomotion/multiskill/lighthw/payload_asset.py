"""生成每次运行独立的载荷 URDF；规范 URDF 和原网格只读。"""
from pathlib import Path
import math
import xml.etree.ElementTree as ET
import numpy as np


def rotation_rpy(rpy):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                     [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr], [-sp, cp*sr, cp*cr]])


def composite_inertia(base_mass, base_com, base_inertia, payload_mass, center, size):
    """合并刚性固定长方体的质量、质心和惯量；返回总质量、合成质心及质心处惯量。

    输入惯量和坐标均在 Base_link 轴系中；使用平行轴定理，不能仅按质量缩放旧惯量。
    """
    values = np.concatenate(([base_mass, payload_mass], np.ravel(base_com), np.ravel(base_inertia), center, size))
    if not np.isfinite(values).all() or base_mass <= 0 or payload_mass < 0 or min(size) <= 0:
        raise ValueError("Invalid composite payload parameters")
    base_com, center, size = np.asarray(base_com), np.asarray(center), np.asarray(size)
    total = base_mass + payload_mass
    com = (base_mass*base_com + payload_mass*center)/total
    payload_i = payload_mass/12.0 * np.diag([size[1]**2+size[2]**2, size[0]**2+size[2]**2, size[0]**2+size[1]**2])
    def shift(mass, delta):
        return mass * (float(delta @ delta)*np.eye(3)-np.outer(delta, delta))
    inertia = base_inertia + shift(base_mass, base_com-com) + payload_i + shift(payload_mass, center-com)
    if np.linalg.eigvalsh(inertia).min() <= 0:
        raise ValueError("Composite inertia is not positive definite")
    return total, com, inertia


def write_payload_urdf(source, destination, mass, center=(0., 0., .20), size=(.30, .25, .20)):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination:
        raise ValueError("Canonical URDF must remain read-only")
    if not math.isfinite(mass) or not 0 <= mass <= 65:
        raise ValueError("Payload must be between 0 and 65 kg")
    tree = ET.parse(source)
    base = tree.getroot().find("link[@name='Base_link']")
    inertial = base.find("inertial")
    origin, it = inertial.find("origin"), inertial.find("inertia")
    c = np.array([float(v) for v in origin.get("xyz", "0 0 0").split()])
    r = rotation_rpy([float(v) for v in origin.get("rpy", "0 0 0").split()])
    i = np.array([[float(it.get(f'i{a}{b}' if a <= b else f'i{b}{a}')) for b in 'xyz'] for a in 'xyz'])
    base_mass = float(inertial.find("mass").get("value"))
    total_before = sum(float(m.get("value")) for m in tree.findall("link/inertial/mass"))
    total, com, tensor = composite_inertia(base_mass, c, r @ i @ r.T, mass, center, size)
    inertial.find("mass").set("value", format(total, '.17g'))
    origin.set("xyz", ' '.join(format(v, '.17g') for v in com)); origin.set("rpy", "0 0 0")
    for row, a in enumerate('xyz'):
        for col, b in enumerate('xyz'):
            if col >= row:
                it.set(f'i{a}{b}', format(tensor[row, col], '.17g'))
    # 运行目录改变后将 mesh 指向当前 checkout；不引用原电脑绝对路径。
    for mesh in tree.findall('.//mesh'):
        value = mesh.get('filename')
        path = source.parents[1] / value.removeprefix('package://').split('/', 1)[1] if value.startswith('package://') else source.parent/value
        path = path.resolve(strict=True)
        if not path.is_relative_to(source.parents[1]):
            raise ValueError(f"Mesh is outside canonical asset: {path}")
        mesh.set('filename', str(path))
    destination.parent.mkdir(parents=True, exist_ok=True)
    tree.write(destination, encoding='utf-8', xml_declaration=True)
    return {"payload_kg": mass, "payload_center_m": list(center), "payload_box_size_m": list(size),
            "base_mass_kg": total, "base_com_m": com.tolist(), "base_inertia_kg_m2": tensor.tolist(),
            "unloaded_robot_mass_kg": total_before, "loaded_robot_mass_kg": total_before+mass,
            "payload_collision_geometry": False, "model": "rigidly attached inertial payload; no thermal model"}
