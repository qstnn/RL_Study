"""Isolated passive R1 trailer geometry; the V6 asset and its contract stay intact."""

from __future__ import annotations

import math
from pathlib import Path
import xml.etree.ElementTree as ET


CART_ROOT_HEIGHT_M = 0.30
CART_BASE_HALF_HEIGHT_M = 0.08
CART_BODY_MASS_KG = 36.0
WHEEL_MASS_KG = 1.0
WHEEL_WIDTH_M = 0.06
WHEEL_CENTERS_XY_M = ((0.25, 0.31), (0.25, -0.31), (-0.25, 0.31), (-0.25, -0.31))


def replace_passive_wheels(urdf_path: Path, radius_m: float) -> dict:
    """Keep all four wheel contact bottoms at the original plane and total mass at 40 kg."""
    if not math.isfinite(radius_m) or not 0.14 <= radius_m <= 0.22:
        raise ValueError("Passive wheel radius must lie in [0.14, 0.22] m")
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    if root.get("name") != "r1_passive_cart":
        raise ValueError("Unexpected trailer asset")
    base = root.find("./link[@name='cart_base']")
    if base is None or float(base.find("./inertial/mass").get("value")) != CART_BODY_MASS_KG:
        raise ValueError("Expected original 36 kg cart body")
    axle_z = radius_m - CART_ROOT_HEIGHT_M
    for index, (x, y) in enumerate(WHEEL_CENTERS_XY_M):
        name = f"cart_wheel_{index}"
        link = root.find(f"./link[@name='{name}']")
        joint = root.find(f"./joint[@name='{name}_joint']")
        if link is None or joint is None:
            raise ValueError(f"Missing {name} wheel/joint")
        if float(link.find("./inertial/mass").get("value")) != WHEEL_MASS_KG:
            raise ValueError(f"Unexpected {name} mass")
        for kind in ("visual", "collision"):
            cylinder = link.find(f"./{kind}/geometry/cylinder")
            if cylinder is None or float(cylinder.get("length")) != WHEEL_WIDTH_M:
                raise ValueError(f"Unexpected {name} wheel geometry")
            cylinder.set("radius", str(radius_m))
        joint.find("./origin").set("xyz", f"{x} {y} {axle_z}")
        transverse = WHEEL_MASS_KG * (3 * radius_m**2 + WHEEL_WIDTH_M**2) / 12
        spin = WHEEL_MASS_KG * radius_m**2 / 2
        inertia = link.find("./inertial/inertia")
        for key, value in (("ixx", transverse), ("iyy", spin), ("izz", transverse)):
            inertia.set(key, str(value))
        if joint.get("type") != "continuous" or float(joint.find("./limit").get("effort")) != 0:
            raise ValueError(f"Expected passive {name} wheel joint")
    tree.write(urdf_path, encoding="utf-8", xml_declaration=True)
    return validate_passive_trailer(urdf_path, radius_m)


def validate_passive_trailer(urdf_path: Path, expected_radius_m: float) -> dict:
    root = ET.parse(urdf_path).getroot()
    links = root.findall("./link")
    joints = root.findall("./joint")
    if len(links) != 5 or len(joints) != 4:
        raise ValueError("Trailer topology must remain one body and four passive wheels")
    masses = [float(link.find("./inertial/mass").get("value")) for link in links]
    if not math.isclose(sum(masses), 40.0, abs_tol=1e-9):
        raise ValueError("Trailer total mass changed")
    radii = [float(link.find("./collision/geometry/cylinder").get("radius")) for link in links[1:]]
    axle_z = [float(value) for joint in joints for value in joint.find("./origin").get("xyz").split()][2::3]
    if any(not math.isclose(r, expected_radius_m, abs_tol=1e-9) for r in radii):
        raise ValueError("Wheel collision radius mismatch")
    if any(not math.isclose(CART_ROOT_HEIGHT_M + z - expected_radius_m, 0.0, abs_tol=1e-9) for z in axle_z):
        raise ValueError("Wheel bottom is not tangent to the initial terrain plane")
    if any(joint.get("type") != "continuous" for joint in joints):
        raise ValueError("Trailer wheel joint is not passive")
    if any(float(joint.find("./limit").get("effort")) != 0 for joint in joints):
        raise ValueError("Trailer has a powered joint")
    return {
        "total_mass_kg": sum(masses),
        "body_mass_kg": masses[0],
        "wheel_mass_kg_each": masses[1],
        "wheel_radius_m": expected_radius_m,
        "wheel_width_m": WHEEL_WIDTH_M,
        "cart_root_normal_height_at_reset_m": CART_ROOT_HEIGHT_M,
        "wheel_axle_normal_height_at_reset_m": expected_radius_m,
        "wheel_bottom_normal_height_at_reset_m": 0.0,
        "cart_box_bottom_normal_height_at_reset_m": CART_ROOT_HEIGHT_M - CART_BASE_HALF_HEIGHT_M,
        "wheel_joint_effort_Nm": 0.0,
        "interpretation": "Four freely rotating cylindrical wheels; no propulsion, no reduction in 40 kg weight or 25 degree gravity load.",
    }
