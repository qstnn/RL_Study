"""Route lifecycle and provenance, usable without Isaac Sim."""

from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
SOURCE = REPO / "source/rl_training/rl_training/tasks/manager_based/locomotion/multiskill/lighthw"
URDF = REPO / "model_lighthw/lighthw_urdf/urdf/lighthw.urdf"


def module_from_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


catalog = module_from_file("_lighthw_multiskill_catalog", SOURCE / "catalog.py")


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def new_run(route, mode):
    record = catalog.implemented_route(route)
    folder = {"train": "runs", "smoke": "validation", "play": "playbacks", "evaluate": "evaluations"}[mode]
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d_%H-%M-%S_%f")
    path = ROOT / "routes" / record["directory"] / folder / stamp
    path.mkdir(parents=True, exist_ok=False)
    return path


def asset_manifest():
    if sha256(URDF) != catalog.CONTRACT["urdf_sha256"]:
        raise ValueError("Canonical URDF changed; review and version the multi-skill contract")
    tree = ET.parse(URDF)
    joints = {j.attrib["name"]: j for j in tree.findall("joint") if j.attrib["type"] != "fixed"}
    if set(joints) != set(catalog.JOINT_NAMES):
        raise ValueError("Canonical URDF joint layout changed")
    linear = module_from_file("_lighthw_multiskill_linear_spec", SOURCE.parents[1] /
        "velocity/config/wheeled/deeprobotics_lighthw/linear_leg_spec.py")
    for name in catalog.LEG_NAMES:
        bounds = joints[name].find("limit").attrib
        if tuple(linear.LIMITS[name]) != (float(bounds["lower"]), float(bounds["upper"])):
            raise ValueError(f"Action clip no longer matches canonical URDF: {name}")
    for name in catalog.WHEEL_NAMES:
        if float(joints[name].find("limit").attrib["velocity"]) < 37.0:
            raise ValueError(f"Wheel target exceeds URDF velocity limit: {name}")
    meshes = {}
    for mesh in tree.findall(".//mesh"):
        value = mesh.attrib["filename"]
        if value.startswith("package://"):
            value = value.removeprefix("package://").split("/", 1)[1]
            path = URDF.parents[1] / value
        else:
            path = URDF.parent / value
        path = path.resolve()
        if not path.is_relative_to(URDF.parents[1]):
            raise ValueError(f"Unexpected mesh location: {path}")
        relative = str(path.relative_to(REPO))
        meshes[relative] = sha256(path)
        expected_path = ROOT / "records/portable_assets.json"
        if expected_path.exists():
            expected = json.loads(expected_path.read_text())["meshes"]
            if expected.get(relative) != meshes[relative]:
                raise ValueError(f"Mesh missing/corrupt or still a Git LFS pointer: {relative}; run git lfs pull study")
    return {"urdf": str(URDF.relative_to(REPO)), "urdf_sha256": sha256(URDF), "meshes": meshes,
            "mass_kg": sum(float(m.attrib["value"]) for m in tree.findall("link/inertial/mass")),
            "joint_count": len(joints)}


def snapshot(run):
    files = set((REPO / "source/rl_training/rl_training").rglob("*.py"))
    files.update(p for p in ROOT.rglob("*.py") if not any(
        part in ("source_snapshot", "runs", "validation", "evaluations", "playbacks") for part in p.relative_to(ROOT).parts))
    files.add(URDF)
    hashes = {}
    for path in sorted(files):
        relative = path.relative_to(REPO)
        destination = run / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        hashes[str(relative)] = sha256(destination)
    write_json(run / "source_manifest.json", hashes)


def validate_checkpoint(checkpoint):
    """Reject legacy/same-shape foreign policies before creating an environment."""
    import torch
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if (saved.get("infos") or {}).get("multiskill_contract") != catalog.CONTRACT:
        raise ValueError("Checkpoint is not this Route01 Stage1 contract; legacy transfer is not implemented")
    for name, tensor in saved["model_state_dict"].items():
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Non-finite model tensor: {name}")
    return saved
