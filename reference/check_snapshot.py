#!/usr/bin/env python3
"""Validate the text-only Git snapshot without meshes or simulator imports."""
import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tomllib
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "reference/source_manifest.json"
ALLOWED = {".py", ".md", ".json", ".toml", ".yaml", ".yml", ".xml", ".urdf",
           ".sh", ".cfg", ".ini", ".txt", ".lock"}
NAMES = {"LICENSE", "LICENSE-robot_lab", "VERSION", ".gitignore", ".gitattributes"}


def main():
    files = json.loads((ROOT / MANIFEST).read_text(encoding="utf-8"))["files"]
    tracked = set(subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=ROOT).decode().rstrip("\0").split("\0"))
    expected = set(files) | {MANIFEST}
    assert tracked == expected, {"missing": sorted(expected - tracked),
                                 "unexpected": sorted(tracked - expected)}
    layout = json.loads((ROOT / "reference/layout_migration.json").read_text(encoding="utf-8"))
    assert not any(Path(path).parts[0].lower().startswith("r5") for path in tracked)
    for old, new in layout["files"].items():
        assert old not in tracked and new in tracked, (old, new)
    counts = Counter()
    for relative in sorted(tracked):
        path = ROOT / relative
        assert not path.is_symlink(), relative
        assert path.suffix in ALLOWED or path.name in NAMES, relative
        data = path.read_bytes()
        assert b"\0" not in data, relative
        text = data.decode("utf-8")
        if relative in files:
            assert hashlib.sha256(data).hexdigest() == files[relative]["sha256"], relative
        counts[path.suffix or path.name] += 1
        if path.suffix == ".py":
            ast.parse(text, filename=relative)
        elif path.suffix == ".json":
            json.loads(text)
        elif path.suffix == ".toml":
            tomllib.loads(text)
        elif path.suffix in {".urdf", ".xml"}:
            ET.fromstring(text)
        elif path.suffix == ".sh":
            subprocess.run(["bash", "-n", str(path)], check=True)

    profile = json.loads((ROOT / "model_lighthw/lighthw_urdf/model_profile.json").read_text())
    urdf = ROOT / profile["urdf_path"]
    assert hashlib.sha256(urdf.read_bytes()).hexdigest() == profile["urdf_sha256"]
    robot = ET.parse(urdf).getroot()
    links, joints = robot.findall("link"), robot.findall("joint")
    assert len(links) == 17 and len(joints) == 16
    masses = [float(link.find("inertial/mass").attrib["value"]) for link in links]
    assert all(math.isfinite(mass) and mass > 0 for mass in masses)
    assert math.isclose(sum(masses), 22.0, abs_tol=1e-10, rel_tol=0)
    assert profile["robot_mass_kg"] == 22.0
    names = {link.attrib["name"] for link in links}
    children = set()
    edges = {}
    for joint in joints:
        parent = joint.find("parent").attrib["link"]
        child = joint.find("child").attrib["link"]
        assert parent in names and child in names and child not in children
        children.add(child)
        edges.setdefault(parent, []).append(child)
    roots = names - children
    assert len(roots) == 1
    visited, pending = set(), list(roots)
    while pending:
        node = pending.pop()
        assert node not in visited
        visited.add(node)
        pending.extend(edges.get(node, []))
    assert visited == names
    print(json.dumps({"status": "text_snapshot_passed", "files": len(tracked),
                      "file_types": dict(sorted(counts.items())),
                      "urdf_mass_kg": sum(masses), "links": len(links),
                      "joints": len(joints), "simulation_run": False}, indent=2))


if __name__ == "__main__":
    main()
