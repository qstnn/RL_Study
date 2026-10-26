"""R4运行目录、源码和模型追溯；不调用其他路线的训练入口。"""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from zoneinfo import ZoneInfo

ROUTE = Path(__file__).resolve().parents[1]
REPO = ROUTE.parents[5]
SOURCE = REPO / "source/rl_training/rl_training/tasks/manager_based/locomotion/multiskill/lighthw/two_leg_balance_v3"
URDF = REPO / "model_lighthw/lighthw_urdf/urdf/lighthw.urdf"
sys.path.insert(0, str(SOURCE.parent))
from two_leg_balance_v3.spec import contract


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def read(path):
    return json.loads(Path(path).read_text())


def source_files():
    return sorted(list(SOURCE.glob("*.py")) + list((ROUTE / "scripts").glob("*.py"))
                     + [ROUTE / "scripts/r4_v3.sh", REPO / "scripts/tools/lighthw_standing_pose.py"])


def manifest():
    return {str(p.relative_to(REPO)): sha(p) for p in source_files()}


def asset_hashes():
    root = URDF.parents[1]
    return {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob("*"))
            if p.is_file() and p.suffix.lower() in (".urdf", ".stl", ".obj")}


def snapshot(run):
    hashes = manifest()
    for relative in hashes:
        dest = run / "source_snapshot" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / relative, dest)
    write(run / "source_manifest.json", hashes)
    # 每个session保存一份独立资产快照；各段硬链接到快照，绝不链接到可编辑canonical输入。
    asset_root = URDF.parents[1]
    assets = asset_hashes()
    pool = run.parent / "shared_assets"
    pool_manifest = run.parent / "shared_assets_manifest.json"
    if pool_manifest.exists() and read(pool_manifest) != assets:
        raise ValueError("Canonical R4 assets changed during this session")
    for relative, digest in assets.items():
        shared = pool / relative
        shared.parent.mkdir(parents=True, exist_ok=True)
        if not shared.exists():
            shutil.copy2(asset_root / relative, shared)
        if sha(shared) != digest:
            raise ValueError(f"R4 asset snapshot changed: {relative}")
        dest = run / "assets" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        os.link(shared, dest)
    write(pool_manifest, assets)
    write(run / "asset_manifest.json", assets)


def checked_checkpoint(path, expected):
    import torch
    value = torch.load(path, map_location="cpu", weights_only=False)
    info = value.get("infos") or {}
    if info.get("contract") != expected:
        raise ValueError("R4 checkpoint contract mismatch (mode/hold/action/observations); no implicit transfer")
    if info.get("source_manifest") != manifest():
        raise ValueError("R4 executable sources changed; resume requires the checkpoint's exact source version")
    if info.get("asset_manifest") != asset_hashes():
        raise ValueError("R4 checkpoint asset/mesh provenance differs from current canonical inputs")
    for key, tensor in value["model_state_dict"].items():
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Non-finite model tensor: {key}")
    return value
