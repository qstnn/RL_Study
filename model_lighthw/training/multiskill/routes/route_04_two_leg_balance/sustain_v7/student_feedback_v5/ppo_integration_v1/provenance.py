"""Archive and reject drift of all new PPO entrypoints and their live dependencies."""

from __future__ import annotations

import shutil
import sys
import os
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
STRUCTURED = HERE.parent / "structured_v1"
if str(STRUCTURED) not in sys.path:
    sys.path.insert(0, str(STRUCTURED))
from common_feedback import REPO, read, require, sha, snapshot, write  # noqa: E402


def dependency_paths():
    lab = Path(os.environ.get("ISAACLAB_PATH", REPO.parent / "Isaaclab")) / "source"
    rsl = importlib.util.find_spec("rsl_rl")
    if rsl is None or not rsl.submodule_search_locations:
        raise FileNotFoundError("Active rsl_rl package not found")
    package = Path(next(iter(rsl.submodule_search_locations))).resolve()
    return {
        **{str(p.relative_to(REPO)): p for p in sorted(HERE.rglob("*"))
           if p.is_file() and p.suffix in (".py", ".sh", ".md")
           and not {"runs", "reviews"}.intersection(p.parts)},
        str((HERE.parent / "ppo_design" / "continuing_driver.py").relative_to(REPO)):
            HERE.parent / "ppo_design" / "continuing_driver.py",
        "external/rsl_rl/algorithms/ppo.py": package / "algorithms" / "ppo.py",
        "external/rsl_rl/modules/actor_critic.py": package / "modules" / "actor_critic.py",
        "external/rsl_rl/storage/rollout_storage.py": package / "storage" / "rollout_storage.py",
        "external/isaaclab_rl/rsl_rl/vecenv_wrapper.py":
            lab / "isaaclab_rl" / "isaaclab_rl" / "rsl_rl" / "vecenv_wrapper.py",
        "external/isaaclab/envs/direct_rl_env.py":
            lab / "isaaclab" / "isaaclab" / "envs" / "direct_rl_env.py",
    }


def manifest():
    paths = dependency_paths()
    missing = [key for key, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing PPO dependency: " + ", ".join(missing))
    return {key: sha(path) for key, path in paths.items()}


def archive(run):
    run = Path(run)
    snapshot(run)  # Includes all frozen R4a source, student sources, geometry and robot assets.
    paths = dependency_paths()
    hashes = manifest()
    for key, path in paths.items():
        destination = run / "source_snapshot" / key
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    write(run / "ppo_dependency_manifest.json", hashes)
    check(run)


def check(run):
    run = Path(run)
    require(run)
    saved = read(run / "ppo_dependency_manifest.json")
    current = manifest()
    if saved != current:
        raise RuntimeError("PPO source/dependency drift; old evidence is not current")
    for key, digest in saved.items():
        if sha(run / "source_snapshot" / key) != digest:
            raise RuntimeError("Archived PPO dependency mismatch: " + key)
    return saved
