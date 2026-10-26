"""Read-only PPO Actor loading and isolated evaluation provenance."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
V5 = HERE.parent
STRUCTURED = V5 / "structured_v1"
PPO = V5 / "ppo_integration_v1"
for directory in (STRUCTURED, PPO):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from common_feedback import (  # noqa: E402
    REPO, URDF, BodyEstimator, read, write, sha, now,
    snapshot as snapshot_feedback, require as require_feedback,
    load_actor as load_bc_actor,
)
from model_feedback import FeedbackActor  # noqa: E402
from provenance import check as check_ppo_provenance  # noqa: E402

VERSION = "r4a_ppo_autonomous_evaluation_v1"
OBSERVATION_CONTRACT = "r4a_raw305_memory2_trigger16_fk2_v1"


def sources() -> dict[str, str]:
    files = sorted(HERE.glob("*.py")) + sorted(HERE.glob("*.sh"))
    return {str(path.relative_to(REPO)): sha(path) for path in files}


def contract() -> dict:
    return dict(version=VERSION, actor_kind="full_ppo_checkpoint_mean_actor",
        actor_input_dim=325, raw_environment_input_dim=305, critic_input_dim=88, actions=16,
        feedback="same causal raw305 feature builder as structured_v1 and PPO training",
        teacher="never constructed", stochastic_sampling=False,
        physics="original sustain_v7 two-slot reset, 1kHz, no automatic return",
        acceptance="independent verify_sustain 1kHz gates; 10/60/300 s held separately")


def _within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def load_ppo_actor(checkpoint: Path, training_run: Path):
    """Load actor.* exactly; BC artifact only supplies the reviewed topology."""
    checkpoint, training_run = Path(checkpoint).resolve(), Path(training_run).resolve()
    if not checkpoint.is_file() or not _within(checkpoint, training_run / "checkpoints"):
        raise ValueError("PPO checkpoint must exist inside training-run/checkpoints")
    if not (training_run / "request.json").is_file() or not (training_run / "result.json").is_file():
        raise ValueError("Training run request/result missing")
    check_ppo_provenance(training_run)
    request, result = read(training_run / "request.json"), read(training_run / "result.json")
    if request["mode"] != "train" or result["mode"] != "train" or not result["technical_pass"]:
        raise ValueError("Checkpoint source is not a technically complete formal training run")
    if (request["num_envs"] != 16 or request["steps_per_env"] != 32
            or request["technical_time_limit_s"] != 360.
            or not request["no_observation_noise"] or not request["no_reset_randomization"]
            or request["stop_on_observation"] or request["is_finite_horizon"]):
        raise ValueError("Training source physics/observation contract changed")
    if request["actor_sha256"] != result["actor_sha256"] or sha(request["actor"]) != request["actor_sha256"]:
        raise ValueError("Source structured BC actor hash differs")
    checkpoint_sha = sha(checkpoint)
    if checkpoint == Path(result["final_checkpoint"]).resolve() and checkpoint_sha != result["final_checkpoint_sha256"]:
        raise ValueError("Final checkpoint differs from training result")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    expected_config = dict(actor_obs_dim=325, critic_obs_dim=88, num_actions=16)
    if (saved["schema"] != "r4a_unconnected_ppo_design_v1"
            or saved["observation_contract"] != OBSERVATION_CONTRACT
            or saved["policy_contract"] != "structured_feedback_bc_" + request["actor_sha256"]
            or any(saved["config"].get(key) != value for key, value in expected_config.items())
            or saved["num_envs"] != 16 or saved["physics_state_saved"] is not False
            or saved["new_physics_episode_on_process_resume"] is not True):
        raise ValueError("PPO checkpoint policy/observation or physical-state contract changed")
    update_count = int(saved["counters"]["completed_updates"])
    if checkpoint.name != f"model_{update_count:06d}.pt" or update_count > result["completed_updates"]:
        raise ValueError("Checkpoint name/update counter is not in the formal training run")
    if checkpoint == Path(result["final_checkpoint"]).resolve() and update_count != result["completed_updates"]:
        raise ValueError("Final checkpoint update count differs from formal result")
    bc_actor, bc_value = load_bc_actor(request["actor"])
    actor = FeedbackActor(bc_actor.prefix, bc_value["variant"]).eval()
    policy_state = saved["model_state_dict"]
    actor_state = {key.removeprefix("actor."): value for key, value in policy_state.items()
                   if key.startswith("actor.")}
    if set(actor_state) != set(actor.state_dict()):
        raise ValueError("PPO actor tensor keys do not strictly match 325-input FeedbackActor")
    if not all(torch.isfinite(value).all() for value in policy_state.values()):
        raise ValueError("PPO checkpoint contains nonfinite policy tensor")
    for key, value in bc_actor.prefix.state_dict().items():
        if not torch.equal(actor_state["prefix." + key], value):
            raise ValueError("PPO checkpoint changed the frozen rising prefix")
    if not torch.equal(actor_state["mask"], bc_actor.mask):
        raise ValueError("PPO checkpoint changed the structured capture mask")
    actor.load_state_dict(actor_state, strict=True)
    metadata = dict(checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_sha,
        checkpoint_updates=update_count, training_run=str(training_run),
        training_request_sha256=sha(training_run / "request.json"),
        training_result_sha256=sha(training_run / "result.json"),
        training_source_manifest_sha256=sha(training_run / "source_manifest.json"),
        training_dependency_manifest_sha256=sha(training_run / "ppo_dependency_manifest.json"),
        source_bc_actor=str(Path(request["actor"]).resolve()),
        source_bc_actor_sha256=request["actor_sha256"], variant=bc_value["variant"],
        observation_contract=saved["observation_contract"],
        policy_contract=saved["policy_contract"], mean_action_only=True)
    return actor, metadata


def snapshot(run: Path, metadata: dict):
    run = Path(run)
    snapshot_feedback(run)
    for relative in sources():
        destination = run / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / relative, destination)
    write(run / "evaluation_sources.json", sources())
    write(run / "evaluation_contract.json", contract())
    training = Path(metadata["training_run"])
    archive = run / "training_evidence"
    archive.mkdir()
    for name in ("request.json", "result.json", "source_manifest.json", "ppo_dependency_manifest.json"):
        shutil.copy2(training / name, archive / name)


def require(run: Path):
    run = Path(run).resolve()
    require_feedback(run)
    if read(run / "evaluation_sources.json") != sources() or read(run / "evaluation_contract.json") != contract():
        raise ValueError("Evaluation source/contract drift")
    for relative, digest in sources().items():
        if sha(run / "source_snapshot" / relative) != digest:
            raise ValueError("Archived evaluation source differs: " + relative)
    request = read(run / "request.json")
    actor, metadata = load_ppo_actor(request["checkpoint"], request["training_run"])
    if metadata != request["ppo_source"]:
        raise ValueError("Evaluation PPO source metadata differs")
    archive = run / "training_evidence"
    for name, key in (("request.json", "training_request_sha256"),
                      ("result.json", "training_result_sha256"),
                      ("source_manifest.json", "training_source_manifest_sha256"),
                      ("ppo_dependency_manifest.json", "training_dependency_manifest_sha256")):
        if sha(archive / name) != metadata[key]:
            raise ValueError("Archived training evidence differs: " + name)
    return actor, metadata
