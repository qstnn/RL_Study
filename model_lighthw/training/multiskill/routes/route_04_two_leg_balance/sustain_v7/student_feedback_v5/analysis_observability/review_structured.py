"""Read-only independent review of the frozen structured capture-head fit."""
import sys
sys.dont_write_bytecode = True
from datetime import datetime
from zoneinfo import ZoneInfo
import importlib.util
from pathlib import Path
import hashlib
import json
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SUSTAIN = HERE.parents[1]
ROOT = SUSTAIN / "student_feedback_v5/structured_v1"
REPO = SUSTAIN.parents[5]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def main():
    torch.set_num_threads(2)
    run = ROOT / "runs/2026-10-23_full_01"
    report = read(run / "result.json")
    checkpoint = torch.load(run / "actor.pt", map_location="cpu", weights_only=False)
    prefix_path = SUSTAIN / "student_memory_v4/runs/2026-10-23_exposed/actor.pt"
    prefix = torch.load(prefix_path, map_location="cpu", weights_only=False)
    data_path = SUSTAIN / "student_memory_v4/datasets/2026-10-23_memory/dataset.npz"
    data_meta = read(data_path.with_suffix(".json"))
    dataset_sha = sha(data_path)
    assert dataset_sha == checkpoint["dataset_sha256"] == prefix["dataset_sha256"] == report["dataset_sha256"] == data_meta["dataset_sha256"]
    assert checkpoint["prefix_sha256"] == sha(prefix_path)
    assert sha(run / "actor.pt") == report["actor_sha256"]
    assert checkpoint["contract"] == read(run / "feedback_contract.json")
    assert checkpoint["sources"] == read(run / "feedback_sources.json")
    for path,digest in checkpoint["sources"].items():
        assert sha(REPO / path) == sha(run / "source_snapshot" / path) == digest, path
    state = checkpoint["actor_state_dict"]
    assert all(torch.equal(state["prefix."+key],value) for key,value in prefix["actor_state_dict"].items())
    assert all(torch.isfinite(value).all() for value in state.values())
    coefficient = state["coefficients"].numpy()
    mask = state["mask"].numpy()
    assert coefficient.shape == mask.shape == (16,6)
    assert not np.any(coefficient[mask==0])
    with np.load(data_path) as z:
        selected = z["teacher_executed"] & (z["capture_memory"][:,1]>=1.)
        raw, actions, stage = z["observation"][selected], z["action"][selected], z["stage"][selected]
        assert np.isfinite(raw).all() and np.isfinite(actions).all()
        selected_memory = z["capture_memory"][selected]
        assert (selected_memory == 1.).all()
        episodes = np.unique(z["episode"][selected],return_counts=True)
    estimator_path = SUSTAIN / "student_feedback_v5/kinematics/body_estimator.py"
    spec = importlib.util.spec_from_file_location("structured_audit_kinematics",estimator_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    estimated = module.BodyEstimator(REPO / "model_lighthw/lighthw_urdf/urdf/lighthw.urdf").estimate(raw)
    f = raw[:,-61:].astype(np.float64)
    gravity, omega = f[:,3:6], f[:,:3]/.2
    derivative = -np.cross(omega,gravity)
    pitch = np.arctan2(gravity[:,0],-gravity[:,2])
    rate = (-gravity[:,2]*derivative[:,0]+gravity[:,0]*derivative[:,2])/np.maximum(gravity[:,0]**2+gravity[:,2]**2,1e-6)
    gap, velocity = estimated["gap_m"], estimated["velocity_m_s"]
    basis = np.stack((np.ones(len(raw)),gap,velocity,pitch,rate,gap.clip(-.12,.12)),-1)
    weights = np.zeros(len(stage))
    for s in np.unique(stage):
        weights[stage==s] = 1./np.sum(stage==s)
    weights /= weights.mean()
    reconstructed = np.zeros((16,6))
    fits=[]
    for j in range(16):
        cols = np.flatnonzero(mask[j])
        valid = abs(actions[:,j])<.999
        a,b,w = basis[valid][:,cols],actions[valid,j].astype(np.float64),weights[valid]
        scale = np.sqrt((a*a*w[:,None]).sum(0)/w.sum()).clip(1e-6)
        aw,bw = a/scale*np.sqrt(w[:,None]),b*np.sqrt(w)
        reconstructed[j,cols] = np.linalg.lstsq(aw,bw,rcond=1e-8)[0]/scale
        fit = report["fits"][j]
        assert fit["features"] == cols.tolist() and fit["fit_frames"] == int(valid.sum())
        fits.append(dict(action=j,rank=int(np.linalg.matrix_rank(aw)),columns=len(cols),condition=float(np.linalg.cond(aw)),
                         fit_frames=int(valid.sum()),saturated_excluded=int((~valid).sum())))
    np.testing.assert_allclose(reconstructed,np.asarray(report["coefficients"]),atol=1e-9,rtol=1e-8)
    np.testing.assert_allclose(coefficient,reconstructed,atol=4e-7,rtol=1e-7)
    error=(basis@reconstructed.T).clip(-1,1)-actions
    mse=float(np.mean(error**2)); rear=float(np.mean(abs(error[:,14:16])))
    assert abs(mse-report["fit_mse"])<1e-12
    assert abs(rear-report["fit_rear_wheel_mae"])<1e-10
    result=dict(created_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
        kind="independent_read_only_structured_fit_review",fit_run=str(run),fit_result_sha256=sha(run/"result.json"),
        actor_sha256=sha(run/"actor.pt"),prefix_sha256=sha(prefix_path),dataset_sha256=dataset_sha,
        source_snapshot_hashes_verified=len(checkpoint["sources"]),root_files_modified=0,
        dataset_hash_guard="bound transitively through unchanged sealed prefix checkpoint.dataset_sha256 and dataset metadata; independently verified here",
        dimensions=dict(raw=305,memory=2,latched_previous_executed_action=16,proprioceptive_estimate=2,actor=325,critic=88,action=16),
        previous_action_latch="trigger raw last-frame indices34:50; blend u=0 repeats last executed action; no old capture-head output used as latch",
        selected_frames=int(selected.sum()),selected_teacher_only=True,blend_complete_only=True,
        stage_counts={str(s):int(np.sum(stage==s)) for s in np.unique(stage)},
        per_episode_frames={str(i):int(n) for i,n in zip(*episodes)},
        stage_weight_sums={str(s):float(weights[stage==s].sum()) for s in np.unique(stage)},
        original_prefix_tensors_unchanged=True,prefix_tensor_count=len(prefix["actor_state_dict"]),
        coefficient_trainable_storage_count=int(coefficient.size),effective_masked_coefficient_count=int(mask.sum()),
        all_coefficients_finite=True,all_fits_full_column_rank=all(f["rank"]==f["columns"] for f in fits),
        fits=fits,coefficient_reconstruction_max_error=float(abs(coefficient-reconstructed).max()),
        fit_mse=mse,fit_rear_wheel_mae=rear,fit_max_error=float(abs(error).max()),
        weighted_fit_mse=float(np.sum((error**2).mean(-1)*weights)/weights.sum()),
        stage_mse={str(s):float(np.mean(error[stage==s]**2)) for s in np.unique(stage)},
        learned_rear_wheel_basis_coefficients=reconstructed[14:].tolist(),
        observations=["All 6980 selected labels were actual executed teacher frames in successful source traces; no new labels or teacher parameter changes.",
                      "96 is coefficient tensor storage/trainable count; mask exposes only26 effective coefficients in full variant.",
                      "Input basis is proprioceptive; true COM/velocity/contact remain 1kHz acceptance truth only.",
                      "Training fit and low imitation error do not establish autonomous capture or PPO admission.",
                      "Head masking and trigger/physical geometry are structural engineering choices; affine numerical coefficients were learned from executed labels."],
        simulator_runs=0,robot_ppo_updates=0,ppo_admission=False,review_script_sha256=sha(__file__))
    (HERE / "structured_review.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    print(json.dumps({key:result[key] for key in ["selected_frames","stage_counts","effective_masked_coefficient_count","all_fits_full_column_rank","fit_mse","fit_rear_wheel_mae","coefficient_reconstruction_max_error"]},indent=2))


if __name__ == '__main__':
    main()
