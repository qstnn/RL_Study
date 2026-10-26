"""CPU-only state-estimation diagnostic; no action supervision or simulator imports.

Train on the unchanged 7726-frame memory-v4 data, hold out a complete source
trace (both slots) at a time, and score failed autonomous v4 traces as OOD.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from datetime import datetime
from zoneinfo import ZoneInfo

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
import torch

HERE = Path(__file__).resolve().parent
SUSTAIN = HERE.parents[1]
MEMORY = SUSTAIN / "student_memory_v4"
DATA = MEMORY / "datasets/2026-10-23_memory"
TARGET_NAMES = ("signed_gap_m", "forward_com_velocity_m_s")
SIGN_THRESHOLDS = (.005, .02)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def memory_from_raw(raw):
    """Independent causal reconstruction, matching deployed 40-control-step latch."""
    out = np.zeros((*raw.shape[:2], 2), np.float32)
    for slot in range(raw.shape[1]):
        trigger = None
        for k, frame in enumerate(raw[:, slot, -61:]):
            if frame[55] >= .5 or frame[60] <= .5:
                trigger = None
            elif trigger is None and np.arctan2(frame[3], -frame[5]) <= np.deg2rad(-5.):
                trigger = k
            if trigger is not None:
                u = min((k - trigger) / 40., 1.)
                out[k, slot] = (1., u * u * (3. - 2. * u))
    return out


def load_trace(path, capture_times=None):
    """Align policy input k to physical sample k*20-1; k=0 has no label."""
    with np.load(path) as z:
        raw = z["policy_raw_observation" if "policy_raw_observation" in z else "policy_observation"].copy()
        active = z["policy_active"].astype(bool)
        mem = memory_from_raw(raw)
        pcount, slots = raw.shape[:2]
        indices = np.arange(pcount) * 20 - 1
        present = indices >= 0
        clipped = indices.clip(0)
        assert indices[-1] < len(z["com"])
        g = z["gravity"][clipped]
        velocity_body = z["ang_vel"][clipped]
        np.testing.assert_allclose(raw[1:, :, -58:-55][active[1:]], g[1:][active[1:]], atol=1e-6, rtol=0)
        np.testing.assert_allclose(raw[1:, :, -61:-58][active[1:]], (.2 * velocity_body[1:])[active[1:]], atol=1e-6, rtol=0)
        np.testing.assert_allclose(z["policy_physics_time"][1:][active[1:]], z["t"][clipped][1:][active[1:]], atol=2e-6, rtol=0)
        wheels = z["wheel_pos"][clipped, :, 2:, :2]
        lateral = wheels[:, :, 1] - wheels[:, :, 0]
        forward = np.stack((lateral[..., 1], -lateral[..., 0]), -1)
        norm = np.maximum(np.linalg.norm(forward, axis=-1, keepdims=True), 1e-6)
        forward = forward / norm
        gap = ((z["com"][clipped, :, :2] - wheels.mean(2)) * forward).sum(-1)
        speed = (z["com_vel"][clipped, :, :2] * forward).sum(-1)
        target = np.stack((gap, speed), -1)
        # Independent Torch transcription verifies target axis/sign and normalization.
        tw = torch.from_numpy(wheels)
        tl = tw[:, :, 1] - tw[:, :, 0]
        tf = torch.stack((tl[..., 1], -tl[..., 0]), -1)
        tf = tf / tf.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        tg = ((torch.from_numpy(z["com"][clipped, :, :2]) - tw.mean(2)) * tf).sum(-1)
        tv = (torch.from_numpy(z["com_vel"][clipped, :, :2]) * tf).sum(-1)
        np.testing.assert_allclose(target, torch.stack((tg, tv), -1).numpy(), atol=2e-6, rtol=2e-5)
        good = active & present[:, None] & z["active"][clipped] & ~z["failed"][clipped]
        task_t = z["policy_time"].copy()
        policy_t = z["policy_physics_time"].copy()
        if capture_times is None:
            stage = np.where(task_t < 1, 0, np.where(task_t < 3.2, 1, 2))
        else:
            cap = np.asarray(capture_times)[None]
            stage = np.where(task_t < 1, 0, np.where(task_t < 3.2, 1, np.where(policy_t < cap, 2, 3)))
        if "policy_capture_memory" in z:
            np.testing.assert_allclose(mem[active], z["policy_capture_memory"][active], atol=3e-7, rtol=0)
    return dict(raw=raw, memory=mem, target=target, good=good, stage=stage,
                policy_time=policy_t, indices=indices, trace_sha256=sha(path))


def scores(y, pred, mask):
    if not np.any(mask):
        return dict(n=0)
    truth, guessed = y[mask], pred[mask]
    out = dict(n=int(mask.sum()))
    for j, name in enumerate(TARGET_NAMES):
        err = np.abs(truth[:, j] - guessed[:, j])
        nonzero = np.abs(truth[:, j]) > SIGN_THRESHOLDS[j]
        any_nonzero = np.abs(truth[:, j]) > 1e-6
        out[name] = dict(mae=float(err.mean()), p95=float(np.quantile(err, .95)), max=float(err.max()),
            rmse=float(np.sqrt(np.mean(err * err))), target_abs_p95=float(np.quantile(abs(truth[:, j]), .95)),
            sign_threshold=SIGN_THRESHOLDS[j], nonzero_n=int(nonzero.sum()),
            nonzero_direction_error=float(np.mean(np.sign(truth[nonzero, j]) != np.sign(guessed[nonzero, j]))) if nonzero.any() else None,
            exact_nonzero_n=int(any_nonzero.sum()),
            exact_nonzero_direction_error=float(np.mean(np.sign(truth[any_nonzero, j]) != np.sign(guessed[any_nonzero, j]))) if any_nonzero.any() else None)
    return out


def by_stage(y, pred, stage):
    return {name: scores(y, pred, mask) for name, mask in (
        ("all", np.ones(len(y), bool)), ("entry", stage == 0), ("rise", stage == 1),
        ("capture", stage == 2), ("transition", (stage == 1) | (stage == 2)), ("post_capture", stage == 3))}


def fit_predict(kind, x, y, stage, targets, steps):
    """Normalization is fitted on training trajectories only; fixed hyperparameters."""
    mean, scale = x.mean(0), np.maximum(x.std(0), 1e-3)
    xx = (x - mean) / scale
    tx = [(v - mean) / scale for v in targets]
    if kind == "ridge":
        model = Ridge(alpha=1., solver="cholesky")
        model.fit(xx.astype(np.float64), y.astype(np.float64))
        return [model.predict(v).astype(np.float32) for v in tx]
    if kind == "knn":
        model = KNeighborsRegressor(n_neighbors=5, weights="distance", algorithm="brute", n_jobs=1)
        model.fit(xx, y)
        return [model.predict(v).astype(np.float32) for v in tx]
    torch.manual_seed(42)
    model = torch.nn.Sequential(torch.nn.Linear(x.shape[1], 128), torch.nn.ELU(),
                                torch.nn.Linear(128, 64), torch.nn.ELU(), torch.nn.Linear(64, 2))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    ym, ys = y.mean(0), np.maximum(y.std(0), 1e-3)
    xt, yt = torch.from_numpy(xx), torch.from_numpy((y - ym) / ys)
    groups = [torch.from_numpy(np.flatnonzero(stage == s)) for s in range(4) if np.any(stage == s)]
    for _ in range(steps):
        ids = torch.cat([g[torch.randint(len(g), (64,))] for g in groups])
        loss = (model(xt[ids]) - yt[ids]).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
    with torch.no_grad():
        return [(model(torch.from_numpy(v)).numpy() * ys + ym).astype(np.float32) for v in tx]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1500)
    args = parser.parse_args()
    torch.set_num_threads(2)
    start = time.time()
    meta = read(DATA / "dataset.json")
    assert sha(DATA / "dataset.npz") == meta["dataset_sha256"]
    with np.load(DATA / "dataset.npz") as z:
        data = {key: z[key].copy() for key in z.files if key != "action"}
    paths = sorted(set(str(Path(e["run"]) / e["trace"]) for e in meta["episodes"]))
    dataset_group = np.full(len(data["stage"]), -1, int)
    y = np.full((len(dataset_group), 2), np.nan, np.float32)
    usable = np.zeros(len(y), bool)
    traces, checks = [], []
    for group, path in enumerate(paths):
        episodes = [(i, e) for i, e in enumerate(meta["episodes"]) if str(Path(e["run"]) / e["trace"]) == path]
        assert sorted(e["slot"] for _, e in episodes) == [0, 1]
        trace = load_trace(path, [next(e["capture_time"] for _, e in episodes if e["slot"] == slot) for slot in (0, 1)])
        assert trace["trace_sha256"] == episodes[0][1]["trace_sha256"]
        for episode_id, episode in episodes:
            take = data["episode"] == episode_id
            k, slot = data["policy_index"][take], episode["slot"]
            np.testing.assert_array_equal(data["observation"][take], trace["raw"][k, slot])
            np.testing.assert_allclose(data["capture_memory"][take], trace["memory"][k, slot], atol=1e-7, rtol=0)
            dataset_group[take] = group
            y[take] = trace["target"][k, slot]
            usable[take] = trace["good"][k, slot]
        traces.append(trace)
        checks.append(dict(group=group, trace=path, sha256=trace["trace_sha256"], slots_held_out_together=[0, 1],
                           full_policy_frames=int(trace["good"].sum()), dataset_frames=int(np.sum((dataset_group == group) & usable))))
    assert (dataset_group >= 0).all()
    assert np.isfinite(y[usable]).all()
    raw, mem, yy, stage, groups = (data["observation"][usable], data["capture_memory"][usable], y[usable],
                                 data["stage"][usable], dataset_group[usable])
    oods = []
    for mode in ("zero", "exposed"):
        path = MEMORY / f"runs/2026-10-23_{mode}_screen/trace_autonomous_seed43.npz"
        tr = load_trace(path)
        good = tr["good"]
        oods.append(dict(name=mode, path=str(path), sha256=tr["trace_sha256"],
                         raw=tr["raw"][good], memory=tr["memory"][good], target=tr["target"][good],
                         stage=tr["stage"][good], slot=np.broadcast_to(np.arange(2), good.shape)[good],
                         policy_time=tr["policy_time"][good]))
    input_hashes = {str(DATA / "dataset.npz"): sha(DATA / "dataset.npz"),
                    str(DATA / "dataset.json"): sha(DATA / "dataset.json"),
                    str(SUSTAIN / "teacher_sustain.py"): sha(SUSTAIN / "teacher_sustain.py"),
                    str(MEMORY / "capture_memory.py"): sha(MEMORY / "capture_memory.py")}
    input_hashes.update({c["trace"]: c["sha256"] for c in checks})
    input_hashes.update({o["path"]: o["sha256"] for o in oods})
    result = dict(kind="state_observability_diagnostic_not_actor", created_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
        timezone="Asia/Shanghai", dataset=str(DATA), dataset_sha256=meta["dataset_sha256"],
        original_dataset_frames=len(y), scored_dataset_frames=int(usable.sum()),
        unscored_initial_no_previous_physics=int(np.sum(data["policy_index"] == 0)),
        original_source_checks=checks, target_names=TARGET_NAMES,
        alignment="policy k to source physical k*20-1; k=0 excluded; active nonfailed first episode only",
        target_definition="teacher_sustain: normalized forward=[rear_lateral_y,-rear_lateral_x]; signed COM-rear-midpoint dot forward; COM_xy velocity dot same forward",
        split="4 leave-one-source-trace-out folds; both slots held out jointly, no shuffled adjacent-frame split",
        hyperparameters=dict(ridge_alpha=1., knn_k=5, mlp=[128,64], mlp_steps=args.steps,
                             mlp_seed=42, mlp_batch="64 per available stage", standardization="train-fold only; scale floor1e-3"),
        simulator_runs=0, robot_ppo_updates=0, action_labels_generated=0, action_arrays_loaded=False,
        script_sha256=sha(__file__), input_hashes=input_hashes, methods={}, ood_sources=[{k:v for k,v in o.items() if k in ("name","path","sha256")} for o in oods])
    archives = dict(dataset_target=yy, dataset_stage=stage, dataset_group=groups,
                    original_dataset_row=np.flatnonzero(usable))
    for inputs in ("raw305", "raw305_memory307"):
        x = raw if inputs == "raw305" else np.concatenate((raw, mem), -1)
        for kind in ("ridge", "knn", "mlp"):
            name = f"{inputs}_{kind}"
            prediction = np.full_like(yy, np.nan)
            folds = []
            for heldout in range(len(paths)):
                train, test = groups != heldout, groups == heldout
                tr = traces[heldout]
                good = tr["good"]
                full = tr["raw"][good] if inputs == "raw305" else np.concatenate((tr["raw"][good], tr["memory"][good]), -1)
                p, pfull = fit_predict(kind, x[train], yy[train], stage[train], [x[test], full], args.steps)
                prediction[test] = p
                folds.append(dict(heldout_group=heldout, train_groups=sorted(set(groups[train].tolist())),
                                   train_n=int(train.sum()), test_n=int(test.sum()),
                                   dataset=by_stage(yy[test], p, stage[test]),
                                   full_trace=by_stage(tr["target"][good], pfull, tr["stage"][good])))
                archives[f"{name}_heldout{heldout}_full_prediction"] = pfull
                print(f"[{name}] held out trace {heldout} transition {folds[-1]['dataset']['transition']}", flush=True)
            assert np.isfinite(prediction).all()
            archives[name + "_oof_prediction"] = prediction
            targets = [o["raw"] if inputs == "raw305" else np.concatenate((o["raw"], o["memory"]), -1) for o in oods]
            oodpred = fit_predict(kind, x, yy, stage, targets, args.steps)
            oscores = []
            for o, p in zip(oods, oodpred):
                oscores.append(dict(name=o["name"], metrics=by_stage(o["target"], p, o["stage"]),
                                   capture_blend=scores(o["target"], p, (o["stage"] == 2) & (o["memory"][:,1] < 1)),
                                   capture_after_blend=scores(o["target"], p, (o["stage"] == 2) & (o["memory"][:,1] >= 1)),
                                   slots={str(slot): by_stage(o["target"][o["slot"]==slot], p[o["slot"]==slot], o["stage"][o["slot"]==slot]) for slot in (0,1)}))
                archives[f"{name}_ood_{o['name']}_prediction"] = p
                archives[f"ood_{o['name']}_target"] = o["target"]
                archives[f"ood_{o['name']}_stage"] = o["stage"]
                archives[f"ood_{o['name']}_slot"] = o["slot"]
                archives[f"ood_{o['name']}_policy_time"] = o["policy_time"]
            result["methods"][name] = dict(out_of_fold=by_stage(yy, prediction, stage), folds=folds, ood=oscores)
            write(HERE / "summary.json", result)
    result["wall_seconds"] = time.time() - start
    # Preserve only numerical state labels/predictions; no policy or action labels are produced.
    np.savez_compressed(HERE / "predictions.npz", **archives)
    result["predictions_sha256"] = sha(HERE / "predictions.npz")
    result["original_inputs_unchanged"] = all(sha(path) == digest for path,digest in input_hashes.items())
    assert result["original_inputs_unchanged"]
    write(HERE / "summary.json", result)
    print(f"COMPLETE in {result['wall_seconds']:.1f}s; {HERE / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
