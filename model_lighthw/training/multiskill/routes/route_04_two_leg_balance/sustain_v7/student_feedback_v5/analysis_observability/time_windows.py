"""Localize state-estimator errors before/after the existing physical departure.

No action labels or simulator; reads numerical predictions and original traces.
"""
from pathlib import Path
import numpy as np
from diagnose import HERE, SUSTAIN, load_trace, read, sha, scores, write


def main():
    summary = read(HERE / "summary.json")
    nominal_path = SUSTAIN / "runs/2026-10-22_sustain_30s_blend800/trace_seed43_blend800.npz"
    with np.load(nominal_path) as z:
        nominal = {k:z[k] for k in ("gravity", "com", "t")}
    with np.load(HERE / "predictions.npz") as z:
        predictions = {k:z[k] for k in z.files}
    out = dict(departure_definition="same physical time gravity difference >5deg OR relative 3D COM displacement difference >3cm versus nominal hybrid; descriptive not causal",
               script_sha256=sha(__file__), nominal_sha256=sha(nominal_path), methods={})
    for ood in summary["ood_sources"]:
        mode, path = ood["name"], Path(ood["path"])
        tr = load_trace(path)
        good = tr["good"]
        truth, tt = tr["target"][good], tr["policy_time"][good]
        slots = np.broadcast_to(np.arange(2), good.shape)[good]
        np.testing.assert_array_equal(truth, predictions[f"ood_{mode}_target"])
        with np.load(path) as z:
            departures = {}
            for slot in (0,1):
                active = z["active"][:,slot].astype(bool)
                g, com, t = z["gravity"][active,slot], z["com"][active,slot], z["t"][active,slot]
                n = min(len(g),len(nominal["gravity"]))
                error = np.rad2deg(np.arccos(np.clip((g[:n] * nominal["gravity"][:n,slot]).sum(-1),-1,1)))
                relative = (com[:n]-com[0])-(nominal["com"][:n,slot]-nominal["com"][0,slot])
                crossing = np.flatnonzero((error>5) | (np.linalg.norm(relative,axis=-1)>.03))
                departures[slot] = float(t[crossing[0]]) if len(crossing) else None
        for name in summary["methods"]:
            pred = predictions[f"{name}_ood_{mode}_prediction"]
            rows=[]
            for slot in (0,1):
                local = slots==slot
                departure = departures[slot]
                capture = local & (tt >= 3.2)
                windows={f"{a:.1f}_{b:.1f}s":scores(truth,pred,local&(tt>=a-1e-6)&(tt<b-1e-6))
                         for a,b in [(3.2,4.4),(4.4,4.7),(4.7,5.0),(4.4,5.0),(5.0,6.0)]}
                windows["before_departure"] = scores(truth,pred,capture&(tt<departure))
                windows["after_departure"] = scores(truth,pred,capture&(tt>=departure))
                sign_bad = capture & (abs(truth[:,1])>.02) & (np.sign(truth[:,1]) != np.sign(pred[:,1]))
                large_error = capture & (abs(truth[:,1]-pred[:,1])>.12)
                indices=np.flatnonzero(sign_bad)
                events=[dict(time_s=float(tt[i]),true_velocity=float(truth[i,1]),estimated_velocity=float(pred[i,1]),
                             relative_departure_s=float(tt[i]-departure), within_4p4_5p0=bool(4.4-1e-6<=tt[i]<5.0-1e-6)) for i in indices]
                first_large=np.flatnonzero(large_error)
                rows.append(dict(slot=slot, first_reference_departure_s=departure,
                                 first_sign_error=events[0] if events else None,sign_error_events=events,
                                 first_velocity_error_gt_0p12_s=float(tt[first_large[0]]) if len(first_large) else None,
                                 windows=windows))
            out["methods"].setdefault(name,{})[mode]=rows
    write(HERE / "time_windows.json",out)
    for name in ["raw305_ridge","raw305_memory307_ridge","raw305_mlp","raw305_memory307_mlp"]:
        print(name)
        for row in out["methods"][name]["exposed"]:
            print('exposed slot',row['slot'],'departure',row['first_reference_departure_s'],
                  'first sign error',row['first_sign_error'],'4.4–5.0',row['windows']['4.4_5.0s'])


if __name__ == '__main__':
    main()
