"""Read-only CPU review of a completed structured-feedback autonomous screen."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import argparse
import hashlib
import json
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SUSTAIN = HERE.parents[1]
STRUCTURED = SUSTAIN / "student_feedback_v5/structured_v1"
sys.path.insert(0,str(STRUCTURED))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run",type=Path,nargs="?",default=STRUCTURED/"runs/2026-10-23_full_01_screen")
    parser.add_argument("--output",type=Path,default=HERE/"screen_review.json")
    args=parser.parse_args();run=args.run.resolve()
    assert args.output.resolve().parent == HERE, "Evidence output must stay in analysis_observability"
    if not (run/"result.json").exists() or not (run/"verification.json").exists():
        print("PENDING: completed result.json and verification.json are required",flush=True)
        return 2
    from verify_feedback import verify
    from verify_sustain import analyse
    from common_feedback import load_actor,require
    torch.set_num_threads(2)
    require(run)
    request,result=read(run/"request.json"),read(run/"result.json")
    expected=read(run/"verification.json")
    recomputed=verify(run)
    assert recomputed==expected, "Read-only full NumPy/actor verification differs"
    actor,value=load_actor(request["actor"])
    assert sha(request["actor"])==request["actor_sha256"]==result["actor_sha256"]
    old_path=SUSTAIN/"student_memory_v4/runs/2026-10-23_exposed_screen/trace_autonomous_seed43.npz"
    with np.load(old_path) as z:
        old={k:z[k] for k in z.files}
    reports=[]
    first_trace=None
    seed_repeats=[]
    for path in sorted(run.glob("trace_*.npz")):
        assert sha(path)==result["trace_hashes"][path.name]
        with np.load(path) as z:
            d={k:z[k] for k in z.files}
        if first_trace is None:
            first_trace=d
        else:
            seed_repeats.append(dict(trace=path.name,
                all_arrays_identical_to_first=all(k in first_trace and np.array_equal(v,first_trace[k]) for k,v in d.items())))
        assert not d["policy_takeover"].any() and not d["policy_teacher_action"].any()
        for case in [c for c in result["cases"] if c["trace"]==path.name]:
            slot=case["trace_env"]
            active=d["active"][:,slot].astype(bool)
            dd={k:v[active,slot] for k,v in d.items() if not k.startswith("policy_")}
            stats,qualified=analyse(dd,case,read(run/"geometry.json"),request["hold_seconds"])
            control=d["policy_active"][:,slot].astype(bool)
            latch=d["policy_capture_memory"][:,slot,0]>.5
            triggered=np.flatnonzero(control & latch)
            assert len(triggered)
            k=int(triggered[0]);trigger_time=float(d["policy_physics_time"][k,slot])
            # The trigger input still reflects the unchanged previous physical prefix.
            raw_equal=np.array_equal(d["policy_raw_observation"][:k+1,slot],old["policy_raw_observation"][:k+1,slot])
            action_equal=np.array_equal(d["policy_action"][:k,slot],old["policy_action"][:k,slot])
            actor_equal=np.array_equal(d["policy_actor_action"][:k,slot],old["policy_actor_action"][:k,slot])
            physical_count=int(round(trigger_time/.001))
            common=[name for name in d if not name.startswith("policy_") and name in old]
            fields=[]
            for name in common:
                if np.array_equal(d[name][:physical_count,slot],old[name][:physical_count,slot]):
                    fields.append(name)
            # Preserve mismatches explicitly: batch shape can change BLAS roundoff.
            mismatches={name:float(np.max(np.abs(d[name][:physical_count,slot].astype(float)-old[name][:physical_count,slot].astype(float))))
                        for name in common if name not in fields}
            assert np.array_equal(d["policy_trigger_action"][k,slot],d["policy_raw_observation"][k,slot,-61+34:-61+50])
            np.testing.assert_allclose(d["policy_action"][k,slot],d["policy_action"][k-1,slot],atol=0,rtol=0)
            wheels=dd["wheel_pos"][:,2:,:2];lateral=wheels[:,1]-wheels[:,0]
            forward=np.stack((lateral[:,1],-lateral[:,0]),-1)
            forward/=np.maximum(np.linalg.norm(forward,axis=-1,keepdims=True),1e-6)
            signed_gap=((dd["com"][:,:2]-wheels.mean(1))*forward).sum(-1)
            forward_speed=(dd["com_vel"][:,:2]*forward).sum(-1)
            speed=np.linalg.norm(dd["com_vel"],axis=-1)
            angular=np.linalg.norm(dd["ang_vel"],axis=-1)
            hf_ratio=(dd["smooth_rms"]/[.02,.05,np.deg2rad(.2),.5]).max(-1)
            tail=np.zeros(len(dd["t"]),bool)
            tail[-min(len(tail),1000):]=True  # exactly1000 samples, independent of float timestamp rounding
            checkpoints=[]
            for when in (3.9,4.4,4.7,5.,6.,8.,10.,float(dd["t"][-1])):
                if when>dd["t"][-1]+1e-5:
                    continue
                i=int(np.argmin(abs(dd["t"]-when)))
                checkpoints.append(dict(t=float(dd["t"][i]),signed_gap_m=float(signed_gap[i]),line_error_m=float(dd["line_error"][i]),
                    forward_com_velocity_m_s=float(forward_speed[i]),com_speed_m_s=float(speed[i]),angular_speed_rad_s=float(angular[i]),
                    actual_hf_ratio=float(hf_ratio[i]),qualified=bool(qualified[i])))
            reports.append(dict(**stats,seed=case["seed"],trigger_time_s=trigger_time,
                blend_complete_time_s=float(d["policy_physics_time"][np.flatnonzero(control&(d["policy_capture_memory"][:,slot,1]>=1))[0],slot])
                    if np.any(control&(d["policy_capture_memory"][:,slot,1]>=1)) else None,
                trigger_prefix=dict(raw_through_trigger_exact=raw_equal,requested_actions_before_trigger_exact=action_equal,
                    actor_output_before_trigger_exact=actor_equal,physical_frames_compared=physical_count,
                    physical_fields_exact=fields,physical_field_mismatches=mismatches,
                    trigger_action_repeats_last_executed_exact=True),
                terminal_last1s=dict(samples=int(tail.sum()),com_speed_mean_m_s=float(speed[tail].mean()),com_speed_max_m_s=float(speed[tail].max()),
                    signed_gap_mean_m=float(signed_gap[tail].mean()),signed_gap_abs_max_m=float(abs(signed_gap[tail]).max()),
                    finite_line_error_mean_m=float(dd["line_error"][tail].mean()),angular_speed_max_rad_s=float(angular[tail].max()),
                    actual_hf_max_ratio=float(hf_ratio[tail].max()),qualified_fraction=float(qualified[tail].mean())),
                checkpoints=checkpoints))
    out=dict(created_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),run=str(run),
        session=read(run/"session.json"),actor_sha256=sha(request["actor"]),old_v4_trace_sha256=sha(old_path),
        independent_verifier_rerun_matches=True,teacher_off=True,source_model_checks_passed=True,
        robot_ppo_updates=0,simulator_runs_added=0,ppo_admission=False,
        cases=reports,seed_repeat_checks=seed_repeats,read_only_analysis_sha256=sha(__file__),
        limitation="State/actor/physics checks are on these finite fixed-condition runs; this CPU audit does not supply missing six-case/PPO admission evidence.")
    args.output.write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    print(json.dumps([{k:c[k] for k in ["slot","capture_time_s","post_capture_uninterrupted_s","first_break_time_s","first_break_gates","observation_pass","trigger_prefix","terminal_last1s"]} for c in reports],ensure_ascii=False,indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
