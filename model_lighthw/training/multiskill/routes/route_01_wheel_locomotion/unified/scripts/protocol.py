"""统一R1固定起停筛查；最终15kg/40kg/25deg/3mps要求按真实组合统计。"""
import math
import numpy as np

DURATION=18.
DT=.02
STEPS=900


def cases():
    return [dict(slope_deg=slope,payload_kg=load,towed_kg=tow,speed=speed)
            for slope in (0.,25.,-25.) for load in (0.,15.) for tow in (0.,40.) for speed in (0.,3.1)]


def summarize(rows,case,reason):
    complete=len(rows)==STEPS and reason is None
    result={"case":case,"samples":len(rows),"complete":complete,"failure_reason":reason,"success":False,
            "sample_hz":50,"final_hardware_acceptance":"not_evaluated"}
    if not rows:
        return result
    if not all(math.isfinite(v) for row in rows for v in row.values()):
        result["failure_reason"]="non_finite_measurement";result["complete"]=False
        return result
    result["observed"]={
        "duration_s":len(rows)*DT,
        "peak_tilt_deg":max(r["tilt_deg"] for r in rows),
        "peak_drawbar_force_N_1kHz":max(r.get("drawbar_force_peak_N_1kHz",abs(r["drawbar_force_N"])) for r in rows),
        "body_contact_samples":sum(r["body_contact"]>0 for r in rows),
        "termination_causes":[key.removeprefix("termination_") for key in rows[0]
                              if key.startswith("termination_") and any(r[key]>0 for r in rows)],
    }
    cruise=[r for r in rows if 5.<=r["t"]<11.]
    hold=[r for r in rows if 13.<=r["t"]<18.]
    if not cruise or not hold:
        return result
    avg=lambda rs,key:float(np.mean([r[key] for r in rs]))
    metrics={"cruise_vx_mps":avg(cruise,"vx"),"cruise_mae_mps":avg(cruise,"velocity_error"),
             "tilt_p95_deg":float(np.percentile([r["tilt_deg"] for r in rows],95)),
             "heading_abs_p95_deg":float(np.percentile([abs(r["heading_rad"])*180/math.pi for r in rows],95)),
             "hold_speed_mps":avg(hold,"speed"),"hold_yaw_rate_rps":avg(hold,"yaw_rate_abs"),
             "hold_four_wheel_fraction":avg(hold,"four_wheel"),
             "height_error_p95_m":float(np.percentile([abs(r["height"]-.416) for r in rows],95)),
             "body_contact_samples":sum(r["body_contact"]>0 for r in rows),
             "peak_torque_Nm_50Hz":max(r["torque_abs_max"] for r in rows),
             "torque_saturation_fraction":avg(rows,"torque_saturation"),
             "wheel_target_clip_fraction":avg(rows,"wheel_target_clip"),
             "peak_drawbar_force_N_50Hz":max(abs(r["drawbar_force_N"]) for r in rows),
             "hold_drift_m":max(math.hypot(r["x"]-hold[0]["x"],r["y"]-hold[0]["y"]) for r in hold)}
    checks={"complete":complete,"tracking":metrics["cruise_mae_mps"]<=.15,
            "actual_3mps":case["speed"]<3. or metrics["cruise_vx_mps"]>=3.,
            "surface_posture":metrics["tilt_p95_deg"]<=15.,"heading":metrics["heading_abs_p95_deg"]<=10.,"height":metrics["height_error_p95_m"]<=.08,
            "hold_motion":metrics["hold_speed_mps"]<=.05 and metrics["hold_yaw_rate_rps"]<=.05,
            "hold_support":metrics["hold_four_wheel_fraction"]>=.95,
            "no_body_contact":metrics["body_contact_samples"]==0,
            "drawbar_guard":result["observed"]["peak_drawbar_force_N_1kHz"]<1499.}
    result.update(metrics=metrics,checks=checks,success=all(checks.values()))
    return result
