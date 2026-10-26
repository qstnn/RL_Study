"""R6固定20秒持载筛查；保留提前终止和非有限值。"""
import math
import numpy as np


def summarize(rows, reason, thresholds, expected_steps=1000):
    finite = all(math.isfinite(float(v)) for row in rows for v in row.values())
    complete = reason is None and len(rows) == expected_steps and finite
    hold = [r for r in rows if r['t'] >= 2.0] if finite else []
    report = {"complete": complete, "reason": reason, "samples": len(rows), "success": False,
              "final_acceptance": "not_evaluated", "sample_dt_s": .02, "thermal_validation": False}
    if not hold:
        return report
    percentile = lambda key: float(np.percentile([r[key] for r in hold], 95))
    report.update(tilt_p95_deg=percentile('tilt_deg'), tilt_max_deg=max(r['tilt_deg'] for r in hold),
                  speed_p95_m_s=percentile('speed'), yaw_rate_p95_rad_s=percentile('yaw_rate'),
                  height_error_p95_m=percentile('height_error'),
                  four_wheel_fraction=float(np.mean([r['four_wheel'] for r in hold])),
                  torque_saturation_fraction=float(np.mean([r['torque_saturation'] for r in hold])),
                  hold_displacement_max_m=max(math.hypot(r['x']-hold[0]['x'], r['y']-hold[0]['y']) for r in hold),
                  torque_peak_nm=max(r['torque_peak'] for r in rows),
                  body_contact_samples=sum(r['body_contact'] > 0 for r in rows))
    report['success'] = complete and report['body_contact_samples'] == 0 and all(
        report[key] >= value if key == 'four_wheel_fraction' else report[key] <= value
        for key, value in thresholds.items())
    return report
