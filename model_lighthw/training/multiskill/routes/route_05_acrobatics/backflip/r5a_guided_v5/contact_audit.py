"""离线 CPU 接触审计：只读取 JSON/NPZ 和模型字节，不载入 checkpoint 或执行归档代码。

audit_session(root, output) 核对完整 session 或精简 USB 包；不完整时明确返回
comparison_complete=False。audit_case 支持外部工作流指定一组有来源关联的记录。
法向取原平地世界 +Z；记录的是 rigid-body COM 速度，不声称接触点速度。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

WHEELS = ("FR_wheel_link", "FL_wheel_link", "RR_wheel_link", "RL_wheel_link")
DT = .001


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _finite_arrays(path):
    with np.load(path, allow_pickle=False) as src:
        arrays = {name: src[name] for name in src.files}
    if not all(np.issubdtype(v.dtype, np.number) or v.dtype == bool for v in arrays.values()):
        raise ValueError("Non-numeric trace array")
    if not all(np.isfinite(v).all() for v in arrays.values()):
        raise ValueError("Non-finite trace array")
    return arrays


def _validate(a, meta):
    required = ("t", "contact_force", "wheel_force_z", "wheel_clearance", "com", "com_velocity",
                "body_com_position", "body_com_velocity", "body_angular_velocity", "joint_position",
                "joint_velocity", "applied_torque", "root_state")
    missing = [k for k in required if k not in a]
    if missing:
        raise ValueError("Missing trace arrays: " + ", ".join(missing))
    t = a["t"]
    if t.ndim != 1 or len(t) < 2 or abs(float(t[0])) > 1e-9 or not np.allclose(np.diff(t), DT, atol=1e-9, rtol=0):
        raise ValueError("Trace must start at zero with complete 1 kHz samples")
    n = len(t)
    if any(v.ndim < 1 or len(v) != n for v in a.values()):
        raise ValueError("Trace arrays have inconsistent sample counts")
    if n != meta.get("physics_samples") or n != meta.get("expected_physics_samples"):
        raise ValueError("Trace length does not match metadata")
    c = meta["contract"]
    if abs(c.get("physics_dt", 0) - DT) > 1e-12 or abs(c.get("policy_dt", 0) - .02) > 1e-12:
        raise ValueError("Not the reviewed 1 kHz physics / 50 Hz control contract")
    if n != meta["steps"] * 20 + 1:
        raise ValueError("Incomplete policy/physics sample coverage")
    names = [meta[k] for k in ("body_names", "contact_names", "joint_names")]
    if any(len(x) != len(set(x)) for x in names):
        raise ValueError("Duplicate body/contact/joint names")
    b, contacts, joints = map(len, names)
    shapes = {"contact_force": (n, contacts, 3), "wheel_force_z": (n, 4), "wheel_clearance": (n, 4),
              "com": (n, 3), "com_velocity": (n, 3), "body_com_position": (n, b, 3),
              "body_com_velocity": (n, b, 3), "body_angular_velocity": (n, b, 3),
              "joint_position": (n, joints), "joint_velocity": (n, joints), "applied_torque": (n, joints),
              "root_state": (n, 13)}
    for k, shape in shapes.items():
        if a[k].shape != shape:
            raise ValueError(f"Unexpected {k} shape: {a[k].shape}, expected {shape}")
    mass = np.asarray(meta["masses"], dtype=float)
    if mass.shape != (b,) or not np.isfinite(mass).all() or (mass <= 0).any():
        raise ValueError("Invalid rigid-body masses")
    ci = [meta["contact_names"].index(k) for k in WHEELS]
    bi = [meta["body_names"].index(k) for k in WHEELS]
    if not np.allclose(a["contact_force"][:, ci, 2], a["wheel_force_z"], atol=2e-4, rtol=1e-6):
        raise ValueError("Wheel force order disagrees with named contact bodies")
    com = (a["body_com_position"] * mass[None, :, None]).sum(1) / mass.sum()
    cv = (a["body_com_velocity"] * mass[None, :, None]).sum(1) / mass.sum()
    if not np.allclose(com, a["com"], atol=2e-5, rtol=1e-5) or not np.allclose(cv, a["com_velocity"], atol=2e-5, rtol=1e-5):
        raise ValueError("Whole COM disagrees with mass-weighted body data")
    return ci, bi, float(mass.sum())


def segment_flight(a, meta):
    """由力、间隙和 COM 速度识别首次起跳/接触，不使用在线 landed 位。"""
    c = meta["contract"]
    t = a["t"]
    free = (np.linalg.norm(a["contact_force"], axis=-1).max(-1) < c.get("airborne_force_N", 2.)) & (a["wheel_clearance"].min(-1) > 0)
    count = 0
    start = None
    release = None
    for i in range(1, len(t)):
        if free[i] and t[i] > c.get("settle_seconds", .4):
            if count == 0 and a["com_velocity"][i, 2] > c.get("min_takeoff_vz", .25):
                start, count = i, 1
            elif count:
                count += 1
        else:
            count = 0
        if count >= c.get("release_samples", 7):
            release = start
            break
    if release is None:
        return None, None
    for i in range(release + c.get("release_samples", 7), len(t)):
        if np.linalg.norm(a["contact_force"][i], axis=-1).max() >= c.get("airborne_force_N", 2.):
            return release, i
    return release, None


def _stats(values):
    values = np.asarray(values, dtype=float)
    return dict(min=values.min(axis=0).tolist(), max=values.max(axis=0).tolist(),
                mean=values.mean(axis=0).tolist(), rms=np.sqrt(np.mean(values ** 2, axis=0)).tolist())


def _integral(values, dt=DT):
    """梯形数值积分；仅用于已记录瞬时功率，不声称接触/关节全部耗散能。"""
    values = np.asarray(values, dtype=float)
    return ((values[:-1] + values[1:]) * (.5 * dt)).sum(axis=0)


def contact_metrics(a, meta):
    ci, bi, mass = _validate(a, meta)
    release, first = segment_flight(a, meta)
    if release is None or first is None:
        raise ValueError("No independently identified completed first flight")
    t = a["t"]
    start, end = first - 100, first + 250
    if start < 1 or end >= len(t):
        raise ValueError("Missing complete contact window [-100 ms, +250 ms]")
    ids = np.arange(start, end + 1)
    post = np.arange(first, end + 1)
    pre = np.arange(start, first)
    wf = a["contact_force"][:, ci].astype(float)
    support = np.maximum(wf[:, :, 2], 0.)
    total = support.sum(-1)
    peak = first + int(total[first:end + 1].argmax())
    whole_peak = first + int(total[first:].argmax())
    jnames = meta["joint_names"]
    velocity, torque = a["joint_velocity"].astype(float), a["applied_torque"].astype(float)
    power = torque * velocity
    drift = np.linalg.norm(a["com"][:, :2] - a["com"][first, :2], axis=-1)
    bw = mass * 9.81
    cumulative, intervals = {}, {}
    previous_ms = 0
    for duration_ms in (20, 50, 250):
        def impulse_slice(begin_ms):
            # i=first 对应的力单独报告；这两个集合都严格排除其左端点。
            selected = slice(first + begin_ms + 1, first + duration_ms + 1)
            each_impulse = (wf[selected, :, 2] * DT).sum(0)
            return dict(start_after_first_contact_ms=begin_ms, end_after_first_contact_ms=duration_ms,
                sample_count=duration_ms - begin_ms, total_wheel_normal_impulse_Ns=float(each_impulse.sum()),
                per_wheel_normal_impulse_Ns={name: float(each_impulse[j]) for j, name in enumerate(WHEELS)},
                interval="(t_first + start, t_first + end]; first-contact sample excluded")
        cumulative[str(duration_ms) + "ms"] = impulse_slice(0)
        intervals[f"{previous_ms}_{duration_ms}ms"] = impulse_slice(previous_ms)
        previous_ms = duration_ms
    each = {}
    for j, name in enumerate(WHEELS):
        active = np.flatnonzero(np.linalg.norm(wf[first:end + 1, j], axis=-1) >= 2.)
        first_wheel = first + int(active[0]) if len(active) else None
        pk = first + int(support[first:end + 1, j].argmax())
        each[name] = dict(first_contact_s=float(t[first_wheel]) if first_wheel is not None else None,
            peak_normal_force_N=float(support[pk, j]), peak_time_s=float(t[pk]),
            normal_impulse_Ns=float((wf[ids[1:], j, 2] * DT).sum()),
            vector_impulse_Ns=(wf[ids[1:], j] * DT).sum(0).tolist(),
            support_fraction_postcontact=float(np.mean(support[post, j] > 5.)),
            COM_world_z_velocity_precontact_m_s=float(a["body_com_velocity"][first - 1, bi[j], 2]),
            COM_world_z_velocity_at_contact_m_s=float(a["body_com_velocity"][first, bi[j], 2]),
            COM_world_z_velocity_before_own_first_contact_m_s=float(a["body_com_velocity"][first_wheel - 1, bi[j], 2]) if first_wheel is not None else None,
            own_first_contact_support_N=float(support[first_wheel, j]) if first_wheel is not None else None)
    joint = {}
    for j, name in enumerate(jnames):
        joint[name] = dict(position_at_first_contact_rad=float(a["joint_position"][first, j]),
            velocity_precontact_rad_s=float(velocity[first - 1, j]), velocity_postcontact_rad_s=float(velocity[first, j]),
            precontact_velocity_rad_s=_stats(velocity[pre, j]), postcontact_velocity_rad_s=_stats(velocity[post, j]),
            peak_actual_torque_Nm=float(np.abs(torque[ids, j]).max()),
            actual_torque_at_peak_Nm=float(torque[peak, j]),
            net_actuator_work_J=float(_integral(power[ids, j])),
            positive_actuator_work_J=float(_integral(np.maximum(power[ids, j], 0.))),
            negative_actuator_work_J=float(_integral(np.minimum(power[ids, j], 0.))),
            postcontact_net_actuator_work_J=float(_integral(power[post, j])))
    wheel_com_vz = a["body_com_velocity"][:, bi, 2]
    window = {k: v[ids] for k, v in a.items()}
    window.update(relative_time_s=t[ids] - t[first], wheel_COM_world_z_velocity=wheel_com_vz[ids],
                  actual_joint_power_W=power[ids], COM_landing_drift_m=drift[ids])
    link_note = "not available: link origin position was not recorded; only wheel body COM velocity is reported"
    if "body_link_position" in a:
        if a["body_link_position"].shape != a["body_com_position"].shape:
            raise ValueError("Invalid body_link_position shape")
        lv = a["body_com_velocity"] + np.cross(a["body_angular_velocity"], a["body_link_position"] - a["body_com_position"])
        window["wheel_link_origin_world_z_velocity"] = lv[ids][:, bi, 2]
        link_note = "computed exactly from recorded v_COM + omega x (p_link - p_COM); wheel link origin, not contact point"
    result = dict(first_release_s=float(t[release]), first_contact_s=float(t[first]),
        flight_s=float(t[first] - t[release]), window_start_s=float(t[start]), window_end_s=float(t[end]),
        window_samples=len(ids), complete_window=True, robot_mass_kg=mass, gravity_m_s2=9.81,
        total_support_peak_N=float(total[peak]), total_support_peak_bodyweights=float(total[peak] / bw),
        total_support_peak_s=float(t[peak]), first_contact_to_peak_ms=float((t[peak] - t[first]) * 1000),
        first_contact_total_wheel_normal_force_N=float(wf[first, :, 2].sum()),
        first_contact_per_wheel_normal_force_N={name: float(wf[first, j, 2]) for j, name in enumerate(WHEELS)},
        postcontact_cumulative_impulse=cumulative, postcontact_interval_impulse=intervals,
        whole_postcontact_peak_bodyweights=float(total[whole_peak] / bw), whole_postcontact_peak_s=float(t[whole_peak]),
        wheel_normal_impulse_Ns=float((wf[ids[1:], :, 2] * DT).sum()),
        all_body_contact_impulse_Ns=(a["contact_force"][ids[1:]].sum(1) * DT).sum(0).tolist(),
        COM_velocity_precontact_m_s=a["com_velocity"][first - 1].tolist(),
        COM_velocity_at_contact_m_s=a["com_velocity"][first].tolist(),
        root_angular_speed_at_contact_rad_s=float(np.linalg.norm(a["root_state"][first, 10:13])),
        maximum_COM_drift_250ms_m=float(drift[post].max()), COM_drift_250ms_m=float(drift[end]),
        four_wheel_support_fraction_250ms=float(np.mean((support[post] > 5.).all(-1))),
        per_wheel=each, per_joint=joint,
        definitions=dict(normal="world +Z of recorded plane; positive wheel support is clipped upward force",
            impulses="sum of 1 kHz reported mean step forces for (window_start, window_end]; signed normal and vector impulses",
            postcontact_impulses="cumulative (t_first, t_first+20/50/250ms], disjoint (0,20], (20,50], (50,250]ms; first-contact sample excluded and its force reported separately",
            actuator_work="trapezoidal integral of applied_torque * joint_velocity; sampled actuator work, not total collision energy or passive dissipation",
            wheel_normal_velocity="recorded wheel rigid-body COM linear velocity projected on world +Z; not ground contact-point or wheel link-origin velocity",
            link_origin_velocity=link_note,
            contact_point_velocity="unavailable: contact-point positions and normals are not recorded",
            precontact_velocity_timing="COM_world_z_velocity_precontact refers to 1 ms before ANY body first lands; before_own_first_contact uses each wheel's own first >=2 N contact",
            support="each named wheel upward force > 5 N", joint_order="metadata joint_names; not policy action_order"))
    return result, window


def _report_checks(report, metrics, a, meta):
    expected = {"samples": len(a["t"]), "duration_s": float(a["t"][-1]), "flight_s": metrics["flight_s"],
                "peak_support_bodyweights": metrics["whole_postcontact_peak_bodyweights"]}
    differences = []
    for key, value in expected.items():
        if key not in report or not np.isclose(report[key], value, rtol=2e-5, atol=2e-6):
            differences.append(key)
    for key, value in (("first_contact_s", metrics["first_contact_s"]),
                       ("post_landing_peak_s", metrics["whole_postcontact_peak_s"]),
                       ("touchdown_root_angular_speed_rad_s", metrics["root_angular_speed_at_contact_rad_s"])):
        saved = report.get("landing_diagnostics", {}).get(key)
        if saved is not None and not np.isclose(saved, value, rtol=2e-5, atol=2e-6):
            differences.append("landing_diagnostics/" + key)
    if differences:
        raise ValueError("Report/trace mismatch: " + ", ".join(differences))


def audit_case(trace_path, metadata_path, report_path, checkpoint_path,
               expected_checkpoint_sha256, *, label="case", case_report=None,
               manifest_root=None, output_dir=None, report_container_path=None):
    """核对一条完整记录；缺失/不一致返回状态，不把摘要当作原始轨迹证据。

    case_report 是 evaluation/result.json 内同 artifact_prefix 的单条记录，其中
    checkpoint_sha256 为保存时的模型关联。旧逐条 .json 没有此字段，需调用方提供。
    """
    paths = {"trace": trace_path, "metadata": metadata_path, "report": report_path, "checkpoint": checkpoint_path}
    if trace_path is not None:
        trace = Path(trace_path)
        policy = trace.with_name(trace.name.removesuffix("_trace.npz") + "_policy.npz")
        if policy.is_file():
            paths["policy"] = policy
    result = dict(label=label, status="incomplete", complete=False, errors=[], files={}, metrics=None)
    missing = [k for k, p in paths.items() if p is None or not Path(p).is_file()]
    if missing:
        result["errors"] = ["Missing " + ", ".join(missing)]
        return result
    try:
        if manifest_root is not None:
            root = Path(manifest_root).resolve()
            for p in [*paths.values(), *([report_container_path] if report_container_path is not None else [])]:
                if not Path(p).resolve().is_relative_to(root):
                    raise ValueError("Evidence path escapes source session")
        for k, p in paths.items():
            result["files"][k] = dict(path=str(Path(p).resolve()), sha256=sha256(p))
        if not expected_checkpoint_sha256 or result["files"]["checkpoint"]["sha256"] != expected_checkpoint_sha256:
            raise ValueError("Checkpoint SHA256 differs from evaluation provenance")
        manifest = None
        if manifest_root is not None and (Path(manifest_root) / "SHA256.json").is_file():
            if not (Path(manifest_root) / "SHA256.json").resolve().is_relative_to(Path(manifest_root).resolve()):
                raise ValueError("Source hash manifest escapes session")
            manifest = read(Path(manifest_root) / "SHA256.json")
            for k, p in paths.items():
                rel = str(Path(p).resolve().relative_to(Path(manifest_root).resolve()))
                if manifest.get(rel) != result["files"][k]["sha256"]:
                    raise ValueError("Missing/mismatched source manifest hash: " + rel)
            if report_container_path is not None:
                rel = str(Path(report_container_path).resolve().relative_to(Path(manifest_root).resolve()))
                if manifest.get(rel) != sha256(report_container_path):
                    raise ValueError("Missing/mismatched evaluation report manifest hash: " + rel)
        a, meta, report = _finite_arrays(trace_path), read(metadata_path), read(report_path)
        provenance = case_report or report
        if provenance.get("checkpoint_sha256") != expected_checkpoint_sha256:
            raise ValueError("Case report is not associated with expected checkpoint")
        prefix = provenance.get("artifact_prefix")
        if prefix and Path(trace_path).name != prefix + "_trace.npz":
            raise ValueError("Artifact prefix/trace filename mismatch")
        for k, value in report.items():
            if k in provenance and value != provenance[k]:
                raise ValueError("Standalone report differs from evaluation case: " + k)
        metrics, window = contact_metrics(a, meta)
        _report_checks(report, metrics, a, meta)
        result.update(status="verified", complete=True, metrics=metrics,
            model_sha256=expected_checkpoint_sha256, source_manifest_verified=manifest is not None,
            binding_basis="case checkpoint SHA and artifact prefix; standalone report content and metrics independently matched to raw trace; current artifact hashes recorded",
            original_artifact_hash_limitation=None if manifest is not None else "No pre-existing SHA256 manifest: computed hashes bind this audit snapshot, not a historical seal",
            acceptance_reported=dict(success=report.get("success"), physics_consistent=report.get("physics_consistent")),
            acceptance_scope="contact-window audit; success/momentum fields remain reported values, not a new protocol or momentum re-evaluation")
        if report_container_path is not None:
            result["files"]["report_container"] = dict(path=str(Path(report_container_path).resolve()), sha256=sha256(report_container_path))
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            safe_label = "".join(c if c.isalnum() or c in "_-" else "_" for c in label)
            path = output_dir / (safe_label + "_contact_window.npz")
            np.savez_compressed(path, **window)
            result["window_file"] = dict(path=str(path.resolve()), sha256=sha256(path))
    except (ValueError, KeyError, OSError, TypeError, IndexError) as error:
        result.update(status="invalid", complete=False, errors=[str(error)])
    return result


def _local_checkpoint(root, saved, expected_sha):
    """只解析会话目录内的模型；不沿 JSON 中任意绝对路径读取外部文件。"""
    candidates = []
    saved = Path(saved) if saved else None
    if saved:
        if not saved.is_absolute():
            candidates.append(root / saved)
        parts = saved.parts
        if root.name in parts:
            candidates.append(root.joinpath(*parts[parts.index(root.name) + 1:]))
        candidates.extend((root / "training" / saved.name, root / "training/training" / saved.name))
    candidates.extend((root / "selected.pt", root / "best_new.pt", root / "baseline.pt",
                       root / "baseline/model.pt", root / "final/model.pt", root / "best_new/model.pt"))
    for path in candidates:
        if path.resolve().is_relative_to(root) and path.is_file() and sha256(path) == expected_sha:
            return path
    return None


def audit_session(session_root, output_dir=None, required_roles=("baseline", "final")):
    root = Path(session_root).resolve(strict=True)
    comparison_path = root / "candidate_comparison.json"
    def session_json(path):
        if not path.resolve().is_relative_to(root):
            raise ValueError("Session JSON path escapes source session")
        return read(path)
    if comparison_path.is_file():
        candidates = session_json(comparison_path)
    else:
        evaluation = root / "final_evaluation/result.json"
        cases = session_json(evaluation).get("cases", []) if evaluation.is_file() else []
        request = root / "final_evaluation/request.json"
        choices = session_json(request).get("candidates", []) if request.is_file() else []
        candidates = [dict(**choice, cases=[c for c in cases if c.get("candidate") == choice["label"]]) for choice in choices]
    finals = [c for c in candidates if c.get("label") != "session_start_baseline"]
    last_updates = max((c.get("completed_updates", -1) for c in finals), default=-1)
    output = dict(schema="r5a_contact_audit_v1", source_session=str(root), comparison_complete=False,
        status="incomplete", required_roles=list(required_roles), roles={}, cases=[],
        scope="CPU raw contact audit only; no new simulation, training, checkpoint loading or archived-code execution")
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    for candidate in candidates:
        label = candidate["label"]
        role = "baseline" if label == "session_start_baseline" else ("final" if candidate.get("completed_updates") == last_updates else "intermediate")
        reports = candidate.get("cases", [])
        if not reports:
            output["cases"].append(dict(label=label, role=role, complete=False, status="incomplete", errors=["No evaluation cases"]));continue
        for case in reports:
            prefix = case.get("artifact_prefix", "")
            if not prefix or Path(prefix).name != prefix:
                output["cases"].append(dict(label=label, role=role, complete=False, status="invalid", errors=["Unsafe/missing artifact prefix"]));continue
            folders = (root / "final_evaluation", root / "selected", root / "best_new", root / "baseline", root / "final")
            folder = next((p for p in folders if (p / (prefix + "_trace.npz")).is_file()), folders[0])
            checkpoint = _local_checkpoint(root, candidate.get("checkpoint"), case.get("checkpoint_sha256"))
            r = audit_case(folder / (prefix + "_trace.npz"), folder / (prefix + "_trace.json"),
                folder / (prefix + ".json"), checkpoint, case.get("checkpoint_sha256"), label=prefix,
                case_report=case, manifest_root=root, output_dir=output_dir,
                report_container_path=comparison_path if comparison_path.is_file() else root / "final_evaluation/result.json")
            r.update(role=role, completed_updates=candidate.get("completed_updates"), candidate=label)
            output["cases"].append(r)
    for role in ("baseline", "final", "intermediate"):
        rows = [r for r in output["cases"] if r.get("role") == role]
        output["roles"][role] = dict(present=bool(rows), verified=bool(rows) and all(r["complete"] for r in rows),
            cases=len(rows), note="middle-training checkpoints are not assumed to have been evaluated" if role == "intermediate" else None)
    output["comparison_complete"] = all(output["roles"].get(role, {}).get("verified", False) for role in required_roles)
    output["all_available_candidate_traces_verified"] = bool(output["cases"]) and all(c["complete"] for c in output["cases"])
    output["passed"] = output["comparison_complete"]
    output["inputs"] = {}
    output["input_manifest"] = {}
    for case in output["cases"]:
        for value in case.get("files", {}).values():
            path = Path(value["path"])
            output["input_manifest"][str(path)] = value["sha256"]
            if path.is_relative_to(root):
                output["inputs"][str(path.relative_to(root))] = value["sha256"]
    output["status"] = "verified" if output["comparison_complete"] else "incomplete"
    output["conclusion"] = ("Required baseline and final contact records verified; causal attribution still requires matched controlled evidence."
        if output["comparison_complete"] else "Candidate contact audit is NOT complete. Do not claim new-candidate contacts were checked or calibrate rewards from missing trajectories.")
    if output_dir is not None:
        dest = Path(output_dir)
        write(dest / "contact_audit.json", output)
        lines = ["# R5A contact audit", "", output["conclusion"], "",
                 "| Case | Role | Status | Contact s | Peak BW | Impulse Ns | Drift 250 ms |", "|---|---|---|---:|---:|---:|---:|"]
        for c in output["cases"]:
            m = c.get("metrics")
            if m:
                lines.append(f"| {c['label']} | {c['role']} | {c['status']} | {m['first_contact_s']:.3f} | {m['total_support_peak_bodyweights']:.2f} | {m['wheel_normal_impulse_Ns']:.3f} | {m['COM_drift_250ms_m']:.4f} |")
            else:
                lines.append(f"| {c['label']} | {c['role']} | {c['status']}: {'; '.join(c['errors'])} | — | — | — | — |")
        lines += ["", "Velocities refer to named wheel rigid-body COM in world +Z, not contact-point velocity. Applied torque × joint speed gives sampled actuator work, not total collision dissipation.",
                  "The 350 ms window contains 351 snapshots; force impulses use 350 recorded step forces in (start, end]. Success and momentum acceptance are not re-evaluated by this contact tool.",
                  "A complete contact check cannot alone distinguish contact geometry, servo tuning, or collision-solver causes."]
        (dest / "CONTACT_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = audit_session(args.session, args.output)
    print("[CONTACT AUDIT]", result["status"], result["conclusion"])
    print("[CONTACT AUDIT FILE]", args.output / "contact_audit.json")
    return 0 if result["comparison_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
