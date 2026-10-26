"""接触审计 CPU 测试：已知冲量/功、坐标语义、缺失/篡改与会话来源。"""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from .contact_audit import WHEELS, audit_case, audit_session, contact_metrics, sha256


def fixture():
    t = np.arange(1201) * .001
    n = len(t)
    forces = np.zeros((n, 4, 3));forces[:501, :, 2] = 10.;forces[800:, :, 2] = 20.
    com = np.zeros((n, 3));com[:, 2] = .4
    vel = np.zeros((n, 3));vel[501:800, 2] = 1.
    clearance = np.zeros((n, 4));clearance[501:800] = .1
    qv = np.tile([2., -3.], (n, 1));tau = np.tile([4., 5.], (n, 1))
    a = dict(t=t, contact_force=forces, wheel_force_z=forces[:, :, 2], wheel_clearance=clearance,
        com=com, com_velocity=vel, body_com_position=np.repeat(com[:, None], 4, axis=1),
        body_com_velocity=np.repeat(vel[:, None], 4, axis=1), body_angular_velocity=np.zeros((n, 4, 3)),
        joint_position=qv * t[:, None], joint_velocity=qv, applied_torque=tau, root_state=np.zeros((n, 13)))
    meta = dict(contract=dict(physics_dt=.001, policy_dt=.02, settle_seconds=.4), physics_samples=n,
        expected_physics_samples=n, steps=60, body_names=list(WHEELS), contact_names=list(WHEELS),
        joint_names=["FR_thigh_joint", "FR_calf_joint"], masses=[1.] * 4, reason="timeout")
    return a, meta


def bundle(folder, label="session_start_baseline", model=None):
    a, meta = fixture()
    m, _ = contact_metrics(a, meta)
    folder.mkdir(parents=True, exist_ok=True)
    prefix = label + "_seed43_case0"
    np.savez_compressed(folder / (prefix + "_trace.npz"), **a)
    (folder / (prefix + "_trace.json")).write_text(json.dumps(meta))
    report = dict(samples=len(a["t"]), duration_s=1.2, flight_s=.299, peak_support_bodyweights=m["whole_postcontact_peak_bodyweights"],
        success=False, physics_consistent=True, landing_diagnostics=dict(first_contact_s=.8, post_landing_peak_s=.8, touchdown_root_angular_speed_rad_s=0.))
    (folder / (prefix + ".json")).write_text(json.dumps(report))
    model = model or folder / "model.pt"
    model.write_bytes(b"opaque model bytes; never unpickle")
    case = dict(**report, checkpoint_sha256=sha256(model), artifact_prefix=prefix, candidate=label)
    return prefix, case, model


class ContactTests(unittest.TestCase):
    def test_analytical_impulse_and_actuator_work(self):
        a, meta = fixture();r, w = contact_metrics(a, meta)
        self.assertAlmostEqual(r["first_contact_s"], .8)
        self.assertAlmostEqual(r["flight_s"], .299)
        self.assertEqual(r["window_samples"], 351)
        self.assertAlmostEqual(r["wheel_normal_impulse_Ns"], .251 * 80.)
        self.assertAlmostEqual(r["per_joint"]["FR_thigh_joint"]["net_actuator_work_J"], .35 * 8.)
        self.assertAlmostEqual(r["per_joint"]["FR_calf_joint"]["negative_actuator_work_J"], -.35 * 15.)
        self.assertEqual(w["relative_time_s"].shape, (351,))

    def test_postcontact_impulse_excludes_first_sample_and_partitions(self):
        a, meta = fixture()
        # 一次很大的首触采样不应污染 (t_first,t_first+T] 三个积分。
        a["contact_force"][800, :, 2] = 1000.
        r, _ = contact_metrics(a, meta)
        self.assertEqual(r["first_contact_total_wheel_normal_force_N"], 4000.)
        for duration in (20, 50, 250):
            section = r["postcontact_cumulative_impulse"][str(duration) + "ms"]
            self.assertEqual(section["sample_count"], duration)
            self.assertAlmostEqual(section["total_wheel_normal_impulse_Ns"], duration * .001 * 80.)
            self.assertAlmostEqual(section["per_wheel_normal_impulse_Ns"][WHEELS[0]], duration * .001 * 20.)
        self.assertAlmostEqual(sum(x["total_wheel_normal_impulse_Ns"] for x in r["postcontact_interval_impulse"].values()), 20.)
        self.assertAlmostEqual(r["wheel_normal_impulse_Ns"], 24.)

    def test_body_com_velocity_is_not_contact_point(self):
        a, meta = fixture();r, w = contact_metrics(a, meta)
        self.assertIn("not ground contact-point", r["definitions"]["wheel_normal_velocity"])
        self.assertEqual(r["per_wheel"][WHEELS[0]]["COM_world_z_velocity_precontact_m_s"], 1.)
        self.assertNotIn("wheel_link_origin_world_z_velocity", w)
        a["body_link_position"] = a["body_com_position"] + [1., 0., 0.]
        a["body_angular_velocity"][:, :, 1] = 2.
        _, w = contact_metrics(a, meta)
        np.testing.assert_allclose(w["wheel_link_origin_world_z_velocity"], w["wheel_COM_world_z_velocity"] - 2.)

    def test_incomplete_physics_rejected(self):
        a, meta = fixture();a["t"][900] += .001
        with self.assertRaisesRegex(ValueError, "1 kHz"):
            contact_metrics(a, meta)

    def test_com_mismatch_rejected(self):
        a, meta = fixture();a["com"][750, 0] = .2
        with self.assertRaisesRegex(ValueError, "Whole COM"):
            contact_metrics(a, meta)

    def test_missing_postcontact_window_rejected(self):
        a, meta = fixture()
        a = {k: v[:1001] for k, v in a.items()}
        meta.update(physics_samples=1001, expected_physics_samples=1001, steps=50)
        with self.assertRaisesRegex(ValueError, "complete contact window"):
            contact_metrics(a, meta)

    def test_named_force_order_and_own_wheel_touch_time(self):
        a, meta = fixture()
        a["contact_force"][800:810, 1] = 0.
        a["body_com_velocity"][800:810, 1, 2] = -2.
        a["com_velocity"][800:810, 2] = -.5
        r, _ = contact_metrics(a, meta)
        wheel = r["per_wheel"][WHEELS[1]]
        self.assertAlmostEqual(wheel["first_contact_s"], .810)
        self.assertEqual(wheel["COM_world_z_velocity_before_own_first_contact_m_s"], -2.)
        a["wheel_force_z"] = a["wheel_force_z"].copy()
        a["wheel_force_z"][805, 1] = 20.
        with self.assertRaisesRegex(ValueError, "Wheel force order"):
            contact_metrics(a, meta)

    def test_missing_model_or_wrong_sha_not_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp);prefix, case, model = bundle(p)
            args = (p / (prefix + "_trace.npz"), p / (prefix + "_trace.json"), p / (prefix + ".json"))
            self.assertEqual(audit_case(*args, None, case["checkpoint_sha256"], case_report=case)["status"], "incomplete")
            self.assertEqual(audit_case(*args, model, "0" * 64, case_report=case)["status"], "invalid")
            self.assertTrue(audit_case(*args, model, case["checkpoint_sha256"], case_report=case)["complete"])

    def test_wrong_case_report_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp);prefix, case, model = bundle(p)
            case["flight_s"] = .4
            r = audit_case(p / (prefix + "_trace.npz"), p / (prefix + "_trace.json"), p / (prefix + ".json"), model,
                           sha256(model), case_report=case)
            self.assertEqual(r["status"], "invalid")

    def test_missing_new_candidate_is_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp);_, case, model = bundle(p / "selected", model=p / "selected.pt")
            candidates = [dict(label="session_start_baseline", completed_updates=0, checkpoint="training/initial.pt", cases=[case]),
                          dict(label="update_003000", completed_updates=3000, checkpoint="training/final.pt",
                               cases=[dict(case, candidate="update_003000", artifact_prefix="update_003000_seed43_case0", checkpoint_sha256="f" * 64)])]
            (p / "candidate_comparison.json").write_text(json.dumps(candidates))
            r = audit_session(p, p / "audit")
            self.assertTrue(r["roles"]["baseline"]["verified"])
            self.assertFalse(r["comparison_complete"])
            self.assertIn("NOT complete", r["conclusion"])

    def test_complete_session_and_manifest_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp);(p / "training").mkdir()
            candidates = []
            for name, number, fname in (("session_start_baseline", 0, "initial.pt"), ("update_003000", 3000, "final.pt")):
                _, case, model = bundle(p / "final_evaluation", name, p / "training" / fname)
                candidates.append(dict(label=name, completed_updates=number, checkpoint="training/" + fname, cases=[case]))
            (p / "candidate_comparison.json").write_text(json.dumps(candidates))
            self.assertTrue(audit_session(p)["comparison_complete"])
            manifest = {str(f.relative_to(p)): sha256(f) for f in p.rglob("*") if f.is_file()}
            (p / "SHA256.json").write_text(json.dumps(manifest))
            self.assertTrue(audit_session(p)["comparison_complete"])
            f = p / "final_evaluation/update_003000_seed43_case0_trace.json"
            f.write_text(f.read_text() + " ")
            self.assertFalse(audit_session(p)["comparison_complete"])

    def test_compact_two_role_layout_can_be_reaudited(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp);candidates = []
            for role, label, number in (("baseline", "session_start_baseline", 0), ("final", "update_003000", 3000)):
                _, case, model = bundle(p / role, label, p / role / "model.pt")
                candidates.append(dict(label=label, completed_updates=number, checkpoint="/other/machine/initial.pt" if role=="baseline" else "/other/machine/final.pt", cases=[case]))
            (p / "candidate_comparison.json").write_text(json.dumps(candidates))
            self.assertTrue(audit_session(p)["passed"])

    def test_symlink_outside_source_session_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp);session = p / "session";session.mkdir()
            prefix, case, model = bundle(p / "outside")
            for suffix in ("_trace.npz", "_trace.json", ".json"):
                (session / (prefix + suffix)).symlink_to(p / "outside" / (prefix + suffix))
            r = audit_case(session / (prefix + "_trace.npz"), session / (prefix + "_trace.json"),
                session / (prefix + ".json"), model, sha256(model), case_report=case, manifest_root=session)
            self.assertEqual(r["status"], "invalid")
            self.assertIn("escapes", r["errors"][0])


if __name__ == "__main__":
    unittest.main()
