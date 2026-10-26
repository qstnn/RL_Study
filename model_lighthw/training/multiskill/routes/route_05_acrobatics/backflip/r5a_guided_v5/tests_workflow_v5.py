"""CPU工作流回归：缺证据拒训、缓存失效、无仿真审计和双候选USB回收。"""
import json
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from . import candidate_prepare as cp
from .io import read, write, sha

FAKE_SOURCE = 'model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/r5a_guided_v4/env.py'
FAKE_SOURCE_SHA = hashlib.sha256(b'verified = True\n').hexdigest()


def fake_session(root, name='guided_v4_one'):
    session = root / 'offline_sessions/route_05a' / name
    write(session / 'session.json', dict(status='completed', contract=dict(version='lighthw_r5a_teacher_guided_v4')))
    rows = []
    for label, update, filename in [('session_start_baseline', 0, 'initial.pt'), ('update_003000', 3000, 'final.pt')]:
        model = session / 'training' / filename; model.parent.mkdir(parents=True, exist_ok=True); model.write_bytes(filename.encode())
        prefix = label + '_seed43_case0'
        case = dict(artifact_prefix=prefix, checkpoint_sha256=sha(model))
        row = dict(label=label, completed_updates=update, checkpoint=str(model), cases=[case], metrics={})
        rows.append(row)
        for suffix in cp.RAW_SUFFIXES:
            target = session / 'final_evaluation' / (prefix + suffix); target.parent.mkdir(parents=True, exist_ok=True); target.write_text('{}')
    write(session / 'candidate_comparison.json', rows)
    (session / 'training/source_snapshot').mkdir(parents=True)
    source = session / 'training/source_snapshot' / FAKE_SOURCE; source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text('verified = True\n')
    write(session / 'training/source_manifest.json', {FAKE_SOURCE: FAKE_SOURCE_SHA})
    return session, rows


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
        self.scope = patch.multiple(cp, ROOT=self.root, implementation=lambda: {'source': 'current'}, legacy_implementation=lambda: {FAKE_SOURCE: FAKE_SOURCE_SHA})
        self.scope.start()
    def tearDown(self):
        self.scope.stop(); self.tmp.cleanup()
    def audit(self, session, contact=True, reward=True, strict=True):
        inputs = cp.audit_inputs(session)
        contact_result = dict(comparison_complete=contact, passed=contact, status='verified' if contact else 'incomplete',
                              source_session=str(session), inputs=inputs, required_roles=list(cp.REQUIRED_ROLES))
        reward_result = dict(passed=reward, source_session=str(session), inputs=inputs)
        with patch('r5a_guided_v5.contact_audit.audit_session', return_value=contact_result) as contact_fn, patch('r5a_guided_v5.reward_audit.audit_session_rewards', return_value=reward_result) as reward_fn:
            record = cp.ensure_candidate_audit(session, export=False, strict=strict)
        return record, contact_fn, reward_fn
    def test_unique_selection_requires_explicit_when_ambiguous(self):
        session, _ = fake_session(self.root)
        self.assertEqual(cp.resolve_v4_session(), session)
        fake_session(self.root, 'guided_v4_two')
        with self.assertRaisesRegex(ValueError, '找到 2'):
            cp.resolve_v4_session()
        self.assertEqual(cp.resolve_v4_session(session), session)
    def test_cache_binds_trace_source_and_reports(self):
        session, rows = fake_session(self.root)
        record, _, _ = self.audit(session)
        self.assertTrue(cp.validate_candidate_audit(record))
        source = session / 'training/source_snapshot' / FAKE_SOURCE; source.write_text('changed = True\n')
        self.assertFalse(cp.validate_candidate_audit(record))
        source.write_text('verified = True\n')
        self.assertTrue(cp.validate_candidate_audit(record))
        Path(record['evidence']['reward']['path']).write_text('{"passed":false}')
        self.assertFalse(cp.validate_candidate_audit(record))
    def test_incomplete_contact_does_not_run_reward_or_allow_ppo(self):
        session, _ = fake_session(self.root)
        record, _, reward = self.audit(session, contact=False, strict=False)
        self.assertFalse(record['passed']); reward.assert_not_called()
        self.assertTrue(Path(record['archive']).is_file())
        with self.assertRaisesRegex(RuntimeError, '没有启动仿真或PPO'):
            self.audit(session, contact=False)
    def test_failed_reward_exports_and_refuses(self):
        session, _ = fake_session(self.root)
        with self.assertRaisesRegex(RuntimeError, '没有启动仿真或PPO'):
            self.audit(session, reward=False)
        self.assertEqual(len(list((self.root / 'usb_review/route_05a').glob('*.zip'))), 1)
    def test_prepare_calls_gate_before_probe(self):
        from . import station
        with patch.object(cp, 'ensure_candidate_audit', side_effect=RuntimeError('closed')), patch.object(station, 'invoke_probe') as probe:
            with self.assertRaisesRegex(RuntimeError, 'closed'):
                station.prepare('cpu', v4_session=self.root)
            probe.assert_not_called()
    def test_direct_cli_and_resume_cannot_skip_audit(self):
        from . import cli
        for extra in ([], ['--checkpoint', str(self.root / 'fake.pt')]):
            with patch.object(cp, 'ensure_candidate_audit', side_effect=RuntimeError('closed')), patch.object(cli, 'child') as child:
                with self.assertRaisesRegex(RuntimeError, 'closed'):
                    cli.main(['train', *extra])
                child.assert_not_called()
    def test_direct_worker_rejects_before_app_import(self):
        import subprocess, sys
        run = self.root / 'worker'; write(run / 'request.json', dict(mode='train'))
        process = subprocess.run([sys.executable, str(Path(__file__).with_name('worker.py')), '--run', str(run)], capture_output=True, text=True, timeout=30)
        self.assertNotEqual(process.returncode, 0)
        error = read(run / 'worker_error.json')
        self.assertEqual(error['stage'], 'inputs')
        self.assertIn('no simulation or PPO started', error['error'])
        self.assertNotIn('AppLauncher', error['traceback'])
    def test_direct_bootstrap_rejects_missing_audit(self):
        from .bootstrap import initialize
        run = self.root / 'bootstrap'; write(run / 'request.json', dict(initial_actor='unused.pt'))
        with self.assertRaisesRegex(ValueError, 'PPO must not start'):
            initialize(object(), run)
    def test_audit_reports_cannot_be_swapped_between_sessions(self):
        session, _ = fake_session(self.root)
        record, _, _ = self.audit(session)
        for role in ('contact', 'reward'):
            path = Path(record['evidence'][role]['path']); original = read(path)
            changed = dict(original, source_session=str(self.root / 'other_session')); write(path, changed)
            record['evidence'][role]['sha256'] = sha(path)
            self.assertFalse(cp.validate_candidate_audit(record))
            changed = dict(original, inputs={'session.json': 'wrong'}); write(path, changed)
            record['evidence'][role]['sha256'] = sha(path)
            self.assertFalse(cp.validate_candidate_audit(record))
            write(path, original); record['evidence'][role]['sha256'] = sha(path)
            self.assertTrue(cp.validate_candidate_audit(record))
    def test_checkpoint_traversal_and_symlink_are_rejected(self):
        session, rows = fake_session(self.root)
        escaped = dict(rows[1], checkpoint=str(session / 'training/../../secret.pt'))
        with self.assertRaisesRegex(ValueError, 'Unsafe checkpoint'):
            cp.local_checkpoint(session, escaped)
        secret = self.root / 'secret.pt'; secret.write_bytes(b'do not export')
        model = Path(rows[1]['checkpoint']); model.unlink(); model.symlink_to(secret)
        with self.assertRaisesRegex(ValueError, 'escapes session boundary'):
            cp.local_checkpoint(session, rows[1])
        with self.assertRaisesRegex(ValueError, 'escapes session boundary'):
            cp.audit_inputs(session)
    def test_raw_and_source_symlinks_are_rejected(self):
        session, rows = fake_session(self.root)
        secret = self.root / 'secret'; secret.write_bytes(b'do not export')
        raw = session / 'final_evaluation' / (rows[1]['cases'][0]['artifact_prefix'] + '_trace.npz')
        raw.unlink(); raw.symlink_to(secret)
        with self.assertRaisesRegex(ValueError, 'escapes session boundary'):
            cp.audit_inputs(session)
        raw.unlink(); raw.write_text('{}')
        source = session / 'training/source_snapshot' / FAKE_SOURCE
        source.unlink(); source.symlink_to(secret)
        with self.assertRaisesRegex(ValueError, 'escapes session boundary'):
            cp.source_evidence(session)
    def test_empty_or_untrusted_source_snapshot_cannot_pass(self):
        session, _ = fake_session(self.root)
        source = session / 'training/source_snapshot' / FAKE_SOURCE
        source.unlink()
        self.assertIn('source_snapshot_manifest', cp.required_evidence_errors(session))
        source.write_text('verified = True\n')
        write(session / 'training/source_manifest.json', {FAKE_SOURCE: 'wrong'})
        self.assertIn('source_snapshot_manifest', cp.required_evidence_errors(session))
    def test_unsafe_input_diagnostic_never_copies_secret(self):
        import zipfile
        session, rows = fake_session(self.root)
        secret = self.root / 'secret'; secret.write_bytes(b'never-copy-this-secret')
        model = Path(rows[1]['checkpoint']); model.unlink(); model.symlink_to(secret)
        record = cp.ensure_candidate_audit(session, strict=False)
        self.assertFalse(record['passed'])
        with zipfile.ZipFile(record['archive']) as archive:
            self.assertNotIn(b'never-copy-this-secret', b''.join(archive.read(n) for n in archive.namelist()))
            self.assertFalse(any(n.endswith('/final/model.pt') for n in archive.namelist()))
    def test_preparation_and_final_zip_keep_self_contained_audit_summaries(self):
        import shutil, zipfile
        from .station import compact_preparation
        from .compact import export_review
        session, _ = fake_session(self.root)
        record, _, _ = self.audit(session)
        audit_root = Path(record['evidence']['contact']['path']).parent
        (audit_root / 'large_old_trace.npz').write_bytes(b'not-a-summary')
        (audit_root / 'old_model.pt').write_bytes(b'not-a-summary')
        preparation = self.root / 'prepare'
        write(preparation / 'preparation.json', dict(status='prepared', candidate_audit=record))
        work = self.root / 'training_session'; work.mkdir()
        compact_preparation(preparation, work / 'preparation')
        frozen = work / 'preparation/candidate_audit'
        self.assertTrue(read(frozen / 'provenance.json')['self_contained'])
        self.assertEqual({p.suffix for p in frozen.iterdir()}, {'.json', '.md'})
        shutil.rmtree(audit_root)
        review = self.root / 'self_contained_review'
        archive = export_review(work, review, dict(status='interrupted', mode='train'))
        manifest = read(review / 'review_manifest.json')
        self.assertEqual(manifest['candidate_audit_summary'], 'candidate_audit/provenance.json')
        proof = read(review / 'candidate_audit/provenance.json')
        for name, info in proof['files'].items():
            self.assertEqual(sha(review / 'candidate_audit' / name), info['sha256'])
        self.assertTrue(read(review / 'candidate_audit/contact.json')['passed'])
        self.assertTrue(read(review / 'candidate_audit/reward.json')['passed'])
        self.assertFalse((review / 'preparation/candidate_audit').exists())
        with zipfile.ZipFile(archive) as zipped:
            self.assertFalse(any(n.endswith(('.npz', '.pt')) for n in zipped.namelist()))
    def test_audit_summary_copy_reports_changed_bound_json(self):
        session, _ = fake_session(self.root)
        record, _, _ = self.audit(session)
        Path(record['evidence']['contact']['path']).write_text('{}')
        summary = cp.copy_audit_summaries(record, self.root / 'summary')
        self.assertFalse(summary['self_contained'])
        self.assertIn('contact/bound_result', summary['missing'])
        self.assertFalse((self.root / 'summary/contact.json').exists())
    def test_archive_keeps_selected_and_new_candidate_when_baseline_wins(self):
        from .compact import export_review
        session, rows = fake_session(self.root)
        metrics = dict(physics_failure_rate=0, momentum_unchecked_rate=0, success_rate=1, qualified_rate=1,
                       mean_raw_recovery_score=1, mean_impact_bodyweights=90, mean_end_tilt_deg=5, mean_stable_s=4)
        for row in rows: row['metrics'] = metrics
        write(session / 'candidate_comparison.json', rows)
        selected = dict(rows[0], run=str(session / 'final_evaluation'), behavior_status='fixed_condition_passed')
        review = self.root / 'review'
        export_review(session, review, dict(status='completed', mode='train'), selected)
        self.assertTrue((review / 'selected.pt').is_file()); self.assertTrue((review / 'secondary.pt').is_file())
        secondary = read(review / 'secondary_selection.json')
        self.assertEqual(secondary['evidence_role'], 'best_new_candidate')
        self.assertEqual(secondary['label'], 'update_003000')
        manifest = read(review / 'review_manifest.json')
        self.assertEqual(len(manifest['retained_evidence_roles']), 2); self.assertEqual(manifest['missing_raw_evidence'], [])
        self.assertEqual(len(list(review.rglob('*_trace.npz'))), 2)
        self.assertEqual(len(list(review.rglob('*.pt'))), 2)
    def test_interrupted_export_without_model_is_diagnostic(self):
        from .compact import export_review
        session = self.root / 'interrupted'; session.mkdir()
        write(session / 'session.json', dict(status='interrupted'))
        review = self.root / 'review'
        export_review(session, review, dict(status='interrupted', mode='train'))
        self.assertEqual(read(review / 'review_manifest.json')['behavior_status'], 'no_evaluated_model')

    def test_archive_keeps_baseline_when_new_candidate_wins(self):
        from .compact import export_review
        session, rows = fake_session(self.root)
        selected = dict(rows[1], run=str(session / 'final_evaluation'), behavior_status='fixed_condition_passed')
        review = self.root / 'review'
        export_review(session, review, dict(status='completed', mode='train'), selected)
        secondary = read(review / 'secondary_selection.json')
        self.assertEqual(secondary['evidence_role'], 'session_start_baseline')
        self.assertEqual(secondary['label'], 'session_start_baseline')


if __name__ == '__main__':
    unittest.main()
