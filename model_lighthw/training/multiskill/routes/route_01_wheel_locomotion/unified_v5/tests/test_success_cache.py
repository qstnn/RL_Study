"""Success-cache acceptance boundaries using the real 20-second protocol."""
import csv
import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
import numpy as np

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))
import build_success_cache as cache


def evaluation(root, *, fail_seed=None, noisy=True, variant='baseline', tamper=False):
    root.mkdir();checkpoint=root/'model.pt';checkpoint.write_bytes(b'test checkpoint, no simulator')
    digest=cache.sha256(checkpoint)
    cache.write_json(root/'request.json',dict(checkpoint=str(checkpoint),checkpoint_sha256=digest,
                   execution_contract={'fixture':True},physics_override={'position_iterations':4},noise=noisy))
    cache.write_json(root/'runtime_check.json',dict(policy_enable_corruption=noisy))
    reports=[]
    for seed in (43,44,45):
        case=dict(slope_deg=25.,payload_kg=0.,towed_kg=40.,speed=0.,goal_distance_m=1.,variant=variant)
        rows=[]
        for step in range(1,1001):
            rows.append(dict(t=step*.02,tilt_deg=0.,drawbar_force_N=170.,body_contact=0.,hitch_clearance_m=.07,
                vx=0.,velocity_error=0.,heading_rad=0.,height=.416,speed=0.,yaw_rate_abs=0.,
                four_wheel=float(seed!=fail_seed),x=0.,y=0.,z=.416,torque_abs_max=10.,command_vx=0.,target_vx=0.))
        prefix=root/f'seed{seed}_case00'
        with prefix.with_suffix('.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        result=cache.PROTOCOL.summarize(rows,case,None);result['seed']=seed
        if tamper:result['success']=not result['success']
        reports.append(result)
        np.savez_compressed(str(prefix)+'_policy_trace.npz',observation=np.zeros((1000,300),np.float32),
                            action=np.zeros((1000,16),np.float32),actor_action=np.zeros((1000,16),np.float32),
                            phase=np.array([0]*150+[3]*850))
    cache.write_json(root/'evaluation.json',dict(sha256=digest,cases=reports))
    return root


class SuccessCacheTests(unittest.TestCase):
    def test_three_noisy_successes_preserve_phase_and_case_labels(self):
        with TemporaryDirectory() as d:
            root=Path(d);source=evaluation(root/'eval');out=root/'success.npz'
            record=cache.build([source],out)
            self.assertEqual(record['case_count'],1);self.assertEqual(record['samples'],3000)
            with np.load(out,allow_pickle=False) as z:
                self.assertEqual(set(z['group_id']),{0,1,2});self.assertTrue(z['successful'].all());self.assertEqual(set(z['stage']),{'initial_settle','static_mid','static_hold'})
                self.assertEqual(z['case_key'].dtype.kind,'U');self.assertEqual(len(set(z['source_episode'])),3)

    def test_one_failed_seed_rejects_the_entire_condition(self):
        with TemporaryDirectory() as d:
            root=Path(d);source=evaluation(root/'eval',fail_seed=44);out=root/'success.npz'
            with self.assertRaisesRegex(ValueError,'No condition'):cache.build([source],out)
            self.assertFalse(out.exists())
            record=json.loads(out.with_suffix('.json').read_text());self.assertEqual(record['cases'][0]['passed'],2)

    def test_clean_repetitions_cannot_claim_noisy_robustness(self):
        with TemporaryDirectory() as d:
            root=Path(d);source=evaluation(root/'eval',noisy=False)
            with self.assertRaisesRegex(ValueError,'No condition'):cache.build([source],root/'success.npz')

    def test_tampered_success_flag_and_implicit_teacher_are_rejected(self):
        for options,error in (({'tamper':True},'does not reproduce'),({'variant':'pose_15'},'Teacher-controlled')):
            with self.subTest(options=options),TemporaryDirectory() as d:
                root=Path(d);source=evaluation(root/'eval',**options)
                with self.assertRaisesRegex(ValueError,error):
                    cache.build([source],root/'success.npz',variant=options.get('variant','baseline'))


if __name__=='__main__':unittest.main()
