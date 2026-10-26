"""固定条件合同、旧模型拒载与无自动评估的真实入口调度。"""
import ast
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from test_protocol import SOURCE
from two_leg_balance_split_v4.spec import contract
from two_leg_balance_v3.spec import contract as old_contract
import run as launcher
from common import checked_checkpoint, sha


class FixedTrainingTests(unittest.TestCase):
    def test_fixed_contract_preserves_physics_and_success(self):
        new, old = contract(), old_contract()
        self.assertFalse(new['observation_noise'])
        self.assertEqual(new['reset_jitter_rad'], 0.)
        for k in ('domain_randomization','external_pushes','init_at_random_ep_len','automatic_frozen_evaluation'):
            self.assertFalse(new[k])
        for k in ('physics_dt','policy_dt','leg_pd','wheel_damping','leg_effort_Nm','wheel_effort_Nm',
                  'leg_velocity_rad_s','wheel_velocity_rad_s','friction','gravity','self_collisions',
                  'solver_iterations','entry_seconds','hold_seconds','recovery_seconds','min_clearance_m',
                  'max_line_error_m','max_target_tilt_deg','max_com_speed_m_s','max_ang_speed_rad_s'):
            self.assertEqual(new[k],old[k],k)
        self.assertNotEqual(new['weights'],old['weights'])
        self.assertEqual(new['mode'],'rear')
        cfg=next(n for n in ast.parse((SOURCE/'env.py').read_text()).body if isinstance(n,ast.ClassDef) and n.name=='TwoLegBalanceCfg')
        constants={n.targets[0].id:ast.literal_eval(n.value) for n in cfg.body if isinstance(n,ast.Assign) and isinstance(n.value,ast.Constant)}
        self.assertFalse(constants['observation_noise'])
        self.assertEqual(constants['reset_jitter'],0.)

    def test_v3_checkpoint_is_rejected_without_implicit_transfer(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'v3.pt';torch.save({'infos':{'contract':old_contract()}},p)
            with self.assertRaisesRegex(ValueError,'contract mismatch'): checked_checkpoint(p,contract())

    def exercise(self,start,skill="R4a",profile="baseline_v4"):
        with tempfile.TemporaryDirectory() as tmp:
            parent=Path(tmp);output=parent/'out';saved=parent/'saved.pt';saved.write_bytes(b'test')
            calls=[]
            def execute(folder,request,geometry):
                calls.append(request.copy());folder.mkdir()
                checkpoint=folder/'final.pt';checkpoint.write_bytes(b'new-test-checkpoint')
                return {'completed_updates':start+request['updates'],'checkpoint_sha256':sha(checkpoint)}
            args=['run.py','train','--max_iterations','205','--save_every','10','--num_envs','2','--output',str(output),'--skill',skill,'--reward_profile',profile]
            if start:args+=['--checkpoint',str(saved)]
            with patch('sys.argv',args),patch.object(launcher,'audit',return_value={'candidates':[{'candidate_pass':True}]}),patch.object(launcher,'execute',side_effect=execute),patch.object(launcher,'checked_checkpoint',return_value={'infos':{'completed_updates':start}}):
                self.assertEqual(launcher.main(),0)
            self.assertEqual(len(calls),1)
            self.assertEqual(calls[0]['mode'],'train')
            self.assertEqual(calls[0]['support_mode'],'rear' if skill=='R4a' else 'front')
            self.assertEqual(calls[0]['skill'],skill)
            self.assertEqual(calls[0]['reward_profile'],profile)
            self.assertEqual(calls[0]['updates'],205)
            self.assertEqual(calls[0]['save_every'],10)
            result=json.loads((output/'result.json').read_text())
            self.assertEqual(result['completed_updates'],start+205)
            self.assertEqual(result['frozen_reviews'],[])
            self.assertEqual(result['review_status'],'awaiting_user_request')
            self.assertFalse(result['behavior_accepted'])
            self.assertFalse(list(output.glob('eval_*')))

    def test_fresh_full_budget_is_one_worker_without_final_review(self):self.exercise(0)
    def test_explicit_resume_keeps_full_additional_budget_without_review(self):self.exercise(200)

    def test_r4b_dispatch_is_front_only_without_review(self):self.exercise(0,"R4b")

    def test_soft_transfer_dispatch_is_fresh_and_without_review(self):
        self.exercise(0, profile='soft_transfer_v2')

    def test_audit_checkpoint_cannot_resume_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / 'audit.pt'
            checkpoint.write_bytes(b'audit-only')
            args = ['run.py', 'train', '--checkpoint', str(checkpoint), '--reward_profile', 'unload_diag_v1']
            with patch('sys.argv', args), patch.object(launcher, 'audit', return_value={'candidates': [{'candidate_pass': True}]}), patch.object(launcher, 'checked_checkpoint', return_value={'infos': {'completed_updates': 200, 'audit_only': True}}), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    launcher.main()
            self.assertEqual(error.exception.code, 2)

    def test_execute_contract_records_selected_reward_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / 'run'
            request = dict(hold_seconds=2.0, support_mode='rear', reward_profile='unload_diag_v1')
            geometry = {'candidates': [{'candidate_pass': True}]}
            with patch.object(launcher, 'snapshot'), patch.object(launcher, 'write') as write_mock:
                # execute is allowed to reach the worker only after metadata is written; stop there.
                with patch('subprocess.Popen', side_effect=RuntimeError('stop after metadata')):
                    with self.assertRaisesRegex(RuntimeError, 'stop after metadata'):
                        launcher.execute(run, request, geometry)
            contract_writes = [call.args for call in write_mock.call_args_list if Path(call.args[0]).name == 'contract.json']
            self.assertEqual(len(contract_writes), 1)
            self.assertEqual(contract_writes[0][1]['reward_profile'], 'unload_diag_v1')

    def test_old_eval_every_option_is_not_silently_enabled(self):
        with patch('sys.argv',['run.py','train','--eval_every','100']),contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:launcher.main()
        self.assertEqual(error.exception.code,2)
