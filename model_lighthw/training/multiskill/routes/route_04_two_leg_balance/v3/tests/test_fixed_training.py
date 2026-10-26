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
from two_leg_balance_v3.spec import contract
from two_leg_balance_v2.spec import contract as old_contract
import run as launcher
from common import checked_checkpoint, sha


class FixedTrainingTests(unittest.TestCase):
    def test_fixed_contract_preserves_physics_reward_and_success(self):
        new, old = contract(), old_contract()
        self.assertFalse(new['observation_noise'])
        self.assertEqual(new['reset_jitter_rad'], 0.)
        for k in ('domain_randomization','external_pushes','init_at_random_ep_len','automatic_frozen_evaluation'):
            self.assertFalse(new[k])
        for k,v in old.items():
            if k not in ('version','task'): self.assertEqual(new[k],v,k)
        cfg=next(n for n in ast.parse((SOURCE/'env.py').read_text()).body if isinstance(n,ast.ClassDef) and n.name=='TwoLegBalanceCfg')
        constants={n.targets[0].id:ast.literal_eval(n.value) for n in cfg.body if isinstance(n,ast.Assign) and isinstance(n.value,ast.Constant)}
        self.assertFalse(constants['observation_noise'])
        self.assertEqual(constants['reset_jitter'],0.)

    def test_v2_checkpoint_is_rejected_without_implicit_transfer(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'v2.pt';torch.save({'infos':{'contract':old_contract()}},p)
            with self.assertRaisesRegex(ValueError,'contract mismatch'): checked_checkpoint(p,contract())

    def exercise(self,start):
        with tempfile.TemporaryDirectory() as tmp:
            parent=Path(tmp);output=parent/'out';saved=parent/'saved.pt';saved.write_bytes(b'test')
            calls=[]
            def execute(folder,request,geometry):
                calls.append(request.copy());folder.mkdir()
                checkpoint=folder/'final.pt';checkpoint.write_bytes(b'new-test-checkpoint')
                return {'completed_updates':start+request['updates'],'checkpoint_sha256':sha(checkpoint)}
            args=['run.py','train','--max_iterations','205','--save_every','10','--num_envs','2','--output',str(output)]
            if start:args+=['--checkpoint',str(saved)]
            with patch('sys.argv',args),patch.object(launcher,'audit',return_value={'candidates':[{'candidate_pass':True}]}),patch.object(launcher,'execute',side_effect=execute),patch.object(launcher,'checked_checkpoint',return_value={'infos':{'completed_updates':start}}):
                self.assertEqual(launcher.main(),0)
            self.assertEqual(len(calls),1)
            self.assertEqual(calls[0]['mode'],'train')
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

    def test_old_eval_every_option_is_not_silently_enabled(self):
        with patch('sys.argv',['run.py','train','--eval_every','100']),contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:launcher.main()
        self.assertEqual(error.exception.code,2)
