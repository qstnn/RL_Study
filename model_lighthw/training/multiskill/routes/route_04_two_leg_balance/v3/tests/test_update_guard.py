"""实际worker更新闭包的事务测试，避免拒绝更新后残留Adam动量。"""
import ast
import copy
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
import numpy as np
import torch
from test_v3 import PhaseActorCritic


class GuardTests(unittest.TestCase):
    def exercise(self, jump):
        torch.manual_seed(123)
        worker=Path(__file__).resolve().parents[1]/'scripts/worker.py'
        node=next(n for n in ast.walk(ast.parse(worker.read_text())) if isinstance(n,ast.FunctionDef) and n.name=='update')
        p=PhaseActorCritic(305,88,16,actor_hidden_dims=[8],critic_hidden_dims=[8],noise_std_type='log',init_noise_std=.1)
        opt=torch.optim.Adam(p.parameters(),lr=1e-4)
        # 建立真实的非空Adam历史，确保测试覆盖动量和步数。
        opt.zero_grad();sum(x.square().sum() for x in p.parameters()).backward();opt.step()
        obs=torch.zeros(2,4,305);obs[:,:,-6]=1.
        p.update_distribution(obs.flatten(0,1))
        storage=NS(observations=obs,mu=p.action_mean.detach().reshape(2,4,16).clone(),sigma=p.action_std.detach().reshape(2,4,16).clone())
        alg=NS(policy=p,optimizer=opt,storage=storage,learning_rate=1e-4)
        owner=NS(alg=alg,completed_updates=0)
        before=copy.deepcopy(p.state_dict());optimizer=copy.deepcopy(opt.state_dict())
        def proposal():
            opt.zero_grad();sum(x.square().sum() for x in p.parameters()).backward();opt.step()
            with torch.no_grad():p.actor[-1].bias.add_(jump)
            return {'value_function':.1,'surrogate':.1,'entropy':.1}
        with tempfile.TemporaryDirectory() as tmp:
            scope=dict(self=owner,run=Path(tmp),original_update=proposal,torch=torch,np=np,copy=copy,json=json)
            exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(worker),'exec'),scope)
            scope['update']()
            row=json.loads((Path(tmp)/'updates.jsonl').read_text())
        return owner,before,optimizer,row

    def test_oversized_kl_restores_model_and_adam(self):
        owner,before,optimizer,row=self.exercise(2.)
        self.assertFalse(row['accepted']);self.assertFalse(row['actor_changed']);self.assertEqual(row['rollout_kl'],0.)
        self.assertGreater(row['proposed_kl'],.03)
        for k,v in before.items():torch.testing.assert_close(owner.alg.policy.state_dict()[k],v,atol=0,rtol=0)
        for i,state in optimizer['state'].items():
            for k,v in state.items():torch.testing.assert_close(owner.alg.optimizer.state_dict()['state'][i][k],v,atol=0,rtol=0)
        self.assertEqual(owner.alg.learning_rate,5e-5)
        self.assertEqual(owner.alg.optimizer.param_groups[0]['lr'],5e-5)
        self.assertEqual(owner.completed_updates,1)

    def test_small_update_is_retained(self):
        owner,before,optimizer,row=self.exercise(0.)
        self.assertTrue(row['accepted']);self.assertTrue(row['actor_changed'])
        self.assertLessEqual(row['rollout_kl'],.03)
        self.assertEqual(owner.alg.learning_rate,1e-4)


if __name__=='__main__':unittest.main()
