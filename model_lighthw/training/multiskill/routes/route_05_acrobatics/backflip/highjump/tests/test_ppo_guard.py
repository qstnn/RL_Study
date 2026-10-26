"""使用真实 RSL-RL PPO 的 CPU 测试：Critic 预热与 KL 失败回退。"""
from pathlib import Path
from copy import deepcopy
import sys
import tempfile
import unittest
import torch

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/scripts/r5a.sh').is_file())
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from worker import make_runner
from launch import spec
torch.set_num_threads(2)


class DummyEnv:
    num_envs=8
    num_actions=16
    def get_observations(self):
        return torch.randn(8,400),{'observations':{'critic':torch.randn(8,109)}}


class GuardTests(unittest.TestCase):
    def test_actual_ppo_freezes_actor_then_rolls_back_large_update(self):
        torch.manual_seed(42)
        cfg=dict(policy=dict(class_name='ActorCritic',actor_hidden_dims=[512,256,128],critic_hidden_dims=[512,256,128],
                    activation='elu',init_noise_std=.05,noise_std_type='log'),
            algorithm=dict(class_name='PPO',num_learning_epochs=2,num_mini_batches=2,learning_rate=1e-5,
                schedule='fixed',desired_kl=.01,entropy_coef=.001),num_steps_per_env=24,save_interval=50,empirical_normalization=False)
        with tempfile.TemporaryDirectory() as directory:
            run=Path(directory);runner=make_runner(run)(DummyEnv(),cfg,device='cpu')
            alg=runner.alg
            def update():
                with torch.inference_mode():
                    for _ in range(24):
                        obs=torch.randn(8,400);critic=torch.randn(8,109)
                        action=alg.act(obs,critic)
                        alg.process_env_step(torch.randn(8)+action[:,0],torch.zeros(8),{})
                    alg.compute_returns(torch.randn(8,109))
                return alg.update()
            before=deepcopy(alg.policy.state_dict());update()
            for k,v in before.items():
                if k.startswith('actor.') or k=='log_std':self.assertTrue(torch.equal(v,alg.policy.state_dict()[k]),k)
            self.assertFalse(torch.equal(before['critic.0.weight'],alg.policy.critic[0].weight))
            runner.completed_updates=spec.CONTRACT['critic_warmup_updates']
            old_lr=spec.CONTRACT['actor_learning_rate'];before=deepcopy(alg.policy.state_dict());adam=deepcopy(alg.optimizer.state_dict())
            try:
                spec.CONTRACT['actor_learning_rate']=.1
                losses=update()
            finally:spec.CONTRACT['actor_learning_rate']=old_lr
            self.assertEqual(losses['accepted_update'],0.)
            for k,v in before.items():self.assertTrue(torch.equal(v,alg.policy.state_dict()[k]),k)
            after=alg.optimizer.state_dict()
            for k,state in adam['state'].items():
                for name,value in state.items():
                    if isinstance(value,torch.Tensor):self.assertTrue(torch.equal(value,after['state'][k][name]))
            self.assertEqual(runner.completed_updates,spec.CONTRACT['critic_warmup_updates']+1)


if __name__=='__main__':unittest.main()
