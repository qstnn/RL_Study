"""Real PPO regression checks for rare-condition reference retention."""
from copy import deepcopy
import unittest
from unittest.mock import patch
import torch
from test_v5_optimizer import make_algorithm, collect, nested_equal

class ReferenceGroupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_rare_group_gets_equal_sampling_mass(self):
        alg=make_algorithm(reference_loss_coef=.1,reference_batch_size=4096)
        x=torch.randn(1001,300)
        with torch.no_grad():y=alg.policy.act_inference(x)
        y[-1]+=alg.reference_action_scale
        groups=torch.tensor([0]*1000+[1])
        alg.set_reference_data(x,y,torch.ones(1001,dtype=torch.bool),group_ids=groups)
        torch.manual_seed(13)
        loss=float(alg._reference_loss())
        self.assertGreater(loss,.45);self.assertLess(loss,.55)
        stats=alg._reference_diagnostics()
        self.assertAlmostEqual(stats['reference_group_mse_max'],1.,places=5)
        self.assertEqual(stats['reference_group_count'],2.)

    def test_group_ids_filter_with_success_and_restore_exactly(self):
        alg=make_algorithm(reference_loss_coef=.1)
        x=torch.randn(10,300);y=torch.randn(10,16);groups=torch.tensor([2]*4+[9]*4+[99]*2)
        alg.set_reference_data(x,y,torch.tensor([True]*8+[False]*2),group_ids=groups)
        self.assertTrue(torch.equal(alg.reference_group_ids,groups[:8]))
        saved=alg.training_state_dict();restored=make_algorithm(reference_loss_coef=.1)
        restored.policy.load_state_dict(alg.policy.state_dict());restored.optimizer.load_state_dict(deepcopy(alg.optimizer.state_dict()))
        restored.load_training_state_dict(saved)
        self.assertTrue(nested_equal(saved,restored.training_state_dict()))
        for invalid in (torch.zeros(10),torch.full((10,),-1,dtype=torch.long),torch.zeros(9,dtype=torch.long)):
            with self.assertRaises(ValueError):alg.set_reference_data(x,y,torch.ones(10,dtype=torch.bool),group_ids=invalid)

    def test_reference_guard_rejects_proposal_even_when_rollout_kl_passes(self):
        alg=make_algorithm(reference_loss_coef=.1,reference_max_group_mse=.02)
        x=torch.randn(8,300)
        with torch.no_grad():y=alg.policy.act_inference(x)
        alg.set_reference_data(x,y,torch.ones(8,dtype=torch.bool),group_ids=torch.tensor([0]*7+[1]))
        before=[p.detach().clone() for p in alg.actor_parameters]
        adam=[deepcopy(alg.optimizer.state.get(p)) for p in alg.actor_parameters]
        with torch.no_grad():alg.policy.actor[-1].bias.add_(100.)
        with patch.object(alg,'_rollout_kl',return_value=0.):result=alg._accept_actor_proposal(before,adam)
        self.assertFalse(result['accepted'])
        self.assertTrue(all(torch.equal(p,v) for p,v in zip(alg.actor_parameters,before)))
        self.assertFalse(any(p in alg.optimizer.state for p in alg.actor_parameters))

    def test_reference_groups_survive_actual_ppo_update_and_resume(self):
        alg=make_algorithm(reference_loss_coef=.1,reference_max_group_mse=.02)
        x=torch.randn(30,300)
        with torch.no_grad():y=alg.policy.act_inference(x)
        alg.set_reference_data(x,y,torch.ones(30,dtype=torch.bool),group_ids=torch.tensor([2]*29+[7]))
        collect(alg);metrics=alg.update()
        self.assertLessEqual(metrics['reference_group_mse_max'],.02+1.e-6)
        restored=make_algorithm(reference_loss_coef=.1,reference_max_group_mse=.02)
        restored.policy.load_state_dict(alg.policy.state_dict());restored.optimizer.load_state_dict(deepcopy(alg.optimizer.state_dict()));restored.load_training_state_dict(alg.training_state_dict())
        collect(alg,seed=72);a=alg.update();collect(restored,seed=72);b=restored.update()
        self.assertEqual(a,b);self.assertTrue(nested_equal(alg.training_state_dict(),restored.training_state_dict()))


class ReferenceCoverageTests(unittest.TestCase):
    def test_baseline_cannot_protect_a_case_missing_from_reference(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from types import SimpleNamespace
        from test_v5_worker_state import worker, quality, CheckpointRunner
        args=dict(device='cpu',ground_friction=4.,rolling_coefficient=.02,leg_torque_limit=96.,wheel_torque_limit=48.,hitch_height=.07,unloaded_acceleration=1.,loaded_acceleration=.3,eval_distance=1.,critic_warmup_updates=50)
        cases=worker.protocol.cases(1.)
        runner=CheckpointRunner();runner.alg.reference_max_group_mse=.02
        runner.initialization_record={'reference_cache':{'case_keys':[quality.case_key(c) for i,c in enumerate(cases) if i not in (10,11)]}}
        env=SimpleNamespace(unwrapped=SimpleNamespace(scene={'robot':SimpleNamespace(data=SimpleNamespace(root_state_w=torch.zeros(16,13)))}))
        def child(cmd,**kwargs):
            out=Path(cmd[cmd.index('--output_dir')+1]);out.mkdir(parents=True)
            checkpoint=Path(cmd[cmd.index('--checkpoint')+1])
            rows=[{'case':c,'success':i!=11} for i,c in enumerate(cases)]
            worker.write_json(out/'evaluation.json',dict(sha256=worker.sha256(checkpoint),cases=rows,pass_count=15,complete_count=15))
            return SimpleNamespace(returncode=0)
        with TemporaryDirectory() as d,patch('subprocess.run',side_effect=child):
            with self.assertRaisesRegex(RuntimeError,'missing.*reference|reference.*missing'):
                worker.periodic_evaluation(runner,env,args,Path(d),-1,baseline=True)

class ReferenceIntegrityTests(unittest.TestCase):
    def test_resume_cannot_silently_drop_group_ids(self):
        alg=make_algorithm(reference_loss_coef=.1)
        x=torch.randn(4,300)
        with torch.no_grad():y=alg.policy.act_inference(x)
        alg.set_reference_data(x,y,torch.ones(4,dtype=torch.bool),group_ids=torch.tensor([0,0,1,1]))
        saved=alg.training_state_dict();saved['reference_group_ids']=None
        with self.assertRaisesRegex(ValueError,'Incomplete reference'):
            alg.load_training_state_dict(saved)

    def test_initialization_must_already_meet_group_bound(self):
        import numpy as np
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from types import SimpleNamespace
        from test_v5_worker_state import worker,quality
        alg=make_algorithm(reference_loss_coef=.1,reference_max_group_mse=.02)
        x=torch.randn(6,300)
        with torch.no_grad():y=alg.policy.act_inference(x)+1.
        with TemporaryDirectory() as directory:
            path=Path(directory)/'success.npz'
            np.savez(path,observation=x.numpy(),action=y.numpy(),successful=np.ones(6,dtype=np.bool_),cache_kind=np.asarray("deployed_actor_success_reference"),
                     group_id=np.zeros(6,dtype=np.int64),stage=np.full(6,'static_hold'),case_key=np.full(6,quality.case_key(worker.protocol.cases()[0])))
            saved={'infos':{'initialization':{'reference_cache':{'path':path.name,'sha256':worker.sha256(path)}}}}
            with self.assertRaisesRegex(ValueError,'already exceeds'):
                worker.install_reference_cache(SimpleNamespace(alg=alg),Path(directory)/'model.pt',saved)


class StageCoverageTests(unittest.TestCase):
    def test_baseline_requires_steady_hold_samples_and_accepts_complete_stages(self):
        import json
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from types import SimpleNamespace
        from test_v5_worker_state import worker,quality,CheckpointRunner
        args=dict(device='cpu',ground_friction=4.,rolling_coefficient=.02,leg_torque_limit=96.,wheel_torque_limit=48.,hitch_height=.07,unloaded_acceleration=1.,loaded_acceleration=.3,eval_distance=1.,critic_warmup_updates=50)
        cases=worker.protocol.cases(1.)
        def child(cmd,**kwargs):
            out=Path(cmd[cmd.index('--output_dir')+1]);out.mkdir(parents=True)
            checkpoint=Path(cmd[cmd.index('--checkpoint')+1])
            worker.write_json(out/'evaluation.json',dict(sha256=worker.sha256(checkpoint),
                cases=[{'case':c,'success':i!=11} for i,c in enumerate(cases)],pass_count=15,complete_count=15))
            return SimpleNamespace(returncode=0)
        for omit_hold in (True,False):
            with self.subTest(omit_hold=omit_hold),TemporaryDirectory() as d,patch('subprocess.run',side_effect=child):
                runner=CheckpointRunner();runner.alg.reference_max_group_mse=.02
                stages={quality.case_key(c):(['initial_settle','static_mid','static_hold'] if c['speed']==0. else
                    ['wait','move','brake','stop_settle','stop_hold']) for i,c in enumerate(cases) if i!=11}
                if omit_hold:stages[quality.case_key(cases[10])].remove('static_hold')
                runner.initialization_record={'reference_cache':{'case_keys':list(stages),'case_stages':stages}}
                env=SimpleNamespace(unwrapped=SimpleNamespace(scene={'robot':SimpleNamespace(data=SimpleNamespace(root_state_w=torch.zeros(16,13)))}))
                if omit_hold:
                    with self.assertRaisesRegex(RuntimeError,'missing reference stages'):
                        worker.periodic_evaluation(runner,env,args,Path(d),-1,baseline=True)
                else:
                    result=worker.periodic_evaluation(runner,env,args,Path(d),-1,baseline=True)
                    self.assertTrue(result['quality']['eligible'])
                record=json.loads((Path(d)/'reference_stage_coverage_check.json').read_text())
                self.assertEqual(record['passed'],not omit_hold)


if __name__=='__main__':unittest.main()
