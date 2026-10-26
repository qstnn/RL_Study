"""CPU数据/版本准入与控制坐标检查；不能替代GPU行为验收。"""
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from common_guided import contract,require_teacher,write
from teacher import candidates,pitch_and_rate,inverse_targets,Teacher
from learning import validate_arrays,imitation_weights,fit_actor,actor_network,load_actor,load_checkpoint,preflight
from learning import archived_learning_sources,corrected_trace_arrays
from two_leg_balance_split_v4.core import targets
from two_leg_balance_split_v4.spec import STAND,LOW,HIGH


def dataset():
    x=np.zeros((100,305),dtype=np.float32);phase=np.arange(100,dtype=np.int64)%5
    x[np.arange(100),299+phase]=1.
    return dict(observations=x,actions=np.zeros((100,16),dtype=np.float32),phase=phase,
                episode=np.zeros(100,dtype=np.int64),time=np.arange(100,dtype=np.float32)*.02)


class GuidedChecks(unittest.TestCase):
    def test_finite_budget(self):
        rows=candidates();self.assertEqual(len(rows),256)
        self.assertEqual([r['id'] for r in rows],list(range(256)))
        self.assertEqual(set(r['ramp_s'] for r in rows),{.75,1.,1.5,2.})

    def test_pitch_rate_matches_gravity_kinematics(self):
        torch.manual_seed(7)
        g=torch.randn(200,3,dtype=torch.float64);g=g/g.norm(dim=-1,keepdim=True)
        g=g[g[:,2]<-.3];w=torch.randn_like(g)
        p,r=pitch_and_rate(g,w);dg=-torch.linalg.cross(w,g)
        nxt,_=pitch_and_rate(g+dg*1e-7,w)
        torch.testing.assert_close((nxt-p)/1e-7,r,atol=1e-5,rtol=1e-4)

    def test_return_candidates_preserve_verified_capture(self):
        capture=candidates()[50];rows=candidates('return')
        self.assertEqual(len(rows),64)
        for row in rows:
            self.assertEqual({k:row[k] for k in capture if k!='id'},
                             {k:v for k,v in capture.items() if k!='id'})
            self.assertLessEqual(row['return_duration'],3.)

    def test_pitch_sign(self):
        p,r=pitch_and_rate(torch.tensor([[-.8,0.,-.6]]),torch.tensor([[0.,-.5,0.]]))
        self.assertLess(float(p),0);self.assertAlmostEqual(float(r),-.5,places=6)

    def test_inverse_mapping_extremes_and_pd_offset(self):
        q=torch.tensor([STAND,LOW,HIGH])
        a=inverse_targets(q)
        torch.testing.assert_close(targets(torch.cat((a,torch.zeros(3,4)),-1)),q)
        self.assertTrue(torch.equal(a[0],torch.zeros(12)))

    def test_contract_no_automatic_evaluation_or_teacher_actor(self):
        c=contract()
        for key in ('observation_noise','domain_randomization','automatic_frozen_evaluation','teacher_in_actor_path'):
            self.assertFalse(c[key])
        self.assertEqual(c['actor_observation_dim'],305)
        self.assertEqual(c['hold_seconds'],2.)
        self.assertFalse(c['missed_hold_early_termination'])

    def test_data_valid(self):validate_arrays(dataset())

    def test_privileged_input_rejected(self):
        d=dataset();d['observations']=np.pad(d['observations'],((0,0),(0,3)))
        with self.assertRaises(ValueError):validate_arrays(d)

    def test_unbounded_action_rejected(self):
        d=dataset();d['actions'][0,0]=1.01
        with self.assertRaises(ValueError):validate_arrays(d)

    def test_nonfinite_rejected(self):
        d=dataset();d['observations'][0,0]=np.nan
        with self.assertRaises(ValueError):validate_arrays(d)

    def test_wrong_phase_rejected(self):
        d=dataset();d['phase'][0]=4
        with self.assertRaises(ValueError):validate_arrays(d)

    def test_partial_protocol_rejected(self):
        d=dataset();d={k:v[:3] for k,v in d.items()}
        with self.assertRaises(ValueError):validate_arrays(d)

    def test_stage_balancing_and_transition_weight(self):
        phase=torch.tensor([0,0,1,1,2,2,3,4]);t=torch.tensor([.5,.8,2.,3.99,5.,4.01,8.,11.])
        w=imitation_weights(phase,t)
        self.assertAlmostEqual(float(w.mean()),1.,places=6)
        self.assertGreater(float(w[3]),float(w[2]))
        self.assertGreater(float(w[5]),float(w[4]))

    def test_partial_teacher_cannot_start_collection(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);write(p/'result.json',dict(kind='teacher_replay',dataset_eligible=False))
            with patch('common_guided.require_sources'):
                with self.assertRaises(ValueError):require_teacher(p)

    def test_legacy_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'model.pt';torch.save(dict(infos={'contract':{'version':'lighthw_r4_split_v4'}}),p)
            with self.assertRaises(ValueError):load_checkpoint(p)

    def test_legacy_actor_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'actor.pt';torch.save(dict(kind='legacy_actor'),p)
            with self.assertRaises(ValueError):load_actor(p)

    def test_no_implicit_fresh_training(self):
        with self.assertRaises(ValueError):preflight(SimpleNamespace(mode='train',actor=None,checkpoint=None))

    def test_small_supervised_fit_is_only_numeric_validation(self):
        d=dataset();torch.manual_seed(42)
        initial=actor_network()
        with torch.no_grad():initial_mse=float(initial(torch.from_numpy(d['observations'])).square().mean())
        model,history=fit_actor(d,20,42)
        with torch.no_grad():final_mse=float(model(torch.from_numpy(d['observations'])).square().mean())
        self.assertLess(final_mse,initial_mse*.25)
        self.assertEqual(history[-1]['epoch'],20)

    def test_correction_uses_pre_action_state_and_excludes_ended_slots(self):
        d={'policy_observation':np.zeros((3,2,305),dtype=np.float32)}
        d['policy_observation'][:,:,299]=1.
        for k in ('gravity','ang_vel','com','com_vel'):
            d[k]=np.zeros((60,2,3),dtype=np.float32)
        d['wheel_pos']=np.zeros((60,2,4,3),dtype=np.float32)
        d['force_z']=np.zeros((60,2,4),dtype=np.float32)
        d['task_t']=np.broadcast_to(np.arange(1,61,dtype=np.float32)[:,None]*.001,(60,2)).copy()
        d['com'][:,:,0]=d['task_t'];d['active']=np.ones((60,2),dtype=bool);d['active'][40:,0]=False
        class Labeler:
            def __init__(self,*args):pass
            def __call__(self,base,t):
                action=torch.zeros(2,16);action[:,0]=base.metrics()['com'][:,0];return action
        with patch('teacher.Teacher',Labeler):r=corrected_trace_arrays(d,{}, {})
        np.testing.assert_allclose(r['actions'][:,0],[.001,.001,.02,.02,.04],atol=1e-7)
        np.testing.assert_allclose(r['time'],[0,0,.02,.02,.04],atol=1e-7)
        self.assertEqual(r['observations'].shape,(5,305));self.assertTrue(r['is_correction'].all())

    def test_archived_corrections_allow_only_learning_code_change(self):
        from common_guided import sources,REPO,sha,contract,write
        from common import asset_hashes
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);recorded=sources(True)
            for rel in recorded:
                dest=root/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True)
                dest.write_bytes((REPO/rel).read_bytes())
            key=str((HERE/'learning.py').relative_to(REPO));file=root/'source_snapshot'/key
            file.write_text(file.read_text()+'\n# prior learning version\n');recorded[key]=sha(file)
            write(root/'source_manifest.json',recorded);write(root/'contract.json',contract())
            write(root/'asset_manifest.json',asset_hashes());self.assertEqual(archived_learning_sources(root),recorded)
            other=next(k for k in recorded if k!=key);recorded[other]='changed'
            write(root/'source_manifest.json',recorded)
            with self.assertRaises(ValueError):archived_learning_sources(root)


if __name__=='__main__':unittest.main()
