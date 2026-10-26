"""Reject teacher demonstration data and non-finite bounds at the training boundary."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import numpy as np
import torch
from test_v5_optimizer import make_algorithm
from test_v5_worker_state import worker, quality

class CacheSafetyTests(unittest.TestCase):
    def test_group_bound_must_be_finite(self):
        for value in (float('nan'),float('inf')):
            with self.subTest(value=value),self.assertRaisesRegex(ValueError,'reference'):
                make_algorithm(reference_loss_coef=.1,reference_max_group_mse=value)

    def test_teacher_labels_cannot_be_installed_as_actor_success(self):
        alg=make_algorithm(reference_loss_coef=.1,reference_max_group_mse=.02)
        x=torch.randn(6,300)
        with torch.no_grad():y=alg.policy.act_inference(x)
        for kind in (None,'teacher_labels_for_offline_fit_only','deployed_actor_success_reference'):
            with self.subTest(kind=kind),TemporaryDirectory() as d:
                p=Path(d)/'reference.npz'
                arrays=dict(observation=x.numpy(),action=y.numpy(),successful=np.ones(6,dtype=np.bool_),
                    group_id=np.zeros(6,dtype=np.int64),stage=np.full(6,'static_hold'),
                    case_key=np.full(6,quality.case_key(worker.protocol.cases()[0])))
                if kind is not None:arrays['cache_kind']=np.asarray(kind)
                np.savez(p,**arrays)
                saved={'infos':{'initialization':{'reference_cache':{'path':p.name,'sha256':worker.sha256(p)}}}}
                if kind=='deployed_actor_success_reference':
                    result=worker.install_reference_cache(SimpleNamespace(alg=alg),Path(d)/'model.pt',saved)
                    self.assertEqual(result['cache_kind'],kind)
                else:
                    with self.assertRaisesRegex(ValueError,'Actor-success'):
                        worker.install_reference_cache(SimpleNamespace(alg=alg),Path(d)/'model.pt',saved)

if __name__=='__main__':unittest.main()
