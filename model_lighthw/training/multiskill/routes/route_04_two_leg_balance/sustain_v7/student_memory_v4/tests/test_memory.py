import unittest
import math
import numpy as np
import torch
from common_memory import initial_actor
from capture_memory import CaptureMemory,augment
from model_memory import MemoryActor
from verify_memory import reconstruct_memory


def raw(degrees,entry=False,stand=True):
    x=torch.zeros(len(degrees),5,61);a=torch.tensor(degrees)*math.pi/180
    x[:,:,3]=a.sin()[:,None];x[:,:,5]=-a.cos()[:,None]
    x[:,:,55 if entry else 57]=1;x[:,:,60]=int(stand)
    return x.flatten(1)


class MemoryTests(unittest.TestCase):
    def test_trigger_zero_then_40_steps(self):
        m=CaptureMemory(2,'cpu');a=m.update(raw([-6.,0.]))
        torch.testing.assert_close(a,torch.tensor([[1.,0.],[0.,0.]]))
        for _ in range(20):a=m.update(raw([-20.,0.]))
        self.assertAlmostEqual(float(a[0,1]),.5)
        for _ in range(20):a=m.update(raw([-56.,0.]))
        self.assertEqual(a.tolist(),[[1.,1.],[0.,0.]])
        for _ in range(100):a=m.update(raw([-56.,0.]))
        self.assertEqual(a.tolist(),[[1.,1.],[0.,0.]])

    def test_latch_survives_reverse_crossing(self):
        m=CaptureMemory(1,'cpu');m.update(raw([-6.]))
        for _ in range(40):a=m.update(raw([8.]))
        self.assertEqual(a.tolist(),[[1.,1.]])

    def test_selective_reset_and_return(self):
        m=CaptureMemory(2,'cpu');m.update(raw([-10.,-10.]));m.update(raw([-10.,-10.]))
        a=m.update(raw([0.,-10.]),torch.tensor([True,False]))
        self.assertEqual(a[0].tolist(),[0.,0.]);self.assertGreater(float(a[1,1]),0.)
        self.assertFalse(m.update(raw([-10.,-10.],stand=False)).any())
        self.assertFalse(m.update(raw([-10.,-10.],entry=True)).any())

    def test_offline_replay_independent_matches(self):
        seq=[raw([0.,0.],entry=True)]*3+[raw([-8.,0.])]+[raw([3.,-8.])]*50
        m=CaptureMemory(2,'cpu');actual=torch.stack([m.update(x) for x in seq]).numpy()
        np.testing.assert_allclose(actual,reconstruct_memory(torch.stack(seq).numpy()),atol=1e-7,rtol=0)

    def test_no_elapsed_episode_signal(self):
        a=CaptureMemory(1,'cpu');b=CaptureMemory(1,'cpu')
        for _ in range(170):b.update(raw([0.]))
        for _ in range(45):torch.testing.assert_close(a.update(raw([-8.])),b.update(raw([-8.])),rtol=0,atol=0)

    def test_initial_migration_and_stateless_forward(self):
        base=initial_actor();a=MemoryActor(base,'zero');b=MemoryActor(base,'exposed')
        for k,v in a.state_dict().items():torch.testing.assert_close(v,b.state_dict()[k],rtol=0,atol=0)
        x=raw([0.,-60.]);memory=torch.tensor([[0.,0.],[1.,.5]])
        torch.testing.assert_close(a(augment(x,memory,'zero')),base(x),rtol=1e-5,atol=1e-7)
        torch.testing.assert_close(b(augment(x,memory,'exposed')),base(x),rtol=1e-5,atol=1e-7)
        torch.testing.assert_close(b(augment(x,memory,'exposed')),b(augment(x,memory,'exposed')),rtol=0,atol=0)
        self.assertEqual(sum(p.numel() for p in a.parameters()),sum(p.numel() for p in b.parameters()))
        with self.assertRaises(ValueError):b(x)


if __name__=='__main__':unittest.main()
