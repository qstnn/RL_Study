import math
import unittest
import torch
from common_phase import actor_network
from model_phase import PhaseActor,observed_capture


def sample(pitch):
    obs=torch.zeros(len(pitch),5,61)
    obs[:,:,55+1]=1;obs[:,:,60]=1
    angle=torch.tensor(pitch)
    obs[:,:,3]=angle.sin()[:,None];obs[:,:,5]=-angle.cos()[:,None]
    return obs.flatten(1)


class PhaseTests(unittest.TestCase):
    def test_observed_gate_no_history_or_privileged_dependence(self):
        x=sample([0.,math.radians(-4.9),math.radians(-5.1),math.radians(-56.)])
        self.assertEqual(observed_capture(x).tolist(),[False,False,True,True])
        x[:,:244]=torch.randn_like(x[:,:244])*100
        self.assertEqual(observed_capture(x).tolist(),[False,False,True,True])
        x[2,-1]=0;x[3,-6]=1
        self.assertFalse(observed_capture(x).any())

    def test_both_modes_preserve_initial_actor(self):
        base=actor_network();x=sample([0.,-1.])
        a,b=PhaseActor(base,'uniform'),PhaseActor(base,'observed')
        torch.testing.assert_close(a(x),base(x),rtol=0,atol=0)
        torch.testing.assert_close(b(x),base(x),rtol=0,atol=0)
        self.assertEqual(sum(p.numel() for p in a.parameters()),sum(p.numel() for p in b.parameters()))

    def test_routing_changes_only_selected_head_gradient(self):
        model=PhaseActor(actor_network(),'observed')
        model(sample([0.])).sum().backward()
        self.assertGreater(float(model.rise.weight.grad.abs().sum()),0)
        self.assertEqual(float(model.capture.weight.grad.abs().sum()),0)
        model.zero_grad();model(sample([-1.])).sum().backward()
        self.assertEqual(float(model.rise.weight.grad.abs().sum()),0)
        self.assertGreater(float(model.capture.weight.grad.abs().sum()),0)

    def test_uniform_equal_heads_stay_equal(self):
        model=PhaseActor(actor_network(),'uniform');opt=torch.optim.Adam(model.parameters(),lr=.0003)
        model(sample([0.,-1.])).square().mean().backward();opt.step()
        for a,b in zip(model.rise.parameters(),model.capture.parameters()):self.assertTrue(torch.equal(a,b))

    def test_invalid_contract(self):
        with self.assertRaises(ValueError):PhaseActor(actor_network(),'teacher')
        with self.assertRaises(ValueError):observed_capture(torch.zeros(2,304))


if __name__=='__main__':unittest.main()
