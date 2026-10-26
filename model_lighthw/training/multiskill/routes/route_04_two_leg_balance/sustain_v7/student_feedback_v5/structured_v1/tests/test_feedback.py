import unittest
import torch
import numpy as np
from common_feedback import prefix_actor
from features_feedback import FeedbackFeatures
from model_feedback import FeedbackActor,feedback_basis


def raw(pitch=0.,entry=False,stand=True,n=2):
    x=torch.zeros(n,5,61);x[:,:,3]=np.sin(pitch);x[:,:,5]=-np.cos(pitch)
    x[:,:,55 if entry else 57]=1;x[:,:,60]=float(stand)
    return x.flatten(1)


class FeedbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4);cls.prefix=prefix_actor()

    def test_basis_units(self):
        x=raw();x[:,-61+1]=.2
        b=feedback_basis(x,torch.tensor([[.2,.4],[-.2,-.4]]))
        torch.testing.assert_close(b[:,4],torch.ones(2))
        torch.testing.assert_close(b[:,5],torch.tensor([.12,-.12]))

    def test_prefix_and_head_boundary(self):
        model=FeedbackActor(self.prefix);builder=FeedbackFeatures(self.prefix,2,'cpu')
        x=builder(raw(),torch.ones(2,dtype=torch.bool))
        with torch.no_grad():torch.testing.assert_close(model(x),self.prefix(x[:,:307]).clamp(-1,1),rtol=0,atol=0)
        trigger=raw(-.2);trigger[:,-27:-11]=.123
        x=builder(trigger,torch.zeros(2,dtype=torch.bool))
        torch.testing.assert_close(x[:,307:323],torch.full((2,16),.123),rtol=0,atol=0)
        start=x[:,307:323].clone()
        torch.testing.assert_close(model(x),start.clamp(-1,1),rtol=0,atol=0)
        with torch.no_grad():model.coefficients[:,0]=.3
        for _ in range(40):x=builder(raw(-.5),torch.zeros(2,dtype=torch.bool))
        torch.testing.assert_close(x[:,307:323],start,rtol=0,atol=0)
        torch.testing.assert_close(model(x),torch.full((2,16),.3),rtol=0,atol=0)

    def test_selective_reset_and_frozen_prefix(self):
        model=FeedbackActor(self.prefix);builder=FeedbackFeatures(self.prefix,2,'cpu')
        builder(raw(-.2),torch.zeros(2,dtype=torch.bool));xraw=raw(-.3);xraw[0]=raw(entry=True,n=1)[0]
        x=builder(xraw,torch.tensor([True,False]))
        self.assertEqual(x[0,305:323].count_nonzero().item(),0);self.assertEqual(x[1,305].item(),1)
        model(x).square().sum().backward()
        self.assertTrue(all(p.grad is None for p in model.prefix.parameters()))
        self.assertIsNotNone(model.coefficients.grad)
        self.assertFalse(model.coefficients.grad[model.mask==0].any())
        with self.assertRaises(ValueError):model(x[:,:307])


if __name__=='__main__':unittest.main()
