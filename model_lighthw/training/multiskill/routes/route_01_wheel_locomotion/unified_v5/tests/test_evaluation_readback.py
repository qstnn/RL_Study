"""Readback preserves trajectory evidence while avoiding hundreds of synchronizations."""
import unittest
from unittest.mock import patch
import numpy as np
import torch
from test_v5_worker_state import worker


class ReadbackTests(unittest.TestCase):
    def test_exact_values_order_and_dtype_with_one_copy_per_group(self):
        values={}
        for i in range(40):
            values[f'f{i}']=torch.arange(16,dtype=torch.float32)[::2]+i
            values[f'i{i}']=torch.arange(8,dtype=torch.int64)*10**12+i
        values['flags']=torch.tensor([True,False]*4)
        values['float64']=torch.tensor([1.+1.e-12]*8,dtype=torch.float64)
        expected={key:value.numpy().copy() for key,value in values.items()}
        calls=[];original=torch.Tensor.cpu
        def counted(tensor,*args,**kwargs):calls.append(tensor.shape);return original(tensor,*args,**kwargs)
        with patch.object(torch.Tensor,'cpu',new=counted):actual=worker.cpu_measurements(values)
        self.assertEqual(list(actual),list(values));self.assertEqual(len(calls),4)
        for key,value in expected.items():
            self.assertEqual(actual[key].dtype,value.dtype);np.testing.assert_array_equal(actual[key],value)
        for value in values.values():value.zero_()
        for key,value in expected.items():np.testing.assert_array_equal(actual[key],value)

    def test_complete_motor_frames_keep_environment_time_joint_axes(self):
        values={name:torch.arange(4*20*16,dtype=torch.float32).reshape(4,20,16)+i*10000 for i,name in enumerate(('applied','requested','velocity'))}
        actual=worker.cpu_measurements(values)
        for name,value in values.items():
            for i in range(4):np.testing.assert_array_equal(actual[name][i],value[i].numpy())


if __name__=='__main__':unittest.main()
