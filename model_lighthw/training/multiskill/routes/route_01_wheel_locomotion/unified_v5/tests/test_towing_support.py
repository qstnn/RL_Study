"""Actual reward functions on controlled force states, without Isaac startup."""
import ast
from pathlib import Path
from types import SimpleNamespace as N
import unittest
import torch

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'source/rl_training').is_dir())
PATH=ROOT/'source/rl_training/rl_training/tasks/manager_based/locomotion/multiskill/lighthw/wheel_unified_v5/mdp.py'

def reward(name):
    nodes=[n for n in ast.parse(PATH.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name]
    if not nodes:raise AssertionError('Missing production reward: '+name)
    ns={'torch':torch};exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(PATH),'exec'),ns)
    return ns[name]

class TowingSupportTests(unittest.TestCase):
    def test_margin_is_towing_only_and_needs_load_above_acceptance_floor(self):
        normals=torch.tensor([[3.,26.,26.,49.],[13.,26.,26.,39.],[3.,26.,26.,49.],[3.,26.,26.,49.]])
        force=torch.zeros(4,4,3);force[:,:,2]=normals
        env=N(scene={'contact':N(data=N(net_forces_w=force))},r1_normal=torch.tensor([[0.,0.,1.]]*4),r1_payload_kg=torch.tensor([0.,0.,0.,15.]),r1_towed_kg=torch.tensor([40.,40.,0.,0.]))
        result=reward('towing_support_margin')(env,N(name='contact',body_ids=[0,1,2,3]))
        self.assertGreater(float(result[0]),float(result[1]));self.assertGreater(float(result[0]),.1)
        self.assertTrue(torch.equal(result[2:],torch.zeros(2)))
        self.assertTrue(torch.isfinite(result).all())

    def test_towing_margin_remains_active_through_start_and_braking(self):
        force=torch.zeros(1,4,3);force[:,:,2]=torch.tensor([[0.,20.,40.,40.]])
        env=N(scene={'contact':N(data=N(net_forces_w=force))},r1_normal=torch.tensor([[0.,0.,1.]]),r1_payload_kg=torch.zeros(1),r1_towed_kg=torch.tensor([40.]))
        fn=reward('towing_support_margin');values=[]
        for speed in (0.,.0001,.03,.3):
            env.command_manager=N(get_command=lambda _:torch.tensor([[speed,0.,0.]]))
            values.append(float(fn(env,N(name='contact',body_ids=[0,1,2,3]))))
        self.assertEqual(values,[values[0]]*4);self.assertGreater(values[0],0.)

if __name__=='__main__':unittest.main()
