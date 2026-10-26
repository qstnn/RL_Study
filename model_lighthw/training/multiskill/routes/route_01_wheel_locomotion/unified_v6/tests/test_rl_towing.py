"""CPU语义回归：条件概率、共同进度、奖励隔离和64工况覆盖。"""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
import numpy as np
import torch
ROOT=next(p for p in Path(__file__).resolve().parents if (p/"source/rl_training").exists())
SRC=ROOT/"source/rl_training/rl_training/tasks/manager_based/locomotion/multiskill/lighthw/wheel_unified_v6"
def module(name):
    spec=importlib.util.spec_from_file_location("_v6_"+name,SRC/(name+".py"));out=importlib.util.module_from_spec(spec);spec.loader.exec_module(out);return out
core=module("core");spec=module("spec");policy=module("policy")
# Isaac math imports require Kit; extract unchanged pure reward bodies for CPU semantic assertions.
tree=ast.parse((SRC/"mdp.py").read_text());names={"loaded_motion_weight","towing_signed_progress","surface_height","towing_failure","towing_distance_reached"}
namespace={"torch":torch};exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[]),str(SRC/"mdp.py"),"exec"),namespace)
class Tests(unittest.TestCase):
    def test_conditional_probabilities(self):
        torch.manual_seed(42)
        p=policy.MotionExplorationActorCritic(300,252,16,actor_hidden_dims=[32],critic_hidden_dims=[32],noise_std_type="log")
        with torch.no_grad():p.log_std.copy_(torch.tensor([.04]*12+[.08]*4).log())
        obs=torch.zeros(5,300);obs[:,42]=torch.tensor([0.,.3,.3,.025,.3]);obs[:,299]=torch.tensor([1.,0.,1.,1.,.5])
        p.update_distribution(obs)
        expected=torch.tensor([.005,.005,.04,.0225,.04])
        torch.testing.assert_close(p.action_std[:,0],expected)
        torch.testing.assert_close(p.action_std[:,12],expected*2)
        action=p.act(obs);direct=torch.distributions.Normal(p.actor(obs),p.std_for_observations(obs))
        torch.testing.assert_close(p.get_actions_log_prob(action),direct.log_prob(action).sum(-1))
        torch.testing.assert_close(p.entropy,direct.entropy().sum(-1))
        # Larger towing exploration changes probabilities, never the deterministic mean action.
        torch.testing.assert_close(p.act_inference(obs),p.actor(obs))
    def test_reward_counterexamples(self):
        n=6;command=torch.tensor([[.3,0,0]]*n);command[4]=0
        robot_v=torch.tensor([[.2,0,0],[.2,0,0],[-.2,0,0],[.2,0,0],[.2,0,0],[.2,0,0]])
        cart_v=robot_v.clone();cart_v[1]=0;cart_v[3,0]=-.1
        robot=NS(data=NS(root_lin_vel_w=robot_v,root_pos_w=torch.tensor([[0.,0.,.38]]*n)))
        env=NS(scene={"robot":robot,"trailer":NS(data=NS(root_lin_vel_w=cart_v))},r1_tangent=torch.tensor([[1.,0,0]]*n),r1_normal=torch.tensor([[0.,0,1.]]*n),r1_towed_kg=torch.tensor([40.,40.,40.,40.,40.,0.]),command_manager=NS(get_command=lambda _:command))
        score=namespace["towing_signed_progress"](env)
        self.assertGreater(float(score[0]),.7);self.assertEqual(float(score[1]),0.);self.assertLess(float(score[2]),0.);self.assertLess(float(score[3]),0.);self.assertEqual(float(score[4]),0.);self.assertEqual(float(score[5]),0.)
        class Scene(dict):pass
        env.scene=Scene(env.scene);env.scene.env_origins=torch.zeros(n,3)
        height=namespace["surface_height"](env);self.assertEqual(float(height[0]),0.);self.assertGreater(float(height[5]),.001)
        env.termination_manager=NS(terminated=torch.tensor([True,False,True,False,True,True]))
        self.assertEqual(namespace["towing_failure"](env).tolist(),[True,False,True,False,True,False])
    def test_sampling(self):
        a,m,t=core.assignments(1024,spec.ANGLES,spec.PAYLOADS,spec.TOWED_MASSES)
        self.assertEqual(len(set(zip(a,m,t))),64)
        self.assertEqual([int(((m==0)&(t==0)).sum()),int((m>0).sum()),int((t>0).sum())],[256,256,512])
        self.assertFalse(np.any((a==-25)&(t>0)));self.assertFalse(np.any((m>0)&(t>0)))
if __name__=="__main__":unittest.main()
