"""测试会改变实际学习方向的反例：模式错配、腾空、四轮停留和未完成保持的返回。"""
import ast
from types import SimpleNamespace as NS
from unittest.mock import patch
import tempfile
from pathlib import Path
import unittest
import torch
from test_protocol import BalanceState, desired_gravity, SOURCE
from test_v4 import metrics
from two_leg_balance_split_v4.rewards import task_rewards
from two_leg_balance_split_v4.spec import contract, STAND, WEIGHTS, SKILLS, weights_for
from common import checked_checkpoint, manifest, asset_hashes
import run as launcher


def example(n=1, mode=1):
    m=metrics(n);m['line_error'][:]=.45
    s=BalanceState(n,'cpu',2.);s.entered[:]=True
    mass=torch.full((n,),11.668111);m['force_z']=mass[:,None].expand(-1,4)*9.81/4
    modes=torch.full((n,),mode,dtype=torch.long)
    g=desired_gravity(torch.full((n,),-.979469 if mode else .909966))
    args=dict(m=m,state=s,phase=torch.full((n,),2),mode=modes,full_target=g,mass=mass,dt=.02,
        joint_pos=torch.tensor(STAND).repeat(n,1),leg_torque=torch.zeros(n,12),leg_speed=torch.zeros(n,12),
        wheel_speed=torch.zeros(n,4),actions=torch.zeros(n,16),previous_actions=torch.zeros(n,16),
        timeouts=torch.zeros(n,dtype=torch.bool),missed_hold=torch.zeros(n,dtype=torch.bool))
    return args


def weighted(args, profile='baseline_v4'):
    terms=task_rewards(**args, reward_profile=profile)
    assert set(terms)==set(WEIGHTS)
    return {k:v*weights_for(profile)[k] for k,v in terms.items()}


class SplitRewardsTests(unittest.TestCase):
    def test_rear_front_identity_and_cross_branch_checkpoint_rejected(self):
        self.assertEqual(SKILLS,{'R4a':'rear','R4b':'front'})
        self.assertEqual(contract()['skill'],'R4a')
        self.assertEqual(contract(mode='front')['skill'],'R4b')
        with self.assertRaises(ValueError):contract(mode='both')
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'rear.pt'
            torch.save({'infos':{'contract':contract(),'source_manifest':manifest(),'asset_manifest':asset_hashes()},
                        'model_state_dict':{'x':torch.ones(1)}},path)
            checked_checkpoint(path,contract())
            with self.assertRaisesRegex(ValueError,'contract mismatch'):checked_checkpoint(path,contract(mode='front'))
            with self.assertRaisesRegex(ValueError,'contract mismatch'):
                checked_checkpoint(path, contract(reward_profile='unload_diag_v1'))

    def test_progress_strictly_increases_with_transfer_for_both_tasks(self):
        for mode in (0,1):
            a=example(4,mode);m=a['m'];weight=a['mass']*9.81
            sign=.909966 if mode==0 else -.979469
            m['gravity']=desired_gravity(torch.tensor([0.,sign*.3,sign*.65,sign]))
            m['line_error']=torch.tensor([.45,.3,.15,0.])
            stance=[2*mode,2*mode+1];swing=[2*(1-mode),2*(1-mode)+1]
            loads=torch.tensor([.5,.35,.15,0.])
            m['force_z'][:,stance]=(weight*(1-loads)/2)[:,None]
            m['force_z'][:,swing]=(weight*loads/2)[:,None]
            m['clearance'][:,swing]=torch.tensor([0.,0.,.02,.08])[:,None]
            t=weighted(a);total=sum(t.values())
            self.assertTrue(torch.all(total[1:]>total[:-1]))
            for key in ('attitude','com_line','support'):
                self.assertTrue(torch.all(t[key][1:]>t[key][:-1]),key)
            self.assertEqual(float(t['attitude'][0]),0.)
            self.assertEqual(float(t['com_line'][0]),0.)
            self.assertGreater(float(total[-1]/a['dt']),35.)

    def test_unload_diagnostic_uses_worst_front_wheel_and_gates_progress(self):
        a = example(3); m = a['m']; w = a['mass'] * 9.81
        a['phase'][:] = 2
        m['gravity'][:] = a['full_target']
        m['line_error'].zero_(); m['clearance'][:] = .08
        m['force_z'][:] = 0.
        m['force_z'][:, 2:] = w[:, None] * .5
        m['force_z'][0, :2] = w[0] * .25       # 四轮承载：诊断门控仍为零
        m['force_z'][1, :2] = torch.tensor([0., .36]) * w[1]  # 单个前轮残余高载
        m['force_z'][2, :2] = w[2, None] * .05  # 两只前轮都开始卸载
        t = weighted(a, 'unload_diag_v1')
        self.assertEqual(float(t['attitude'][0]), 0.)
        self.assertEqual(float(t['attitude'][1]), 0.)
        self.assertGreater(float(t['attitude'][2]), 0.)
        self.assertLess(float(t['swing_load'][1]), float(t['swing_load'][2]))
        self.assertLess(float(t['swing_asymmetry'][1]), float(t['swing_asymmetry'][2]))

    def test_airborne_wrong_pair_one_leg_and_four_wheel_cannot_earn_pair_reward(self):
        a=example(4);m=a['m'];m['gravity']=a['full_target'].clone();m['line_error'].zero_();m['clearance'][:]=.08
        w=a['mass']*9.81
        m['force_z'][0]=0 # 腾空
        m['force_z'][1]=torch.tensor([.5,.5,0.,0.])*w[1] # 错误前支撑
        m['force_z'][2]=torch.tensor([0.,0.,1.,0.])*w[2] # 后单轮
        t=weighted(a)
        for name in ('support','capture','clearance','pair','balance'):
            self.assertEqual(t[name].tolist(),[0.]*4,name)
        for name in ('attitude','com_line'):
            self.assertEqual(t[name][:3].tolist(),[0.]*3,name)

    def test_return_and_quiet_require_completed_hold(self):
        a=example(4);a['phase']=torch.tensor([3,4,3,4]);a['state'].held[2:]=True
        t=weighted(a)
        for key in ('return_pose','return_attitude','quiet'):
            self.assertEqual(t[key][:2].tolist(),[0.,0.],key)
        self.assertGreater(float(t['return_pose'][2]),0.)
        self.assertGreater(float(t['quiet'][3]),0.)
        for key in ('attitude','com_line','support','clearance','capture','pair'):
            self.assertEqual(t[key].tolist(),[0.]*4,key)

    def test_initial_entry_stays_rewarded_and_failed_state_has_no_positive_task_rewards(self):
        a=example(2);a['phase'][:]=0;a['state'].failed[1]=True
        t=weighted(a)
        self.assertGreater(float(t['entry_quiet'][0]+t['entry_pose'][0]),0.)
        for key,w in WEIGHTS.items():
            if w>0:self.assertEqual(float(t[key][1]),0.,key)
        self.assertEqual(float(t['failure'][1]),-20.)

    def test_missed_hold_penalty_occurs_once_in_actual_env_reward(self):
        a=example(1)
        tree=ast.parse((SOURCE/'env.py').read_text())
        node=next(x for x in ast.walk(tree) if isinstance(x,ast.FunctionDef) and x.name=='_get_rewards')
        from two_leg_balance_split_v4.core import masks
        scope=dict(torch=torch,desired_gravity=desired_gravity,masks=masks,WEIGHTS=WEIGHTS,task_rewards=task_rewards)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'env_reward','exec'),scope)
        env=NS(current_metrics=a['m'],state=a['state'],robot=NS(data=NS(joint_pos=a['joint_pos'],applied_torque=torch.zeros(1,16),joint_vel=torch.zeros(1,16))),
            cfg=NS(target_pitches=(.909966,-.979469)),mode=a['mode'],reward_phase=torch.tensor([2]),mass=a['mass'],step_dt=.02,
            actions=a['actions'],previous_actions=a['previous_actions'],leg_ids=list(range(12)),wheel_ids=list(range(12,16)),
            reset_time_outs=a['timeouts'],episode_rewards={k:torch.zeros(1) for k in WEIGHTS},extras={'log':{}},
            command=lambda:(torch.tensor([3]),a['full_target']))
        scope['_get_rewards'](env);self.assertEqual(float(env.reward_terms['missed_hold'][0]),-8.)
        scope['_get_rewards'](env);self.assertEqual(float(env.reward_terms['missed_hold'][0]),0.)

    def test_joint_mode_cli_is_rejected_before_creating_run(self):
        import contextlib,io
        with patch('sys.argv',['run.py','train','--support_mode','both']),contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):launcher.main()
