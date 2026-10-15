"""CPU checks for R6 physics, checkpoint isolation and evaluation failures."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common
sys.path.insert(0, str(common.ROOT/'routes/route_06_heavy_payload/scripts'))
import launch
import metrics


def test_r6_registration_is_explicit_and_separate_from_r1():
    import gymnasium as gym
    common.catalog.register_tasks()
    before = dict(gym.registry)
    launch.spec.register_task(); launch.spec.register_task()
    assert gym.spec(launch.spec.TASK).kwargs['env_cfg_entry_point'].endswith('payload_env_cfg:HeavyPayloadEnvCfg')
    assert all(gym.registry[k] is v for k, v in before.items())
    assert launch.spec.CONTRACT != common.catalog.CONTRACT
    assert common.catalog.implemented_route('06')['task_id'] == launch.spec.TASK


def test_parallel_axis_composite_matches_closed_form():
    mass, com, inertia = launch.asset.composite_inertia(2., np.zeros(3), np.eye(3), 2., [0.,0.,1.], [1.,1.,1.])
    assert mass == 4
    np.testing.assert_allclose(com, [0, 0, .5])
    np.testing.assert_allclose(inertia, np.diag([7/3, 7/3, 4/3]))
    mass, com, inertia = launch.asset.composite_inertia(2., np.array([.1,.2,.3]), np.eye(3), 0., [0.,0.,1.], [1.,1.,1.])
    np.testing.assert_allclose(com, [.1,.2,.3]); np.testing.assert_allclose(inertia, np.eye(3))


@pytest.mark.parametrize('kg', launch.spec.STAGES_KG)
def test_generated_asset_preserves_joints_geometry_and_canonical_source(tmp_path, kg):
    before = common.sha256(common.URDF)
    path = tmp_path/'loaded.urdf'
    report = launch.asset.write_payload_urdf(common.URDF, path, kg)
    original, loaded = ET.parse(common.URDF), ET.parse(path)
    assert common.sha256(common.URDF) == before
    assert [ET.tostring(x) for x in loaded.findall('joint')] == [ET.tostring(x) for x in original.findall('joint')]
    for a, b in zip(original.findall('link'), loaded.findall('link')):
        if a.get('name') != 'Base_link':
            assert ET.tostring(a.find('inertial')) == ET.tostring(b.find('inertial'))
    assert np.linalg.eigvalsh(np.array(report['base_inertia_kg_m2'])).min() > 0
    assert report['loaded_robot_mass_kg'] == pytest.approx(report['unloaded_robot_mass_kg']+kg)
    assert len(loaded.findall('.//mesh')) == len(original.findall('.//mesh'))
    assert all(Path(m.get('filename')).is_file() for m in loaded.findall('.//mesh'))
    with pytest.raises(ValueError, match='read-only'):
        launch.asset.write_payload_urdf(common.URDF, common.URDF, kg)


@pytest.mark.parametrize('mass', [-1, 66, float('nan'), float('inf')])
def test_bad_mass_is_rejected(mass):
    with pytest.raises(ValueError): launch.spec.validate_mass(mass)


def test_payload_channel_and_load_cost_use_real_inputs():
    mdp = common.module_from_file('_r6_test_mdp', common.SOURCE/'payload_mdp.py')
    env = SimpleNamespace(num_envs=2, device='cpu', cfg=SimpleNamespace(payload_mass_kg=65.))
    torch.testing.assert_close(mdp.payload_observation(env), torch.ones(2,1))
    forces = torch.tensor([[[0.,0.,10.]]*4, [[0.,0.,40.],[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]]])
    env.scene = {'contact': SimpleNamespace(data=SimpleNamespace(net_forces_w=forces)),
                 'robot': SimpleNamespace(data=SimpleNamespace(applied_torque=torch.tensor([[0.,0.],[48.,48.]]), joint_effort_limits=torch.full((2,2),48.)))}
    costs = mdp.load_balance_cost(env, SimpleNamespace(name='contact',body_ids=[0,1,2,3]))
    assert costs[0] == 0 and costs[1] > .5
    costs = mdp.torque_margin_cost(env, SimpleNamespace(name='robot',joint_ids=[0,1]))
    assert costs[0] == 0 and costs[1] > 0


def test_r6_checkpoint_rejects_r1_and_nonfinite(tmp_path):
    path = tmp_path/'model.pt'
    saved = {'model_state_dict': {'weight': torch.ones(2)}, 'infos': {'multiskill_contract': common.catalog.CONTRACT}}
    torch.save(saved,path)
    with pytest.raises(ValueError, match='R6 checkpoint'): launch.spec.checkpoint(path)
    saved['infos'] = {'r6_contract': deepcopy(launch.spec.CONTRACT)}
    torch.save(saved,path); launch.spec.checkpoint(path)
    saved['model_state_dict']['weight'][0] = float('nan'); torch.save(saved,path)
    with pytest.raises(ValueError, match='Non-finite'): launch.spec.checkpoint(path)


def test_exit_zero_without_result_is_not_training_success(tmp_path):
    with pytest.raises(FileNotFoundError): launch.verify_completion(tmp_path, 'smoke')
    common.write_json(tmp_path/'runtime_check.json', {'dimensions':[58,237,16]})
    with pytest.raises(FileNotFoundError): launch.verify_completion(tmp_path, 'smoke')
    common.write_json(tmp_path/'result.json', {'mode':'smoke', 'status':'completed'})
    launch.verify_completion(tmp_path, 'smoke')


def test_eval_does_not_hide_failure_or_nonfinite():
    rows = [dict(t=(i+1)*.02,tilt_deg=1.,speed=0.,yaw_rate=0.,height_error=0.,four_wheel=1.,torque_saturation=0.,x=0.,y=0.,torque_peak=3.,body_contact=0.) for i in range(1000)]
    assert metrics.summarize(rows,None,launch.spec.SCREEN)['success']
    assert not metrics.summarize(rows[:-1],None,launch.spec.SCREEN)['success']
    assert not metrics.summarize(rows,'terminated',launch.spec.SCREEN)['success']
    for key, value in [('four_wheel',0.),('torque_saturation',1.),('body_contact',1.),('speed',float('nan'))]:
        bad = deepcopy(rows)
        for row in bad: row[key]=value
        assert not metrics.summarize(bad,None,launch.spec.SCREEN)['success']


def test_route06_cli_dispatches_to_r6_without_simulator():
    result = subprocess.run([sys.executable,str(common.ROOT/'run.py'),'check','--route','06','--payload-kg','65'],capture_output=True,text=True)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report['task'] == launch.spec.TASK and report['payload']['payload_kg'] == 65
    assert not report['gpu_verified']


def test_real_rsl_cpu_save_and_stage_resume(tmp_path):
    """合成CPU数据仅证明PPO集成，不能作为机器人仿真或承载证据。"""
    from training import make_runner_class, resume_runner
    torch.set_num_threads(1)
    class CpuEnv:
        num_envs, num_actions, max_episode_length = 4, 16, 100
        device = 'cpu'
        cfg = {}
        def __init__(self):
            self.episode_length_buf = torch.zeros(4,dtype=torch.long)
            self.r6_samples = []
        def get_observations(self):
            return torch.zeros(4,58), {'observations': {'critic':torch.zeros(4,237)}}
        def step(self, action):
            self.episode_length_buf += 1
            self.r6_samples.append(dict(terminated=0,timeout=0,tilt_deg=0.,speed=0.,height=.416,four_wheel=1.,torque_peak=2.,torque_saturation=0.))
            obs, extras = self.get_observations()
            return obs, -action.square().mean(-1), torch.zeros(4,dtype=torch.long), extras
    cfg = dict(num_steps_per_env=4,save_interval=1,empirical_normalization=False,logger='tensorboard',
        policy=dict(class_name='ActorCritic',init_noise_std=.15,noise_std_type='log',actor_hidden_dims=[8],critic_hidden_dims=[8],activation='elu'),
        algorithm=dict(class_name='PPO',learning_rate=3e-4,num_learning_epochs=1,num_mini_batches=1,schedule='fixed'))
    first = tmp_path/'first';first.mkdir()
    runner = make_runner_class(first,0)(CpuEnv(),deepcopy(cfg),log_dir=str(first/'training'),device='cpu')
    runner.learn(2);runner.writer.close()
    path = first/'training/model_1.pt'
    saved = launch.spec.checkpoint(path)
    assert saved['infos']['payload_kg']==0 and saved['optimizer_state_dict']['state']
    assert len((first/'training_metrics.csv').read_text().splitlines())==3
    before = common.sha256(path)
    second=tmp_path/'second';second.mkdir()
    later = make_runner_class(second,5)(CpuEnv(),deepcopy(cfg),log_dir=str(second/'training'),device='cpu')
    resume_runner(later,str(path),saved)
    assert later.current_learning_iteration == 2
    for key,tensor in runner.alg.policy.state_dict().items():
        torch.testing.assert_close(later.alg.policy.state_dict()[key],tensor)
    assert later.alg.optimizer.state_dict()['state'].keys()==saved['optimizer_state_dict']['state'].keys()
    for key,state in later.alg.optimizer.state_dict()['state'].items():
        for name,value in state.items():
            torch.testing.assert_close(value,saved['optimizer_state_dict']['state'][key][name])
    later.learn(1);later.writer.close()
    next_saved = launch.spec.checkpoint(second/'training/model_2.pt')
    assert next_saved['infos']['payload_kg']==5 and next_saved['iter']==2
    assert common.sha256(path)==before
