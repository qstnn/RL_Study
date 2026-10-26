"""统一R1的CPU物理/采样/模型迁移检查；不冒充GPU行为验收。"""
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import numpy as np
import pytest
import torch

SCRIPTS=Path(__file__).resolve().parents[1]/"scripts"
sys.path.insert(0,str(SCRIPTS))
import launch
protocol=launch.PROTOCOL

spec=launch.SPEC
core=launch.module_from_file("_test_r1_core",launch.SOURCE/"wheel_unified/core.py")
cart=launch.CART


def test_mixture_contains_hard_combinations_from_first_batch():
    slope,mass,tow=core.assignments(840,spec.SLOPE_MIX,spec.PAYLOADS,spec.TOWED_MASSES)
    assert len(set(zip(slope,mass,tow)))==11*4*3
    assert (25.,15.,40.) in set(zip(slope,mass,tow))
    assert (-25.,15.,40.) in set(zip(slope,mass,tow))
    assert np.mean(slope==0)==pytest.approx(.8*4/14)
    assert np.mean(mass==15)==pytest.approx(.8*.25+.2)
    assert np.mean(tow==40)==pytest.approx(.8/3+.2)
    first=core.assignments(2,spec.SLOPE_MIX,spec.PAYLOADS,spec.TOWED_MASSES)
    assert set(first[0])=={-25.,25.} and set(first[1])=={15.} and set(first[2])=={40.}


def test_exact_25_degree_mesh_and_surface_reward_coordinates():
    for degrees in (-25,0,25):
        vertices,faces=core.slope_mesh(degrees,14.)
        top=vertices[:4]
        normal=np.cross(top[1]-top[0],top[2]-top[0]);normal/=np.linalg.norm(normal)
        angle=torch.tensor([degrees*np.pi/180],dtype=torch.float64)
        tangent,n=core.plane_basis(angle)
        np.testing.assert_allclose(normal,n[0].numpy(),atol=1e-12)
        velocity=core.tangent_velocity(3.1*tangent,tangent,n)
        torch.testing.assert_close(velocity,torch.tensor([[3.1,0.]],dtype=torch.float64))
        assert float((3.1*tangent*n).sum())==pytest.approx(0.,abs=1e-12)
        assert len(faces)==12


def test_payload_preserves_zero_mass_and_parallel_axis_moments():
    base_mass=5.;base_com=np.array([.03,-.01,.02]);inertia=np.diag([.2,.3,.4])
    total,com,tensor=core.compose_inertia(base_mass,base_com,inertia,0.,[0,0,.2],[.3,.25,.2])
    assert total==base_mass
    np.testing.assert_allclose(com,base_com);np.testing.assert_allclose(tensor,inertia)
    total,com,tensor=core.compose_inertia(base_mass,base_com,inertia,15.,[0,0,.2],[.3,.25,.2])
    assert total==20.
    np.testing.assert_allclose(com,(5*base_com+15*np.array([0,0,.2]))/20)
    assert np.linalg.eigvalsh(tensor).min()>0
    # 对同一参考原点的惯量守恒，独立计算验证平行轴项。
    shift=lambda mass,x:mass*((x@x)*np.eye(3)-np.outer(x,x))
    expected=inertia+shift(5,base_com)+15/12*np.diag([.25**2+.2**2,.3**2+.2**2,.3**2+.25**2])+shift(15,np.array([0,0,.2]))
    np.testing.assert_allclose(tensor+shift(total,com),expected,atol=1e-12)


def test_cart_has_real_mass_four_passive_wheels_and_no_drive(tmp_path):
    path=tmp_path/"cart.urdf";cart.write_cart_urdf(path)
    root=ET.parse(path)
    assert sum(float(m.get("value")) for m in root.findall("link/inertial/mass"))==40.
    joints=root.findall("joint")
    assert len(joints)==4 and all(j.get("type")=="continuous" for j in joints)
    assert all(j.find("limit").get("effort")=="0" for j in joints)
    assert len(root.findall("link/collision/geometry/cylinder"))==4


def test_drawbar_zero_rest_passivity_and_disabled_cases():
    rp=torch.tensor([[1.,0,0],[1.,0,0],[2.,0,0]])
    cp=torch.zeros_like(rp)
    rv=torch.tensor([[0.,0,0],[1.,0,0],[1.,0,0]])
    cv=torch.zeros_like(rv)
    active=torch.tensor([True,True,False])
    force,length,tension=cart.drawbar_force(rp,cp,rv,cv,active)
    torch.testing.assert_close(force[0],torch.zeros(3))
    torch.testing.assert_close(force[2],torch.zeros(3))
    assert float((force[1]*(cv[1]-rv[1])).sum())<0 # 阻尼仅耗散能量。
    assert tension[2]==0
    stretched=torch.tensor([[1.1,0,0]])
    force,_,_=cart.drawbar_force(stretched,cp[:1],rv[:1],cv[:1],active[:1])
    assert float(force[0,0])==pytest.approx(200.,abs=.001)


def test_high_speed_has_dense_signal_and_surface_target_has_full_reward():
    command=torch.tensor([3.1,3.1])
    error=torch.tensor([[-3.1,0.],[-2.8,0.]])
    reward=core.velocity_score(error,command)
    assert 0<reward[0]<reward[1]<1
    assert core.velocity_score(torch.zeros_like(error),command).tolist()==[1.,1.]


def test_real_stage1_transfer_preserves_actor_and_critic_function():
    from rsl_rl.modules import ActorCritic
    torch.set_num_threads(2)
    kwargs=dict(actor_hidden_dims=[512,256,128],critic_hidden_dims=[512,256,128],activation="elu",noise_std_type="log")
    source=launch.validate_checkpoint(launch.BASELINE)
    old=ActorCritic(57,236,16,**kwargs);old.load_state_dict(source["model_state_dict"])
    new=ActorCritic(290,242,16,**kwargs)
    new.load_state_dict(spec.transfer_stage1(source,new.state_dict()))
    obs=torch.randn(64,290);critic=torch.randn(64,242)
    with torch.inference_mode():
        torch.testing.assert_close(old.act_inference(obs[:,spec.latest_source_columns()]),new.act_inference(obs),atol=1e-6,rtol=1e-5)
        torch.testing.assert_close(old.evaluate(critic[:,:236]),new.evaluate(critic),atol=1e-6,rtol=1e-5)
    assert torch.count_nonzero(new.actor[0].weight[:,-5:])==0


def test_registration_does_not_replace_stage1_task():
    import gymnasium as gym
    launch.module_from_file("_test_stage1_catalog",launch.SOURCE/"catalog.py").register_tasks()
    before=gym.spec("MultiSkill-LightHW-Wheel-Stage1-v0")
    spec.register_task();spec.register_task()
    assert gym.spec("MultiSkill-LightHW-Wheel-Stage1-v0") is before
    assert gym.spec(spec.TASK).entry_point.endswith("UnifiedEnv")


def rows(vx=3.05):
    return [dict(t=i*.02,vx=vx if 3<=i*.02<11 else 0.,velocity_error=.05,speed=0.,yaw_rate_abs=0.,
                 tilt_deg=2.,heading_rad=0.,height=.416,four_wheel=1.,body_contact=0.,torque_abs_max=8.,torque_saturation=0.,
                 wheel_target_clip=.25,drawbar_force_N=180.,x=0.,y=0.) for i in range(900)]


def test_acceptance_checks_actual_3mps_and_keeps_incomplete_failures():
    case=dict(slope_deg=25.,payload_kg=15.,towed_kg=40.,speed=3.1)
    assert protocol.summarize(rows(),case,None)["success"]
    assert not protocol.summarize(rows(2.99),case,None)["success"]
    assert not protocol.summarize(rows()[:-1],case,None)["success"]
    assert not protocol.summarize(rows(),case,"terminated")["success"]
    deviating=rows()
    for row in deviating:row["heading_rad"]=.3
    assert not protocol.summarize(deviating,case,None)["checks"]["heading"]
    incomplete=rows()[:10]
    incomplete[-1]["termination_cart_failure"]=1.
    for row in incomplete[:-1]:row["termination_cart_failure"]=0.
    report=protocol.summarize(incomplete,case,"terminated")
    assert report["observed"]["termination_causes"]==["cart_failure"]
    assert len(protocol.cases())==24


def test_traction_requirement_is_not_confused_with_40kg_gravity_drag():
    robot_mass=11.66811+15.;cart_mass=40.;theta=np.deg2rad(25)
    required=((robot_mass+cart_mass)*np.sin(theta)+.02*cart_mass*np.cos(theta))/(robot_mass*np.cos(theta))
    assert 1.19<required<1.21 # μ=1不足；1.3是显式的未标定高附着假设。
    flat_cart_drag=.02*cart_mass*9.81
    assert flat_cart_drag==pytest.approx(7.848)
