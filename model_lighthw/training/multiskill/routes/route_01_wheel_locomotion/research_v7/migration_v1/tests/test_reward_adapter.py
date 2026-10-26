import torch
from types import SimpleNamespace

from migration_v1.reward_adapter import height_error, nominal_point_slip, support_score


def test_low_towing_posture_does_not_penalize_protected_height_target():
    height = torch.tensor([.28217066, .416, .2, .5, .28217066])
    towing = torch.tensor([True, True, True, True, False])
    assert torch.allclose(height_error(height, towing), torch.tensor([0., 0., .06, .06, -.13382934]))


def test_support_accepts_partial_wheels_but_not_airborne():
    forces = torch.tensor([[0., 0., 60., 60.], [30., 30., 30., 30.], [0., 0., 0., 0.], [-1., -1., -1., -1.]])
    assert torch.equal(support_score(forces, torch.full((4,), 120.)), torch.tensor([1., 1., 0., 0.]))


def test_slip_distinguishes_rolling_locked_wheel_and_unloaded_spin():
    normal = torch.tensor([[0., 0., 1.]]).repeat(3, 1)
    loads = torch.tensor([[10., 10., 0., 0.]]).repeat(3, 1)
    velocity = torch.zeros(3, 4, 3); velocity[:, :, 0] = .3
    angular = torch.zeros(3, 4, 3)
    angular[0, :, 1] = .3/.0875
    angular[2, :, 1] = .3/.0875; angular[2, 2:, 1] = 1000.
    cost = nominal_point_slip(loads, velocity, angular, normal)
    assert torch.allclose(cost, torch.tensor([0., 1., 0.]), atol=1.e-6)


def test_mixed_batch_preserves_original_non_towing_term_outputs(monkeypatch):
    """有限合成状态验证分支/批维度；不冒充物理保留能力。"""
    from migration_v1 import reward_adapter as reward
    n=6;original=torch.tensor([.12,.23,.34,.45,.56,.67])
    robot=SimpleNamespace(data=SimpleNamespace(root_pos_w=torch.tensor([[0.,0.,.3]]).repeat(n,1),
        body_link_lin_vel_w=torch.zeros(n,4,3),body_ang_vel_w=torch.zeros(n,4,3)))
    forces=torch.zeros(n,4,3);forces[:,:,2]=30.
    class Scene(dict):pass
    scene=Scene(robot=robot,contact_forces=SimpleNamespace(data=SimpleNamespace(net_forces_w=forces)))
    scene.env_origins=torch.zeros(n,3)
    term=SimpleNamespace(towing_preparation_complete=torch.tensor([False,True,False,True,False,True]))
    env=SimpleNamespace(scene=scene,r1_towed_kg=torch.tensor([0.,0.,40.,40.,0.,0.]),
        r1_payload_kg=torch.tensor([0.,15.,0.,0.,5.,10.]),r1_normal=torch.tensor([[0.,0.,1.]]).repeat(n,1),
        cfg=SimpleNamespace(hitch_margin_m=.05),r1_migration_slip_ids=(list(range(4)),list(range(4))),
        command_manager=SimpleNamespace(get_term=lambda _:term,get_command=lambda _:torch.zeros(n,3)))
    mdp=SimpleNamespace(**{name:(lambda *args,**kwargs:original.clone()) for name in
        ('tracking','surface_height','stand_support','stand_motion','stand_drift','stand_wheels','stand_pose')},
        state=lambda _:(robot,torch.zeros(n,2),torch.zeros(n,3)),heading_error=lambda _:torch.zeros(n,1),
        posture_cost=lambda _:torch.zeros(n),hitch_clearance=lambda _:torch.ones(n)*.1,
        loaded_motion_weight=lambda command:torch.zeros_like(command),standing=lambda _:torch.ones(n,dtype=torch.bool),
        core=SimpleNamespace(velocity_score=lambda error,command,fine:torch.ones(n)))
    monkeypatch.setattr(reward,'_mdp',lambda:mdp)
    sensor=SimpleNamespace(name='contact_forces',body_ids=list(range(4)))
    protected=env.r1_towed_kg==0
    calls=[(reward.tracking,()),(reward.tracking,(True,)),(reward.surface_height,()),
        (reward.stand_support,(sensor,)),(reward.stand_motion,()),(reward.stand_drift,()),
        (reward.stand_wheels,(None,)),(reward.stand_pose,(None,))]
    for function,args in calls:
        result=function(env,*args)
        assert result.shape==(n,) and torch.isfinite(result).all()
        assert torch.equal(result[protected],original[protected])
    for result in (reward.towing_support_margin(env,sensor),reward.towing_slip(env)):
        assert result.shape==(n,) and torch.equal(result[protected],torch.zeros_like(result[protected]))
