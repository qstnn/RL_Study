from .common import read, write
from two_leg_balance_split_v4.spec import LEG_JOINTS, WHEELS, LOW, HIGH

def runtime(env, run):
    import torch
    import isaaclab.sim as sim
    from pxr import Usd
    b, d = env.unwrapped, env.unwrapped.robot.data
    assert b.cfg.mode in ("rear", "front")
    assert (b.mode == int(b.cfg.mode == "rear")).all()
    assert [env.num_obs, env.num_privileged_obs, env.num_actions] == [305, 88, 16], (env.num_obs, env.num_privileged_obs, env.num_actions)
    assert [b.robot.joint_names[i] for i in b.joint_ids] == list(LEG_JOINTS + WHEELS)
    torch.testing.assert_close(b.mass, torch.full_like(b.mass, read(run / "geometry.json")["mass_kg"]), atol=.002, rtol=1e-4)
    import xml.etree.ElementTree as ET
    links = {link.get("name"): link for link in ET.parse(run / "assets/urdf/lighthw.urdf").findall("link")}
    actual_inertia = b.robot.root_physx_view.get_inertias()[0].cpu().reshape(-1, 3, 3)
    for i, name in enumerate(b.robot.body_names):
        inertial = links[name].find("inertial")
        mass = float(inertial.find("mass").get("value"))
        torch.testing.assert_close(b.masses[0, i].cpu(), torch.tensor(mass), atol=2e-4, rtol=1e-3)
        com = [float(x) for x in inertial.find("origin").get("xyz").split()]
        torch.testing.assert_close(d.body_com_pos_b[0, i].cpu(), torch.tensor(com), atol=2e-4, rtol=1e-3)
        iv = {k: float(v) for k, v in inertial.find("inertia").attrib.items()}
        matrix = torch.tensor([[iv['ixx'], iv['ixy'], iv['ixz']], [iv['ixy'], iv['iyy'], iv['iyz']],
                               [iv['ixz'], iv['iyz'], iv['izz']]])
        torch.testing.assert_close(torch.linalg.eigvalsh(actual_inertia[i]), torch.linalg.eigvalsh(matrix), atol=2e-5, rtol=.005)
    expected_effort = [24. if "wheel" in n else 48. for n in b.robot.joint_names]
    expected_speed = [37.7 if "wheel" in n else 12.57 for n in b.robot.joint_names]
    torch.testing.assert_close(d.joint_effort_limits[0], d.joint_effort_limits.new_tensor(expected_effort))
    torch.testing.assert_close(d.joint_vel_limits[0], d.joint_vel_limits.new_tensor(expected_speed))
    assert (d.joint_pos_limits[0, b.leg_ids, 0] <= d.joint_pos_limits.new_tensor(LOW)).all()
    assert (d.joint_pos_limits[0, b.leg_ids, 1] >= d.joint_pos_limits.new_tensor(HIGH)).all()
    props = b.robot.root_physx_view.get_material_properties()
    torch.testing.assert_close(props[..., :2], torch.ones_like(props[..., :2]))
    self_collision, angular_caps, gyroscopic = [], [], []
    for prim in Usd.PrimRange(sim.get_current_stage().GetPrimAtPath("/World/envs/env_0/Robot")):
        attr = prim.GetAttribute("physxArticulation:enabledSelfCollisions")
        if attr and attr.HasAuthoredValueOpinion():
            self_collision.append(bool(attr.Get()))
        cap = prim.GetAttribute("physxRigidBody:maxAngularVelocity")
        if cap and cap.HasAuthoredValueOpinion():
            angular_caps.append(float(cap.Get()))
        gyro = prim.GetAttribute("physxRigidBody:enableGyroscopicForces")
        if gyro and gyro.HasAuthoredValueOpinion():
            gyroscopic.append(bool(gyro.Get()))
    assert self_collision and all(self_collision)
    assert len(angular_caps) == len(b.robot.body_names) and all(abs(x - 100000.) < .01 for x in angular_caps)
    assert len(gyroscopic) == len(b.robot.body_names) and all(gyroscopic)
    assert b.cfg.observation_noise is False and b.cfg.reset_jitter == 0.
    assert b.cfg.events is None, "Fixed baseline must not introduce randomized events"
    assert b.cfg.action_noise_model is None and b.cfg.observation_noise_model is None
    # 不推进物理的两次全量reset：验证固定姿态/速度和观测重复性。
    # Isaac的DelayedPD即使[0,0]延迟也调用randint；检查实际延迟为0，而非误判为环境扰动。
    fixed = []
    b.episode_count.zero_()
    for _ in range(2):
        b._reset_idx(torch.arange(b.num_envs, device=b.device))
        b.scene.write_data_to_sim()
        b.sim.forward()
        for actuator in b.robot.actuators.values():
            assert actuator.cfg.min_delay == actuator.cfg.max_delay == 0
            for buffer in (actuator.positions_delay_buffer, actuator.velocities_delay_buffer, actuator.efforts_delay_buffer):
                assert (buffer.time_lags == 0).all()
        cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state(b.device)
        reset_obs, _ = env.get_observations()
        expected_root = d.default_root_state.clone()
        expected_root[:, :3] += b.scene.env_origins
        torch.testing.assert_close(d.root_pos_w, expected_root[:, :3], atol=0, rtol=0)
        torch.testing.assert_close(d.root_quat_w, expected_root[:, 3:7], atol=0, rtol=0)
        torch.testing.assert_close(d.root_vel_w, torch.zeros_like(d.root_vel_w), atol=0, rtol=0)
        torch.testing.assert_close(d.joint_pos, d.default_joint_pos, atol=0, rtol=0)
        torch.testing.assert_close(d.joint_vel, torch.zeros_like(d.joint_vel), atol=0, rtol=0)
        torch.testing.assert_close(torch.get_rng_state(), cpu_rng, atol=0, rtol=0)
        torch.testing.assert_close(torch.cuda.get_rng_state(b.device), cuda_rng, atol=0, rtol=0)
        fixed.append(reset_obs.clone())
    torch.testing.assert_close(fixed[0], fixed[1], atol=0, rtol=0)
    obs, extras = env.get_observations()
    again, _ = env.get_observations()
    torch.testing.assert_close(obs, again, atol=0, rtol=0)
    assert torch.isfinite(obs).all() and torch.isfinite(extras["observations"]["critic"]).all()
    owned = obs.clone()
    env.step(torch.zeros(env.num_envs, 16, device=env.device))
    torch.testing.assert_close(obs, owned, atol=0, rtol=0)
    # 对一个槽做局部reset，其他槽的时钟/历史必须保持；重置槽的五帧不能带入旧回合。
    other_clock = b.physics_steps[1:].clone()
    other_history = b.history[1:].clone()
    b._reset_idx(torch.tensor([0], device=b.device))
    b.scene.write_data_to_sim()
    b.sim.forward()
    env.get_observations()
    torch.testing.assert_close(b.physics_steps[1:], other_clock)
    torch.testing.assert_close(b.history[1:], other_history)
    torch.testing.assert_close(b.history[0], b.history[0, :1].expand(5, -1))
    env.reset()
    b.episode_count.zero_()
    b.last_reports = []
    write(run / "runtime_check.json", dict(
        dimensions=[305, 88, 16], support_mode=b.cfg.mode, reward_profile=b.cfg.reward_profile,
        fixed_single_task=True, mass_kg=float(b.mass[0]), effort_limits_Nm=expected_effort,
        velocity_limits_rad_s=expected_speed, robot_friction=props[0].cpu().tolist(),
        self_collisions=self_collision, rigid_angular_caps_deg_s=angular_caps, gyroscopic_forces=gyroscopic,
        urdf_body_mass_com_inertia_checked=len(links), observation_ownership=True, partial_reset_history=True,
        physics_dt=b.physics_dt, policy_dt=b.step_dt, observation_noise=b.cfg.observation_noise,
        reset_jitter_rad=b.cfg.reset_jitter, fixed_reset_and_observations_exact=True,
        observation_rng_unchanged=True, actuator_delay_steps=[0, 0], randomized_events=False, behavior_accepted=False))

