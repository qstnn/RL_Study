"""学生只读记录接线；复用原物理动作类/原生CoP记录器，不实现控制器。"""


def recorded_wheel_action_type():
    """仅在AppLauncher之后调用。原类负责所有外力公式，子类只读前态。"""
    import torch
    from isaaclab.utils.math import quat_apply
    from rl_training.tasks.manager_based.locomotion.multiskill.lighthw.wheel_unified_v6.cart import TowingWheelAction

    class StudentRecordedWheelAction(TowingWheelAction):
        def apply_actions(self):
            super().apply_actions()
            env = self._env
            if not getattr(env, 'r1_physics_ready', False):
                return
            r, c = env.scene['robot'], env.scene['trailer']
            rh = quat_apply(r.data.root_quat_w, env.r1_robot_hitch)+r.data.root_pos_w
            ch = quat_apply(c.data.root_quat_w, env.r1_cart_hitch)+c.data.root_pos_w
            ra, ca = rh-r.data.body_com_pos_w[:, env.r1_base_id], ch-c.data.body_com_pos_w[:, env.r1_cart_base_id]
            env.r1_student_cart_audit = dict(
                robot_force_w_N=r._external_force_b[:, env.r1_base_id].clone(),
                robot_torque_w_Nm=r._external_torque_b[:, env.r1_base_id].clone(),
                cart_forces_w_N=c._external_force_b.clone(), cart_torques_w_Nm=c._external_torque_b.clone(),
                cart_masses_kg=env.r1_cart_masses.clone(), robot_hitch_pre_w=rh.clone(), cart_hitch_pre_w=ch.clone(),
                robot_arm_pre_w=ra.clone(), cart_arm_pre_w=ca.clone(),
                robot_hitch_velocity_pre_w=(r.data.root_com_lin_vel_w+torch.cross(r.data.root_ang_vel_w, ra, dim=-1)).clone(),
                cart_hitch_velocity_pre_w=(c.data.root_com_lin_vel_w+torch.cross(c.data.root_ang_vel_w, ca, dim=-1)).clone())
    return StudentRecordedWheelAction


def attach_recorders(base, evaluation_worker, research_common, out, write_json):
    """复用历史PhysicsRecorder/record_contact_geometry；局部扩展拖车只读字段。"""
    import torch
    from contact_state import record_contact_geometry
    from momentum_probe import record_momentum
    module = research_common.launch.module_from_file('_r1_student_physics_recorder',
        research_common.OLD/'coordinated_control.py')
    recorder = module.PhysicsRecorder(base, evaluation_worker)
    record_contact_geometry(recorder, base)
    record_momentum(recorder, base, out, write_json)
    cart = base.scene['trailer']
    wheels = cart.find_bodies([f'cart_wheel_{i}' for i in range(4)], preserve_order=True)[0]
    joints = cart.find_joints([f'cart_wheel_{i}_joint' for i in range(4)], preserve_order=True)[0]
    shapes = dict(cart_pos_w=(3,), cart_velocity_w=(3,), cart_wheel_center_velocity_w=(4,3),
        cart_wheel_angular_velocity_w=(4,3), cart_joint_velocity=(4,), cart_rolling_torque_Nm=(4,),
        real_cart_cart_quat_w=(4,), real_cart_cart_wheel_pos_w=(4,3),
        cart_ground_friction_w=(5,3), cart_ground_normal_w=(5,3), cart_ground_point_count=(5,),
        cart_ground_min_separation_m=(5,))
    external = dict(robot_force_w_N=(3,), robot_torque_w_Nm=(3,), cart_forces_w_N=(5,3),
        cart_torques_w_Nm=(5,3), cart_masses_kg=(5,), robot_hitch_pre_w=(3,), cart_hitch_pre_w=(3,),
        robot_arm_pre_w=(3,), cart_arm_pre_w=(3,), robot_hitch_velocity_pre_w=(3,), cart_hitch_velocity_pre_w=(3,))
    shapes.update({'real_cart_'+key: shape for key, shape in external.items()})
    for key, shape in shapes.items():
        recorder.buffers[key] = torch.zeros((base.num_envs, base.cfg.decimation, *shape), device=base.device)
        recorder.history[key] = []
    original_torque_sample = base.r1_torque_recorder.sample
    def sample_external():
        index = base.r1_torque_recorder.cursor
        original_torque_sample()
        if base.r1_torque_recorder.cursor > index:
            for key, value in base.r1_student_cart_audit.items():
                recorder.buffers['real_cart_'+key][:, index] = value
    base.r1_torque_recorder.sample = sample_external

    sensor = base.scene['trailer_contacts']
    paths = list(sensor.body_physx_view.prim_paths)
    names = [f'cart_wheel_{i}' for i in range(4)]+['cart_base']
    requested = [next(path for path in paths if f'/env_{i}/' in path and path.endswith('/'+name))
                 for i in range(base.num_envs) for name in names]
    if len(set(requested)) != len(requested):
        raise RuntimeError('Duplicate student cart contact path')
    ground = recorder.contact_meta['ground_filters']
    view = sensor._physics_sim_view.create_rigid_contact_view(requested,
        filter_patterns=[ground]*len(requested), max_contact_data_count=len(requested)*128)
    actual = list(view.sensor_paths)
    if len(actual) != len(requested) or set(actual) != set(requested):
        raise RuntimeError('Student cart native contact view differs')
    order = torch.tensor([actual.index(path) for path in requested], device=base.device)
    offsets = torch.arange(128, device=base.device)
    write_json(out/'cart_contact_view.json', dict(requested_paths=requested, actual_paths=actual,
        ground_filters=ground, capacity=view.max_contact_data_count, body_order=names))
    previous = recorder.sample
    def sample_cart():
        index = recorder.cursor
        previous()
        friction, _, counts, starts = view.get_friction_data(dt=base.physics_dt)
        counts, starts = counts.long(), starts.long()
        if (counts > len(offsets)).any():
            raise RuntimeError('Student cart friction evidence truncated')
        valid = offsets[None,None,:] < counts[:,:,None]
        ix = (starts[:,:,None]+offsets[None,None,:]).clamp(0, len(friction)-1)
        tangential = torch.where(valid[:,:,:,None], friction[ix], 0.).sum((1,2))
        normal = view.get_contact_force_matrix(dt=base.physics_dt).sum(1)
        _, _, _, separation, point_counts, point_starts = view.get_contact_data(dt=base.physics_dt)
        point_counts, point_starts = point_counts.long(), point_starts.long()
        if (point_counts > len(offsets)).any():
            raise RuntimeError('Student cart point evidence truncated')
        valid_points = offsets[None,None,:] < point_counts[:,:,None]
        indices = (point_starts[:,:,None]+offsets[None,None,:]).clamp(0, len(separation)-1)
        gap = torch.where(valid_points, separation[indices,0], 1000.).flatten(1).min(1).values
        for key, value in dict(cart_ground_friction_w=tangential, cart_ground_normal_w=normal,
            cart_ground_point_count=point_counts.sum(1), cart_ground_min_separation_m=gap).items():
            recorder.buffers[key][:,index] = value[order].reshape(base.num_envs,5,*value.shape[1:])
        values = dict(cart_pos_w=cart.data.root_pos_w, cart_velocity_w=cart.data.root_com_lin_vel_w,
            cart_wheel_center_velocity_w=cart.data.body_lin_vel_w[:,wheels],
            cart_wheel_angular_velocity_w=cart.data.body_ang_vel_w[:,wheels],
            cart_joint_velocity=cart.data.joint_vel[:,joints], cart_rolling_torque_Nm=base.r1_rolling_torque,
            real_cart_cart_quat_w=cart.data.root_quat_w, real_cart_cart_wheel_pos_w=cart.data.body_pos_w[:,wheels])
        for key, value in values.items():
            recorder.buffers[key][:,index] = value
    recorder.sample = sample_cart
    return recorder
