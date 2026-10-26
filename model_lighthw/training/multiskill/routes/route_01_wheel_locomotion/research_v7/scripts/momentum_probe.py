"""只读原生刚体质心速度/惯量，用于区分接触冲量与内部姿态运动。"""
def record_momentum(physical,base,out,write_json):
    import torch
    robot=base.scene['robot']
    write_json(out/'native_inertia.json',dict(body_names=robot.body_names,
        mass_kg=robot.root_physx_view.get_masses().cpu().tolist(),
        inertia_mass_frame=robot.root_physx_view.get_inertias().cpu().tolist(),
        armature=robot.root_physx_view.get_dof_armatures().cpu().tolist(),
        note='Read-only; inertia about each native COM, rotated by body_com_quat_w.'))
    shapes=dict(momentum_body_com_velocity_w=(robot.num_bodies,3),
        momentum_body_com_quaternion_w=(robot.num_bodies,4))
    for key,shape in shapes.items():
        physical.buffers[key]=torch.zeros((base.num_envs,base.cfg.decimation,*shape),device=base.device)
        physical.history[key]=[]
    previous=physical.sample
    def sample():
        index=physical.cursor;previous()
        physical.buffers['momentum_body_com_velocity_w'][:,index]=robot.data.body_com_lin_vel_w
        physical.buffers['momentum_body_com_quaternion_w'][:,index]=robot.data.body_com_quat_w
    physical.sample=sample


def native_dynamics(base):
    """控制前的原生动力学；动作顺序重排，不进行状态写入。"""
    import numpy as np
    robot=base.scene['robot'];view=robot.root_physx_view
    cpu=lambda x:x.detach().cpu().numpy()
    ids=np.asarray(base.r1_joint_ids);order=np.r_[np.arange(6),ids+6]
    bodies=[robot.body_names.index(name) for name in ('Base_link','FR_wheel_link','FL_wheel_link','RR_wheel_link','RL_wheel_link')]
    mass=cpu(view.get_generalized_mass_matrices()[0])[np.ix_(order,order)]
    mass=mass+np.diag(np.r_[np.zeros(6),cpu(view.get_dof_armatures()[0])[ids]])
    bias=cpu(view.get_gravity_compensation_forces()[0]+view.get_coriolis_and_centrifugal_compensation_forces()[0])[order]
    jacobian=cpu(view.get_jacobians()[0])[bodies][:,:,order]
    return dict(dynamics_mass=mass,dynamics_bias=bias,dynamics_jacobian=jacobian,
        dynamics_pre_velocity=np.r_[cpu(robot.data.root_com_lin_vel_w[0]),cpu(robot.data.root_ang_vel_w[0]),cpu(robot.data.joint_vel[0])[ids]],
        dynamics_pre_com=cpu(robot.data.body_com_pos_w[0])[bodies])
