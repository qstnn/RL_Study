"""R1统一环境MDP：坡面相对奖励、刚性载荷、固定工况出生与真实单位日志。"""
import math
import xml.etree.ElementTree as ET
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from isaaclab.utils.math import quat_apply,quat_apply_inverse
from . import core,spec


def heading_error(env):
    """可由IMU航向估计获取的相对初始航向；返回每环境1维弧度，不输入地形真值。"""
    forward=quat_apply(env.scene["robot"].data.root_quat_w,env.r1_axis_x)
    return torch.atan2(forward[:,1],forward[:,0]).unsqueeze(-1)


def privileged_context(env):
    """仅Critic使用坡面法向和真实质量；返回6维特权工况，Actor不读取。"""
    normal=quat_apply_inverse(env.scene["robot"].data.root_quat_w,env.r1_normal)
    return torch.cat((normal,env.r1_payload_kg[:,None]/15.,heading_error(env),env.r1_towed_kg[:,None]/40.),dim=-1)


def state(env):
    robot=env.scene["robot"]
    forward=quat_apply(robot.data.root_quat_w,env.r1_axis_x)
    up=quat_apply(robot.data.root_quat_w,env.r1_axis_z)
    v=core.tangent_velocity(robot.data.root_lin_vel_w,forward,env.r1_normal)
    return robot,v,up


def tracking(env,fine=False):
    """所有工况激活；沿当前机身投影到坡面的切向坐标跟踪指令，返回[0,1]。"""
    _,v,_=state(env)
    command=env.command_manager.get_command("base_velocity")
    return core.velocity_score(v-command[:,:2],command[:,0],fine)


def tracking_l1(env):
    """所有工况激活；返回平滑L1误差，给高速严重欠速提供不饱和的学习信号。"""
    _,v,_=state(env)
    delta=(v-env.command_manager.get_command("base_velocity")[:,:2]).abs()
    return torch.where(delta<.5,delta.square(),delta-.25).sum(-1)


def yaw_tracking(env):
    """绕坡面法向跟踪角速度；所有工况激活，返回指数奖励。"""
    rate=(env.scene["robot"].data.root_ang_vel_w*env.r1_normal).sum(-1)
    target=env.command_manager.get_command("base_velocity")[:,2]
    return torch.exp(-(rate-target).square()/.35**2)


def surface_tilt(env):
    """保持机身与坡面协调而非强制世界水平；返回法向夹角的有界代价[0,2]。"""
    _,_,up=state(env)
    return 1.-(up*env.r1_normal).sum(-1).clamp(-1.,1.)


def surface_height(env):
    """所有工况激活；以坡面法向离地高度0.416m为目标，返回米平方。"""
    delta=env.scene["robot"].data.root_pos_w-env.scene.env_origins
    return ((delta*env.r1_normal).sum(-1)-.416).square()


def normal_motion(env):
    """只惩罚垂直坡面的跳动；所有工况激活，返回法向速度平方。"""
    return (env.scene["robot"].data.root_lin_vel_w*env.r1_normal).sum(-1).square()


def tilt_rate(env):
    """惩罚相对坡面的翻滚角速度；所有工况激活，返回去除法向旋转后的角速平方。"""
    omega=env.scene["robot"].data.root_ang_vel_w
    return (omega-env.r1_normal*(omega*env.r1_normal).sum(-1,keepdim=True)).square().sum(-1)


def heading_cost(env):
    """直线任务保持初始航向；所有工况激活，返回航向误差平方，输入在Actor可见。"""
    return heading_error(env).squeeze(-1).square()


def effort_usage(env):
    """按额定力矩归一化，避免重载所需静态力矩压过速度目标；返回16关节使用率平方均值。"""
    data=env.scene["robot"].data
    return (data.applied_torque/data.joint_effort_limits.clamp_min(1.)).square().mean(-1)


def target_overflow(env):
    """只惩罚超过限位的未裁剪目标；允许使用边界目标，返回按各关节范围归一化的越界平方。"""
    values=[]
    for name in ("joint_pos","joint_vel"):
        term=env.action_manager.get_term(name)
        raw=term.raw_actions*term._scale+term._offset
        low,high=term._clip[:,:,0],term._clip[:,:,1]
        excess=(low-raw).clamp_min(0.)+(raw-high).clamp_min(0.)
        values.append((excess/(high-low).clamp_min(1e-3)).square())
    return torch.cat(values,dim=-1).mean(-1)


def standing(env):
    return env.command_manager.get_command("base_velocity").norm(dim=-1)<.05


def stand_motion(env):
    """只在零指令时抑制沿坡漂移和角运动；返回切向平速平方加角速平方。"""
    robot,v,_=state(env)
    return standing(env)*(v.square().sum(-1)+robot.data.root_ang_vel_w.square().sum(-1))


def stand_support(env,sensor_cfg):
    """只在停车时鼓励四轮共同承载；返回四轮法向载荷达阈值的最小归一化分数。"""
    forces=env.scene[sensor_cfg.name].data.net_forces_w[:,sensor_cfg.body_ids]
    normal=(forces*env.r1_normal[:,None,:]).sum(-1)
    # 随机器人+载荷质量缩放阈值；每轮至少名义四分之一法向重力的10%。
    threshold=(spec.CONTRACT["robot_mass_kg"]+env.r1_payload_kg)*9.81*env.r1_normal[:,2]/4.*.1
    return standing(env)*(normal/threshold[:,None]).clamp(0.,1.).amin(-1)


def bad_surface_orientation(env,limit=math.pi/4):
    return surface_tilt(env)>1.-math.cos(limit)


def outside_lane(env):
    delta=env.scene["robot"].data.root_pos_w-env.scene.env_origins
    return (delta[:,0]<-18.)|(delta[:,0]>98.)|(delta[:,1].abs()>4.5)


def cart_failure(env):
    cart=env.scene["trailer"]
    up=quat_apply(cart.data.root_quat_w,env.r1_axis_z)
    return (env.r1_towed_kg>0)&(((up*env.r1_normal).sum(-1)<.5)|
           env.r1_drawbar_guard|(env.r1_drawbar_length>1.6))


def set_payload_properties(env,env_ids=None):
    """启动/固定评估时更新质量、质心和主惯量；无比例缩放旧机身惯量的近似。"""
    robot,cart=env.scene["robot"],env.scene["trailer"]
    ids=torch.arange(env.num_envs,dtype=torch.long) if env_ids is None else torch.as_tensor(env_ids,device="cpu")
    if not hasattr(env,"r1_unloaded_mass"):
        env.r1_unloaded_mass=robot.root_physx_view.get_masses().clone()
        env.r1_unloaded_com=robot.root_physx_view.get_coms().clone()
        env.r1_unloaded_inertia=robot.root_physx_view.get_inertias().clone()
        env.r1_original_cart_mass=cart.root_physx_view.get_masses().clone()
        env.r1_original_cart_inertia=cart.root_physx_view.get_inertias().clone()
        env.r1_original_cart_com=cart.root_physx_view.get_coms().clone()
        # 与URDF原始Base_link惯量交叉核对getter轴系，而非只验证一次自洽变换。
        inertial=ET.parse(env.cfg.scene.robot.spawn.asset_path).find("link[@name='Base_link']/inertial")
        element=inertial.find("inertia")
        declared=np.array([[float(element.get("i"+min(a,b)+max(a,b))) for b in "xyz"] for a in "xyz"])
        rotation=Rotation.from_euler("xyz",[float(v) for v in inertial.find("origin").get("rpy","0 0 0").split()]).as_matrix()
        declared=rotation@declared@rotation.T
        base_id=robot.find_bodies("Base_link")[0][0]
        if not np.allclose(env.r1_unloaded_inertia[0,base_id].reshape(3,3).numpy(),declared,rtol=1e-3,atol=1e-5):
            raise RuntimeError("Unloaded inertia getter does not match canonical URDF link frame")
        env.r1_unloaded_inertia_urdf_verified=True
    masses=env.r1_unloaded_mass.clone()
    coms=env.r1_unloaded_com.clone()
    # 批量setter涵盖全部连杆：原getter返回link轴矩阵，先转换回各COM主轴，
    # 防止只改Base_link却对腿部原惯量再次施加主轴旋转。
    original=env.r1_unloaded_inertia.numpy().reshape(env.num_envs,robot.num_bodies,3,3)
    axes=Rotation.from_quat(coms[:,:,3:].numpy().reshape(-1,4)).as_matrix().reshape(env.num_envs,robot.num_bodies,3,3)
    principal=np.swapaxes(axes,-1,-2)@original@axes
    inertias=torch.as_tensor(principal.reshape(env.num_envs,robot.num_bodies,9),dtype=torch.float32).clone()
    base=robot.find_bodies("Base_link")[0][0]
    payload=env.r1_payload_kg.cpu().numpy()
    audit=[]
    expected_tensors={}
    for index in ids.tolist():
        mass=float(masses[index,base])
        c=coms[index,base,:3].numpy()
        # 本机PhysX回读为相对质心、但沿link轴表达的完整惯量。
        # 已用规范URDF和主轴写入/回读探针核对，不能再乘一次COM旋转。
        inertia=original[index,base].copy()
        total,center,tensor=core.compose_inertia(mass,c,inertia,float(payload[index]),spec.CONTRACT["payload_center_m"],spec.CONTRACT["payload_size_m"])
        expected_tensors[index]=tensor
        eigen,basis=np.linalg.eigh(tensor)
        if np.linalg.det(basis)<0:
            basis[:,0]*=-1
        masses[index,base]=total
        coms[index,base,:3]=torch.as_tensor(center,dtype=coms.dtype)
        coms[index,base,3:]=torch.as_tensor(Rotation.from_matrix(basis).as_quat(),dtype=coms.dtype)
        inertias[index,base]=torch.as_tensor(np.diag(eigen).ravel(),dtype=inertias.dtype)
        if index<6:
            audit.append({"env":index,"payload_kg":float(payload[index]),"base_mass_kg":total,
                          "base_com_m":center.tolist(),"inertia_body_frame":tensor.tolist()})
    robot.root_physx_view.set_masses(masses,ids)
    # 先写对角主惯量，再写质心主轴姿态；set_coms会使回读矩阵按link轴旋转。
    robot.root_physx_view.set_inertias(inertias,ids)
    robot.root_physx_view.set_coms(coms,ids)
    ratio=torch.where(env.r1_towed_kg.cpu()>0,env.r1_towed_kg.cpu()/40.,1.)
    cart_masses=env.r1_original_cart_mass*ratio[:,None]
    cart.root_physx_view.set_masses(cart_masses,ids)
    cart_com=env.r1_original_cart_com.clone()
    cart_axes=Rotation.from_quat(cart_com[:,:,3:].numpy().reshape(-1,4)).as_matrix().reshape(env.num_envs,cart.num_bodies,3,3)
    cart_expected=(env.r1_original_cart_inertia*ratio[:,None,None]).numpy().reshape(env.num_envs,cart.num_bodies,3,3)
    principal=np.swapaxes(cart_axes,-1,-2)@cart_expected@cart_axes
    cart.root_physx_view.set_inertias(torch.as_tensor(principal.reshape(env.num_envs,cart.num_bodies,9),dtype=torch.float32),ids)
    cart.root_physx_view.set_coms(cart_com,ids)
    if not np.allclose(cart.root_physx_view.get_inertias()[ids].numpy().reshape(-1,cart.num_bodies,3,3),cart_expected[ids],rtol=1e-3,atol=1e-5):
        raise RuntimeError("Physical trailer inertia write failed")
    env.r1_cart_masses=cart.root_physx_view.get_masses().to(env.device)
    actual=robot.root_physx_view.get_masses().sum(-1)
    expected=env.r1_unloaded_mass.sum(-1)+env.r1_payload_kg.cpu()
    if not torch.allclose(actual[ids],expected[ids],atol=1e-4,rtol=1e-5):
        raise RuntimeError("Physical payload mass did not match configured mass")
    if not torch.allclose(robot.root_physx_view.get_coms()[ids,:,:3],coms[ids,:,:3],atol=1e-5):
        raise RuntimeError("Physical COM write failed")
    actual_com=robot.root_physx_view.get_coms()
    actual_inertia=robot.root_physx_view.get_inertias()
    unchanged=[index for index in range(robot.num_bodies) if index!=base]
    if not torch.allclose(actual_inertia[ids][:,unchanged],env.r1_unloaded_inertia[ids][:,unchanged],atol=1e-6,rtol=1e-4):
        raise RuntimeError("Payload update changed non-base link inertia")
    env.r1_nonbase_inertia_unchanged=True
    for index in ids.tolist():
        physical=actual_inertia[index,base].reshape(3,3).numpy()
        if not np.allclose(physical,expected_tensors[index],atol=1e-5,rtol=1e-4):
            raise RuntimeError(f"Physical inertia mismatch: env={index}, expected={expected_tensors[index].tolist()}, physical={physical.tolist()}, actual_com={actual_com[index,base].tolist()}, requested_com={coms[index,base].tolist()}, actual_I={actual_inertia[index,base].tolist()}, requested_I={inertias[index,base].tolist()}")
    env.r1_payload_audit=audit
    env.r1_inertia_verified=True


def reset_ramp(env,env_ids):
    """在对应坡面、法向高度0.43m出生；关节仍使用规范默认角，不改公共资产。"""
    ids=torch.as_tensor(env_ids,device=env.device)
    robot,cart=env.scene["robot"],env.scene["trailer"]
    angle=env.r1_slope_rad[ids]
    pose=robot.data.default_root_state[ids].clone()
    pose[:,:3]=env.scene.env_origins[ids]+env.r1_normal[ids]*.43
    pose[:,3:7]=0.
    pose[:,3]=torch.cos(angle/2.)
    pose[:,5]=-torch.sin(angle/2.)
    pose[:,7:]=0.
    robot.write_root_pose_to_sim(pose[:,:7],env_ids=ids)
    robot.write_root_velocity_to_sim(pose[:,7:],env_ids=ids)
    trailer_pose=cart.data.default_root_state[ids].clone()
    trailer_pose[:,:3]=env.scene.env_origins[ids]+env.r1_normal[ids]*.30-env.r1_tangent[ids]*1.73
    trailer_pose[:,3:7]=pose[:,3:7]
    trailer_pose[:,7:]=0.
    inactive=env.r1_towed_kg[ids]==0
    trailer_pose[inactive,1]=500.
    trailer_pose[inactive,2]=1.
    cart.write_root_pose_to_sim(trailer_pose[:,:7],env_ids=ids)
    cart.write_root_velocity_to_sim(trailer_pose[:,7:],env_ids=ids)
    cart.write_joint_state_to_sim(cart.data.default_joint_pos[ids],torch.zeros_like(cart.data.default_joint_vel[ids]),env_ids=ids)
    env.r1_drawbar_force[ids]=0.
    env.r1_drawbar_length[ids]=1.
    env.r1_drawbar_guard[ids]=False
    env.r1_drawbar_peak[ids]=0.


def measurements(env,command=None):
    """50Hz真实单位诊断；调用方保留终止前状态，避免把自动重置后的出生帧计入成功。"""
    robot,v,up=state(env)
    data=robot.data
    command=env.command_manager.get_command("base_velocity") if command is None else command
    forces=env.scene["contact_forces"].data.net_forces_w
    wheel_normal=(forces[:,env.r1_wheel_sensor_ids]*env.r1_normal[:,None,:]).sum(-1)
    threshold=(spec.CONTRACT["robot_mass_kg"]+env.r1_payload_kg)*9.81*env.r1_normal[:,2]/4.*.1
    targets=[];clips=[]
    for key in ("joint_pos","joint_vel"):
        term=env.action_manager.get_term(key)
        targets.append(term.processed_actions)
        clips.append(((term.processed_actions<=term._clip[:,:,0]+1e-5)|(term.processed_actions>=term._clip[:,:,1]-1e-5)).float())
    cart_up=quat_apply(env.scene["trailer"].data.root_quat_w,env.r1_axis_z)
    result={"cart_tilt_deg":torch.acos((cart_up*env.r1_normal).sum(-1).clamp(-1.,1.))*180/math.pi,"slope_deg":env.r1_slope_rad*180/math.pi,"payload_kg":env.r1_payload_kg,"towed_kg":env.r1_towed_kg,
            "command_vx":command[:,0],"vx":v[:,0],"vy":v[:,1],"speed":v.norm(dim=-1),
            "velocity_error":(v-command[:,:2]).norm(dim=-1),
            "yaw_rate_abs":(data.root_ang_vel_w*env.r1_normal).sum(-1).abs(),
            "tilt_deg":torch.acos((up*env.r1_normal).sum(-1).clamp(-1.,1.))*180/math.pi,
            "height":((data.root_pos_w-env.scene.env_origins)*env.r1_normal).sum(-1),
            "heading_rad":heading_error(env).squeeze(-1),"x":data.root_pos_w[:,0],"y":data.root_pos_w[:,1],
            "four_wheel":(wheel_normal>threshold[:,None]).all(-1).float(),
            "body_contact":(forces[:,env.r1_body_sensor_ids].norm(dim=-1)>1.).any(-1).float(),
            "torque_abs_max":data.applied_torque.abs().amax(-1),
            "torque_saturation":(data.applied_torque.abs()>=data.joint_effort_limits*.99).float().mean(-1),
            "leg_target_clip":clips[0].mean(-1),"wheel_target_clip":clips[1].mean(-1),
            "drawbar_force_N":env.r1_drawbar_force,"drawbar_length_m":env.r1_drawbar_length,
            "drawbar_force_peak_N_1kHz":env.r1_drawbar_peak,
            "rolling_torque_abs_Nm":env.r1_rolling_torque.abs().sum(-1)}
    joint_targets=torch.cat(targets,dim=-1)
    for i,name in enumerate(spec.CONTRACT["action_order"]):
        result["target_"+name]=joint_targets[:,i]
        result["torque_"+name]=data.applied_torque[:,env.r1_joint_ids[i]]
    for i,name in enumerate(("FR","FL","RR","RL")):
        result["force_n_"+name]=wheel_normal[:,i]
    for name in env.termination_manager.active_terms:
        result["termination_"+name]=env.termination_manager.get_term(name).float()
    return result
