"""R1统一环境MDP：坡面相对奖励、刚性载荷、固定工况出生与真实单位日志。"""
import math
import xml.etree.ElementTree as ET
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from isaaclab.utils.math import quat_apply,quat_apply_inverse
from . import core,spec,torque


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
    v=core.route_velocity(robot.data.root_lin_vel_w,env.r1_tangent,env.r1_normal)
    return robot,v,up


def tracking_gate(env):
    """正向跟踪奖励的平滑安全系数；姿态/航向/净高恶化时减分，最低0.05保留学习信号。

    仅缩放正奖励，不屏蔽碰撞、终止、力矩或其他负代价；移动时不要求四轮同时承载。
    """
    heading=heading_error(env).squeeze(-1)
    height=((env.scene["robot"].data.root_pos_w-env.scene.env_origins)*env.r1_normal).sum(-1)
    orientation=torch.exp(-heading.square()/.35**2-posture_cost(env)/.08-(height-.416).square()/.1**2)
    clearance=(hitch_clearance(env)/env.cfg.hitch_margin_m).clamp(0.,1.)
    clearance=torch.where(env.r1_towed_kg>0,clearance,torch.ones_like(clearance))
    return .05+.95*orientation*clearance


def tracking(env,fine=False):
    """空载/停车按路线速度跟踪；负载移动按安全前向进度给分，不设置最低移动速度。

    负载达到参考速度后进度分饱和，避免以无限加速刷分；横向滑移降低进度分。
    返回[0,1]正奖励，并乘姿态/航向/牵引净高的平滑系数。
    """
    _,velocity,_=state(env)
    command=env.command_manager.get_command("base_velocity")
    loaded=(env.r1_payload_kg>0)|(env.r1_towed_kg>0)
    moving=command[:,0]>.05
    progress=(velocity[:,0]/command[:,0].clamp_min(.2)).clamp(0.,1.)
    progress*=torch.exp(-velocity[:,1].square()/.25**2)
    score=core.velocity_score(velocity-command[:,:2],command[:,0],fine)
    score=torch.where(loaded&moving,progress,score)
    return score*tracking_gate(env)


def tracking_l1(env):
    """空载使用平滑L1跟踪误差；负载移动仅约束横向和后退，不强制精确参考速度。"""
    _,velocity,_=state(env)
    command=env.command_manager.get_command("base_velocity")
    loaded=(env.r1_payload_kg>0)|(env.r1_towed_kg>0)
    delta=(velocity-command[:,:2]).abs()
    delta[:,0]=torch.where(loaded&(command[:,0]>.05),(-velocity[:,0]).clamp_min(0.),delta[:,0])
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



def posture_cost(env):
    """坡面允许主动调姿；返回横滚与超出俯仰容许带的角度平方代价。

    上坡25度时允许机身相对坡面低头0到10度，给前腿收缩/后腿伸长留空间；
    下坡只给正负5度自由度，不强制反向姿态。平地回到原参考。
    该项不改变45度翻覆、碰撞和牵引净高保护，也不锁定某套关节角。
    """
    robot,_,up=state(env)
    forward=quat_apply(robot.data.root_quat_w,env.r1_axis_x)
    vertical=(forward*env.r1_normal).sum(-1)
    planar=(forward-env.r1_normal*vertical[:,None]).norm(dim=-1)
    pitch=torch.atan2(vertical,planar)
    tolerance=(env.r1_slope_rad.abs()*.2).clamp_max(math.radians(5.))
    center=torch.where(env.r1_slope_rad>0,-tolerance,torch.zeros_like(tolerance))
    pitch_error=((pitch-center).abs()-tolerance).clamp_min(0.)
    lateral=torch.cross(env.r1_normal,env.r1_tangent,dim=-1)
    roll_sine=(up*lateral).sum(-1)
    return .5*(roll_sine.square()+pitch_error.square())


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
    """所有工况激活；按显式PD真实上限归一化，返回关节力矩使用率平方均值。

    上限来自actuator.effort_limit；求解器的1e9保护值不表示电机能力。
    保留轻度正则，允许策略为完成任务使用必要的承载力矩。
    """
    robot=env.scene["robot"]
    return (robot.data.applied_torque/torque.actuator_limits(robot)).square().mean(-1)


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
    term=env.command_manager.get_term("base_velocity")
    return (term.command.norm(dim=-1)<1e-4)&(term.target_vx.abs()<1e-4)


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


def stand_drift(env):
    """实际指令归零2秒后，惩罚坡面内位置漂移；返回按5cm归一化并限幅的平方。

    漂移仅作训练奖励和验收真值，不新增到Actor观测；不宣称有额外部署定位传感器。
    """
    term=env.command_manager.get_term("base_velocity")
    delta=env.scene["robot"].data.root_pos_w-term.hold_origin
    planar=delta-env.r1_normal*(delta*env.r1_normal).sum(-1,keepdim=True)
    return standing(env)*(term.hold_elapsed>=2.)*(planar.norm(dim=-1)/.05).clamp_max(3.).square()


def stand_wheels(env,asset_cfg):
    """仅实际零指令且目标也为零时抑制轮速；返回四轮角速度平方和。"""
    return standing(env)*env.scene[asset_cfg.name].data.joint_vel[:,asset_cfg.joint_ids].square().sum(-1)


def stand_pose(env,asset_cfg):
    """仅平地停车轻度约束默认腿姿；坡面取消该项，允许前后腿主动调姿。返回平方和。"""
    robot=env.scene[asset_cfg.name]
    return standing(env)*(env.r1_slope_rad.abs()<math.radians(2.))*(robot.data.joint_pos[:,asset_cfg.joint_ids]-robot.data.default_joint_pos[:,asset_cfg.joint_ids]).square().sum(-1)


def bad_surface_orientation(env,limit=math.pi/4):
    return surface_tilt(env)>1.-math.cos(limit)


def outside_lane(env):
    delta=env.scene["robot"].data.root_pos_w-env.scene.env_origins
    return (delta[:,0]<-18.)|(delta[:,0]>98.)|(delta[:,1].abs()>4.5)


def hitch_clearance(env):
    """两端牵引点到所属坡面的最小法向距离；用于拒绝穿地的低位牵引安装。"""
    robot,cart=env.scene["robot"],env.scene["trailer"]
    rh=quat_apply(robot.data.root_quat_w,env.r1_robot_hitch)+robot.data.root_pos_w
    ch=quat_apply(cart.data.root_quat_w,env.r1_cart_hitch)+cart.data.root_pos_w
    h1=((rh-env.scene.env_origins)*env.r1_normal).sum(-1)
    h2=((ch-env.scene.env_origins)*env.r1_normal).sum(-1)
    return torch.minimum(h1,h2)


def hitch_clearance_cost(env):
    """拖车激活且连接点不足8cm时提前惩罚；返回[0,1]的间隙缺额平方。

    8cm是学习用余量，5mm仍是实际失败边界；无拖车时返回0。
    在穿地前连续提供信号，避免仅靠终止才知道牵引点过低。
    """
    margin=env.cfg.hitch_margin_m
    deficit=((margin-hitch_clearance(env))/margin).clamp(0.,1.)
    return deficit.square()*(env.r1_towed_kg>0)


def drawbar_ground(env):
    """仅有拖车时激活；牵引点法向净高小于5mm判失败，不允许靠穿地模型获益。"""
    return (env.r1_towed_kg>0)&(hitch_clearance(env)<.005)


def cart_failure(env):
    cart=env.scene["trailer"]
    up=quat_apply(cart.data.root_quat_w,env.r1_axis_z)
    return (env.r1_towed_kg>0)&(((up*env.r1_normal).sum(-1)<.5)|
           env.r1_drawbar_guard|(env.r1_drawbar_length>1.6))


def set_payload_properties(env,env_ids=None):
    """启动/固定评估时更新质量、质心和主惯量；无比例缩放旧机身惯量的近似。"""
    robot,cart=env.scene["robot"],env.scene["trailer"]
    ids=torch.arange(env.num_envs,dtype=torch.long) if env_ids is None else torch.as_tensor(env_ids,device="cpu")
    if bool(((env.r1_payload_kg>0)&(env.r1_towed_kg>0)).any()):
        raise ValueError("V2 forbids simultaneous payload and towing")
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
    if hasattr(env,"r1_eval_phase"):
        env.r1_eval_phase[ids]=0
        env.r1_eval_stop_time[ids]=-1.
        env.r1_eval_origin[ids]=pose[:,:3]
        env.r1_eval_progress[ids]=0.


def _measurements_v2(env,command=None):
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
            "torque_saturation":(data.applied_torque.abs()>=torque.actuator_limits(robot)*.99).float().mean(-1),
            "leg_target_clip":clips[0].mean(-1),"wheel_target_clip":clips[1].mean(-1),
            "drawbar_force_N":env.r1_drawbar_force,"drawbar_length_m":env.r1_drawbar_length,
            "drawbar_force_peak_N_1kHz":env.r1_drawbar_peak,
            "rolling_torque_abs_Nm":env.r1_rolling_torque.abs().sum(-1),
            "hitch_clearance_m":torch.where(env.r1_towed_kg>0,hitch_clearance(env),torch.ones_like(env.r1_towed_kg)),
            "eval_phase":env.r1_eval_phase.float(),"eval_progress_m":env.r1_eval_progress,
            "eval_stop_time_s":env.r1_eval_stop_time}
    stats=env.r1_torque_recorder.summaries(env.r1_motor_limits)
    result.update(torque_peak_Nm_1kHz=stats["peak"].amax(-1),
                  torque_saturation_1kHz=stats["saturation"].mean(-1),
                  torque_clipped_1kHz=stats["clipped"].mean(-1),
                  torque_request_peak_Nm_1kHz=stats["request_peak"].amax(-1))
    joint_targets=torch.cat(targets,dim=-1)
    for i,name in enumerate(spec.CONTRACT["action_order"]):
        result["target_"+name]=joint_targets[:,i]
        result["torque_"+name]=data.applied_torque[:,env.r1_joint_ids[i]]
        for key in ("peak","rms","request_peak","saturation","clipped"):
            result["motor_"+key+"_"+name]=stats[key][:,i]
    for i,name in enumerate(("FR","FL","RR","RL")):
        result["force_n_"+name]=wheel_normal[:,i]
    for name in env.termination_manager.active_terms:
        result["termination_"+name]=env.termination_manager.get_term(name).float()
    return result


def measurements(env,command=None):
    """保留原物理/终止前指标，新增原始目标、坡面路线速度和停车漂移。"""
    values=_measurements_v2(env,command)
    term=env.command_manager.get_term("base_velocity")
    robot=env.scene["robot"]
    forward=quat_apply(robot.data.root_quat_w,env.r1_axis_x)
    values["body_pitch_relative_slope_deg"]=torch.asin((forward*env.r1_normal).sum(-1).clamp(-1.,1.))*180/math.pi
    values["posture_cost"]=posture_cost(env)
    values["z"]=robot.data.root_pos_w[:,2]
    values["target_vx"]=getattr(env,"r1_transition_target",term.target_vx).clone()
    values["command_mode_step"]=torch.full_like(term.target_vx,float(env.cfg.command_profile=="step"))
    delta=env.scene["robot"].data.root_pos_w-term.hold_origin
    planar=delta-env.r1_normal*(delta*env.r1_normal).sum(-1,keepdim=True)
    values["stand_drift_m"]=planar.norm(dim=-1)
    values["hold_elapsed_s"]=term.hold_elapsed.clone()
    return values
