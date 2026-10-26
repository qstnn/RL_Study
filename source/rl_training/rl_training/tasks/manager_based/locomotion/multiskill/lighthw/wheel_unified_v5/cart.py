"""四轮被动车的物理子步外力，不参与Actor动作输出。"""
import torch
from isaaclab.envs.mdp.actions import JointVelocityAction
from isaaclab.utils.math import quat_apply
from .cart_asset import RADIUS,ROBOT_HITCH,CART_HITCH,drawbar_force

def apply_cart_forces(env):
    robot,cart=env.scene["robot"],env.scene["trailer"]
    active=env.r1_towed_kg>0
    rh=quat_apply(robot.data.root_quat_w,env.r1_robot_hitch)+robot.data.root_pos_w
    ch=quat_apply(cart.data.root_quat_w,env.r1_cart_hitch)+cart.data.root_pos_w
    rr=rh-robot.data.body_com_pos_w[:,env.r1_base_id]
    cr=ch-cart.data.body_com_pos_w[:,env.r1_cart_base_id]
    rv=robot.data.root_com_lin_vel_w+torch.cross(robot.data.root_ang_vel_w,rr,dim=-1)
    cv=cart.data.root_com_lin_vel_w+torch.cross(cart.data.root_ang_vel_w,cr,dim=-1)
    force,length,tension=drawbar_force(rh,ch,rv,cv,active)
    robot.set_external_force_and_torque((-force)[:,None,:],torch.cross(rr,-force,dim=-1)[:,None,:],
                                       body_ids=[env.r1_base_id],is_global=True)
    # 被动车关闭引擎重力，在每个1ms物理步按每个link真实质量施加同等重力。
    # 这样0kg牵引样本的未连接拖车可停放在远处且不自由落体干扰场景。
    forces=torch.zeros((env.num_envs,cart.num_bodies,3),device=env.device)
    forces[:,:,2]=-9.81*env.r1_cart_masses*active[:,None]
    forces[:,env.r1_cart_base_id]+=force
    torques=torch.zeros_like(forces)
    torques[:,env.r1_cart_base_id]=torch.cross(cr,force,dim=-1)
    sensor=env.scene["trailer_contacts"]
    normals=(sensor.data.net_forces_w[:,env.r1_cart_sensor_ids]*env.r1_normal[:,None,:]).sum(-1).clamp_min(0.)
    speed=cart.data.joint_vel[:,env.r1_cart_joint_ids]
    resistance=-env.cfg.rolling_coefficient*normals*RADIUS*torch.tanh(speed/.2)*active[:,None]
    axis=quat_apply(cart.data.root_quat_w,env.r1_cart_axis)
    torques[:,env.r1_cart_wheel_ids]+=resistance[:,:,None]*axis[:,None,:]
    cart.set_external_force_and_torque(forces,torques,is_global=True)
    env.r1_drawbar_length=length
    env.r1_drawbar_force=tension
    env.r1_drawbar_guard|=tension.abs()>=1499.
    env.r1_drawbar_peak=torch.maximum(env.r1_drawbar_peak,tension.abs())
    env.r1_rolling_torque=resistance


class TowingWheelAction(JointVelocityAction):
    """保留原轮速映射；每个物理子步更新被动车外力，避免20ms延迟弹簧不稳定。"""
    def apply_actions(self):
        super().apply_actions()
        if getattr(self._env,"r1_physics_ready",False):
            apply_cart_forces(self._env)
