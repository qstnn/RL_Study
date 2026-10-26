"""隔离物理试验：捕获/实时Actor腿参考、接触前馈和有限幅姿态修正。

只输出原关节位置目标增量，不修改质量、外力或接触，不要求同时四轮承载。
"""
from dataclasses import asdict, dataclass
import numpy as np
from scipy.spatial.transform import Rotation
from model import Model
from research_common import LEGS

SCALE=np.tile([.08,.14,.14],4)


@dataclass(frozen=True)
class BalanceOptions:
    reference_mode: str='captured'
    capture_start_s: float=13.
    capture_end_s: float=15.
    blend_s: float=1.
    pitch_gain: float=0.
    roll_gain: float=0.
    rate_gain: float=0.
    forward_shift_m: float=0.
    rear_foot_back_m: float=0.
    body_lower_m: float=0.
    stance_widen_m: float=0.
    leg_force_gain: float=0.
    contact_moment_gain: float=0.
    hip_force_scale: float=1.
    joint_feedback_gain: float=0.
    joint_integral_gain: float=0.
    hip_feedback_gain: float=0.
    hip_damping_s: float=0.
    force_pair_average: float=0.
    force_filter_s: float=.12
    rate_filter_s: float=0.
    diagonal_admittance_m_Ns: float=0.
    horizontal_attitude_gain: float=0.
    geometry_mode: str='linear'
    load_geometry: str='reference'
    max_joint_delta_rad: float=.22
    max_leg_rate_rad_s: float=.6

    def __post_init__(self):
        if self.reference_mode not in ('captured','actor'):raise ValueError('Invalid leg reference mode')
        if self.geometry_mode not in ('linear','iterative') or self.load_geometry not in ('reference','desired'):raise ValueError('Invalid posture geometry mode')
        if not all(np.isfinite(v) for k,v in asdict(self).items() if k not in ('reference_mode','geometry_mode','load_geometry')):raise ValueError('Nonfinite balance option')
        if not 3<=self.capture_start_s<self.capture_end_s or self.blend_s<.5:raise ValueError('Invalid capture/blend window')
        if not all(0<=v<=2 for v in (self.pitch_gain,self.roll_gain,self.rate_gain)):raise ValueError('Invalid feedback gain')
        if not 0<=self.leg_force_gain<=1.:raise ValueError('Invalid force feedforward gain')
        if not 0<=self.contact_moment_gain<=1.:raise ValueError('Invalid contact moment gain')
        if not 0<=self.hip_force_scale<=1. or not 0<=self.joint_feedback_gain<=4. or not 0<=self.joint_integral_gain<=10.:
            raise ValueError('Invalid joint feedback gain')
        if not 0<=self.hip_feedback_gain<=4. or not 0<=self.hip_damping_s<=.2:
            raise ValueError('Invalid lateral joint feedback')
        if not 0<=self.force_pair_average<=1.:raise ValueError('Invalid paired force compensation')
        if not .02<=self.force_filter_s<=.5:raise ValueError('Invalid force filter duration')
        if self.rate_filter_s!=0 and not .02<=self.rate_filter_s<=.5:raise ValueError('Invalid angular rate filter duration')
        if not 0<=self.diagonal_admittance_m_Ns<=.02:raise ValueError('Invalid diagonal load admittance')
        if not 0<=self.horizontal_attitude_gain<=1.:raise ValueError('Invalid horizontal attitude correction')
        if not 0<=self.rear_foot_back_m<=.2:raise ValueError('Invalid rear support extension')
        if not 0<=self.forward_shift_m<=.16 or not 0<=self.body_lower_m<=.18 or not 0<=self.stance_widen_m<=.08 or not 0<self.max_joint_delta_rad<=1.5 or not 0<self.max_leg_rate_rad_s<=5:
            raise ValueError('Invalid geometry/rate bound')


def attitude(quat,normal,omega_body):
    """正俯仰为抬头，正横滚为左侧抬起；弧度及反馈用的小角度角速度近似。"""
    R=Rotation.from_quat(np.asarray(quat)[[1,2,3,0]]).as_matrix()
    pitch=np.arcsin(np.clip(R[:,0]@normal,-1,1))
    roll=np.arctan2(R[:,1]@normal,R[:,2]@normal)
    return np.array([pitch,roll]),np.array([-omega_body[1],omega_body[0]],dtype=float)


def foot_adjustments(centers,error,rates,options):
    """抬起的一侧收腿以降低该侧机身；全轮后移参考让机身相对支撑前移。"""
    correction=np.array([options.pitch_gain,options.roll_gain])*error+options.rate_gain*rates
    result=np.zeros((4,3));result[:,0]=-options.forward_shift_m
    result[2:,0]-=options.rear_foot_back_m
    result[:,1]=np.sign(centers[:,1])*options.stance_widen_m
    result[:,2]=options.body_lower_m+np.clip(centers[:,0]*correction[0]+centers[:,1]*correction[1],-.05,.05)
    # 小角度逆旋转的水平分量：横滚恢复同时需要髋关节侧向配合。
    # 默认0保留原仅高度修正；1采用同一姿态反馈的完整三维位移。
    result[:,0]+=options.horizontal_attitude_gain*np.clip(-centers[:,2]*correction[0],-.05,.05)
    result[:,1]+=options.horizontal_attitude_gain*np.clip(-centers[:,2]*correction[1],-.05,.05)
    return result


def solve_foot_pose(model,reference_q,centers,offset):
    """以原URDF关节界求四轮中心IK；用于较大姿态调整，不移动仿真根状态。"""
    q=np.asarray(reference_q,dtype=float).copy();target=np.asarray(centers)+offset
    for _ in range(8):
        state=model.state(np.zeros(3),[1,0,0,0],q)
        error=[]
        for i,leg in enumerate(LEGS):
            center=state['trans'][leg+'_wheel_link'][:3,3];e=target[i]-center;error.append(e)
            J=state['jac'](leg+'_wheel_link',center)[:3,6+3*i:9+3*i]
            dq=np.linalg.pinv(J,rcond=1.e-4)@e
            q[3*i:3*i+3]+=np.clip(dq,-.2,.2)
        q[:12]=np.clip(q[:12],model.limits[:12,0],model.limits[:12,1])
        if np.max(abs(np.asarray(error)))<1.e-7:break
    return q


def support_joint_torque(model,state,force_body,moment_body):
    """静态腿关节力矩g-Jv^TF-Jw^TM；仅生成位置伺服前馈，不外加助力。"""
    R=state['trans']['Base_link'][:3,:3]
    gravity=sum(state['jac'](name,state['com'][name])[:3,6:18].T@np.array([0.,0.,mass*9.81]) for name,mass in model.mass.items())
    for i,leg in enumerate(LEGS):
        center=state['trans'][leg+'_wheel_link'][:3,3]
        J=state['jac'](leg+'_wheel_link',center)[:,6:18]
        gravity-=J[:3].T@(R@force_body[i])+J[3:].T@(R@moment_body[i])
    return gravity


class PartialSupportControl:
    def __init__(self,options,force_calibration=None,servo_stiffness=None):
        self.options=options;self.samples=[];self.model=Model();self.reference=None;self.previous=None
        self.servo_stiffness=np.full(12,80.) if servo_stiffness is None else np.asarray(servo_stiffness,dtype=float)
        if self.servo_stiffness.shape!=(12,) or not np.isfinite(self.servo_stiffness).all() or np.any(self.servo_stiffness<=0):
            raise ValueError('Expected positive actual 12-joint servo stiffness')
        self.force_calibration=force_calibration
        if force_calibration is not None:
            if options.load_geometry!='reference':raise ValueError('Fixed load calibration requires reference geometry')
            if options.contact_moment_gain:raise ValueError('Calibrated load and measured moment compensation are separate modes')
            self.load_slope=np.asarray(force_calibration['joint_delta_rad_per_N'],dtype=float)
            self.load_intercept=np.asarray(force_calibration['joint_delta_intercept_rad'],dtype=float)
            self.calibration_stiffness=np.asarray(force_calibration.get('source_servo_stiffness_Nm_rad',[80.]*12),dtype=float)
            if any(x.shape!=(12,) or not np.isfinite(x).all() for x in (self.load_slope,self.load_intercept)):
                raise ValueError('Expected finite 12-joint load calibration')
            if self.calibration_stiffness.shape!=(12,) or not np.isfinite(self.calibration_stiffness).all() or np.any(self.calibration_stiffness<=0):
                raise ValueError('Invalid calibration source stiffness')

    def step(self,time_s,leg_action,joints,quat,normal,omega_body,force_world=None,moment_world=None,joint_velocity=None,known_load_N=None):
        opt=self.options;leg_action=np.asarray(leg_action,dtype=float)
        if time_s<opt.capture_start_s and np.linalg.norm(quat)<1.e-8:
            angles,rates=np.zeros(2),np.zeros(2)
        else:angles,rates=attitude(quat,normal,omega_body)
        trace=dict(balance_active=0.,balance_angles_rad=angles,balance_reference_rad=np.zeros(2),
                   balance_leg_delta_rad=np.zeros(12),balance_foot_delta_m=np.zeros((4,3)),balance_force_delta_rad=np.zeros(12))
        if force_world is None:
            if opt.leg_force_gain:raise ValueError('Force compensation requires real contact forces')
            force_body=np.zeros((4,3))
        elif time_s==0 and np.linalg.norm(quat)<1.e-8:force_body=np.zeros((4,3))
        else:
            force_world=np.asarray(force_world)
            if force_world.shape!=(4,3) or not np.isfinite(force_world).all():raise ValueError('Invalid contact force')
            R=Rotation.from_quat(np.asarray(quat)[[1,2,3,0]]).as_matrix()
            force_body=force_world@R
        moment_body=np.zeros((4,3))
        if opt.contact_moment_gain:
            if moment_world is None or np.asarray(moment_world).shape!=(4,3) or not np.isfinite(moment_world).all():
                raise ValueError('Full wrench compensation requires real wheel contact moments')
            if np.linalg.norm(quat)>1.e-8:
                moment_body=np.asarray(moment_world)@Rotation.from_quat(np.asarray(quat)[[1,2,3,0]]).as_matrix()
        if opt.capture_start_s<=time_s<opt.capture_end_s:
            self.samples.append((leg_action.copy(),np.asarray(joints).copy(),angles.copy(),force_body.copy(),moment_body.copy()))
        if time_s<opt.capture_end_s:return leg_action.copy(),trace
        if self.reference is None:
            if len(self.samples)<10:raise RuntimeError('No complete stable reference capture')
            self.reference=np.mean([s[0] for s in self.samples],axis=0)
            q=np.mean([s[1] for s in self.samples],axis=0)
            self.reference_angles=np.mean([s[2] for s in self.samples],axis=0)
            self.reference_force=np.mean([s[3] for s in self.samples],axis=0)
            self.filtered_force=self.reference_force.copy()
            self.reference_moment=np.mean([s[4] for s in self.samples],axis=0)
            self.filtered_moment=self.reference_moment.copy()
            state=self.model.state(np.zeros(3),np.array([1.,0,0,0]),q)
            self.centers=np.array([state['trans'][leg+'_wheel_link'][:3,3] for leg in LEGS])
            self.inverse=[];self.jacobians=[];self.angular_jacobians=[]
            for i,leg in enumerate(LEGS):
                J=state['jac'](leg+'_wheel_link',self.centers[i])[:3,6+3*i:9+3*i]
                self.inverse.append(np.linalg.pinv(J,rcond=1.e-4))
                self.jacobians.append(J)
                self.angular_jacobians.append(state['jac'](leg+'_wheel_link',self.centers[i])[3:,6+3*i:9+3*i])
            self.previous=leg_action.copy()
            self.reference_q=q[:12].copy();self.integral=np.zeros(12)
            self.reference_q_full=q.copy();self.reference_quat=np.asarray(quat).copy()
            if opt.load_geometry=='desired':
                reference_state=self.model.state(np.zeros(3),quat,q)
                self.reference_support_torque=support_joint_torque(self.model,reference_state,self.reference_force,opt.contact_moment_gain*self.reference_moment)
            self.filtered_rates=rates.copy()
            self.diagonal_offset=0.
            self.filtered_normals=self.reference_force@(Rotation.from_quat(np.asarray(quat)[[1,2,3,0]]).as_matrix().T@normal)
        if opt.rate_filter_s:
            alpha=-np.expm1(-.02/opt.rate_filter_s)
            self.filtered_rates+=alpha*(rates-self.filtered_rates)
            rates=self.filtered_rates.copy()
        deltafoot=foot_adjustments(self.centers,angles-self.reference_angles,rates,opt)
        if opt.diagonal_admittance_m_Ns:
            normal_body=Rotation.from_quat(np.asarray(quat)[[1,2,3,0]]).as_matrix().T@normal
            self.filtered_normals+=(.02/.12)*(force_body@normal_body-self.filtered_normals)
            sign=np.array([1.,-1.,-1.,1.])
            imbalance=float(self.filtered_normals@sign)/4.
            self.diagonal_offset=float(np.clip(self.diagonal_offset+.02*opt.diagonal_admittance_m_Ns*imbalance,-.02,.02))
            # 对角内力调节：抬高较重的一对、降低较轻的一对；不要求四轮门，
            # 不直接施力，也不改变四轮平均高度或前后/左右平均高度。
            deltafoot[:,2]+=sign*self.diagonal_offset
        delta=np.concatenate([J@foot for J,foot in zip(self.inverse,deltafoot)])
        if opt.geometry_mode=='iterative':
            pose_q=solve_foot_pose(self.model,self.reference_q_full,self.centers,deltafoot)
            delta=pose_q[:12]-self.reference_q
        self.filtered_force+=(.02/opt.force_filter_s)*(force_body-self.filtered_force)
        self.filtered_moment+=(.02/opt.force_filter_s)*(moment_body-self.filtered_moment)
        # 静力关节项 tau = g - Jv^T F - Jw^T M；按本次实际Kp反解位置伺服。
        # 完整力矩映射的静态一致性不保证这个有时延的闭环稳定；各增益默认0。
        force_delta=-opt.leg_force_gain*np.concatenate([J.T@f for J,f in zip(self.jacobians,self.filtered_force-self.reference_force)])/self.servo_stiffness
        force_delta-=opt.leg_force_gain*opt.contact_moment_gain*np.concatenate([J.T@m for J,m in zip(self.angular_jacobians,self.filtered_moment-self.reference_moment)])/self.servo_stiffness
        if opt.load_geometry=='desired':
            desired_q=self.reference_q_full.copy();desired_q[:12]+=delta
            desired_state=self.model.state(np.zeros(3),quat,desired_q)
            desired_torque=support_joint_torque(self.model,desired_state,self.filtered_force,opt.contact_moment_gain*self.filtered_moment)
            force_delta=opt.leg_force_gain*(desired_torque-self.reference_support_torque)/self.servo_stiffness
        if self.force_calibration is not None:
            if known_load_N is None or not np.isfinite(known_load_N) or known_load_N<0:
                raise ValueError('Calibrated feedforward needs a finite nonnegative known hitch load')
            # 仅用预先归档的负载曲线，不追随当前接触脉冲；外力仍由真实物理加载。
            force_delta=opt.leg_force_gain*(self.load_slope*known_load_N+self.load_intercept)*self.calibration_stiffness/self.servo_stiffness
        force_delta[::3]*=opt.hip_force_scale
        # 平均左右腿的矢状面补偿，保留共同受载项，避免追随交替轮载而
        # 抵消原伺服对左右差动形变的恢复力。横滚仍由显式姿态反馈处理。
        pairs=force_delta.reshape(2,2,3)
        pairs[:,:,1:]=(1-opt.force_pair_average)*pairs[:,:,1:]+opt.force_pair_average*pairs[:,:,1:].mean(1,keepdims=True)
        # 闭环只改变位置目标，仍由本次声明的伺服执行并受原力矩限制。
        # 积分用于抵消恒载引起的弹性偏差；限幅防止长期失接触时累积。
        error=self.reference_q+delta-np.asarray(joints)[:12]
        self.integral=np.clip(self.integral+opt.joint_integral_gain*.02*error,-.4,.4)
        delta+=opt.joint_feedback_gain*error+self.integral
        delta[::3]+=opt.hip_feedback_gain*error[::3]
        if opt.hip_damping_s:
            if joint_velocity is None or not np.isfinite(joint_velocity).all():
                raise ValueError('Hip damping requires finite measured joint velocity')
            delta[::3]-=opt.hip_damping_s*np.asarray(joint_velocity)[:12:3]
        delta+=force_delta
        delta=np.clip(delta,-opt.max_joint_delta_rad,opt.max_joint_delta_rad)
        nominal=self.reference if opt.reference_mode=='captured' else leg_action
        wanted=nominal+delta/SCALE
        u=np.clip((time_s-opt.capture_end_s)/opt.blend_s,0,1);blend=u*u*(3-2*u)
        wanted=(1-blend)*leg_action+blend*wanted
        limited=self.previous+np.clip((wanted-self.previous)*SCALE,-opt.max_leg_rate_rad_s*.02,opt.max_leg_rate_rad_s*.02)/SCALE
        self.previous=limited.copy()
        trace.update(balance_active=blend,balance_reference_rad=self.reference_angles,
                     balance_leg_delta_rad=(limited-self.reference)*SCALE,balance_foot_delta_m=deltafoot,
                     balance_force_delta_rad=force_delta)
        return limited,trace
