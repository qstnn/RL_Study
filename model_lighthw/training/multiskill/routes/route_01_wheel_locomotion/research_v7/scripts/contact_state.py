"""教师接触状态：几何闭合与一个原生控制周期的承载联合判断。"""
import numpy as np

MAX_CONTACT_GAP_M = 1.e-4
MAX_PENETRATION_M = 1.e-3
MINIMUM_MEAN_LOAD_N = 1.

def recovery_gaps(wheel_center_history, native_gap_history, point_count_history, normal, origin):
    """Estimate a wheel's current surface gap only when the native view has no point.

    Each wheel's effective radius is calibrated from its own recent, valid native
    contacts. A missing calibration remains a large invalid gap rather than being
    guessed from the URDF radius, which differs from the PhysX contact geometry.
    """
    centers=np.asarray(wheel_center_history);gaps=np.asarray(native_gap_history)
    counts=np.asarray(point_count_history);n=np.asarray(normal);o=np.asarray(origin)
    if centers.ndim!=3 or centers.shape[1:]!=(4,3) or gaps.shape!=centers.shape[:2] or counts.shape!=gaps.shape or n.shape!=(3,) or o.shape!=(3,):
        raise ValueError('Expected matching wheel/contact history, ground normal and origin')
    height=(centers-o)@n
    result=gaps[-1].copy()
    for wheel in range(4):
        if counts[-1,wheel]>0:continue
        valid=(counts[:,wheel]>0)&np.isfinite(gaps[:,wheel])&(abs(gaps[:,wheel])<=.001)
        if not np.any(valid):continue
        effective_radius=np.median(height[valid,wheel]-gaps[valid,wheel])
        result[wheel]=height[-1,wheel]-effective_radius
    return result

def check_contact(force_history, normal, point_count, minimum_separation):
    """接管后使用最近20个1kHz样本；瞬时零冲量不直接等同离地。

    返回支持状态和可审计诊断。接触点还在contactOffset内并不足够：
    同时要求间隙<=0.1mm、穿入<=1mm、每轮平均载荷>=1N。
    这是教师控制器的接触估计，不改完整行为验收的支撑率门槛。
    """
    f=np.asarray(force_history);n=np.asarray(normal);count=np.asarray(point_count);gap=np.asarray(minimum_separation)
    if f.shape!=(20,4,3) or n.shape!=(3,) or count.shape!=(4,) or gap.shape!=(4,):
        raise ValueError('Expected 20 physical samples and four wheel contacts')
    finite=all(np.isfinite(x).all() for x in (f,n,count,gap))
    mean=(f@n).mean(0)
    supported=finite and np.all(count>0) and np.all(gap<=MAX_CONTACT_GAP_M) and np.all(gap>=-MAX_PENETRATION_M) and np.all(mean>=MINIMUM_MEAN_LOAD_N)
    return dict(supported=bool(supported),mean_normal_N=mean,latest_normal_N=f[-1]@n,
        point_count=count,minimum_separation_m=gap,window_s=.02,
        max_gap_m=MAX_CONTACT_GAP_M,max_penetration_m=MAX_PENETRATION_M)

def record_contact_geometry(physical, base):
    """只读原生接触点；在已有1kHz记录后采样，不改物理或动作。"""
    import torch
    shapes=dict(native_contact_count=(4,),native_contact_min_separation_m=(4,),
        native_patch_force_w=(4,3),native_contact_centroid_w=(4,3),
        native_contact_cop_w=(4,3),native_contact_moment_com_w=(4,3),
        native_friction_moment_com_w=(4,3),body_angular_velocity_w=(17,3))
    for key,shape in shapes.items():
        physical.buffers[key]=torch.zeros((base.num_envs,base.cfg.decimation,*shape),device=base.device)
        physical.history[key]=[]
    sample=physical.sample
    def with_geometry():
        index=physical.cursor;sample()
        force,points,normals,separation,counts,starts=physical.contact_view.get_contact_data(dt=base.physics_dt)
        counts=counts.long();starts=starts.long();offset=physical.contact_offsets
        if (counts>len(offset)).any():raise RuntimeError('Contact point evidence truncated')
        valid=offset[None,None,:]<counts[:,:,None]
        ix=(starts[:,:,None]+offset).clamp(0,len(force)-1)
        f=torch.where(valid,force[ix,0],0.)
        vector=(f[:,:,:,None]*normals[ix]).sum((1,2));count=counts.sum(1)
        # 无接触使用有限哨兵1000m，必须结合count判断，不把它当实测间隙。
        gap=torch.where(valid,separation[ix,0],1000.).flatten(1).min(1).values
        centroid=torch.where(valid[:,:,:,None],points[ix],0.).sum((1,2))/count.clamp_min(1)[:,None]
        # 逐点法向力矩与摩擦锚点力矩分别求和，不用几何质心代替压力中心。
        normal_vectors=f[:,:,:,None]*normals[ix]
        com=(physical.robot.data.body_com_pos_w*physical.weight[:,:,None]).sum(1)
        ordered_com=com[:,None,:].expand(-1,4,-1).reshape(-1,3)
        raw_com=torch.empty_like(ordered_com);raw_com[physical.contact_order]=ordered_com
        normal_moment=torch.cross(points[ix]-raw_com[:,None,None,:],normal_vectors,dim=-1).sum((1,2))
        total=f.sum((1,2));cop=(f[:,:,:,None]*points[ix]).sum((1,2))/total.clamp_min(1.e-12)[:,None]
        cop=torch.where((total>1.e-8)[:,None],cop,centroid)
        friction,fp,fc,fs=physical.contact_view.get_friction_data(dt=base.physics_dt)
        fc=fc.long();fs=fs.long()
        if (fc>len(offset)).any():raise RuntimeError('Friction point evidence truncated')
        fv=offset[None,None,:]<fc[:,:,None];fi=(fs[:,:,None]+offset).clamp(0,len(friction)-1)
        force_vectors=torch.where(fv[:,:,:,None],friction[fi],0.)
        friction_moment=torch.cross(fp[fi]-raw_com[:,None,None,:],force_vectors,dim=-1).sum((1,2))
        values=dict(native_contact_count=count,native_contact_min_separation_m=gap,
            native_patch_force_w=vector,native_contact_centroid_w=centroid,native_contact_cop_w=cop,
            native_contact_moment_com_w=normal_moment,native_friction_moment_com_w=friction_moment)
        for key,value in values.items():
            physical.buffers[key][:,index]=value[physical.contact_order].reshape(base.num_envs,*shapes[key])
        physical.buffers['body_angular_velocity_w'][:,index]=physical.robot.data.body_ang_vel_w
    physical.sample=with_geometry
