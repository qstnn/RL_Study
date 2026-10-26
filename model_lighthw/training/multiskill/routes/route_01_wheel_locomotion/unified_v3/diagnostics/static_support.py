"""静力辅助探针；使用已知质量和坡度，仅验证可行性，不是Actor部署控制器。"""
import math
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation


def nominal_support(urdf, slope_deg, hitch_height, towed_kg=40.):
    root=ET.parse(urdf);links={x.get('name'):x for x in root.findall('link')}
    default={f'{leg}_{j}_joint':v for leg,values in (('FR',(0,.6,-1.5)),('FL',(0,.6,-1.5)),('RR',(0,-.6,1.5)),('RL',(0,-.6,1.5)))
             for j,v in zip(('hip','thigh','calf'),values)}
    transform={'Base_link':np.eye(4)};transform['Base_link'][2,3]=.416223
    joint_data={};remaining=list(root.findall('joint'));parent_map={}
    while remaining:
        done=[]
        for joint in remaining:
            parent,child=joint.find('parent').get('link'),joint.find('child').get('link')
            if parent not in transform:continue
            origin=joint.find('origin');local=np.eye(4)
            local[:3,3]=np.fromstring(origin.get('xyz','0 0 0'),sep=' ')
            local[:3,:3]=Rotation.from_euler('xyz',np.fromstring(origin.get('rpy','0 0 0'),sep=' ')).as_matrix()
            frame=transform[parent]@local;axis=np.fromstring(joint.find('axis').get('xyz','0 0 1'),sep=' ') if joint.find('axis') is not None else np.array([0.,0.,1.])
            name=joint.get('name');joint_data[name]=(frame[:3,3].copy(),frame[:3,:3]@axis,child)
            rotation=np.eye(4);rotation[:3,:3]=Rotation.from_rotvec(axis*default.get(name,0.)).as_matrix()
            transform[child]=frame@rotation;parent_map[child]=parent;done.append(joint)
        if not done:raise ValueError('Unresolved kinematic tree')
        for joint in done:remaining.remove(joint)
    angle=math.radians(slope_deg);gravity=np.array([-9.81*math.sin(angle),0.,-9.81*math.cos(angle)])
    bodies={}
    for name,link in links.items():
        inertia=link.find('inertial')
        if inertia is None:continue
        origin=inertia.find('origin');local=np.r_[np.fromstring(origin.get('xyz','0 0 0'),sep=' '),1.]
        bodies[name]=(float(inertia.find('mass').get('value')),(transform[name]@local)[:3])
    mass=sum(v[0] for v in bodies.values());com=sum(m*x for m,x in bodies.values())/mass
    hitch=np.array([-.38,0.,hitch_height-.43+.416223]);dh=hitch[2]-hitch_height
    tow_force=-towed_kg*9.81*math.sin(angle)
    extension=-tow_force/2000.;dx=math.sqrt((1.+extension)**2-dh**2)
    external=np.array([tow_force,0.,tow_force*dh/dx])
    total=mass*gravity+external;moment=np.cross(com,mass*gravity)+np.cross(hitch,external)
    feet=np.array([transform[leg+'_wheel_link'][:3,3]-[0.,0.,.0875] for leg in ('FR','FL','RR','RL')])
    matrix=np.stack((np.ones(4),feet[:,0],feet[:,1]))
    normals=np.linalg.lstsq(matrix,np.array([-total[2],moment[1],-moment[0]]),rcond=None)[0]
    forces=normals[:,None]*np.array([-total[0]/(-total[2]),0.,1.])[None,:]
    def descends(name,ancestor):
        while name!=ancestor and name in parent_map:name=parent_map[name]
        return name==ancestor
    order=[f'{leg}_{j}_joint' for leg in ('FR','FL','RR','RL') for j in ('hip','thigh','calf')]+[f'{leg}_wheel_joint' for leg in ('FR','FL','RR','RL')]
    torques=[]
    for name in order:
        position,axis,child=joint_data[name];qforce=0.
        for link,(m,center) in bodies.items():
            if descends(link,child):qforce+=axis@np.cross(center-position,m*gravity)
        for i,leg in enumerate(('FR','FL','RR','RL')):
            if descends(leg+'_wheel_link',child):qforce+=axis@np.cross(feet[i]-position,forces[i])
        torques.append(-qforce)
    return dict(torques_Nm=torques,wheel_normals_N=normals.tolist(),friction_required=float(abs(total[0]/total[2])),
                static_nonnegative_normals=bool((normals>=0).all()),preload_extension_m=extension,
                nominal_com_m=com.tolist(),hitch_actual_m=hitch[2],assumption='Nominal pose; passive drawbar preload; zero-speed rolling resistance is zero. GPU dynamics must still validate.')
