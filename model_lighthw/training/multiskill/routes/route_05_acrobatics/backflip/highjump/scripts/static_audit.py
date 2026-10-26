"""可复算的名义动力学边界；不是实机能力或后空翻可行性证明。"""
import math
import xml.etree.ElementTree as ET
import numpy as np


def rpy(values):
    r,p,y=values
    cr,sr,cp,sp,cy,sy=math.cos(r),math.sin(r),math.cos(p),math.sin(p),math.cos(y),math.sin(y)
    return np.array([[cy*cp,cy*sp*sr-sy*cr,cy*sp*cr+sy*sr],
                     [sy*cp,sy*sp*sr+cy*cr,sy*sp*cr-cy*sr],[-sp,cp*sr,cp*cr]])


def origin(element):
    t=np.eye(4)
    if element is not None:
        t[:3,3]=np.fromstring(element.get('xyz','0 0 0'),sep=' ')
        t[:3,:3]=rpy(np.fromstring(element.get('rpy','0 0 0'),sep=' '))
    return t


def fk(tree,q):
    transforms={'Base_link':np.eye(4)}
    remaining=list(tree.findall('joint'))
    while remaining:
        updated=[]
        for j in remaining:
            parent=j.find('parent').get('link')
            if parent not in transforms: continue
            matrix=origin(j.find('origin'))
            if j.get('type')!='fixed':
                axis=np.fromstring(j.find('axis').get('xyz'),sep=' '); axis/=np.linalg.norm(axis)
                x,y,z=axis; cross=np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
                theta=q.get(j.get('name'),0.)
                rotation=np.eye(3)+math.sin(theta)*cross+(1.-math.cos(theta))*(cross@cross)
                matrix[:3,:3]=matrix[:3,:3]@rotation
            transforms[j.find('child').get('link')]=transforms[parent]@matrix
            updated.append(j)
        if not updated: raise ValueError('URDF is not a connected Base_link tree')
        remaining=[j for j in remaining if j not in updated]
    return transforms


def audit(urdf,spec):
    tree=ET.parse(urdf)
    joints={j.get('name'):j for j in tree.findall('joint') if j.get('type')!='fixed'}
    if set(joints)!=set(spec.JOINTS): raise ValueError('R5A joint names differ from contract')
    poses={}; total=None
    tuck=tuple(a+spec.REFERENCE_PROFILES[0]['tuck_fraction']*(b-a) for a,b in zip(spec.STAND,spec.CROUCH))
    for label,angles in (('stand',spec.STAND),('crouch',spec.CROUCH),('extend',spec.EXTEND),('tuck',tuck)):
        q=dict(zip(spec.LEG_JOINTS,angles))
        for name,value in q.items():
            limit=joints[name].find('limit')
            if not float(limit.get('lower'))<=value<=float(limit.get('upper')): raise ValueError(f'{label}: {name} exceeds URDF')
        tf=fk(tree,q); bodies=[]
        for link in tree.findall('link'):
            inertial=link.find('inertial')
            if inertial is None: continue
            m=float(inertial.find('mass').get('value'))
            local=origin(inertial.find('origin')); world=tf[link.get('name')]@local
            a=inertial.find('inertia').attrib
            inertia=np.array([[float(a['ixx']),float(a['ixy']),float(a['ixz'])],
                              [float(a['ixy']),float(a['iyy']),float(a['iyz'])],
                              [float(a['ixz']),float(a['iyz']),float(a['izz'])]])
            bodies.append((m,world[:3,3],world[:3,:3]@inertia@world[:3,:3].T))
        total=sum(b[0] for b in bodies); com=sum(m*c for m,c,_ in bodies)/total
        inertia=sum(i+m*((c-com)@(c-com)*np.eye(3)-np.outer(c-com,c-com)) for m,c,i in bodies)
        poses[label]={'com_in_base_m':com.tolist(),'centroidal_inertia_base_axes_kg_m2':inertia.tolist(),
                      'wheel_centres_in_base_m':{n:tf[n][:3,3].tolist() for n in spec.WHEEL_BODIES},
                      'reference_joint_angles_rad':q}
    ballistic=[]
    pitch_inertia=poses['stand']['centroidal_inertia_base_axes_kg_m2'][1][1]
    for height in (.05,.10,.20,.30,.40):
        flight=2*math.sqrt(2*height/9.81); rate=2*math.pi/flight
        ballistic.append(dict(com_rise_m=height,takeoff_vz_m_s=math.sqrt(2*9.81*height),
            flight_s=flight,vertical_energy_J=total*9.81*height,one_turn_mean_pitch_rate_rad_s=rate,
            rigid_stand_pose_rotation_energy_J=.5*pitch_inertia*rate*rate))
    return dict(status='static_nominal_model_checked',mass_kg=total,poses=poses,ballistic=ballistic,
        assumptions=['Equal takeoff and landing COM height; ideal ballistic flight',
                     'Rotation energy is a fixed standing-pose estimate; tuck/inertia changes not solved',
                     '48/24 Nm and 12.57/37.7 rad/s are rectangular simulation limits, not measured motor curves'],
        unresolved=['Full backflip centroidal angular momentum and collision sweep',
                    'Measured torque-speed envelope, electrical power, thermal and structural/landing limits'],
        full_backflip_feasibility='not_established',hardware_acceptance='not_evaluated')
