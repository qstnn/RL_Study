"""只读URDF逆运动学：固定四轮接触位置，比较机身相对坡面低头0/5/10度的可达性。"""
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares


def solve(urdf,bias_deg):
    xml=ET.parse(urdf);joints=xml.findall('joint')
    order=[f'{leg}_{j}_joint' for leg in ('FR','FL','RR','RL') for j in ('hip','thigh','calf')]
    q0=np.array([0,.6,-1.5]*2+[0,-.6,1.5]*2,dtype=float)
    limits={j.get('name'):(float(j.find('limit').get('lower')),float(j.find('limit').get('upper')))
            for j in joints if j.find('limit') is not None and j.get('type')!='continuous'}
    def fk(q,bias):
        trans={'Base_link':np.eye(4)}
        trans['Base_link'][:3,:3]=Rotation.from_euler('y',bias,degrees=True).as_matrix();trans['Base_link'][2,3]=.416223
        remaining=list(joints);angles=dict(zip(order,q))
        while remaining:
            done=[]
            for j in remaining:
                parent,child=j.find('parent').get('link'),j.find('child').get('link')
                if parent not in trans:continue
                origin=j.find('origin');local=np.eye(4)
                local[:3,3]=np.fromstring(origin.get('xyz','0 0 0'),sep=' ')
                local[:3,:3]=Rotation.from_euler('xyz',np.fromstring(origin.get('rpy','0 0 0'),sep=' ')).as_matrix()
                axis=np.fromstring(j.find('axis').get('xyz','0 0 1'),sep=' ') if j.find('axis') is not None else np.array([0.,0.,1.])
                rot=np.eye(4);rot[:3,:3]=Rotation.from_rotvec(axis*angles.get(j.get('name'),0.)).as_matrix()
                trans[child]=trans[parent]@local@rot;done.append(j)
            if not done:raise ValueError('Unresolved joint tree')
            for j in done:remaining.remove(j)
        return trans
    original=fk(q0,0);feet=np.array([original[f'{l}_wheel_link'][:3,3] for l in ('FR','FL','RR','RL')])
    def residual(q):
        trans=fk(q,bias_deg)
        return (np.array([trans[f'{l}_wheel_link'][:3,3] for l in ('FR','FL','RR','RL')])-feet).ravel()
    lo,hi=np.array([limits[n] for n in order]).T
    solution=least_squares(residual,q0,bounds=(lo+1e-5,hi-1e-5),xtol=1e-12,ftol=1e-12,gtol=1e-12)
    trans=fk(solution.x,bias_deg)
    base=trans['Base_link'];hitch=(base@np.array([-.38,0.,-.33,1.]))[:3]
    return dict(bias_deg=bias_deg,joint_names=order,q=solution.x.tolist(),default=q0.tolist(),
        max_foot_error_m=float(abs(residual(solution.x)).max()),minimum_joint_limit_margin_rad=float(np.minimum(solution.x-lo,hi-solution.x).min()),
        robot_hitch_clearance_m=float(hitch[2]),
        front_wheel_to_body_z_m=float((np.linalg.inv(base)@np.r_[feet[0],1.])[2]),
        rear_wheel_to_body_z_m=float((np.linalg.inv(base)@np.r_[feet[2],1.])[2]))
