"""URDF 几何/质量交叉校验及几何接触滚动矩阵，全部采用世界坐标。

广义速度=[根刚体COM线速度(3), 世界角速度(3), action顺序关节速度(16)]。
接触点随构型迁移，A(q)v=0 的方向导数包含此迁移；不是固定材料点零加速度。
"""
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from research_common import URDF, NAMES, LEGS
RADIUS = .0875

def skew(x):
    x,y,z=x
    return np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])

def rotation(q):
    return Rotation.from_quat(np.roll(q,-1)).as_matrix()

def pose_matrix(node):
    T=np.eye(4)
    if node is not None:
        T[:3,3]=np.fromstring(node.get('xyz','0 0 0'),sep=' ')
        T[:3,:3]=Rotation.from_euler('xyz',np.fromstring(node.get('rpy','0 0 0'),sep=' ')).as_matrix()
    return T

class Model:
    def __init__(self,urdf=URDF):
        root=ET.parse(urdf).getroot()
        self.links={l.get('name'):l for l in root.findall('link')}
        self.joints={j.get('name'):j for j in root.findall('joint')}
        self.names=list(NAMES);self.index={n:i for i,n in enumerate(self.names)}
        self.mass={n:float(l.find('inertial/mass').get('value')) for n,l in self.links.items()}
        self.inertial={n:pose_matrix(l.find('inertial/origin')) for n,l in self.links.items()}
        self.inertia={}
        for n,l in self.links.items():
            a=l.find('inertial/inertia').attrib
            self.inertia[n]=np.array([[float(a['ixx']),float(a['ixy']),float(a['ixz'])],
                [float(a['ixy']),float(a['iyy']),float(a['iyz'])],[float(a['ixz']),float(a['iyz']),float(a['izz'])]])
        self.base_offset=self.inertial['Base_link'][:3,3]
        self.limits=np.array([[float(self.joints[n].find('limit').get(k)) if self.joints[n].find('limit') is not None and self.joints[n].find('limit').get(k) is not None else (-np.inf if k=='lower' else np.inf) for k in ('lower','upper')] for n in self.names])

    def state(self,pos,quat,q):
        T=np.eye(4);T[:3,:3]=rotation(quat);T[:3,3]=pos
        trans={'Base_link':T};axes={};origins={};ancestry={};pending=list(self.joints.values())
        while pending:
            progressed=False
            for j in list(pending):
                parent=j.find('parent').get('link');child=j.find('child').get('link')
                if parent not in trans:continue
                name=j.get('name');pre=trans[parent]@pose_matrix(j.find('origin'))
                a=np.fromstring(j.find('axis').get('xyz'),sep=' ')
                axes[name]=pre[:3,:3]@a;origins[name]=pre[:3,3]
                R=np.eye(4);R[:3,:3]=Rotation.from_rotvec(a*q[self.index[name]]).as_matrix()
                trans[child]=pre@R;ancestry[child]=ancestry.get(parent,[])+[name]
                pending.remove(j);progressed=True
            if not progressed:raise ValueError('Unsupported URDF topology')
        com={n:(trans[n]@self.inertial[n])[:3,3] for n in self.links}
        ref=com['Base_link']
        def jac(name,point):
            J=np.zeros((6,22));J[:3,:3]=np.eye(3);J[:3,3:6]=-skew(point-ref);J[3:,3:6]=np.eye(3)
            for n in ancestry.get(name,[]):
                k=6+self.index[n];J[:3,k]=np.cross(axes[n],point-origins[n]);J[3:,k]=axes[n]
            return J
        return dict(trans=trans,axes=axes,origins=origins,com=com,jac=jac,ref=ref)

    def dynamics(self,state):
        """独立 URDF M/g，仅用于静力筛选和 PhysX 坐标/质量交叉检查。"""
        M=np.zeros((22,22));g=np.zeros(22)
        for n in self.links:
            J=state['jac'](n,state['com'][n]);R=(state['trans'][n]@self.inertial[n])[:3,:3]
            I=R@self.inertia[n]@R.T
            M+=self.mass[n]*J[:3].T@J[:3]+J[3:].T@I@J[3:]
            g+=J[:3].T@np.array([0.,0.,self.mass[n]*9.81])
        return M,g

    def contacts(self,state,normal):
        A=[];points=[];centers=[]
        for leg in LEGS:
            name=leg+'_wheel_link';center=state['trans'][name][:3,3]
            axis=state['axes'][leg+'_wheel_joint']
            radial=normal-axis*np.dot(normal,axis)
            if np.linalg.norm(radial)<.2:raise ValueError('Wheel axis nearly normal to ground')
            lever=-RADIUS*radial/np.linalg.norm(radial)
            point=center+lever;A.append(state['jac'](name,point)[:3]);points.append(point);centers.append(center)
        return np.vstack(A),np.array(points),np.array(centers)

    def contact_derivative(self,pos,quat,q,v,normal,epsilon=1.e-5):
        """沿当前广义速度的中心差分；同时更新轮架、腿和迁移接触点。"""
        R=rotation(quat);root_com=pos+R@self.base_offset;As=[]
        for sign in (-1.,1.):
            dt=sign*epsilon;Rn=Rotation.from_rotvec(v[3:6]*dt).as_matrix()@R
            pn=root_com+dt*v[:3]-Rn@self.base_offset
            qn=np.roll(Rotation.from_matrix(Rn).as_quat(),1)
            A,_,_=self.contacts(self.state(pn,qn,q+v[6:]*dt),normal);As.append(A)
        return (As[1]-As[0])/(2*epsilon)

    def whole_com(self,state):
        return sum(self.mass[n]*state['com'][n] for n in self.links)/sum(self.mass.values())
