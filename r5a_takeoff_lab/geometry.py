"""固定收腿姿态的全角度碰撞包络，仅作离地空间筛查。"""
from pathlib import Path
import json
import sys
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial import ConvexHull

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/highjump/scripts'))
from static_audit import fk,origin


def mesh_vertices(path):
    """读取项目二进制 STL；每个网格先取凸包顶点，保留任意方向支撑函数。"""
    raw=Path(path).read_bytes();n=int.from_bytes(raw[80:84],'little')
    if len(raw)!=84+50*n:raise ValueError('Expected binary STL: '+str(path))
    dtype=np.dtype([('normal','<f4',(3,)),('vertices','<f4',(3,3)),('attribute','<u2')])
    vertices=np.frombuffer(raw,offset=84,count=n,dtype=dtype)['vertices'].reshape(-1,3).astype(float)
    vertices=np.unique(vertices,axis=0)
    return vertices[ConvexHull(vertices).vertices]


def envelope(urdf,thigh=1.23,calf=-2.31):
    urdf=Path(urdf);tree=ET.parse(urdf);q={}
    for leg,sign in [('FR',1),('FL',1),('RR',-1),('RL',-1)]:
        q.update({leg+'_thigh_joint':sign*thigh,leg+'_calf_joint':sign*calf})
    transforms=fk(tree,q);points=[];weighted=np.zeros(3);mass=0.
    for link in tree.findall('link'):
        tf=transforms[link.get('name')];inertial=link.find('inertial')
        if inertial is not None:
            m=float(inertial.find('mass').get('value'));mass+=m
            weighted+=m*(tf@origin(inertial.find('origin')))[:3,3]
        for collision in link.findall('collision'):
            mesh=collision.find('geometry/mesh')
            if mesh is None:raise ValueError('Expected mesh collision for fixed-pose envelope')
            vertices=mesh_vertices((urdf.parent/mesh.get('filename')).resolve())
            vertices*=np.fromstring(mesh.get('scale','1 1 1'),sep=' ')
            matrix=tf@origin(collision.find('origin'))
            points.append(vertices@matrix[:3,:3].T+matrix[:3,3])
    com=weighted/mass;points=np.concatenate(points)-com
    points=points[ConvexHull(points).vertices]
    theta=np.linspace(0,2*np.pi,721)
    # R_y(-theta) 的 z 行：[sin(theta),0,cos(theta)]，正 theta 为后翻。
    z=np.sin(theta)[:,None]*points[None,:,0]+np.cos(theta)[:,None]*points[None,:,2]
    required=-z.min(-1)
    return theta,required,dict(mass_kg=mass,tuck_thigh=thigh,tuck_calf=calf,com_base=com.tolist(),
        max_required_com_z_m=float(required.max()),max_required_angle_deg=float(theta[required.argmax()]*180/np.pi),
        at_quarter_turn_m=float(required[180]),at_inversion_m=float(required[360]),
        assumption='Fixed tuck pose with original collision mesh convex support; excludes moving joints and landing geometry.')


if __name__=='__main__':
    theta,required,report=envelope(ROOT/'model_lighthw/lighthw_urdf/urdf/lighthw.urdf')
    print(json.dumps(report,indent=2))
