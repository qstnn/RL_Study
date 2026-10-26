"""Static locked-inertia candidate generator; no claim of dynamic/contact feasibility."""
from pathlib import Path
import sys,xml.etree.ElementTree as ET,json
import numpy as np
from scipy.optimize import differential_evolution
ROOT=Path(__file__).resolve().parents[7]
sys.path.insert(0,str(ROOT/'model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/highjump/scripts'))
from static_audit import fk,origin
T=ET.parse(ROOT/'model_lighthw/lighthw_urdf/urdf/lighthw.urdf')
links=[]
for link in T.findall('link'):
 inertial=link.find('inertial');m=float(inertial.find('mass').get('value'));a=inertial.find('inertia').attrib
 I=np.array([[float(a['ixx']),float(a['ixy']),float(a['ixz'])],[float(a['ixy']),float(a['iyy']),float(a['iyz'])],[float(a['ixz']),float(a['iyz']),float(a['izz'])]])
 links.append((link.get('name'),m,origin(inertial.find('origin')),I))
def inertia(x):
 q={}
 for leg,sign,offset in [('FR',1,0),('FL',1,0),('RR',-1,2),('RL',-1,2)]:
  q[leg+'_thigh_joint']=sign*x[offset];q[leg+'_calf_joint']=sign*x[offset+1]
 tf=fk(T,q);positions=[];moments=[];masses=[]
 for name,m,O,I in links:
  w=tf[name]@O;positions.append(w[:3,3]);moments.append((w[:3,:3]@I@w[:3,:3].T)[1,1]);masses.append(m)
 positions=np.array(positions);masses=np.array(masses);com=np.sum(positions*masses[:,None],axis=0)/sum(masses)
 return float(sum(moments)+np.sum(masses*((positions-com)[:,[0,2]]**2).sum(1)))
if __name__=='__main__':
 r=differential_evolution(inertia,[(-1.45,1.45),(-2.4,-.12)]*2,seed=42,maxiter=35,popsize=8,tol=1e-5,polish=True)
 out=dict(parameters=r.x.tolist(),minimum_locked_Iyy=float(r.fun),standing_Iyy=inertia([.6,-1.5,.6,-1.5]),legacy_tuck_Iyy=inertia([1.23,-2.31,1.23,-2.31]),note='Geometric lower candidate only; collisions and 12.57rad/s reachability are NOT validated')
 (Path(__file__).parent/'results/tuck_geometry.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
