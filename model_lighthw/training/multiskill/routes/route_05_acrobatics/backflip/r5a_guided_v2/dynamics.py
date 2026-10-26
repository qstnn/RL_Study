"""CPU 整机角动量复算；包含各连杆自转和绕整机 COM 的轨道项。"""
import numpy as np


def rotation(q):
    """wxyz 单位四元数 -> 局部到世界旋转矩阵，支持批量。"""
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                     2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                     2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), -1).reshape(q.shape[:-1]+(3,3))


def centroidal(position, velocity, quaternion, omega, masses, inertia):
    """返回世界系 COM、速度、角动量及锁定关节整机惯量。

    激活：每个已完成的物理步。inertia 是 PhysX get_inertias 的
    连杆 COM 处、以 rigid-body-prim 轴表示的完整张量；不是主轴对角张量。
    H=sum(R I R^T omega + (r-COM) cross m(v-VCOM))。
    锁定惯量不能直接把运动中的多体系统当作刚体预测机身转速。
    """
    p, v, w = [np.asarray(x, dtype=np.float64) for x in (position, velocity, omega)]
    m = np.asarray(masses, dtype=np.float64)
    i = np.asarray(inertia, dtype=np.float64).reshape(-1,3,3)
    c = (p*m[:,None]).sum(-2)/m.sum()
    cv = (v*m[:,None]).sum(-2)/m.sum()
    r = p-c[...,None,:]
    rot = rotation(quaternion)
    iw = rot@i@np.swapaxes(rot,-1,-2)
    spin = np.einsum('...bij,...bj->...bi',iw,w)
    orbital = np.cross(r,(v-cv[...,None,:])*m[:,None])
    eye = np.eye(3)
    locked = (iw+m[:,None,None]*((r*r).sum(-1)[...,None,None]*eye-r[..., :,None]*r[...,None,:])).sum(-3)
    return c, cv, (spin+orbital).sum(-2), locked, spin, orbital


def self_test():
    """验证平移不变、刚体 H=Iω、轨道项与坐标变换，非实现镜像断言。"""
    p=np.array([[-1.,0.,0.],[1.,0.,0.]])
    w=np.tile([0.,2.,0.],(2,1));v=np.cross(w,p)
    q=np.tile([1.,0.,0.,0.],(2,1));m=np.ones(2);i=np.tile(np.diag([.1,.2,.3]),(2,1,1))
    c,cv,h,locked,_,_=centroidal(p,v,q,w,m,i)
    np.testing.assert_allclose(h,[0.,4.8,0.],atol=1e-12)
    np.testing.assert_allclose(h,locked@w[0],atol=1e-12)
    shifted=centroidal(p+[5.,-7.,2.],v+[3.,4.,5.],q,w,m,i)
    np.testing.assert_allclose(shifted[2],h,atol=1e-12)
    angle=.7;rz=rotation([np.cos(angle/2),0,0,np.sin(angle/2)])
    qq=np.tile([np.cos(angle/2),0,0,np.sin(angle/2)],(2,1))
    turned=centroidal(p@rz.T,v@rz.T,qq,w@rz.T,m,i)
    np.testing.assert_allclose(turned[2],rz@h,atol=1e-12)
    print('centroidal analytical checks passed')


if __name__=='__main__':self_test()
