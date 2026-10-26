"""CPU 检查：分腿参考连续性和角度范围。"""
from types import SimpleNamespace
import torch
from control import Reference,baseline
from dynamics import self_test

def test_reference():
    p=baseline();p.update(unload_rear=1,unload_time=-.03)
    ref=Reference(SimpleNamespace(device='cpu',num_envs=1),[p])
    for edge in [.5,.775,.82,.85,.90,1.01]:
        left=ref.pose(torch.tensor([edge-1e-6]));right=ref.pose(torch.tensor([edge+1e-6]))
        assert torch.max(torch.abs(left-right))<.0002,(edge,left,right)
    for time in torch.linspace(0,1.5,100):
        pose=ref.pose(time.reshape(1))
        for j,sign in ((0,1),(3,1),(6,-1),(9,-1)):
            assert -1.57<=float(pose[0,j+1])*sign<=1.57
            assert -2.5<=float(pose[0,j+2])*sign<=-.087
    print('reference continuity/envelope passed')

def test_air_segments():
    import numpy as np
    from analyze import air_segments
    mask=np.zeros(50,dtype=bool)
    mask[2]=True;mask[10:13]=True;mask[20:35]=True
    assert air_segments(mask)==[(2,3),(10,13),(20,35)]
    valid=[(a,b) for a,b in air_segments(mask) if b-a>=7]
    assert valid==[(20,35)]
    # 力先归零、几何净空下一帧才转正：不能整段拒绝。
    force_clear=np.zeros(50,dtype=bool);force_clear[19:35]=True
    geometry=np.ones(50,dtype=bool);geometry[19]=False
    assert [(a,b) for a,b in air_segments(force_clear&geometry) if b-a>=7]==[(20,35)]
    print('short-gap and delayed-clearance checks passed')

if __name__=='__main__':self_test();test_reference();test_air_segments()
