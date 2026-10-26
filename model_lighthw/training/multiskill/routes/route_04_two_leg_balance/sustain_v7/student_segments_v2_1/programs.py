"""有限、明确的接管/释放时序；仅采集用，不进入Actor输入或自主评估。"""
import torch

PROGRAMS={
    'early':[[2.5,2.9],[3.2,3.6],[3.8,None]],
    'late':[[2.8,3.2],[3.4,3.8],[4.0,None]],
    'guarded':[[2.5,3.2],[3.3,3.8],[3.9,None]],
}


def validate(windows):
    assert len(windows)>=3
    last=0.
    for i,(start,end) in enumerate(windows):
        assert 1.<=start<=6. and start>last
        assert abs(start/.02-round(start/.02))<1e-6
        if end is None:
            assert i==len(windows)-1
        else:
            assert start<end<=6. and abs(end/.02-round(end/.02))<1e-6
            last=end
    assert windows[-1][1] is None


def takeover(task_time,windows):
    """全阶段返回教师执行掩码；窗口端点按50Hz控制边界，释放时不重置任何状态。"""
    mask=torch.zeros_like(task_time,dtype=torch.bool)
    if windows is not None:
        for start,end in windows:
            row=task_time>=start-1e-6
            if end is not None:row &= task_time<end-1e-6
            mask |= row
    return mask
