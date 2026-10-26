"""V7验收扩为18边界/128全条件；旧16项阈值不降低。"""
from research_common import launch
legacy=launch.PROTOCOL
DT=legacy.DT

def cases(distance_m=1.,suite='boundary'):
    if suite=='teacher':
        return [dict(slope_deg=s,payload_kg=0.,towed_kg=40.,speed=v,goal_distance_m=distance_m)
                for s in (25.,-20.) for v in (0.,.3)]
    if suite=='boundary':
        return legacy.cases(distance_m)+[dict(slope_deg=-20.,payload_kg=0.,towed_kg=40.,speed=v,goal_distance_m=distance_m) for v in (0.,.3)]
    if suite!='full':raise ValueError(suite)
    return [dict(slope_deg=s,payload_kg=p,towed_kg=t,speed=v,goal_distance_m=distance_m)
            for s in launch.SPEC.ANGLES for p,t in ((0.,0.),(5.,0.),(10.,0.),(15.,0.),(0.,20.),(0.,40.))
            if not (s==-25. and t>0) for v in (0.,3.1 if p==0 and t==0 else .3)]

max_steps=legacy.max_steps
finished=legacy.finished

def summarize(rows,case,reason):
    result=legacy.summarize(rows,case,reason)
    if case['towed_kg'] and case['speed'] and rows:
        # 完成1m机器人运动不应只靠拉长弹簧；拖车真实净进度单列验收。
        available=all('cart_progress_m' in r for r in rows)
        progress=max([0.]+[r.get('cart_progress_m',0.) for r in rows])
        result.setdefault('observed',{})['cart_progress_m']=progress
        result.setdefault('checks',{})['cart_distance']=available and progress>=case.get('goal_distance_m',1.)
        result['failed_checks']=[k for k,v in result['checks'].items() if not v]
        result['success']=bool(result.get('success') and result['checks']['cart_distance'])
    return result


def track_completion(callback, case_list, closed):
    """只在原协议完成后关闭该槽控制，避免其他槽继续时污染已完成回合。"""
    ids={id(case):i for i,case in enumerate(case_list)}
    def tracked(rows,case):
        complete=callback(rows,case)
        if complete:closed.add(ids[id(case)])
        return complete
    return tracked
