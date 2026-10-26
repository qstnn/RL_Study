"""冻结质量门：相同初态逐例比较，同时保留真实跳跃、净空、落地和末段稳定。"""
import math
import statistics
from launch import spec


def summarize(result):
    cases=result.get('cases',[])
    if not cases:raise ValueError('No frozen cases')
    out=dict(cases=len(cases),successes=sum(bool(c['success']) for c in cases),
        valid_jumps=sum(bool(c['launched'] and c['landed'] and not c['failed'] and c['duration_s']>=3.999) for c in cases),
        high_jumps=sum(bool(c['height_qualified']) for c in cases))
    for field,label in [('flight_rise','flight_rise_m'),('free_flight_time','free_flight_s'),
        ('wheel_clearance_peak','wheel_clearance_m'),('first_flight_peak_rise','standing_rise_m'),
        ('flight_time','airtime_s'),('stable_time','terminal_stable_s'),('impact_peak','impact_N')]:
        values=[c[field] for c in cases]
        if not all(math.isfinite(v) for v in values):raise ValueError('Nonfinite quality metrics: '+field)
        out['min_'+label]=min(values);out['median_'+label]=statistics.median(values);out['max_'+label]=max(values)
    return out


def initialization_ok(result):
    """只要求起点具备可训练的跳跃与停稳，不要求已经达到最终优化目标。"""
    return compare(result,result)['eligible'] and all(c['flight_rise']>=spec.CONTRACT['initialization_min_flight_rise_m']
        and c['free_flight_time']>=.25 and c['wheel_clearance_peak']>=.15
        and c['first_flight_peak_rise']>=.12 and c['stable_time']>=1.-1e-6 for c in result['cases'])


def compare(result,baseline):
    """质量保护阈值用于仿真训练筛选，不是机械结构安全限。"""
    cases=result['cases'];old=baseline['cases'];goal=spec.CONTRACT
    if [(c['seed'],c['episode']) for c in cases]!=[(c['seed'],c['episode']) for c in old]:
        raise ValueError('Quality evaluations must have identical seed/episode keys')
    failures=[]
    for c,b in zip(cases,old):
        key=f"seed{c['seed']}_episode{c['episode']}"
        if not (c['uniform_physics_trace'] and c['reason']=='timeout' and c['duration_s']>=3.999 and c['launched'] and c['landed'] and not c['failed']):failures.append(key+':physical')
        if c['flight_time']<.10-1e-6:failures.append(key+':airtime')
        for field,target,floor in [('flight_rise',goal['target_flight_rise_m'],.08),
            ('free_flight_time',goal['target_free_flight_s'],.25),
            ('wheel_clearance_peak',goal['target_wheel_clearance_m'],.15),
            ('first_flight_peak_rise',goal['target_standing_rise_m'],.12)]:
            if c[field]+1e-6<max(floor,.95*min(b[field],target)):failures.append(key+':'+field)
        if b['height_qualified'] and not c['height_qualified']:failures.append(key+':height')
        if b['success'] and not c['success']:failures.append(key+':success')
        if b['stable_time']>=1.-1e-6 and c['stable_time']<1.-1e-6:failures.append(key+':hold')
        if c['impact_peak']>min(spec.SCREEN['max_landing_impact_N'],max(b['impact_peak']*1.25,b['impact_peak']+200.)):failures.append(key+':impact')
    return dict(eligible=not failures,failures=failures,summary=summarize(result))


def rank(result):
    s=summarize(result);g=spec.CONTRACT
    distance=sum(abs(s['median_'+metric]-g[target])/g[target] for metric,target in
        [('wheel_clearance_m','target_wheel_clearance_m'),('standing_rise_m','target_standing_rise_m'),
         ('flight_rise_m','target_flight_rise_m'),('free_flight_s','target_free_flight_s')])
    return (s['successes'],s['high_jumps'],-distance,-s['median_impact_N'])
