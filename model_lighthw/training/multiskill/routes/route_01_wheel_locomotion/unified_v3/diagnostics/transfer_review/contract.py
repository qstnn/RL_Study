"""迁移起点准入，阈值在新GPU对照前固定，区别于最终任务验收。"""
from pathlib import Path
REPO=Path(__file__).resolve().parents[8]
V3=Path(__file__).resolve().parents[2]
ROUTES=REPO/'model_lighthw/training/routes'
SOURCES={
 'route53':dict(checkpoint=ROUTES/'route_53_ppo_mixed_recovery/runs/2026-10-15_12-26-34_975760_lighthw_mixed_recovery_v1_mixed/training/model_999.pt',sha256='0e8ccc8d2c832cdd101cb68968c7aca0a658d6eead2b947c6cc397a3639af411',input_dim=60),
 'route38':dict(checkpoint=ROUTES/'route_38_ppo_stair_flights/runs/2026-10-10_02-16-06_263239_lighthw_stair_flights_from_selected_v1/training/model_9999.pt',sha256='dc570667b6c03f2d875eb8eae340a0feb8d431e0324ddfeab49bd7d7322c890d',input_dim=60),
 'control_v3':dict(checkpoint=V3/'runs/2026-10-16_16-29-49.804831/training/model_499.pt',sha256='8f23efdbeeafb77542c7b8c7ed109a5a381c28136a763cfe86488ee04afa377b',input_dim=290)}
GATE=dict(version='r1_transfer_screen_v1',flat_zero='existing20s protocol all checks',
 flat_moving='complete3.1command protocol; actual>=2.5; all checks except final speed/MAE',
 payload15='all six protocols complete; three moving cases >=1m and posture/heading/height/contact checks',
 tow40_up25='survive initial3s zero command without termination',final_unloaded_speed_mps=3.,
 teacher_diagnostic_is_deployable=False,training_requires_deployable_adapter_verification=True)

def gate(evaluation):
    """返回教师准入检查；教师通过不能直接放行部署Actor或PPO。"""
    cases=evaluation['cases']
    def pick(s,p,t,v):
        found=[c for c in cases if all(c['case'][k]==a for k,a in zip(('slope_deg','payload_kg','towed_kg','speed'),(s,p,t,v)))]
        if len(found)!=1:raise ValueError('Expected exactly one fixed case')
        return found[0]
    zero=pick(0.,0.,0.,0.);fast=pick(0.,0.,0.,3.1);tow=pick(25.,0.,40.,0.)
    payload=[c for c in cases if c['case']['payload_kg']==15.];moving=[c for c in payload if c['case']['speed']>0]
    safety=('loaded_distance','surface_posture','heading','height','no_body_contact')
    checks=dict(flat_zero_full_pass=zero['success'],
      flat_moving_complete_safe_hold=fast['complete'] and bool(fast.get('metrics')) and all(v for k,v in fast['checks'].items() if k not in ('unloaded_tracking','unloaded_actual_3mps')),
      flat_moving_actual_at_least_2p5=fast.get('metrics',{}).get('cruise_vx_mps',-1)>=2.5,
      payload15_all_six_complete=len(payload)==6 and all(c['complete'] for c in payload),
      payload15_three_moving_safe=len(moving)==3 and all(all(c.get('checks',{}).get(k,False) for k in safety) for c in moving),
      tow40_up25_survives_startup=tow['observed']['duration_s']>=3. and not(tow['failure_reason'] and tow['observed']['duration_s']<=3.))
    return dict(teacher_pre_screen_passed=all(checks.values()),checks=checks,failed_checks=[k for k,v in checks.items() if not v],training_ready=False,
      note='Teacher diagnostic only; a deployable adapter must pass the same physical screen before training.')
