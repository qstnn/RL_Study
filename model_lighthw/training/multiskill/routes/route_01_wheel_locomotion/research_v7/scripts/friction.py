"""R1许可条件的库仑附着下限；平行坡面拖杆、四轮均接触，非实测材料参数。"""
import math
from model import Model
from settings import GROUND_FRICTION,ROLLING_COEFFICIENT,LOADED_ACCELERATION,TEACHER_ACCELERATION_LIMIT,FRICTION_MARGIN
from research_common import SPEC,launch


def requirement(slope,payload,tow,acceleration,*,moving=True):
    mass=sum(Model().mass.values())+payload
    angle=math.radians(slope);g=9.81
    normal=mass*g*math.cos(angle)
    # 被动车4实心圆柱轮：在40kg标定质量下sum(I/r²)=2kg，质量缩放时同比变化。
    tow_rotation=2.*tow/40.
    tension=tow*g*math.sin(angle)+(ROLLING_COEFFICIENT*tow*g*math.cos(angle) if moving else 0.)+(tow+tow_rotation)*acceleration
    ground_force=mass*g*math.sin(angle)+mass*acceleration+tension
    circular=abs(ground_force)/normal
    return dict(slope_deg=slope,payload_kg=payload,towed_kg=tow,acceleration_mps2=acceleration,moving=moving,
        robot_and_payload_mass_kg=mass,normal_N=normal,drawbar_signed_N=tension,ground_longitudinal_signed_N=ground_force,
        mu_circular_lower_bound=circular,mu_16sided_conservative_bound=circular/math.cos(math.pi/16))


def calculate(path):
    records=[]
    for slope in SPEC.ANGLES:
        for payload,tow in ((0.,0.),(5.,0.),(10.,0.),(15.,0.),(0.,20.),(0.,40.)):
            if slope==-25. and tow:continue
            limit=LOADED_ACCELERATION if payload or tow else 1.
            for a in (-limit,0.,limit):records.append(requirement(slope,payload,tow,a))
            records.append(requirement(slope,payload,tow,0.,moving=False))
    nominal=max(records,key=lambda r:r['mu_circular_lower_bound'])
    # 教师纠偏加速度的范围比指令斜率更大，余量计算也覆盖该显式控制范围。
    teacher=[requirement(s,0.,40.,a) for s in (25.,-20.) for a in (-TEACHER_ACCELERATION_LIMIT,TEACHER_ACCELERATION_LIMIT)]
    worst=max(records+teacher,key=lambda r:r['mu_16sided_conservative_bound'])
    raw=worst['mu_16sided_conservative_bound']*(1+FRICTION_MARGIN)
    selected=math.ceil(raw*10-1.e-10)/10
    if selected!=GROUND_FRICTION:raise ValueError(f'Friction setting {GROUND_FRICTION} differs from computed {selected}; review settings')
    result=dict(selected_static_friction=selected,selected_dynamic_friction=selected,combine_mode='min',rolling_coefficient=ROLLING_COEFFICIENT,
        nominal_condition_count=64,nominal_envelope=nominal,teacher_correction_envelope=worst,nominal_cases=records,teacher_correction_cases=teacher,
        requested_margin=FRICTION_MARGIN,actual_margin_over_16sided_bound=selected/worst['mu_16sided_conservative_bound']-1.,
        selected_mu_unrounded=raw,round_up_step=.1,
        steady_uphill=requirement(25.,0.,40.,0.),downhill_braking=requirement(-20.,0.,40.,-.3),
        assumptions='Known URDF masses; parallel drawbar; all four wheels supported; quasi-static normal load; no lateral demand. Per-wheel forces, torque and geometry checked separately in poses.json. Dynamic success unproved.')
    launch.write_json(path,result);return result

if __name__=='__main__':
    import argparse
    from pathlib import Path
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();calculate(a.output)
