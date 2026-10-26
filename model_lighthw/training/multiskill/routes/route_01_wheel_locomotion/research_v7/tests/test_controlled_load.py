import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from controlled_load import LoadSchedule, analyze_run


def test_staircase_has_complete_constant_plateaus_and_no_overshoot():
    schedule=LoadSchedule((0.,50.,173.))
    time=np.arange(0,schedule.duration_s,.001)
    load=schedule.value(time)
    assert np.all(np.diff(load)>=-1.e-9)
    assert load.min()==0 and load.max()==173
    for stage in schedule.stages():
        window=(time>=stage['plateau_start_s'])&(time<stage['plateau_end_s'])
        assert window.sum()==5000
        np.testing.assert_allclose(load[window],stage['level_N'],atol=1.e-10)


def test_constant_target_is_ramped_from_zero_after_preparation():
    schedule=LoadSchedule((173.,))
    np.testing.assert_allclose(schedule.value([0,5,6,7,12]),[0,0,86.5,173,173])


def test_long_staircase_requires_explicit_budget_and_preserves_prefix():
    levels=(0,60,80,90,100,110,120,130,140,150,160,173,173)
    with pytest.raises(ValueError):LoadSchedule(levels,hold_s=10)
    long=LoadSchedule(levels,hold_s=10,maximum_duration_s=180)
    assert long.duration_s==161
    short=LoadSchedule(levels[:4],hold_s=10)
    time=np.arange(0,short.duration_s,.001)
    np.testing.assert_array_equal(long.value(time),short.value(time))
    assert long.value(160.999)==173
    with pytest.raises(ValueError):LoadSchedule(levels*3,maximum_duration_s=180)


@pytest.mark.parametrize('levels',[(50.,0.),(-1.,),(float('nan'),),(251.,),()])
def test_rejects_invalid_load_direction_or_magnitude(levels):
    with pytest.raises(ValueError):LoadSchedule(levels)


def test_native_audit_rejects_truncated_plateau_and_propulsive_force(tmp_path):
    import json
    from scipy.spatial.transform import Rotation
    schedule=LoadSchedule((50.,))
    count=12000;t=np.arange(count)*.001
    a=np.deg2rad(25.);tangent=np.array([np.cos(a),0.,np.sin(a)])
    normal=np.array([-np.sin(a),0.,np.cos(a)]);side=np.array([0.,1.,0.])
    R=np.column_stack((tangent,side,normal));q=np.roll(Rotation.from_matrix(R).as_quat(),1)
    velocity=np.broadcast_to(.2*tangent,(count,3)).copy()
    centers=np.broadcast_to(.0875*normal,(count,4,3)).copy()
    force=-schedule.value(t)[:,None]*tangent
    z=dict(controlled_load_time_s=t,controlled_load_force_w_N=force,
        controlled_load_torque_w_Nm=np.zeros((count,3)),controlled_load_arm_w_m=np.zeros((count,3)),
        controlled_load_buffer_force_w_N=force.copy(),controlled_load_buffer_torque_w_Nm=np.zeros((count,3)),
        controlled_load_cart_force_norm_N=np.zeros(count),controlled_load_cart_torque_norm_Nm=np.zeros(count),
        controlled_load_cart_distance_m=np.full(count,100.),whole_com_velocity_w=velocity,
        whole_com_w=t[:,None]*velocity,body_pos_w=t[:,None]*velocity+.4*normal,
        hitch_robot_w=t[:,None]*velocity+.07*normal,
        native_ground_normal_w=np.broadcast_to(26.*normal,(count,4,3)),
        native_contact_count=np.ones((count,4)),body_quat_w=np.broadcast_to(q,(count,4)),
        wheel_pos_w=centers,native_contact_cop_w=np.zeros((count,4,3)),
        wheel_center_velocity_w=np.broadcast_to(.2*tangent,(count,4,3)),
        wheel_angular_velocity_w=np.broadcast_to(.2/.0875*side,(count,4,3)),
        friction_force_w=np.broadcast_to(25.*tangent,(count,4,3)))
    def save(length):
        np.savez(tmp_path/'seed43_case00_physics_trace.npz',**{k:v[:length] for k,v in z.items()})
        np.savez(tmp_path/'seed43_case00_motor_trace.npz',applied=np.zeros((length,16)),limits=np.ones(16))
        (tmp_path/'seed43_case00.json').write_text(json.dumps(dict(samples=length//20)))
        (tmp_path/'seed43_case00.csv').write_text('t,body_contact,terminated,height\n'+''.join(f'{i*.02},0,0,0.4\n' for i in range(1,length//20+1)))
    save(count)
    assert analyze_run(tmp_path,schedule)['highest_passed_plateau_N']==50.
    # Raising the command must not turn a slower trajectory into a tracking success.
    (tmp_path/'seed43_case00.json').write_text(json.dumps(dict(samples=count//20,case=dict(speed=.4))))
    report=analyze_run(tmp_path,schedule)
    assert report['highest_passed_plateau_N'] is None
    assert report['stages'][0]['speed_mae_mps']==pytest.approx(.2)
    save(count)
    path=tmp_path/'seed43_case00.csv'
    path.write_text(path.read_text().replace('8.0,0,0,0.4','8.0,1,0,0.4'))
    assert analyze_run(tmp_path,schedule)['highest_passed_plateau_N'] is None
    save(10000)
    assert analyze_run(tmp_path,schedule)['highest_passed_plateau_N'] is None
    z['controlled_load_force_w_N']=-force
    z['controlled_load_buffer_force_w_N']=-force
    save(count)
    # The revised audit must recompute actual force direction without rewriting old evidence.
    from rolling_load_audit import analyze_rolling_run
    historical=(tmp_path/'controlled_load_analysis.json').read_bytes()
    rolling=analyze_rolling_run(tmp_path,schedule)
    assert not rolling['stages'][0]['checks']['force_audit']
    assert (tmp_path/'controlled_load_analysis.json').read_bytes()==historical
    report=analyze_run(tmp_path,schedule)
    assert not report['load_audit_passed'] and report['highest_passed_plateau_N'] is None
