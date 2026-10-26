import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from review_entry_gate import readiness


def row(slope=25.,device='cuda:0',initial='ordinary',passed=True,valid=True):
    return dict(run='example',slope_deg=slope,device=device,initial_state=initial,complete_teacher_pass=passed,source_valid=valid)


def test_preloaded_success_is_not_cold_start_evidence():
    result=readiness([row(initial='loaded')])
    assert result['gates']['uphill_cuda_loaded']
    assert not result['ready_for_teacher_collection']
    assert 'uphill_cuda_ordinary' in result['missing']


def test_cpu_success_and_failed_long_distance_do_not_certify_gpu():
    result=readiness([row(device='cpu'),row(passed=False),row(valid=False)])
    assert result['gates']['uphill_cpu_ordinary'] and not result['gates']['uphill_cuda_ordinary']


def test_full_physical_matrix_only_permits_collection_not_actor_or_ppo_acceptance():
    result=readiness([row(slope=s,device=d) for s in [25.,-20.] for d in ['cpu','cuda:0']])
    assert result['ready_for_teacher_collection'] and not result['missing']
    assert not result['unified_actor_accepted'] and not result['formal_ppo_authorized']


def test_different_solver_profiles_cannot_be_combined_into_one_gate():
    rows=[dict(row(slope=s,device=d),physics_profile=('4/1' if s==25. else '16/4'))
        for s in [25.,-20.] for d in ['cpu','cuda:0']]
    result=readiness(rows)
    assert all(result['gates'][name] for name in result['required_gates'])
    assert not result['ready_for_teacher_collection'] and result['missing']
