"""Independent k/c variant audit retaining every original behavior threshold."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

from compliant_towing import DrawbarParameters, VARIANT, source_hashes
from real_cart_audit import analyze as analyze_original


def declared_parameters(request):
    declaration = request.get('diagnostic_drawbar')
    if not isinstance(declaration, dict) or declaration.get('variant') != VARIANT:
        raise ValueError('Expected an explicit isolated compliant drawbar diagnostic')
    parameters = DrawbarParameters(declaration['stiffness_Npm'], declaration['damping_Nspm'])
    if declaration != parameters.metadata() or parameters == DrawbarParameters():
        raise ValueError('Compliant declaration or preserved physical guards differ')
    if (request.get('initial_state_mode') != 'ordinary' or not request.get('ordinary_live_entry')
            or request.get('execution_physics', {}).get('drawbar') != declaration
            or request.get('physics_variant') != VARIANT):
        raise ValueError('Physical and controller compliant diagnostic profiles disagree')
    return parameters


def drawbar_errors(z, parameters):
    def a(key): return np.asarray(z['real_cart_'+key], dtype=np.float64)
    delta = a('robot_hitch_pre_w')-a('cart_hitch_pre_w')
    length = np.maximum(np.linalg.norm(delta, axis=1), 1.e-6)
    direction = delta/length[:, None]
    rate = np.sum((a('robot_hitch_velocity_pre_w')-a('cart_hitch_velocity_pre_w'))*direction, axis=1)
    tension = parameters.stiffness_Npm*(length-1.)+parameters.damping_Nspm*rate
    force = np.clip(tension, -1500., 1500.)[:, None]*direction
    errors = dict(spring_force_max_N=float(abs(force+a('robot_force_w_N')).max()),
        robot_arm_torque_max_Nm=float(abs(np.cross(a('robot_arm_pre_w'), -force)-a('robot_torque_w_Nm')).max()),
        cart_arm_torque_max_Nm=float(abs(np.cross(a('cart_arm_pre_w'), force)-a('cart_torques_w_Nm')[:, 0]).max()))
    return errors, dict(initial_length_m=float(length[0]), initial_extension_rate_mps=float(rate[0]),
        initial_spring_energy_J=float(.5*parameters.stiffness_Npm*(length[0]-1.)**2),
        length_range_m=[float(length.min()), float(length.max())], raw_tension_range_N=[float(tension.min()), float(tension.max())])


def apply_variant_audit(original, z, parameters):
    """Retain original measurements/checks; replace only the declared force law."""
    report = deepcopy(original)
    errors, mechanics = drawbar_errors(z, parameters)
    report['legacy_independent_drawbar_errors'] = report['independent_drawbar_errors']
    report['independent_drawbar_errors'] = errors
    report['checks']['independent_spring_and_moment'] = max(errors.values()) < .001
    report['real_cart_1m_pass'] = bool(all(report['checks'].values()))
    hold = report['hold_2s_transition_5s']
    report['move_brake_hold_pass'] = bool(report['real_cart_1m_pass'] and hold and hold['pass'])
    report['initial_spring_energy_J'] = mechanics['initial_spring_energy_J']
    report.update(artifact_kind='diagnostic_compliant_towing_execution', physics_variant=VARIANT,
        diagnostic_drawbar=parameters.metadata(), compliant_mechanics=mechanics,
        original_teacher_dataset_eligible=False, route_acceptance=False, formal_ppo_authorized=False,
        behavior_thresholds_changed=False,
        analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        original_analysis_source_sha256=original['analysis_source_sha256'])
    report['limitations'] += ' Explicit compliant spring/damping physical variant; not accepted by the original2000/150 teacher dataset. '
    return report


def analyze(path, write_report=True):
    path = Path(path)
    request = json.loads((path/'request.json').read_text())
    parameters = declared_parameters(request)
    runtime = json.loads((path/'drawbar_runtime.json').read_text())
    if runtime.get('parameters') != parameters.metadata():
        raise ValueError('Executed compliant force parameters differ from request')
    declared_sources = request.get('diagnostic_drawbar_sources', {})
    if set(declared_sources) != set(source_hashes()) or runtime.get('sources') != declared_sources:
        raise ValueError('Missing or inconsistent compliant execution source snapshot')
    for relative, digest in declared_sources.items():
        source = path/'source_snapshot'/relative
        if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != digest:
            raise ValueError('Compliant source snapshot differs: '+relative)
    with np.load(path/'seed43_case00_physics_trace.npz', allow_pickle=False) as data:
        z = dict(data)
    with np.load(path/'seed43_case00_policy_trace.npz', allow_pickle=False) as data:
        frames = len(data['executed_action'])
    if (runtime.get('physics_substeps') != len(z['joint_pos'])
            or runtime.get('live_estimator_frames') != frames):
        raise ValueError('Physical or teacher compliant force call count differs from actual traces')
    original = analyze_original(path, write_report=False)
    report = apply_variant_audit(original, z, parameters)
    # The unchanged legacy auditor must reject this physical candidate, even if
    # a copied or edited result JSON claims success. Do not modify its contract.
    report['legacy_original_force_audit_rejects'] = not bool(original['checks'].get('independent_spring_and_moment', False))
    if not report['legacy_original_force_audit_rejects']:
        raise ValueError('Variant is not distinguishable by the original force audit; do not mix it with baseline teacher evidence')
    report['drawbar_runtime'] = runtime
    if write_report:
        (path/'compliant_cart_analysis.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--require-pass', action='store_true')
    parser.add_argument('--require-hold', action='store_true')
    args = parser.parse_args()
    report = analyze(args.run)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.require_pass and not report['real_cart_1m_pass']:
        parser.exit(2, 'Compliant candidate did not pass the original1m protocol.\n')
    if args.require_hold and not report['move_brake_hold_pass']:
        parser.exit(2, 'Compliant candidate did not pass the original move/brake/2s/5s protocol.\n')
