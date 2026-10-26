"""V5训练前的只读候选接触/奖励审计；缺失证据不触发仿真或放行PPO。"""
from pathlib import Path
import shutil
from .io import ROOT, read, write, sha, stamp, implementation, seal, pack_review

REQUIRED_ROLES = ('baseline', 'final')
RAW_SUFFIXES = ('.json', '_trace.npz', '_trace.json', '_policy.npz')


RAW_FOLDERS = ('final_evaluation', 'selected', 'baseline', 'final', 'best_new', 'secondary')


def safe_path(root, path):
    root = Path(root).resolve()
    path = Path(path)
    if '..' in path.parts:
        raise ValueError('Parent traversal is forbidden: ' + str(path))
    path = path if path.is_absolute() else root / path
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError('Path escapes session boundary: ' + str(path))
    return resolved


def legacy_implementation():
    from r5a_guided_v4.io import implementation as original
    return original()


def source_evidence(session):
    """完整源码清单验真，并和本机受信的原版v4实现核对；空目录不能通过。"""
    for relative in ('final_evaluation/source_snapshot', 'training/source_snapshot', 'source_snapshot'):
        folder = safe_path(session, relative)
        if not folder.is_dir():
            continue
        manifest_path = safe_path(session, folder.parent / 'source_manifest.json')
        if not manifest_path.is_file():
            raise ValueError('Missing source_manifest: ' + str(manifest_path))
        manifest = read(manifest_path)
        expected = legacy_implementation()
        if not isinstance(manifest, dict) or not manifest or not expected or any(manifest.get(k) != h for k, h in expected.items()):
            raise ValueError('Source manifest does not match the trusted v4 implementation')
        paths = [manifest_path]
        for relative, expected_sha in manifest.items():
            source = safe_path(folder, relative)
            if not source.is_file() or sha(source) != expected_sha:
                raise ValueError('Source snapshot missing or modified: ' + relative)
            paths.append(source)
        for source in folder.rglob('*'):
            safe_path(session, source)
        return folder, manifest_path, paths
    raise ValueError('Missing source_snapshot')


def resolve_v4_session(session=None):
    if session is not None:
        path = Path(session).expanduser().resolve(strict=True)
        state = read(safe_path(path, 'session.json'))
        if state.get('status') != 'completed' or state.get('contract', {}).get('version') != 'lighthw_r5a_teacher_guided_v4':
            raise ValueError('需要已完成的Guided v4 session目录；不能使用未完成训练或其他路线。')
        return path
    candidates = []
    for path in sorted((ROOT / 'offline_sessions/route_05a').glob('guided_v4_*')):
        try:
            state = read(safe_path(path, 'session.json'))
        except (OSError, ValueError):
            continue
        if state.get('status') == 'completed' and state.get('contract', {}).get('version') == 'lighthw_r5a_teacher_guided_v4':
            candidates.append(path.resolve())
    if len(candidates) != 1:
        raise ValueError(f'找到 {len(candidates)} 个已完成v4 session；请用 --v4-session 明确指定完整目录。候选：' + ', '.join(map(str, candidates)))
    return candidates[0]


def local_checkpoint(session, row):
    """允许项目整体搬迁；任何..或向session外逃逸的符号链接均拒绝。"""
    session = Path(session).resolve()
    original = Path(row.get('checkpoint', ''))
    if '..' in original.parts:
        raise ValueError('Unsafe checkpoint path')
    choices = []
    if not original.is_absolute() or original.is_relative_to(session):
        choices.append(safe_path(session, original))
    if session.name in original.parts:
        relative = Path(*original.parts[original.parts.index(session.name) + 1:])
        choices.append(safe_path(session, relative))
    choices.extend(safe_path(session, p) for p in (
        session / 'training' / ('initial.pt' if row['label'] == 'session_start_baseline' else original.name),
        session / 'training/training' / original.name))
    for path in choices:
        if path.is_file():
            return path
    expected = {c.get('checkpoint_sha256') for c in row.get('cases', [])}
    for relative in ('selected.pt', 'secondary.pt', 'baseline/model.pt', 'final/model.pt'):
        path = safe_path(session, relative)
        if path.is_file() and len(expected) == 1 and sha(path) in expected:
            return path
    return None


def evidence_rows(session):
    comparison = read(safe_path(session, 'candidate_comparison.json'))
    if not isinstance(comparison, list) or any(not isinstance(r, dict) or 'label' not in r for r in comparison):
        raise ValueError('Malformed candidate comparison')
    baseline = next((r for r in comparison if r['label'] == 'session_start_baseline'), None)
    trained = [r for r in comparison if r['label'] != 'session_start_baseline']
    final = max(trained, key=lambda r: r['completed_updates']) if trained else None
    return {'baseline': baseline, 'final': final}


def audit_inputs(session):
    """缓存绑定实际模型、物理/策略轨迹、报告和单份运行源码，不依赖修改时间。"""
    paths = [session / f for f in ('session.json', 'candidate_comparison.json', 'selection.json', 'final_summary.json', 'source_manifest.json', 'SHA256.json')]
    for run_name in ('final_evaluation', 'training'):
        run = session / run_name
        paths.extend(run / f for f in ('result.json', 'request.json', 'contract.json', 'source_manifest.json', 'runtime_check.json', 'model.json', 'assets.json'))
    rows = read(safe_path(session, 'candidate_comparison.json'))
    for row in rows:
        if not row:
            continue
        model = local_checkpoint(session, row)
        if model:
            paths.append(model)
        for case in row.get('cases', []):
            prefix = str(case['artifact_prefix'])
            if Path(prefix).name != prefix:
                raise ValueError('Unsafe artifact prefix')
            for folder in RAW_FOLDERS:
                paths.extend(session / folder / (prefix + suffix) for suffix in RAW_SUFFIXES)
    _, _, source_paths = source_evidence(session)
    paths.extend(source_paths)
    safe = {safe_path(session, p) for p in paths}
    return {str(p.relative_to(session)): sha(p) for p in sorted(safe) if p.is_file()}


def required_evidence_errors(session):
    errors = []
    for role, row in evidence_rows(session).items():
        if row is None:
            errors.append(role + '/candidate'); continue
        if local_checkpoint(session, row) is None:
            errors.append(role + '/model.pt')
        if not row.get('cases'):
            errors.append(role + '/cases')
        for case in row.get('cases', []):
            prefix = case['artifact_prefix']
            if Path(prefix).name != prefix:
                errors.append(role + '/unsafe_prefix'); continue
            for suffix in RAW_SUFFIXES:
                if not any(safe_path(session, Path(folder) / (prefix + suffix)).is_file() for folder in RAW_FOLDERS):
                    errors.append(role + '/' + prefix + suffix)
    try:source_evidence(session)
    except (OSError, ValueError, TypeError):errors.append('source_snapshot_manifest')
    return errors


def validate_candidate_audit(record):
    """每个入口都重新核对证据和实现；旧准备缓存不能绕过候选复核。"""
    try:
        if not record or record.get('passed') is not True or record.get('required_roles') != list(REQUIRED_ROLES):
            return False
        if record.get('implementation_sha256') != implementation():
            return False
        session = resolve_v4_session(record['session'])
        if required_evidence_errors(session):
            return False
        if not record.get('inputs') or record['inputs'] != audit_inputs(session):
            return False
        evidence = record.get('evidence', {})
        if set(evidence) != {'contact', 'reward'}:
            return False
        reports = {}
        for role, item in evidence.items():
            path = Path(item['path'])
            if not path.is_file() or sha(path) != item['sha256']:
                return False
            reports[role] = read(path)
        for report in reports.values():
            if report.get('source_session') != str(session) or not report.get('inputs'):
                return False
            for relative, expected in report['inputs'].items():
                path = safe_path(session, relative)
                if str(path.relative_to(session)) != relative or record['inputs'].get(relative) != expected:
                    return False
        contact = reports['contact']
        if contact.get('required_roles') != list(REQUIRED_ROLES):
            return False
        return contact.get('comparison_complete') is True and contact.get('passed') is True and reports['reward'].get('passed') is True
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, AttributeError):
        return False


def _copy(source, dest, root):
    source = safe_path(root, source)
    if source.is_file():
        dest = Path(dest); dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, dest)


def _copy_tree(source, destination, root):
    source = safe_path(root, source)
    paths = list(source.rglob('*'))
    for path in paths:
        safe_path(root, path)
    for path in paths:
        if path.is_file() and not any(x in path.parts for x in ('__pycache__', 'asset_cache')):
            _copy(path, destination / path.relative_to(source), root)


def copy_audit_summaries(record, destination):
    """冻结小型审计结论及来源；最终USB无需依赖训练机的外部绝对路径。"""
    destination = Path(destination); destination.mkdir(parents=True, exist_ok=True)
    provenance = dict(schema='guided_v5_audit_summaries_v1', source_session=record.get('session'),
                      original_candidate_audit=record, files={}, missing=[],
                      scope='Verified contact/reward result JSON and Markdown only; no old v4 models, raw arrays or per-step reward histories')
    for role in ('contact', 'reward'):
        item = record.get('evidence', {}).get(role, {})
        source = Path(item.get('path', ''))
        if not source.is_file() or sha(source) != item.get('sha256'):
            provenance['missing'].append(role + '/bound_result'); continue
        target = destination / (role + '.json')
        _copy(source, target, source.parent)
        provenance['files'][target.name] = dict(sha256=sha(target), original_path=str(source), bound_report_sha256=item['sha256'])
        title = 'CONTACT_AUDIT.md' if role == 'contact' else 'REWARD_AUDIT.md'
        markdown = safe_path(source.parent, Path(role) / title)
        md_target = destination / title
        if markdown.is_file():
            _copy(markdown, md_target, source.parent)
            provenance['files'][title] = dict(sha256=sha(md_target), original_path=str(markdown), derived=False)
        else:
            result = read(target)
            md_target.write_text('# ' + role.capitalize() + ' audit\n\n' +
                'Passed: ' + str(result.get('passed')) + '\n\n' +
                '完整结论、来源和数值见同目录 `' + target.name + '`；本页从已绑定JSON生成。\n')
            provenance['files'][title] = dict(sha256=sha(md_target), derived_from=target.name, derived=True)
    provenance['self_contained'] = not provenance['missing']
    write(destination / 'provenance.json', provenance)
    return provenance


def export_contact_review(session, output, record):
    """最多两套原始候选证据；缺失项写清楚，不用旧基线替代新候选。"""
    review = ROOT / 'usb_review/route_05a' / (output.name + '_contact')
    review.mkdir(parents=True, exist_ok=False)
    _copy_tree(output, review / 'audit', output)
    missing = []; retained = {}
    def copy(source, destination):
        try:_copy(source, destination, session)
        except (ValueError,OSError) as error:missing.append(str(error))
    if session is not None:
        for name in ('session.json', 'candidate_comparison.json', 'selection.json', 'final_summary.json'):
            copy(session / name, review / name)
        try:
            rows = evidence_rows(session)
        except (OSError, ValueError, KeyError):
            rows = {}
        for role in REQUIRED_ROLES:
            row = rows.get(role)
            if row is None:
                missing.append(role + '/candidate'); continue
            retained[role] = row
            try:model = local_checkpoint(session, row)
            except (ValueError,OSError) as error:
                missing.append(str(error));model=None
            if model:
                copy(model, review / role / 'model.pt')
            else:
                missing.append(role + '/model.pt')
            for case in row.get('cases', [])[:1]:
                prefix = case['artifact_prefix']
                if Path(prefix).name != prefix:
                    missing.append(role + '/unsafe_prefix');continue
                for suffix in RAW_SUFFIXES:
                    choices = [session / folder / (prefix + suffix) for folder in RAW_FOLDERS]
                    try:source = next((safe_path(session,p) for p in choices if safe_path(session,p).is_file()), None)
                    except (ValueError,OSError) as error:
                        missing.append(str(error));source=None
                    if source:
                        copy(source, review / role / (prefix + suffix))
                    else:
                        missing.append(role + '/' + prefix + suffix)
            if not row.get('cases'):
                missing.append(role + '/cases')
        for run in ('final_evaluation', 'training'):
            for name in ('request.json', 'result.json', 'contract.json', 'runtime_check.json', 'model.json', 'assets.json', 'source_manifest.json'):
                copy(session / run / name, review / run / name)
        try:
            folder, manifest_path, _ = source_evidence(session)
            _copy_tree(folder, review / 'source_snapshot', session)
            copy(manifest_path, review / 'source_manifest.json')
        except (ValueError,OSError,TypeError) as error:
            missing.append('source_snapshot: ' + str(error))
    else:
        missing.append('v4_session')
    write(review / 'retained_candidates.json', retained)
    write(review / 'review_manifest.json', dict(export_kind='guided_v5_contact_audit', status='incomplete' if missing else ('complete' if record.get('passed') else 'audit_failed'),
          missing=missing, required_roles=list(REQUIRED_ROLES), raw_case_limit_per_role=1, no_simulation=True, no_training=True, original_session=str(session) if session else None))
    (review / 'README.md').write_text('本包只读核对已完成v4的baseline和final；不启动仿真或训练。\n缺失证据见review_manifest.json。只有接触审计和奖励审计均通过，V5训练才允许启动。\n')
    seal(review)
    archive = pack_review(review)
    print('[USB CONTACT REVIEW ZIP]', archive, flush=True)
    return archive


def ensure_candidate_audit(v4_session=None, *, export=True, strict=True):
    """先物理接触复核，再离线重放奖励；任何缺失都导出诊断并拒绝PPO。"""
    output = ROOT / 'offline_sessions/route_05a' / ('guided_v5_candidate_audit_' + stamp())
    output.mkdir(parents=True, exist_ok=False)
    record = dict(passed=False, required_roles=list(REQUIRED_ROLES), implementation_sha256=implementation(), inputs={}, evidence={})
    session = None
    try:
        session = resolve_v4_session(v4_session)
        record['session'] = str(session)
        before = audit_inputs(session)
        from .contact_audit import audit_session
        contact = audit_session(session, output / 'contact', required_roles=REQUIRED_ROLES)
        # 统一存实际返回报告，避免依赖审计器内部输出文件名。
        write(output / 'contact_result.json', contact)
        missing = required_evidence_errors(session)
        record['missing_evidence'] = missing
        if contact.get('comparison_complete') is True and contact.get('passed') is True and not missing:
            from .reward_audit import audit_session_rewards
            reward = audit_session_rewards(session, output / 'reward')
        else:
            reward = dict(passed=False, status='not_run', reason='Candidate contact evidence is incomplete or invalid')
        write(output / 'reward_result.json', reward)
        after = audit_inputs(session)
        record.update(inputs=after, passed=contact.get('comparison_complete') is True and contact.get('passed') is True and not missing and reward.get('passed') is True and before == after,
                      evidence={name: dict(path=str(output / (name + '_result.json')), sha256=sha(output / (name + '_result.json'))) for name in ('contact', 'reward')})
        if before != after:
            record['error'] = 'v4 input changed during audit'
    except Exception as error:
        record['error'] = f'{type(error).__name__}: {error}'
    if record['passed'] and not validate_candidate_audit(record):
        record['passed'] = False
        record['error'] = 'Audit reports do not bind to this session, inputs, and implementation'
    write(output / 'candidate_audit.json', record)
    if export or not record['passed']:
        record['archive'] = str(export_contact_review(session, output, record))
        write(output / 'candidate_audit.json', record)
    if strict and not validate_candidate_audit(record):
        raise RuntimeError('V5候选接触/奖励审计未通过；没有启动仿真或PPO。请回收 [USB CONTACT REVIEW ZIP]，保留v4原始session。' + (' ' + record.get('error', '')))
    return record
