import sys
import json
from pathlib import Path

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from student_prefix import StudentPrefix, prefix_steps, load_prefix_student, CONSTRAINED_SCHEMA


@pytest.mark.parametrize('value', [0., .01, .03, 1.02, float('nan'), float('inf')])
def test_prefix_rejects_ambiguous_or_invalid_boundaries(value):
    with pytest.raises(ValueError):
        prefix_steps(value)


def test_executed_actions_switch_once_shadow_teacher_is_never_selected_in_prefix():
    class Policy:
        actor = torch.nn.Identity()
        def __call__(self, obs):
            return obs[:, :16]+2.
    prefix = StudentPrefix.__new__(StudentPrefix)
    prefix.steps = prefix_steps(.04); prefix.calls = 0; prefix.teacher_frames = 0
    prefix.policy = Policy()
    obs = torch.zeros(1, 300); teacher = torch.ones(1, 16)
    for i in range(5):
        action, trace = prefix.select(i*.02, obs, teacher)
        assert torch.equal(action, torch.full_like(teacher, 2. if i < 2 else 1.))
        assert trace['student_prefix_active'].item() == (i < 2)
        assert trace['teacher_executed'].item() == (i >= 2)
        assert trace['student_prefix_takeover_frame'].item() == (i == 2)
        assert torch.equal(trace['student_prefix_teacher_raw'], teacher)
    assert (prefix.calls, prefix.teacher_frames) == (2, 3)
    with pytest.raises(ValueError, match='discontinuous'):
        prefix.select(.12, obs, teacher)


@pytest.mark.parametrize('constrained', [False, True])
def test_explicit_schema_uses_its_strict_loader_and_complete_source_set(monkeypatch, tmp_path, constrained):
    from migration_v1 import student, constrained_student, bc, constrained_refit
    checkpoint = tmp_path/'student.pt'; checkpoint.write_text('synthetic routing test only')
    schema = CONSTRAINED_SCHEMA if constrained else bc.SCHEMA
    checkpoint.with_name('manifest.json').write_text(json.dumps(dict(schema=schema)))
    sources = constrained_refit.source_signatures() if constrained else bc.source_signatures()
    manifest = dict(schema=schema,sources=sources,teacher_bundle=dict(episodes=[
        dict(case=dict(slope_deg=slope),command_timing=dict(actual_move_start_s=start))
        for slope,start in [(25.,3.),(-20.,7.48)]]))
    selected = constrained_student if constrained else student
    rejected = student if constrained else constrained_student
    calls = []
    def load(path, **kwargs):
        calls.append((path,kwargs)); return object(),manifest
    monkeypatch.setattr(selected,'load_student',load)
    monkeypatch.setattr(rejected,'load_student',lambda *a,**k:pytest.fail('Wrong schema loader reached'))
    prefix = StudentPrefix(checkpoint,.02,25.,3.)
    assert calls == [(checkpoint,dict(device='cpu',inference_mode='row'))]
    assert set(sources) < set(prefix.metadata['source_sha256'])
    assert len(prefix.metadata['source_sha256']) == (16 if constrained else 12)
    if constrained:
        assert prefix.metadata['student_schema']==CONSTRAINED_SCHEMA
        assert prefix.metadata['student_loader']=='migration_v1.constrained_student.load_student'
    else:
        assert 'student_schema' not in prefix.metadata and 'student_loader' not in prefix.metadata


def test_unknown_schema_never_falls_back_to_nominal_loader(monkeypatch,tmp_path):
    from migration_v1 import student, constrained_student
    checkpoint=tmp_path/'student.pt';checkpoint.write_text('synthetic routing test only')
    checkpoint.with_name('manifest.json').write_text('{"schema":"unknown"}')
    for module in (student,constrained_student):
        monkeypatch.setattr(module,'load_student',lambda *a,**k:pytest.fail('Unknown schema reached model loading'))
    with pytest.raises(ValueError,match='Unsupported student-prefix'):
        load_prefix_student(checkpoint)
