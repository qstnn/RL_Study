"""CPU-only keyboard, step-boundary, recording and GUI-model checks."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / 'scripts/keyboard_payload.py'
SPEC = importlib.util.spec_from_file_location('r6_keyboard_payload_under_test', MODULE_PATH)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class Runtime:
    def __init__(self, mass=65.):
        self.mass_kg = mass
        self.total_mass_kg = mass + 11.668111
        self.calls = []

    def set_mass(self, mass):
        self.calls.append(mass)
        self.mass_kg = mass
        self.total_mass_kg = mass + 11.668111
        return {'payload_kg': mass, 'physical_properties_verified': True,
                'actual': {'loaded_robot_mass_kg': self.total_mass_kg}}


class PlotStub:
    def __init__(self):
        self.selected_leg = 'FL'
        self.frames = []
        self.labels = []
        self.selections = []
        self.close_count = 0
        self.finish_count = 0

    def select_leg(self, leg):
        self.selected_leg = leg
        self.selections.append(leg)

    def begin_episode(self, label):
        self.labels.append(label)
        self.frames.clear()

    def update(self, frame):
        self.frames.append(frame)

    def finish_episode(self):
        self.finish_count += 1

    def close(self):
        self.close_count += 1


@pytest.fixture
def controller(tmp_path, monkeypatch):
    def build_ui(self):
        self._labels = {key: SimpleNamespace(text='') for key in
                        ['mass', 'pose', 'peak', 'status', *self.joint_names]}
    monkeypatch.setattr(module.KeyboardPayload, '_build_ui', build_ui)
    monkeypatch.setattr(module.KeyboardPayload, '_build_torque_plot',
                        lambda self: setattr(self, '_torque_plot', PlotStub()))
    monkeypatch.setattr(module.KeyboardPayload, '_subscribe_keyboard', lambda _: None)
    names = [f'{leg}_{joint}_joint' for leg in ['FL', 'FR', 'RL', 'RR']
             for joint in ['hip', 'thigh', 'calf', 'wheel']]
    base = SimpleNamespace(scene={'robot': SimpleNamespace(joint_names=names)})
    obj = module.KeyboardPayload(base, tmp_path, Runtime())
    yield obj
    obj.close()


def key(controller, name, kind='KEY_PRESS'):
    event = SimpleNamespace(input=SimpleNamespace(name=name), type=SimpleNamespace(name=kind))
    return controller._on_keyboard_event(event)


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_only_press_and_boundaries():
    state = module.PayloadKeyState(65.)
    assert not state.enqueue_key('LEFT_BRACKET', 'KEY_REPEAT')
    assert not state.enqueue_key('LEFT_BRACKET', 'KEY_RELEASE')
    for navigation_key in ('W', 'A', 'S', 'D', 'Q', 'E', 'LEFT', 'UP'):
        assert not state.enqueue_key(navigation_key)
    assert state.enqueue_key(']')
    assert state.drain() == ['RIGHT_BRACKET']
    assert state.operation('RIGHT_BRACKET') == ('mass', 65.)
    state.commit_mass(0.)
    assert state.operation('LEFT_BRACKET') == ('mass', 0.)
    assert state.operation('MINUS') == ('mass', 0.)
    assert state.operation('KEY_6') == ('mass', 65.)
    for bad in (-1., 66., float('nan'), float('inf')):
        with pytest.raises(ValueError):
            state.commit_mass(bad)


def test_callback_never_changes_mass_and_order_is_preserved(controller):
    controller.begin_episode('case0')
    for name in ('LEFT_BRACKET', 'MINUS', 'KEY_0', 'RIGHT_BRACKET', 'KEY_6'):
        assert key(controller, name) is True
    assert controller.runtime.calls == []
    assert controller.before_step(7, .02)
    assert controller.runtime.calls == [60., 59., 0., 5., 65.]
    actual = rows(controller.run / 'keyboard_payload_events.jsonl')
    events = actual[1:]
    assert [event['old_mass_kg'] for event in events] == [65., 60., 59., 0., 5.]
    assert all(event['step'] == 7 and event['simulation_time_s'] == .14 for event in events)
    assert all(event['runtime_verification']['physical_properties_verified'] for event in events)
    assert rows(controller.run / 'case0_payload_events.jsonl') == actual
    assert not controller.before_step(8, .02)


def test_equal_mass_does_not_call_setter(controller):
    controller.begin_episode('case0')
    for name in ('KEY_6', 'RIGHT_BRACKET', 'EQUAL'):
        key(controller, name)
    assert not controller.before_step(0, .02)
    assert controller.runtime.calls == []
    assert [r['event'] for r in rows(controller.run / 'keyboard_payload_events.jsonl')] == [
        'episode_begin', 'mass_unchanged', 'mass_unchanged', 'mass_unchanged']
    key(controller, 'KEY_0')
    assert controller.before_step(1, .02)
    for name in ('KEY_0', 'LEFT_BRACKET', 'MINUS'):
        key(controller, name)
    assert not controller.before_step(2, .02)
    assert controller.runtime.calls == [0.]


def test_repeat_does_not_repeatedly_change_mass(controller):
    controller.begin_episode('case0')
    key(controller, 'LEFT_BRACKET')
    for _ in range(20):
        key(controller, 'LEFT_BRACKET', 'KEY_REPEAT')
    key(controller, 'LEFT_BRACKET', 'KEY_RELEASE')
    controller.before_step(0, .02)
    assert controller.runtime.calls == [60.]


def test_reset_starts_new_label_and_preserves_current_mass(controller):
    controller.begin_episode('case0')
    for name in ('KEY_0', 'R', 'KEY_6'):
        key(controller, name)
    assert controller.before_step(3, .02)
    assert controller.reset_requested
    assert not controller.stop_requested
    assert controller.mass_kg == 0.
    controller.finish_episode('keyboard_reset')
    assert json.loads((controller.run / 'case0_payload.json').read_text())['final_payload_kg'] == 0.
    key(controller, 'KEY_6')  # pending non-stop key is cleared on the next begin
    controller.begin_episode('case1')
    assert not controller.reset_requested
    assert not controller.before_step(0, .02)
    assert controller.mass_kg == 0.
    key(controller, 'F10')
    key(controller, 'KEY_6')
    assert not controller.before_step(1, .02)
    assert controller.stop_requested
    assert controller.mass_kg == 0.
    controller.finish_episode('keyboard_stop')
    metadata = json.loads((controller.run / 'case1_payload.json').read_text())
    assert metadata['initial_payload_kg'] == 0.
    assert metadata['reason'] == 'keyboard_stop'
    assert metadata['stop_requested']
    assert {row['label'] for row in rows(controller.run / 'keyboard_payload_events.jsonl')} == {'case0', 'case1'}


def test_failed_mass_change_is_flushed_before_exception(controller):
    controller.begin_episode('case0')
    def fail(_):
        raise RuntimeError('inertia verification failed')
    controller.runtime.set_mass = fail
    key(controller, 'KEY_0')
    with pytest.raises(RuntimeError, match='inertia verification failed'):
        controller.before_step(9, .02)
    assert controller.state.mass_kg == 65.
    event = rows(controller.run / 'keyboard_payload_events.jsonl')[-1]
    assert event['event'] == 'mass_change_failed'
    assert event['new_mass_kg'] == 0.
    assert event['runtime_mass_kg'] == 65.
    assert event['error_type'] == 'RuntimeError'


@pytest.mark.parametrize('use_fallback', [False, True])
def test_close_unsubscribes_once_without_changing_mass(controller, use_fallback):
    calls = []
    method = 'unsubscribe_from_keyboard_events' if use_fallback else 'unsubscribe_to_keyboard_events'
    controller._input = SimpleNamespace(**{method: lambda *args: calls.append(args)})
    controller._keyboard, controller._subscription = 'fake-keyboard', 0
    controller.begin_episode('case0')
    key(controller, 'KEY_0')  # closing must discard, never execute
    controller.close()
    controller.close()
    assert calls == [('fake-keyboard', 0)]
    assert controller.runtime.calls == []
    assert controller.mass_kg == 65.
    assert json.loads((controller.run / 'case0_payload.json').read_text())['reason'] == 'controller_closed'


def test_display_uses_recorded_signed_torque_and_throttles(controller):
    controller.begin_episode('case0')
    frame = {'t': .02, 'metrics': {'roll_deg': -1.2, 'pitch_deg': .7, 'height': .396},
             'applied_torque': list(range(-8, 8)), 'computed_torque': [999.] * 16}
    controller.update_display(frame)
    assert 'Roll -1.20 deg' in controller._labels['pose'].text
    assert 'Height 0.396 m' in controller._labels['pose'].text
    assert '-8.000' in controller._labels[controller.joint_names[0]].text
    assert 'legs 8.00 N m' in controller._labels['peak'].text
    assert 'wheels 7.00 N m' in controller._labels['peak'].text
    assert '76.668 kg' in controller._labels['mass'].text
    frame['metrics']['height'] = .4
    frame['t'] = .04
    controller.update_display(frame)
    assert '0.396 m' in controller._labels['pose'].text
    frame['t'] = .12
    controller.update_display(frame)
    assert '0.400 m' in controller._labels['pose'].text
    controller.set_status('Failed: press R to reset at current payload.')
    controller.before_step(6, .02)
    assert controller._labels['status'].text.startswith('Failed: press R')
    key(controller, 'KEY_0')
    controller.before_step(7, .02)
    assert '11.668 kg' in controller._labels['mass'].text


def test_episode_lifecycle_rejects_unsafe_labels(controller):
    with pytest.raises(RuntimeError, match='begin_episode'):
        controller.before_step(0, .02)
    with pytest.raises(ValueError):
        controller.begin_episode('../escape')
    controller.begin_episode('case0')
    with pytest.raises(RuntimeError, match='Finish'):
        controller.begin_episode('case1')
    controller.finish_episode('complete')
    controller.close()
    with pytest.raises(RuntimeError, match='closed'):
        controller.begin_episode('case2')


def test_failed_trial_allows_change_then_reset_without_reopening_old_trace(controller):
    controller.begin_episode('failed_case')
    controller.before_step(37, .02)
    controller.finish_episode('terminated')
    old_events = (controller.run / 'failed_case_payload_events.jsonl').read_bytes()
    old_metadata = (controller.run / 'failed_case_payload.json').read_bytes()
    key(controller, 'KEY_0')
    key(controller, 'RIGHT_BRACKET')
    key(controller, 'R')
    assert controller.before_step(37, .02)
    assert controller.runtime.calls == [0., 5.]
    assert controller.reset_requested
    between = rows(controller.run / 'keyboard_payload_events.jsonl')[-3:]
    assert [r['event'] for r in between] == ['mass_changed', 'mass_changed', 'reset_requested']
    assert all(r['phase'] == 'between_trials' and r['label'] is None for r in between)
    assert all(r['step'] is None and r['simulation_time_s'] is None for r in between)
    assert all(r['after_trial_label'] == 'failed_case' for r in between)
    assert (controller.run / 'failed_case_payload_events.jsonl').read_bytes() == old_events
    assert (controller.run / 'failed_case_payload.json').read_bytes() == old_metadata
    controller.begin_episode('retry_case')
    assert not controller.reset_requested
    assert not controller.stop_requested
    assert controller.mass_kg == 5.
    assert not controller.before_step(0, .02)
    controller.finish_episode('terminated')
    # R must not hide a subsequent F10 in the same callback batch.
    key(controller, 'R')
    key(controller, 'F10')
    controller.before_step(0, .02)
    assert controller.stop_requested
    controller.begin_episode('stopping_case')
    assert not controller.reset_requested
    assert controller.stop_requested


def test_begin_keeps_f10_queued_during_reset_render(controller):
    controller.begin_episode('case0')
    controller.finish_episode('terminated')
    key(controller, 'F10')
    controller.begin_episode('case1')
    assert controller.stop_requested
    events = rows(controller.run / 'keyboard_payload_events.jsonl')
    assert events[-2]['event'] == 'stop_requested'
    assert events[-2]['phase'] == 'between_trials'
    assert events[-1]['event'] == 'episode_begin'


def test_payload_press_prints_current_mass_once_including_clamped_key(controller, capsys):
    capsys.readouterr()
    controller.begin_episode('console_case')
    for name in ('RIGHT_BRACKET', 'LEFT_BRACKET', 'KEY_0', 'MINUS'):
        key(controller, name)
        key(controller, name, 'KEY_REPEAT')
        key(controller, name, 'KEY_RELEASE')
    assert capsys.readouterr().out == ''  # callback only queues
    assert controller.before_step(0, .02)
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 4
    for line, mass in zip(lines, (65., 60., 0., 0.)):
        assert line.startswith('[R6 负载] ')
        assert f'当前负载 {mass:.1f} kg' in line
        assert f'整机总质量 {mass + 11.668111:.3f} kg' in line
    assert '已到边界' in lines[0] and '已到边界' in lines[-1]
    controller.before_step(1, .02)
    assert capsys.readouterr().out == ''


def test_failed_change_prints_retained_actual_mass(controller, capsys):
    controller.begin_episode('failure_console')
    capsys.readouterr()
    def fail(_):
        raise RuntimeError('rejected')
    controller.runtime.set_mass = fail
    key(controller, 'KEY_0')
    with pytest.raises(RuntimeError, match='rejected'):
        controller.before_step(0, .02)
    text = capsys.readouterr().out
    assert text.count('[R6 负载]') == 1
    assert '上次核验负载 65.0 kg' in text and '当前物理状态未确认' in text
    assert '当前负载' not in text


def test_leg_keys_switch_only_plot_and_are_recorded(controller, capsys):
    controller.begin_episode('plot_switch')
    capsys.readouterr()
    for key_name in ('KEY_2', 'KEY_3', 'KEY_4', 'KEY_1'):
        key(controller, key_name)
        key(controller, key_name, 'KEY_REPEAT')
    assert not controller.before_step(4, .02)
    assert controller._torque_plot.selections == ['FR', 'RL', 'RR', 'FL']
    assert controller.runtime.calls == []
    events = rows(controller.run / 'keyboard_payload_events.jsonl')[1:]
    assert [e['selected_leg'] for e in events] == ['FR', 'RL', 'RR', 'FL']
    assert all(e['event'] == 'plot_leg_selected' and e['new_mass_kg'] == 65. for e in events)
    printed = capsys.readouterr().out
    assert printed.count('[R6 力矩图]') == 4 and '[R6 负载]' not in printed
    key(controller, 'KEY_4')
    controller.before_step(5, .02)
    controller.finish_episode('user_reset')
    controller.begin_episode('plot_next')
    assert controller._torque_plot.selected_leg == 'RR'
    assert controller._torque_plot.labels == ['plot_switch', 'plot_next']
    controller.finish_episode('user_finished')
    meta = json.loads((controller.run/'plot_next_payload.json').read_text())
    assert meta['live_torque_plot']['selected_leg'] == 'RR'
    controller.close()
    assert controller._torque_plot.close_count == 1
    assert controller._torque_plot.finish_count == 2


def test_every_policy_frame_reaches_plot_before_label_throttle(controller):
    controller.begin_episode('plot_samples')
    for i in range(1, 8):
        controller.update_display({'t': i*.02, 'applied_torque': [float(i)]*16,
                                  'metrics': {'roll_deg': 0., 'pitch_deg': 0., 'height': .4}})
    assert len(controller._torque_plot.frames) == 7
    assert controller._torque_plot.frames[-1]['t'] == .14
    controller.finish_episode('user_reset')
    controller.begin_episode('plot_reset')
    assert controller._torque_plot.frames == []
