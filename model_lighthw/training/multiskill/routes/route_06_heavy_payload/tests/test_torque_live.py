"""CPU tests for exact joint mapping, bounded history and live plot data."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / 'scripts/torque_live.py'
SPEC = importlib.util.spec_from_file_location('r6_torque_live_under_test', MODULE_PATH)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)

# Deliberately joint-major rather than leg-major to expose assumed array order.
NAMES = [f'{leg}_{joint}_joint' for joint in reversed(module.JOINTS)
         for leg in reversed(module.LEGS)]


def frame(t, offset=0.):
    return {'t': t, 'applied_torque': [(-1) ** i * (i + offset) for i in range(len(NAMES))]}


class Plot:
    def __init__(self):
        self.data = []
        self.visible = False
        self.calls = 0

    def set_xy_data(self, data):
        self.data = list(data)
        self.calls += 1


@pytest.fixture
def plot(monkeypatch):
    def build_ui(self):
        self.destroy_calls = 0
        def destroy():
            self.destroy_calls += 1
        self._window = SimpleNamespace(destroy=destroy)
        self._plots = {joint: Plot() for joint in module.JOINTS}
        self._zero_plots = {joint: Plot() for joint in module.JOINTS}
        keys = ['leg', 'episode', 'time_start', 'time_mid', 'time_end']
        keys += list(module.JOINTS)
        keys += [f'{joint}_{bound}' for joint in module.JOINTS for bound in ['min', 'max']]
        self._labels = {key: SimpleNamespace(text='') for key in keys}
    monkeypatch.setattr(module.LiveLegTorquePlot, '_build_ui', build_ui)
    instance = module.LiveLegTorquePlot(NAMES)
    yield instance
    instance.close()


def test_exact_joint_names_preserve_signed_torque():
    history = module.LegTorqueHistory(NAMES)
    history.begin_episode('first')
    first, second = frame(.02), frame(.04, 20)
    history.update(first)
    history.update(second)
    for leg in module.LEGS:
        for joint in module.JOINTS:
            index = NAMES.index(f'{leg}_{joint}_joint')
            assert history.series(leg)[joint] == (first['applied_torque'][index],
                                                 second['applied_torque'][index])
    assert any(value < 0 for value in history.series('FL')['hip'])


def test_history_is_exact_recent_ten_seconds_and_bounded():
    history = module.LegTorqueHistory(NAMES)
    history.begin_episode('long')
    for step in range(30001):
        history.update(frame(step * .02, step))
    assert len(history.times) == 501
    assert history.times[0] == pytest.approx(590.)
    assert history.times[-1] == pytest.approx(600.)
    # A simulation-time gap also expires old samples, not merely sample count.
    history.update(frame(611.))
    assert history.times == (611.,)


def test_history_reset_keeps_leg_without_cross_episode_lines():
    history = module.LegTorqueHistory(NAMES)
    history.begin_episode('old')
    history.select_leg('RR')
    history.update(frame(9.))
    history.begin_episode('new')
    assert history.selected_leg == 'RR'
    assert history.times == ()
    history.update(frame(.02, 9))
    assert history.times == (.02,)
    assert all(len(values) == 1 for values in history.series().values())


@pytest.mark.parametrize('names', [NAMES[:-1], NAMES + [NAMES[0]]])
def test_joint_mapping_rejects_missing_or_duplicate_names(names):
    with pytest.raises(ValueError):
        module.LegTorqueHistory(names)


@pytest.mark.parametrize('bad_frame', [
    {'t': .02, 'applied_torque': [0.] * 15},
    {'t': float('nan'), 'applied_torque': [0.] * 16},
    {'t': -.02, 'applied_torque': [0.] * 16},
    {'t': .04, 'applied_torque': [float('inf')] + [0.] * 15},
    {'t': .02, 'applied_torque': [0.] * 16},
])
def test_invalid_frames_are_rejected_before_history_mutation(bad_frame):
    history = module.LegTorqueHistory(NAMES)
    history.begin_episode('one')
    history.update(frame(.02))
    before = history.times, history.series()
    with pytest.raises(ValueError):
        history.update(bad_frame)
    assert (history.times, history.series()) == before


def test_updates_require_episode_and_leg_is_explicit():
    history = module.LegTorqueHistory(NAMES)
    with pytest.raises(RuntimeError):
        history.update(frame(.02))
    with pytest.raises(ValueError):
        history.select_leg('left')
    assert history.selected_leg == 'FL'


def test_all_fifty_hz_samples_collected_with_ten_hz_redraw(plot):
    plot.begin_episode('one')
    for step in range(1, 7):
        plot.update(frame(step * .02, step))
    assert len(plot.history.times) == 6
    assert all(widget.calls == 2 for widget in plot._plots.values())
    assert plot.display_snapshot['times'] == tuple(step * .02 for step in range(1, 7))
    plot.update(frame(.14, 7))
    assert len(plot.history.times) == 7
    assert len(plot.display_snapshot['times']) == 6
    assert all(widget.calls == 2 for widget in plot._plots.values())


def test_switch_leg_immediately_draws_that_legs_existing_history(plot):
    plot.begin_episode('one')
    plot.update(frame(.02))
    plot.update(frame(.04, 30))  # not due for ordinary display refresh
    before_times = plot.history.times
    for leg in module.LEGS:
        plot.select_leg(leg)
        shown = plot.display_snapshot
        assert shown['leg'] == leg
        assert shown['times'] == before_times
        for joint in module.JOINTS:
            assert shown['joint_names'][joint] == f'{leg}_{joint}_joint'
            assert plot._plots[joint].data == list(zip(before_times, plot.history.series(leg)[joint]))
            assert shown['series'][joint] == plot.history.series(leg)[joint]
            assert leg in plot._labels[joint].text


def test_reset_clears_visible_lines_and_retains_leg(plot):
    plot.begin_episode('old')
    plot.select_leg('RL')
    plot.update(frame(9.))
    plot.begin_episode('new')
    assert plot.selected_leg == 'RL'
    assert plot.display_snapshot['episode'] == 'new'
    assert plot.display_snapshot['times'] == ()
    assert all(not widget.visible for widget in plot._plots.values())
    assert all(not widget.visible for widget in plot._zero_plots.values())
    plot.update(frame(.02, 100))
    assert plot.display_snapshot['times'] == (.02,)
    assert all(len(widget.data) == 1 for widget in plot._plots.values())


def test_display_axes_signed_values_and_no_mutable_snapshot_alias(plot):
    plot.begin_episode('test')
    plot.update(frame(.02, 30))
    shown = plot.display_snapshot
    assert shown['units'] == 'N m'
    assert shown['source'] == 'signed applied_torque'
    for joint in module.JOINTS:
        low, high = shown['scale_limits'][joint]
        assert low == -high
        assert low <= min(shown['series'][joint]) <= max(shown['series'][joint]) <= high
        assert plot._plots[joint].scale_min == low
        assert plot._plots[joint].scale_max == high
        assert plot._zero_plots[joint].data == [(.02, 0.), (.02, 0.)]
    shown['series']['hip'] = ()
    shown['joint_names']['hip'] = 'wrong'
    shown['leg'] = 'RR'
    assert plot.display_snapshot['series']['hip']
    assert plot.display_snapshot['joint_names']['hip'] == 'FL_hip_joint'
    assert plot.display_snapshot['leg'] == 'FL'
    assert plot._labels['time_end'].text == '0.02 s'


def test_close_is_idempotent_and_stops_updates(plot):
    plot.close()
    plot.close()
    assert plot.destroy_calls == 1
    assert plot._window is None
    with pytest.raises(RuntimeError):
        plot.begin_episode('late')
    with pytest.raises(RuntimeError):
        plot.select_leg('FR')
    with pytest.raises(RuntimeError):
        plot.update(frame(.02))


def test_episode_finish_flushes_latest_frame_before_next_display_period(plot):
    plot.begin_episode('ending')
    plot.select_leg('FR')
    for step in range(1, 4):
        plot.update(frame(step * .02, step))
    assert plot.display_snapshot['times'] == (.02,)
    before_times, before_series = plot.history.times, plot.history.series()
    plot.finish_episode()
    assert plot.selected_leg == 'FR'
    assert plot.history.times == before_times == (.02, .04, .06)
    assert plot.history.series() == before_series
    assert plot.display_snapshot['times'] == before_times
    assert plot.display_snapshot['series'] == before_series
    for joint in module.JOINTS:
        assert plot._plots[joint].data == list(zip(before_times, before_series[joint]))
        assert plot._plots[joint].visible
    plot.begin_episode('next')
    assert plot.selected_leg == 'FR'
    assert plot.display_snapshot['times'] == ()
