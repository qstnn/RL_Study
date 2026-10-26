"""CPU orchestration checks for interactive payload play; no simulator or GUI."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / 'routes/route_06_heavy_payload/scripts'


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_events(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def exercise(monkeypatch, tmp_path, fail_first):
    """Run the real playback loop and keyboard controller with only the plant stubbed."""
    keyboard = load_module('r6_keyboard_integration', SCRIPTS / 'keyboard_payload.py')
    written = {}

    def write_json(path, value):
        written[Path(path).name] = value
        Path(path).write_text(json.dumps(value, allow_nan=False))

    monkeypatch.setitem(sys.modules, 'launch', SimpleNamespace(
        URDF=tmp_path / 'unused.urdf', spec=SimpleNamespace(CONTRACT={}), write_json=write_json))
    play = load_module('r6_payload_play_integration', SCRIPTS / 'payload_play.py')
    monkeypatch.setattr(play.time, 'sleep', lambda _: None)
    names = [f'{leg}_{joint}_joint' for leg in ('FL', 'FR', 'RL', 'RR')
             for joint in ('hip', 'thigh', 'calf', 'wheel')]
    robot = SimpleNamespace(joint_names=names, data=SimpleNamespace(root_state_w=torch.zeros(1, 13)))
    playing_checks = 0

    def is_playing():
        nonlocal playing_checks
        playing_checks += 1
        # First scenario changes mass while paused, then resumes at the same policy step.
        return fail_first or playing_checks > 1

    sim = SimpleNamespace(is_stopped=lambda: False, is_playing=is_playing, render=lambda: None)
    base = SimpleNamespace(scene={'robot': robot}, cfg=SimpleNamespace(payload_mass_kg=65.),
                           step_dt=.02, sim=sim)
    captured = []

    class Audit:
        def __init__(self):
            self.frames = []
            self.current = None

        def begin(self, run, label, record_video, physics_trace):
            assert record_video and physics_trace
            self.current, self.frames = label, []

        def finish(self, reason):
            captured.append({'label': self.current, 'reason': reason, 'frames': list(self.frames)})
            self.current = None

    base.r6_review = Audit()

    class Runtime:
        def __init__(self, actual_base, urdf, contract):
            assert actual_base is base
            self.mass_kg = 65.
            self.changes = []
            self.verifications = []

        @property
        def total_mass_kg(self):
            return self.mass_kg + 11.668111

        def set_mass(self, mass):
            previous = self.mass_kg
            self.changes.append(mass)
            self.mass_kg = base.cfg.payload_mass_kg = mass
            # Same aliasing contract as the real runtime adapter.
            for obs in base.obs_buf.values():
                obs[:, -1] = mass / 65.
            return {'previous_payload_kg': previous, 'payload_kg': mass,
                    'physical_properties_verified': True}

        def verify_current(self):
            self.verifications.append(self.mass_kg)
            assert base.cfg.payload_mass_kg == self.mass_kg
            assert torch.allclose(base.obs_buf['policy'][:, -1], torch.tensor([self.mass_kg / 65.]))
            return {'payload_kg': self.mass_kg, 'verification_phase': 'independent_readback'}

    monkeypatch.setitem(sys.modules, 'payload_runtime', SimpleNamespace(RuntimePayload=Runtime))

    def ui(self):
        self._labels = {k: SimpleNamespace(text='') for k in ['mass', 'pose', 'peak', 'status', *names]}

    monkeypatch.setattr(keyboard.KeyboardPayload, '_build_ui', ui)
    monkeypatch.setattr(keyboard.KeyboardPayload, '_build_torque_plot', lambda self:
        setattr(self, '_torque_plot', SimpleNamespace(selected_leg='FL',
            begin_episode=lambda label: None, update=lambda frame: None,
            finish_episode=lambda: None, close=lambda: None)))
    monkeypatch.setattr(keyboard.KeyboardPayload, '_subscribe_keyboard', lambda _: None)
    original_before = keyboard.KeyboardPayload.before_step
    injected = set()
    controllers = []
    original_init = keyboard.KeyboardPayload.__init__

    def init(self, *args):
        original_init(self, *args)
        controllers.append(self)

    monkeypatch.setattr(keyboard.KeyboardPayload, '__init__', init)
    schedule = ({('seed42_episode0', 0): ['LEFT_BRACKET'], ('seed42_episode0', 1): ['R'],
                 ('seed42_episode1', 0): ['RIGHT_BRACKET'], ('seed42_episode1', 1): ['F10']}
                if not fail_first else
                {(None, 1): ['KEY_0', 'RIGHT_BRACKET', 'R'], ('seed42_episode1', 1): ['F10']})

    def before(self, step, dt):
        signature = (self._label, step)
        if signature not in injected:
            injected.add(signature)
            for name in schedule.get(signature, []):
                self._on_keyboard_event(SimpleNamespace(input=SimpleNamespace(name=name),
                    type=SimpleNamespace(name='KEY_PRESS')))
        return original_before(self, step, dt)

    monkeypatch.setattr(keyboard.KeyboardPayload, 'before_step', before)
    monkeypatch.setitem(sys.modules, 'keyboard_payload', keyboard)
    policy_inputs = []

    class Runner:
        def load(self, checkpoint, load_optimizer):
            assert checkpoint == 'frozen_model.pt'
            assert load_optimizer is False

        def get_inference_policy(self, device):
            def policy(obs):
                policy_inputs.append(obs.detach().clone())
                return torch.zeros((1, 16))
            return policy

    class Env:
        unwrapped = base
        device = 'cpu'

        def __init__(self):
            self.reset_count = self.steps = 0
            self.new_observations()

        def new_observations(self):
            base.obs_buf = {'policy': torch.zeros((1, 58)), 'critic': torch.zeros((1, 237))}
            for obs in base.obs_buf.values():
                obs[:, -1] = base.cfg.payload_mass_kg / 65.

        def seed(self, seed):
            assert seed == 42

        def reset(self):
            self.reset_count += 1
            self.new_observations()
            robot.data.root_state_w[0, 2] = .43
            return base.obs_buf['policy'], {}

        def step(self, action):
            self.steps += 1
            assert self.steps < 10, 'Scripted input did not stop the playback loop'
            runtime = base.r6_payload_runtime
            failed = fail_first and self.steps == 1
            height = .05 if failed else .396
            # Recorder captures terminal data before the simulated auto-reset.
            base.r6_review.frames.append({'t': .02 * (len(base.r6_review.frames) + 1),
                'metrics': {'roll_deg': 0., 'pitch_deg': 0., 'height': height},
                'payload_kg': runtime.mass_kg, 'loaded_mass_kg': runtime.total_mass_kg,
                'applied_torque': [0.] * 16, 'terminated': failed, 'timeout': False,
                'policy_observation_before': policy_inputs[-1].clone()})
            if failed:
                robot.data.root_state_w[0, 2] = .43
            self.new_observations()
            return base.obs_buf['policy'], torch.ones(1), torch.tensor([failed]), {}

    env = Env()
    result = play.play_variable_payload(env, Runner(),
        {'checkpoint': 'frozen_model.pt', 'play_seconds': 1., 'seed': 42, 'payload_kg': 65.},
        tmp_path, SimpleNamespace(is_running=lambda: True))
    return SimpleNamespace(result=result, captured=captured, env=env, policy_inputs=policy_inputs,
                           runtime=base.r6_payload_runtime, control=controllers[0], written=written)


def test_payload_change_reset_and_finish_split_trials_with_matching_policy_input(monkeypatch, tmp_path):
    out = exercise(monkeypatch, tmp_path, fail_first=False)
    assert out.runtime.changes == [60., 65.]
    assert out.env.reset_count == 2
    assert [c['reason'] for c in out.captured] == ['user_reset', 'user_finished']
    assert [c['duration_s'] for c in out.result['cases']] == [.02, .02]
    assert out.result['recorded_duration_s'] == .04
    assert out.result['success_rate'] is None
    assert out.result['final_acceptance'] == 'manual_variable_payload_not_standard_screen'
    assert out.result['exit_reason'] == 'user_finished'
    assert out.result['ppo_updates'] == 0
    assert out.control._closed
    for case, mass in zip(out.captured, (60., 65.)):
        frame, = case['frames']
        assert frame['payload_kg'] == mass
        assert frame['policy_observation_before'][0, -1].item() == pytest.approx(mass / 65.)
    verification = read_events(tmp_path / 'payload_after_step.jsonl')
    assert [row['verification']['payload_kg'] for row in verification] == [60., 65.]
    assert [row['label'] for row in verification] == ['seed42_episode0', 'seed42_episode1']


def test_terminal_trial_is_closed_before_between_trial_changes_and_restart(monkeypatch, tmp_path):
    out = exercise(monkeypatch, tmp_path, fail_first=True)
    assert out.runtime.changes == [0., 5.]
    assert out.env.reset_count == 2
    assert out.result['cases'][0]['reason'] == 'terminated'
    assert out.result['cases'][0]['final_payload_kg'] == 65.
    assert out.result['cases'][1]['initial_payload_kg'] == 5.
    assert out.captured[0]['frames'][0]['metrics']['height'] == .05
    assert out.captured[0]['frames'][0]['terminated']
    assert out.captured[1]['frames'][0]['policy_observation_before'][0, -1].item() == pytest.approx(5. / 65.)
    events = read_events(tmp_path / 'keyboard_payload_events.jsonl')
    between = [event for event in events if event['phase'] == 'between_trials']
    assert [row['event'] for row in between] == ['mass_changed', 'mass_changed', 'reset_requested']
    assert all(row['label'] is None and row['simulation_time_s'] is None for row in between)
    first_events = read_events(tmp_path / 'seed42_episode0_payload_events.jsonl')
    assert [row['event'] for row in first_events] == ['episode_begin', 'episode_end']
    assert first_events[-1]['reason'] == 'terminated'
    assert out.result['exit_reason'] == 'user_finished'
    assert out.result['success_rate'] is None
    verified = read_events(tmp_path / 'payload_after_step.jsonl')
    assert len(verified) == 1
    assert verified[0]['label'] == 'seed42_episode1'
    assert verified[0]['verification']['payload_kg'] == 5.

