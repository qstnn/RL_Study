"""R6 变载回放：键盘只入队，策略步之间改变刚体属性并保留审查记录。

本模块不导入训练入口，不控制相机。Carb/Omni 仅在 GUI 实例初始化时导入，
使按键边界、事件顺序、失败留痕和退订可在 CPU 环境独立验证。
"""
from collections import deque
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
from threading import Lock


KEYS = {
    'LEFT_BRACKET': ('delta', -5.0), 'RIGHT_BRACKET': ('delta', 5.0),
    'MINUS': ('delta', -1.0), 'EQUAL': ('delta', 1.0),
    'KEY_0': ('mass', 0.0), 'KEY_6': ('mass', 65.0),
    'R': ('reset', None), 'F10': ('stop', None),
    'KEY_1': ('leg', 'FL'), 'KEY_2': ('leg', 'FR'),
    'KEY_3': ('leg', 'RL'), 'KEY_4': ('leg', 'RR'),
}
ALIASES = {'[': 'LEFT_BRACKET', ']': 'RIGHT_BRACKET', '-': 'MINUS',
           '=': 'EQUAL', '0': 'KEY_0', '6': 'KEY_6',
           '1': 'KEY_1', '2': 'KEY_2', '3': 'KEY_3', '4': 'KEY_4'}


def _enum_name(value):
    return str(getattr(value, 'name', value)).rsplit('.', 1)[-1].upper()


def _mass(value):
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 65.0:
        raise ValueError('Keyboard payload must be finite and within 0..65 kg')
    return value


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, 'tolist'):
        return _json_safe(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class PayloadKeyState:
    """只处理离散按键和载荷边界；不会访问 PhysX、相机或磁盘。"""

    def __init__(self, mass_kg):
        self.mass_kg = _mass(mass_kg)
        self._pending = deque()
        self._lock = Lock()

    def enqueue_key(self, key, event_type='KEY_PRESS'):
        if _enum_name(event_type) != 'KEY_PRESS':
            return False
        key = _enum_name(key)
        key = ALIASES.get(key, key)
        if key not in KEYS:
            return False
        with self._lock:
            self._pending.append(key)
        return True

    def enqueue_event(self, event):
        return self.enqueue_key(event.input, event.type)

    def drain(self):
        with self._lock:
            pending = list(self._pending)
            self._pending.clear()
        return pending

    def clear(self):
        with self._lock:
            self._pending.clear()

    def operation(self, key):
        kind, value = KEYS[key]
        if kind == 'delta':
            return 'mass', min(65.0, max(0.0, self.mass_kg + value))
        return kind, value

    def commit_mass(self, mass_kg):
        self.mass_kg = _mass(mass_kg)


class KeyboardPayload:
    """单环境 GUI 控制器；调用者在每个策略步前调用 before_step。

    reset_requested 交给外层保存本回合并 reset；begin_episode 清除此标志。
    finish_episode 后仍接收变载/R/F10，只写全局 between_trials 事件。
    stop_requested 交给外层正常收尾。任何 setter 异常写入事件后原样抛出。
    """

    def __init__(self, base, run, runtime):
        self.base, self.run, self.runtime = base, Path(run), runtime
        self.state = PayloadKeyState(runtime.mass_kg)
        self.reset_requested = self.stop_requested = False
        self._closed = False
        self._label = None
        self._has_started = False
        self._last_episode_label = None
        self._input = self._keyboard = self._subscription = self._window = None
        self._episode_stream = None
        self._torque_plot = None
        self._labels = {}
        self._step = 0
        self._sim_time = 0.0
        self._last_display_t = -math.inf
        self._event_count = 0
        self._mass_change_count = 0
        self._status_override = None
        self.run.mkdir(parents=True, exist_ok=True)
        self._stream = (self.run / 'keyboard_payload_events.jsonl').open('a', encoding='utf-8')
        robot = base.scene['robot']
        self.joint_names = list(robot.joint_names)
        try:
            self._build_ui()
            self._build_torque_plot()
            self._subscribe_keyboard()
        except BaseException:
            self.close()
            raise
        print('[R6 变载回放] 单击仿真视口后操作：[ / ] 减/加 5 kg；- / = 减/加 1 kg；'
              '0 空载，6 为 65 kg；R 按当前载荷重新开始，F10 保存并结束。'
              '1/2/3/4 切换左前/右前/左后/右后腿力矩图。'
              '每次按下生效一次，载荷限制 0～65 kg；相机仍可自由移动。', flush=True)
        self._refresh_mass_labels()
        self._print_payload('启动', unchanged=False)

    @property
    def mass_kg(self):
        return float(self.runtime.mass_kg)

    def _build_ui(self):
        import omni.ui as ui
        self._window = ui.Window('R6 Payload / Actual Joint Torque', width=440, height=635)
        with self._window.frame:
            with ui.VStack(spacing=3):
                ui.Label('Click simulation viewport to use keys.', height=20)
                ui.Label('[ / ]: -/+5 kg     - / =: -/+1 kg', height=20)
                ui.Label('0: empty    6: 65 kg    R: reset    F10: finish', height=20)
                ui.Label('Torque plot: 1 FL / 2 FR / 3 RL / 4 RR', height=20)
                self._labels['mass'] = ui.Label('', height=25)
                self._labels['pose'] = ui.Label('Roll / Pitch / Height: waiting for first step', height=25)
                self._labels['peak'] = ui.Label('Peak |torque| (instantaneous): waiting', height=25)
                ui.Label('Applied torque [N m], signed; 10 Hz display', height=22)
                for name in self.joint_names:
                    self._labels[name] = ui.Label(f'{name}: --', height=21)
                self._labels['status'] = ui.Label('Policy checkpoint is frozen; no training.', height=24)

    def _build_torque_plot(self):
        from torque_live import LiveLegTorquePlot
        self._torque_plot = LiveLegTorquePlot(self.joint_names)

    def _print_payload(self, key, *, unchanged=False, failed=False):
        """每次已处理的载荷按键反馈一次，使用核验后的实际值；边界按键也反馈。"""
        if failed:
            # setter可能连回滚也失败；缓存值不能被称为当前物理质量。
            print(f'[R6 负载] {key}: 调整失败；上次核验负载 {self.state.mass_kg:.1f} kg。'
                  '当前物理状态未确认，请查看错误诊断。', flush=True)
            return
        suffix = '（已到边界或与当前值相同）' if unchanged else ''
        print(f'[R6 负载] {key}: 当前负载 {self.mass_kg:.1f} kg | '
              f'整机总质量 {float(self.runtime.total_mass_kg):.3f} kg{suffix}', flush=True)

    def _subscribe_keyboard(self):
        import carb.input
        import omni.appwindow
        window = omni.appwindow.get_default_app_window()
        if window is None:
            raise RuntimeError('Keyboard payload requires an Isaac Sim GUI window')
        self._keyboard = window.get_keyboard()
        if self._keyboard is None:
            raise RuntimeError('Isaac Sim GUI did not provide a keyboard device')
        self._input = carb.input.acquire_input_interface()
        self._subscription = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_keyboard_event)

    def _on_keyboard_event(self, event, *_):
        # 回调可在 render 内发生；这里只入队，不写物理属性或执行 reset。
        if not self._closed and self._has_started:
            self.state.enqueue_event(event)
        return True

    def _emit(self, event):
        between = self._label is None
        row = _json_safe(dict(schema_version=1, label=self._label,
                              phase='between_trials' if between else 'trial',
                              after_trial_label=self._last_episode_label if between else None,
                              step=None if between else self._step,
                              simulation_time_s=None if between else self._sim_time,
                              recorded_at=datetime.now(timezone.utc).isoformat(), **event))
        line = json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n'
        self._stream.write(line)
        self._stream.flush()
        if self._episode_stream is not None:
            self._episode_stream.write(line)
            self._episode_stream.flush()
        self._event_count += 1

    def begin_episode(self, label):
        if self._closed:
            raise RuntimeError('Keyboard payload controller is closed')
        if self._label is not None:
            raise RuntimeError('Finish the current keyboard payload episode before starting another')
        if not re.fullmatch(r'[A-Za-z0-9_-]+', str(label)):
            raise ValueError('Episode label must be a simple filename component')
        pending = self.state.drain()
        # reset 的 render 期间也可能收到 F10；清理旧按键时必须保留退出请求。
        if 'F10' in pending and not self.stop_requested:
            self.stop_requested = True
            self._emit({'event': 'stop_requested', 'key': 'F10',
                        'old_mass_kg': self.mass_kg, 'new_mass_kg': self.mass_kg})
        self._label = str(label)
        self._has_started = True
        self.state.commit_mass(self.runtime.mass_kg)
        self.reset_requested = False
        self._step, self._sim_time = 0, 0.0
        self._last_display_t = -math.inf
        self._event_count = self._mass_change_count = 0
        self._status_override = None
        self._initial_mass_kg = self.mass_kg
        self._torque_plot.begin_episode(self._label)
        self._episode_stream = (self.run / f'{self._label}_payload_events.jsonl').open('w', encoding='utf-8')
        self._emit({'event': 'episode_begin', 'old_mass_kg': self.mass_kg, 'new_mass_kg': self.mass_kg})
        self._refresh_mass_labels()

    def before_step(self, step, dt):
        if self._closed or not self._has_started:
            raise RuntimeError('begin_episode must precede before_step')
        self._step, self._sim_time = int(step), int(step) * float(dt)
        changed = False
        for key in self.state.drain():
            if self.stop_requested:
                break
            if self.reset_requested and key != 'F10':
                continue
            kind, target = self.state.operation(key)
            old = self.mass_kg
            event = {'event': kind, 'key': key, 'old_mass_kg': old, 'new_mass_kg': old}
            if kind == 'mass':
                event['new_mass_kg'] = target
                if target == old:
                    event['event'] = 'mass_unchanged'
                else:
                    try:
                        record = self.runtime.set_mass(target)
                        self.state.commit_mass(self.runtime.mass_kg)
                        if self.state.mass_kg != target:
                            raise RuntimeError('Runtime payload did not retain the requested mass')
                    except BaseException as error:
                        self._emit(dict(event, event='mass_change_failed', error_type=type(error).__name__,
                                        error=str(error), runtime_mass_kg=float(self.runtime.mass_kg)))
                        self._print_payload(key, failed=True)
                        raise
                    event.update(event='mass_changed', runtime_verification=record)
                    self._mass_change_count += 1
                    changed = True
                self._print_payload(key, unchanged=(target == old))
            elif kind == 'leg':
                previous_leg = self._torque_plot.selected_leg
                self._torque_plot.select_leg(target)
                event.update(event='plot_leg_selected', previous_leg=previous_leg, selected_leg=target)
                leg_name = {'FL': '左前腿', 'FR': '右前腿', 'RL': '左后腿', 'RR': '右后腿'}[target]
                print(f'[R6 力矩图] {key}: {leg_name} ({target})，显示该腿四个电机的实际力矩。', flush=True)
            elif kind == 'reset':
                self.reset_requested = True
                event['event'] = 'reset_requested'
            else:
                self.stop_requested = True
                event['event'] = 'stop_requested'
            self._emit(event)
        self._refresh_mass_labels()
        return changed

    def _refresh_mass_labels(self):
        if 'mass' in self._labels:
            self._labels['mass'].text = (f'Payload: {self.mass_kg:.1f} / 65.0 kg   |   '
                                         f'Total: {float(self.runtime.total_mass_kg):.3f} kg')
        if 'status' in self._labels:
            status = 'Finishing and saving...' if self.stop_requested else (
                'Reset requested; current payload retained.' if self.reset_requested else
                self._status_override or 'Policy checkpoint is frozen; no training.')
            self._labels['status'].text = status

    def set_status(self, text=None):
        """外层可提示等待 R / 正在运行；不会修改控制或物理状态。"""
        self._status_override = None if text is None else str(text)
        self._refresh_mass_labels()

    def update_display(self, frame):
        """读取记录器帧，不额外读/改物理状态；使用实际 applied_torque。"""
        # 图形保存每个策略帧；数值标签和图形重绘分别限为10Hz。
        self._torque_plot.update(frame)
        self._sim_time = float(frame['t'])
        if self._sim_time - self._last_display_t < .1 - 1e-9:
            return
        self._last_display_t = self._sim_time
        values = frame['metrics']
        self._labels['pose'].text = (f'Roll {float(values["roll_deg"]):+.2f} deg   '
                                     f'Pitch {float(values["pitch_deg"]):+.2f} deg   '
                                     f'Height {float(values["height"]):.3f} m')
        torques = [float(v) for v in frame['applied_torque']]
        if len(torques) != len(self.joint_names):
            raise ValueError('Recorded torque count does not match robot joint order')
        leg, wheel = [], []
        for name, torque in zip(self.joint_names, torques):
            self._labels[name].text = f'{name:<20s} {torque:+8.3f}'
            (wheel if 'wheel' in name.lower() else leg).append(abs(torque))
        self._labels['peak'].text = (f'Peak |torque|: legs {max(leg, default=0.):.2f} N m   '
                                     f'wheels {max(wheel, default=0.):.2f} N m')
        self._refresh_mass_labels()

    def finish_episode(self, reason):
        if self._label is None:
            return
        self._torque_plot.finish_episode()  # 失败等待时也显示最后一帧，不滞后于归档终帧。
        self._emit({'event': 'episode_end', 'reason': str(reason),
                    'old_mass_kg': self.mass_kg, 'new_mass_kg': self.mass_kg})
        metadata = {'schema_version': 1, 'label': self._label, 'reason': str(reason),
                    'initial_payload_kg': self._initial_mass_kg, 'final_payload_kg': self.mass_kg,
                    'last_step': self._step, 'simulation_time_s': self._sim_time,
                    'event_count': self._event_count, 'mass_change_count': self._mass_change_count,
                    'mass_range_kg': [0., 65.], 'reset_requested': self.reset_requested,
                    'stop_requested': self.stop_requested, 'joint_order': self.joint_names,
                    'events': f'{self._label}_payload_events.jsonl',
                    'mass_change_model': 'instantaneous rigid-body mass/COM/inertia change; not object placement',
                    'live_torque_plot': {'selected_leg': self._torque_plot.selected_leg,
                                         'keys': {'1': 'FL', '2': 'FR', '3': 'RL', '4': 'RR'},
                                         'window_seconds': 10, 'display_hz': 10,
                                         'source': 'every recorded policy frame, signed applied_torque',
                                         'reset_clears_history': True},
                    'display_hz': 10, 'display_torque': 'signed applied torque; maxima across joints at displayed frame'}
        (self.run / f'{self._label}_payload.json').write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        self._episode_stream.close()
        self._episode_stream = None
        self._last_episode_label = self._label
        self._label = None
        # 最后一次 render 收到的按键留给 between_trials；不写已关闭的采样记录。

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self.finish_episode('controller_closed')
        finally:
            try:
                if self._input is not None and self._subscription is not None:
                    unsubscribe = getattr(self._input, 'unsubscribe_to_keyboard_events', None)
                    if unsubscribe is None:
                        unsubscribe = self._input.unsubscribe_from_keyboard_events
                    unsubscribe(self._keyboard, self._subscription)
            finally:
                self._subscription = None
                if self._episode_stream is not None:
                    self._episode_stream.close()
                    self._episode_stream = None
                self._stream.close()
                try:
                    if self._torque_plot is not None:
                        self._torque_plot.close()
                finally:
                    if self._window is not None:
                        self._window.destroy()
                        self._window = None
