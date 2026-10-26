"""R6 回放的实时单腿实际电机力矩图；仅消费记录器数据，不访问物理状态。

历史保存每个 50 Hz 策略帧，窗口按 10 Hz 刷新；切腿立即显示该腿已有历史。
Omni UI 只在创建窗口时导入，历史模型可在没有 Isaac Sim 的 CPU 环境验证。
"""
from collections import deque
import math


LEGS = ('FL', 'FR', 'RL', 'RR')
JOINTS = ('hip', 'thigh', 'calf', 'wheel')
LEG_LABELS = {'FL': 'front left', 'FR': 'front right',
              'RL': 'rear left', 'RR': 'rear right'}
HISTORY_SECONDS = 10.0
MAX_SAMPLES = 501  # 50 Hz 的 10 秒窗口，包含左右端点；限制内存与绘图开销。
DISPLAY_PERIOD = .1


class LegTorqueHistory:
    """按实际 joint_names 映射保留带符号的 applied_torque，不按数组位置猜腿。"""

    def __init__(self, joint_names):
        self.joint_names = tuple(joint_names)
        if len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError('Robot joint names must be unique')
        self.joint_indices = {}
        for leg in LEGS:
            expected = tuple(f'{leg}_{joint}_joint' for joint in JOINTS)
            missing = [name for name in expected if name not in self.joint_names]
            if missing:
                raise ValueError(f'Missing named motor joints: {missing}')
            self.joint_indices[leg] = tuple(self.joint_names.index(name) for name in expected)
        self.selected_leg = 'FL'
        self.episode = None
        self._samples = deque(maxlen=MAX_SAMPLES)

    @property
    def times(self):
        return tuple(sample[0] for sample in self._samples)

    def series(self, leg=None):
        leg = self.selected_leg if leg is None else self._validate_leg(leg)
        return {joint: tuple(sample[1][index] for sample in self._samples)
                for joint, index in zip(JOINTS, self.joint_indices[leg])}

    @staticmethod
    def _validate_leg(leg):
        if leg not in LEGS:
            raise ValueError(f'Unknown leg {leg!r}; expected one of {LEGS}')
        return leg

    def select_leg(self, leg):
        self.selected_leg = self._validate_leg(leg)

    def begin_episode(self, label):
        self.episode = str(label)
        self._samples.clear()

    def update(self, frame):
        if self.episode is None:
            raise RuntimeError('begin_episode must precede torque samples')
        time_s = float(frame['t'])
        if not math.isfinite(time_s) or time_s < 0:
            raise ValueError('Torque frame time must be finite and non-negative')
        if self._samples and time_s <= self._samples[-1][0]:
            raise ValueError('Torque frame time must increase; reset history at episode boundaries')
        torques = tuple(float(value) for value in frame['applied_torque'])
        if len(torques) != len(self.joint_names):
            raise ValueError('Recorded torque count does not match robot joint order')
        if not all(math.isfinite(value) for value in torques):
            raise ValueError('Applied torques must be finite')
        self._samples.append((time_s, torques))
        while self._samples and self._samples[0][0] < time_s - HISTORY_SECONDS - 1e-9:
            self._samples.popleft()


class LiveLegTorquePlot:
    """四幅小图分别显示一条腿的 hip/thigh/calf/wheel 实际力矩，单位 N m。

    公开的 history 与 display_snapshot 分别描述逐帧缓存和最后实际送入绘图器的
    数据，便于从同一份录像轨迹独立核对。各图对称自动缩放，并显示数值刻度。
    """

    def __init__(self, joint_names):
        self.history = LegTorqueHistory(joint_names)
        self._last_display_t = -math.inf
        self._closed = False
        self._window = None
        self._plots, self._zero_plots, self._labels = {}, {}, {}
        self._display_snapshot = None
        try:
            self._build_ui()
            self._redraw()
        except BaseException:
            self.close()
            raise

    @property
    def selected_leg(self):
        return self.history.selected_leg

    @property
    def display_snapshot(self):
        """返回不可变数据的独立容器，修改返回值不会影响窗口或历史。"""
        if self._display_snapshot is None:
            return None
        result = dict(self._display_snapshot)
        result['series'] = dict(result['series'])
        result['scale_limits'] = dict(result['scale_limits'])
        result['joint_names'] = dict(result['joint_names'])
        return result

    def _build_ui(self):
        import omni.ui as ui
        colors = (0xFFFFD45A, 0xFF60C9F2, 0xFF8EE49B, 0xFFE49CBF)
        self._window = ui.Window('R6 Live Leg Torque', width=650, height=650,
                                 position_x=470, position_y=80)
        with self._window.frame:
            with ui.VStack(spacing=3):
                self._labels['leg'] = ui.Label('', height=25)
                ui.Label('1: FL   2: FR   3: RL   4: RR | Click viewport, then press key.', height=21)
                ui.Label('Signed applied torque [N m] | 50 Hz samples / 10 Hz display', height=21)
                self._labels['episode'] = ui.Label('', height=20)
                for joint, color in zip(JOINTS, colors):
                    self._labels[joint] = ui.Label('', height=22, style={'color': color})
                    with ui.HStack(height=91, spacing=4):
                        with ui.VStack(width=58):
                            self._labels[f'{joint}_max'] = ui.Label('', height=18)
                            ui.Spacer()
                            ui.Label('0', height=18)
                            ui.Spacer()
                            self._labels[f'{joint}_min'] = ui.Label('', height=18)
                        with ui.ZStack():
                            self._zero_plots[joint] = ui.Plot(
                                ui.Type.LINE2D, -1., 1., 0., 0., height=91,
                                style={'color': 0xFF626262}, visible=False)
                            self._plots[joint] = ui.Plot(
                                ui.Type.LINE2D, -1., 1., 0., 0., height=91,
                                style={'color': color}, visible=False)
                with ui.HStack(height=20):
                    ui.Spacer(width=62)
                    self._labels['time_start'] = ui.Label('', alignment=ui.Alignment.LEFT)
                    self._labels['time_mid'] = ui.Label('', alignment=ui.Alignment.CENTER)
                    self._labels['time_end'] = ui.Label('', alignment=ui.Alignment.RIGHT)
                ui.Label('Simulation time [s] | Latest 10 s | Reset starts a new trace.', height=21)

    def begin_episode(self, label):
        self._require_open()
        self.history.begin_episode(label)
        self._last_display_t = -math.inf
        self._redraw()

    def finish_episode(self):
        """立即绘出终止帧，等待 R 时仍保留完整末段曲线与当前腿。"""
        self._require_open()
        self._redraw()

    def select_leg(self, leg):
        self._require_open()
        self.history.select_leg(leg)
        self._redraw()  # 切腿即刻重画全部缓存，不等下一次 10 Hz 刷新。

    def update(self, frame):
        self._require_open()
        self.history.update(frame)
        time_s = float(frame['t'])
        if time_s - self._last_display_t >= DISPLAY_PERIOD - 1e-9:
            self._last_display_t = time_s
            self._redraw()

    def _require_open(self):
        if self._closed:
            raise RuntimeError('Live torque plot is closed')

    def _redraw(self):
        leg, times = self.selected_leg, self.history.times
        series = self.history.series()
        scales = {joint: max(1., math.ceil(max(map(abs, values), default=0.) * 1.1))
                  for joint, values in series.items()}
        joint_names = {joint: self.history.joint_names[index]
                       for joint, index in zip(JOINTS, self.history.joint_indices[leg])}
        self._labels['leg'].text = f'{leg} - {LEG_LABELS[leg]} | Actual motor torque'
        self._labels['episode'].text = f'Episode: {self.history.episode or "waiting"}'
        for joint in JOINTS:
            values, bound = series[joint], scales[joint]
            current = f'{values[-1]:+.3f} N m' if values else 'waiting for first step'
            self._labels[joint].text = f'{joint_names[joint]}: {current}'
            self._labels[f'{joint}_max'].text = f'+{bound:g}'
            self._labels[f'{joint}_min'].text = f'-{bound:g}'
            for plot in (self._plots[joint], self._zero_plots[joint]):
                plot.scale_min, plot.scale_max = -bound, bound
                plot.visible = bool(times)
            if times:
                self._plots[joint].set_xy_data(list(zip(times, values)))
                self._zero_plots[joint].set_xy_data([(times[0], 0.), (times[-1], 0.)])
        if times:
            axis = (times[0], (times[0] + times[-1]) / 2., times[-1])
            for key, value in zip(('time_start', 'time_mid', 'time_end'), axis):
                self._labels[key].text = f'{value:.2f} s'
        else:
            for key in ('time_start', 'time_mid', 'time_end'):
                self._labels[key].text = '--'
        self._display_snapshot = dict(
            leg=leg, episode=self.history.episode, times=times, series=series,
            joint_names=joint_names, scale_limits={joint: (-scales[joint], scales[joint]) for joint in JOINTS},
            history_seconds=HISTORY_SECONDS, sample_hz=50, display_hz=10,
            source='signed applied_torque', units='N m')

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self._window is not None:
            self._window.destroy()
            self._window = None
