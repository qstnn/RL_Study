"""LightHW 前进课程的终端摘要；只读取训练统计，不参与奖励或升阶决策。"""

from __future__ import annotations

from contextlib import contextmanager, redirect_stdout
from dataclasses import dataclass
import math
import os
import re
import shutil
import sys
import unicodedata


class _NativeSummaryFilter:
    """仅过滤原生 RSL 的整块训练摘要，其他输出（包括 warning）原样透传。

    原生 log 仍执行所有 writer.add_scalar 和时间计数。不识别的输出格式自动
    回退为完整输出；不重写或 monkey-patch 安装目录中的 RSL-RL。
    """

    def __init__(self, stream, width):
        self.stream, self.width = stream, width
        self._skip_newline = False

    def write(self, text):
        if (text.startswith("#" * self.width + "\n") and "Learning iteration" in text
                and "Total timesteps:" in text and "Computation:" in text):
            self._skip_newline = True
            return len(text)
        if self._skip_newline and text == "\n":
            self._skip_newline = False
            return len(text)
        self._skip_newline = False
        return self.stream.write(text)

    def __getattr__(self, name):
        return getattr(self.stream, name)


@contextmanager
def compact_native_summary(width=80):
    # 作用域仅覆盖单次原生 log 调用；课程更新、其他启动输出和 stderr 不经过过滤。
    with redirect_stdout(_NativeSummaryFilter(sys.stdout, width)):
        yield


@dataclass(frozen=True)
class GateCheck:
    key: str
    label: str
    value: float | None
    limit: float
    minimum: bool = False
    unit: str = ""

    @property
    def valid(self):
        return self.value is not None and math.isfinite(self.value) and self.value >= 0

    @property
    def gap(self):
        """正数表示还需改善多少；负数表示达标余量，单位与原指标相同。"""
        if not self.valid:
            return math.inf
        return self.limit - self.value if self.minimum else self.value - self.limit

    @property
    def passed(self):
        return self.gap <= 0

    def description(self):
        if not self.valid:
            return f"{self.label} 数据缺失/无效"
        if self.unit == "%":
            return f"{self.label} {100*self.value:.3f}% {'还差' if self.minimum else '超出'} {_positive_gap(100*self.gap, 3)}个百分点"
        if self.unit == "count":
            return f"{self.label} {self.value:.0f}/{self.limit:.0f}"
        return f"{self.label} {self.value:.4f} {'还差' if self.minimum else '超出'} {_positive_gap(self.gap, 4)} {self.unit}"


def _positive_gap(value, digits):
    formatted = f"{value:.{digits}f}"
    return f"{value:.2g}" if value > 0 and float(formatted) == 0 else formatted


GROUP_LABELS = {"forward": "前进", "fast": "上限组", "stop": "停车"}


def quality_checks(metrics, *, max_speed, parameters, min_samples):
    """展示与 forward_quality_gate 相同的全部门槛；不把未完成窗口当作正式通过。

    原课程函数仍是唯一升速决策者。回归测试逐项比较这里的展示与生产门控，
    防止遗漏偏航、横漂、样本量等导致“显示达标但课程不动”。
    """
    gate, checks = parameters, []
    if not math.isfinite(max_speed) or max_speed <= 0:
        checks.append(GateCheck("max_speed", "阶段速度上限", None, 0))
    for group, label in GROUP_LABELS.items():
        moving = group != "stop"
        prefix = "moving" if moving else "stop"
        limits = (
            ("samples", "样本", min_samples, True, "count"),
            ("speed_error", "速度误差" if moving else "漂移速度", max(gate["moving_speed_error_min"],
                gate["moving_speed_error_fraction"] * max_speed) if moving else gate["stop_speed_error"], False, "m/s"),
            ("four_wheel_support", "四轮支撑", gate[f"{prefix}_four_wheel_support"], True, "%"),
            ("tilt_bad_fraction", "倾角超限", gate[f"{prefix}_tilt_bad_fraction"], False, "%"),
            ("body_contact_fraction", "非轮接触", gate["body_contact_fraction"], False, "%"),
            ("yaw_rate", "偏航角速度", gate[f"{prefix}_yaw_rate"], False, "rad/s"),
            ("lateral_speed", "横向速度", max(gate["moving_lateral_min"], gate["moving_lateral_fraction"] * max_speed)
                if moving else gate["stop_lateral_speed"], False, "m/s"),
        )
        for key, name, limit, minimum, unit in limits:
            full_key = f"{group}/{key}"
            checks.append(GateCheck(full_key, f"{label}{name}", metrics.get(full_key), limit, minimum, unit))
        # 与生产 gate 保持一致：即使是“越高越好”，支撑率也不能超过 100%。
        key = f"{group}/four_wheel_support"
        checks.append(GateCheck(key + "_valid", f"{label}支撑率有效范围", metrics.get(key), 1.0, unit="%"))
    checks.extend((
        GateCheck("completed_episodes", "完整回合", metrics.get("completed_episodes"), gate["min_completed_episodes"], True, "count"),
        GateCheck("failed_episode_fraction", "失败回合比例", metrics.get("failed_episode_fraction"), gate["failed_episode_fraction"], unit="%"),
        GateCheck("nonfinite_samples", "NaN/Inf样本", metrics.get("nonfinite_samples"), 0, unit="count"),
    ))
    return checks


def _number(metrics, key, digits=3, *, factor=1, signed=False):
    value = metrics.get(key)
    if value is None or not math.isfinite(value):
        return "--"
    return format(value * factor, f"{'+' if signed else ''}.{digits}f")


def _display_width(text):
    """纯文本的终端列宽：中文占两列，组合音标/格式字符不单独占列。"""
    return sum(0 if unicodedata.category(char) in ("Mn", "Me", "Cf") else
               2 if unicodedata.east_asian_width(char) in ("W", "F") else 1
               for char in text)


def _terminal_columns():
    # 每次摘要重新读取，窗口缩放后即可生效；重定向/无TTY时仍输出普通文本。
    try:
        return os.get_terminal_size(sys.stdout.fileno()).columns
    except (AttributeError, OSError, ValueError):
        return shutil.get_terminal_size(fallback=(120, 24)).columns


def center_summary_rows(rows, *, width=None):
    """每组一行并按可见列居中；窄终端按字段换行，不截去指标或门槛。

    不发光标/清屏控制码，warning及历史输出保持可追溯。预留两列避免刚好
    占满终端触发自动换行；超宽字段才按字符分段，续行重复当前组标签。
    """
    columns = max(4, int(_terminal_columns() if width is None else width))
    available = columns - 2
    output = []
    for row in rows:
        row = row.strip()
        if not row:
            output.append("")
            continue
        match = re.match(r"(\[[^\]]+\])\s*", row)
        prefix = match.group(1) + " " if match else ""
        # 极窄窗口容不下组名时，将完整文本分段，仍保留全部内容。
        if _display_width(prefix) >= available:
            prefix = ""
        body = row[len(match.group(0)):] if match and prefix else row
        fields = re.split(r" \| |；", body)
        current = prefix
        wrapped = []
        for field in fields:
            candidate = current + (" | " if current != prefix else "") + field
            if current != prefix and _display_width(candidate) > available:
                wrapped.append(current)
                current = prefix
            if current != prefix:
                current += " | "
            for char in field:
                if _display_width(current) + _display_width(char) > available:
                    wrapped.append(current)
                    current = prefix
                current += char
        if current != prefix:
            wrapped.append(current)
        for line in wrapped:
            output.append(" " * max(0, (columns - _display_width(line)) // 2) + line)
    return "\n".join(output)


def format_grouped_exploration(data, *, width=None):
    """腿/轮探索各一行；保留当前批、条件目标估计和轮目标裁剪前的含义。"""
    return center_summary_rows([
        f"[腿部探索/当前批] raw σ={data['leg_raw_std_mean']:.3f} | "
        f"目标σ估计={data['leg_target_std_rad_mean']:.6f} rad | "
        f"12腿均值贴边={100*data['leg_saturation_fraction']:.1f}%",
        f"[轮部探索/裁剪前] 运动目标σ={data['wheel_moving_target_std_rad_s']:.3f} rad/s | "
        f"停车目标σ={data['wheel_stop_target_std_rad_s']:.3f} rad/s",
    ], width=width)


def format_training_summary(*, metrics, context, progress, training, width=None):
    """所有窗口数值对应 evaluated_stage，升阶后的新上限只显示在进度行。"""
    gate = context["gate_parameters"]
    cap, stage = context["max_speed"], context["stage"]
    checks = quality_checks(metrics, max_speed=cap, parameters=gate, min_samples=context["min_samples"])
    limits = {check.key: check.limit for check in checks}
    lines = [
        f"[LightHW PPO] iteration {training['iteration']}/{training['last_iteration']} | "
        f"累计更新 {progress['total_updates']}",
        f"[阶段] 本窗口阶段 {stage}/{context['last_stage']} vx=0~{cap:.2f} m/s",
        f"[窗口] {'完整评估' if context['due'] else '采集中，非正式判定'} "
        f"{context['window_updates']}/{context['evaluation_interval']} 更新",
        f"[统计口径] 50 Hz 含探索噪声 | 命令稳定 {gate['settle_seconds']:g}s | "
        f"上限组 cmd≥{gate['fast_command_fraction']*cap:.2f} m/s",
    ]
    for group, label in GROUP_LABELS.items():
        n = metrics.get(f"{group}/samples", 0)
        if not math.isfinite(n) or n <= 0:
            lines.append(f"[{label}速度] 等待稳定样本，实际速度/接触/姿态 --")
            continue
        def val(key, digits=3, **kwargs):
            return _number(metrics, f"{group}/{key}", digits, **kwargs)
        error = "平面漂移" if group == "stop" else "vx MAE"
        lines.append(
            f"[{label}速度] cmd/vx={val('command_vx')}/{val('actual_vx', signed=True)} m/s | "
            f"{error} {val('speed_error', 4)} (≤{limits[group+'/speed_error']:.3f})"
        )
    def grouped(key, digits=3, factor=1):
        return "/".join(_number(metrics, f"{group}/{key}", digits, factor=factor)
            if metrics.get(f"{group}/samples", 0) > 0 else "--" for group in GROUP_LABELS)
    lines.extend((
        "[分组顺序] 以下三组数据依次为：前进 / 上限组 / 停车",
        f"[四轮支撑] {grouped('four_wheel_support', factor=100)}% | "
        f"门槛≥{100*limits['forward/four_wheel_support']:.0f}/"
        f"{100*limits['fast/four_wheel_support']:.0f}/{100*limits['stop/four_wheel_support']:.0f}%",
        f"[非轮接触] {grouped('body_contact_fraction', factor=100)}% | 门槛≤{100*gate['body_contact_fraction']:g}%",
        f"[机身倾角] 平均 {grouped('tilt_deg', 2)}° | "
        f"超限界限 {gate['moving_tilt_degrees']:g}/{gate['moving_tilt_degrees']:g}/{gate['stop_tilt_degrees']:g}°",
        f"[倾角超限] {grouped('tilt_bad_fraction', factor=100)}%",
        f"[偏航角速度] |wz|={grouped('yaw_rate')} rad/s",
        f"[横向漂移] |vy|={grouped('lateral_speed')} m/s",
        f"[稳定样本] n={grouped('samples', 0)} | 每组要求≥{context['min_samples']:.0f}",
        f"[回合] 完成 {_number(metrics, 'completed_episodes', 0)} | 失败 {_number(metrics, 'failed_episode_fraction', factor=100)}% "
        f"| NaN/Inf {_number(metrics, 'nonfinite_samples', 0)}",
    ))
    blockers = [check for check in checks if not check.passed]
    # 没有稳定样本时不把空窗口的零均值解释为速度/接触质量。
    blockers = [check for check in blockers if '/' not in check.key or check.key.endswith('/samples')
        or metrics.get(check.key.split('/')[0] + '/samples', 0) > 0]
    if blockers:
        for check in blockers:
            lines.append("[升速差距] " + check.description())
    else:
        lines.append("[升速差距] 当前统计各项均达标" + ("。" if context['due'] else "，仍需完整窗口确认。"))
    if progress["stage"] != stage:
        lines.append(f"[课程] 本窗口通过，已升至阶段 {progress['stage']}，vx上限 {progress['max_speed']:.2f} m/s | 新窗口重新采集。")
    elif stage == context["last_stage"]:
        lines.append("[课程] 已处于最终速度阶段 | 实际效果仍需回放验收。")
    else:
        remaining = max(0, context["min_stage_updates"] - progress["stage_updates"])
        passes = min(progress["consecutive_passes"], context["required_passes"])
        decision = ("通过" if metrics.get("window_passed") else "未通过") if context["due"] else "尚未判定"
        lines.append(f"[课程] 本窗口{decision} | 连续通过 {passes}/{context['required_passes']} | "
            f"阶段最少更新还差 {remaining} | 下一上限 {context['next_speed']:.2f} m/s")
    lines.extend((
        f"[PPO] reward={training['reward']} | lr={training['learning_rate']:.2g}",
        f"[运行] {training['fps']} steps/s | ETA={training['eta']}",
    ))
    if not training.get("grouped_noise", False):
        lines.append(f"[探索噪声] 腿raw σ={training['leg_std']:.3f} | "
                     f"轮目标σ(裁剪前)={training['wheel_std']:.3f} rad/s")
    return "\n" + center_summary_rows(lines, width=width)
