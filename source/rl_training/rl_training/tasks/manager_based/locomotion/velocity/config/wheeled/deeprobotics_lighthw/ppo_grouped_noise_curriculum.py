"""Unchanged PPO actions and quality gates for the grouped exploration route."""

from __future__ import annotations

from isaaclab.utils import configclass

from .ppo_forward_speed_curriculum import LightHWForwardSpeedLegAction, LightHWForwardSpeedLegActionCfg
from .ppo_grouped_noise_core import GROUPED_NOISE_SPEEDS, PROFILE_ID, validate_route12_transfer_state


class LightHWGroupedNoiseLegAction(LightHWForwardSpeedLegAction):
    """保持12个腿动作与4个轮动作，固定1.0上限以单独观察分组探索效果。"""

    def _signature_payload(self) -> dict:
        payload = super()._signature_payload()
        payload["profile_id"] = PROFILE_ID
        return payload

    def validate_route12_transfer(self, state: dict) -> dict:
        return validate_route12_transfer_state(state, self._signature_payload())

    def initialize_route12_transfer(self, state: dict):
        """校验后重收1.0窗口；保留普通缓冲，兼容inference_mode内外的写入。"""
        source = self.validate_route12_transfer(state)
        self.schedule.load_state_dict({
            "stage": 0, "stage_updates": 0, "total_updates": 0,
            "consecutive_passes": 0, "window_updates": 0,
        }, reset_window=True)
        self._transfer_source = source
        self._control_steps = 0
        self._set_limits()
        self._training_curriculum = True
        self.last_quality_metrics = {}
        self._clear_quality_window()
        self._apply_command_stage(resample=True)
        print("[LightHW PPO Grouped Noise] route 12 stage-1 policy transferred; "
              "hold at 1.0 m/s, stage=0, counters=0; unchanged actions/rewards/gates", flush=True)


@configclass
class LightHWGroupedNoiseLegActionCfg(LightHWForwardSpeedLegActionCfg):
    class_type: type = LightHWGroupedNoiseLegAction
    stage_hip_limits: tuple[float, ...] = (0.04,)
    stage_leg_limits: tuple[float, ...] = (0.10,)
    stage_max_speeds: tuple[float, ...] = GROUPED_NOISE_SPEEDS
    leg_noise_std: float = 0.12
    wheel_noise_std: float = 0.25
