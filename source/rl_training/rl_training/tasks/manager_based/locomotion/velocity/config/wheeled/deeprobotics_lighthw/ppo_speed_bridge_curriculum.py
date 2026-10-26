"""PPO-only refinement of the route-11 policy's validated 0.8 m/s stage."""

from __future__ import annotations

from isaaclab.utils import configclass

from .ppo_forward_speed_curriculum import LightHWForwardSpeedLegAction, LightHWForwardSpeedLegActionCfg
from .ppo_speed_bridge_core import BRIDGE_SPEEDS, PROFILE_ID, validate_route11_transfer_state


class LightHWSpeedBridgeLegAction(LightHWForwardSpeedLegAction):
    """复用route 11的动作/质量采样，只细化0.8到1.2之间的速度课程。"""

    def _signature_payload(self) -> dict:
        payload = super()._signature_payload()
        payload["profile_id"] = PROFILE_ID
        return payload

    def validate_route11_transfer(self, state: dict) -> dict:
        return validate_route11_transfer_state(state, self._signature_payload())

    def initialize_route11_transfer(self, state: dict):
        """受检迁移后在0.8重新积累窗口；保持普通缓冲，避免推理模式写入错误。"""
        source = self.validate_route11_transfer(state)
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
        print("[LightHW PPO Speed Bridge] route 11 stage-2 policy transferred; "
              "0.8 -> 1.0 -> 1.2 m/s, stage=0, counters=0; unchanged actions/rewards/gates", flush=True)


@configclass
class LightHWSpeedBridgeLegActionCfg(LightHWForwardSpeedLegActionCfg):
    class_type: type = LightHWSpeedBridgeLegAction
    stage_hip_limits: tuple[float, ...] = (0.04,) * len(BRIDGE_SPEEDS)
    stage_leg_limits: tuple[float, ...] = (0.10,) * len(BRIDGE_SPEEDS)
    stage_max_speeds: tuple[float, ...] = BRIDGE_SPEEDS
