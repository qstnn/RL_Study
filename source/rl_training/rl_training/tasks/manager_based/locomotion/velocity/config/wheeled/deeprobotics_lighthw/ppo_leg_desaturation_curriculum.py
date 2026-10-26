"""Route-14 action metadata: unchanged actions plus a checked reward identity."""

from __future__ import annotations

from isaaclab.utils import configclass

from .adapted_rewards import lighthw_joint_action_saturation_l1
from .ppo_grouped_noise_curriculum import LightHWGroupedNoiseLegAction, LightHWGroupedNoiseLegActionCfg
from .ppo_leg_desaturation_core import (
    DESATURATION_REWARD_NAME,
    PROFILE_ID,
    validate_desaturation_reward_cfg,
    validate_route12_transfer_state,
)


class LightHWLegDesaturationLegAction(LightHWGroupedNoiseLegAction):
    """保留16维PPO动作；将实际新增奖励纳入检查点签名，禁止跨奖励静默恢复。"""

    def _signature_payload(self) -> dict:
        payload = super()._signature_payload()
        configured = getattr(self._env.cfg.rewards, DESATURATION_REWARD_NAME, None)
        fingerprint = validate_desaturation_reward_cfg(configured, lighthw_joint_action_saturation_l1)
        # runner创建时reward manager已经就绪；此时额外核对真正执行的term，
        # 包括运行中set_term_cfg造成的变化，而非只相信最初env cfg中的常量。
        manager = getattr(self._env, "reward_manager", None)
        if manager is not None:
            if DESATURATION_REWARD_NAME not in manager.active_terms:
                raise ValueError("Leg-desaturation reward is absent from the active reward manager")
            actual = manager.get_term_cfg(DESATURATION_REWARD_NAME)
            fingerprint = validate_desaturation_reward_cfg(actual, lighthw_joint_action_saturation_l1)
        payload.update(profile_id=PROFILE_ID, leg_desaturation_reward=fingerprint)
        return payload

    def validate_route12_transfer(self, state: dict) -> dict:
        return validate_route12_transfer_state(state, self._signature_payload())

    def initialize_route12_transfer(self, state: dict):
        """同源model900迁移后重收窗口；仅新增八关节raw惩罚，动作保持原样。"""
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
        print("[LightHW PPO Leg Desaturation] checked route12-stage1 transfer; "
              "hold at 1.0 m/s, counters=0; same grouped noise/actions/gates; "
              "added thigh/calf raw-action L1 reward", flush=True)


@configclass
class LightHWLegDesaturationLegActionCfg(LightHWGroupedNoiseLegActionCfg):
    class_type: type = LightHWLegDesaturationLegAction
