"""早期Rough线性腿映射和逐关节URDF限位，所有目标仍来自Actor。"""
import hashlib,json
from copy import deepcopy
import torch
from isaaclab.utils import configclass
from .ppo_reference_env_cfg import LightHWReferenceLegAction,LightHWReferenceLegActionCfg
from . import linear_leg_spec as spec


class LightHWLinearURDFLegAction(LightHWReferenceLegAction):
    """复用观测/奖励监控/累计更新管理；旧stage_*_limits不参与目标计算。"""
    def process_actions(self,actions):
        self._raw_actions.copy_(actions)
        self._previous_targets.copy_(self._processed_actions)
        target=self._offset+actions*self._scale
        self._processed_actions.copy_(target.clamp(self._clip[:,:,0],self._clip[:,:,1]))

    def _signature_payload(self):
        return spec.mapped_signature(super()._signature_payload())

    def export_curriculum_state(self):
        state=super().export_curriculum_state()
        state.pop('hip_limit_rad',None);state.pop('leg_limit_rad',None)
        state['linear_leg_mapping']=spec.contract()
        return state

    def validate_curriculum_state(self,state):
        expected=self.export_curriculum_state()
        digest=hashlib.sha256(json.dumps(state.get('action_mapping'),sort_keys=True).encode()).hexdigest()
        if state.get('version')!=1 or state.get('signature')!=expected['signature'] or digest!=state['signature']:
            raise ValueError('Wrong linear-URDF action semantics or corrupt signature')
        if state.get('linear_leg_mapping')!=spec.contract():raise ValueError('Wrong leg linear scale/URDF/defaults')
        restored=deepcopy(self.schedule);restored.load_state_dict(state['schedule'],reset_window=False)
        if state.get('max_speed_m_s')!=restored.max_speeds[restored.stage]:raise ValueError('Wrong command cap')
        steps=state.get('control_steps')
        if isinstance(steps,bool) or not isinstance(steps,int) or steps<0:raise ValueError('Invalid control counter')
        return restored

    def advance_curriculum(self):
        values=super().advance_curriculum()
        values.pop('hip_limit_rad',None);values.pop('leg_limit_rad',None)
        values.update(hip_linear_scale=.08,thigh_calf_linear_scale=.14)
        return values

    def validate_action_layout(self):
        super().validate_action_layout()
        if list(self._joint_names)!=spec.ORDER:raise ValueError('Wrong named linear leg order')
        target=self._clip[0]
        expected=target.new_tensor([spec.LIMITS[n] for n in spec.ORDER])
        torch.testing.assert_close(target,expected,rtol=0,atol=1e-6)
        actual=self._asset.root_physx_view.get_dof_limits()[0,self._joint_ids].to(expected.device)
        torch.testing.assert_close(actual,expected,rtol=0,atol=1e-5)


@configclass
class LightHWLinearURDFLegActionCfg(LightHWReferenceLegActionCfg):
    class_type:type=LightHWLinearURDFLegAction
