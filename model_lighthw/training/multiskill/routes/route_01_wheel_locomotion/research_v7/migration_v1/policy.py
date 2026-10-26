"""隔离的部署均值容器：冻结旧策略，仅给名义拖载增加可训练残差。

300 -> 16 是同一个部署图，内部为 frozen base + towing residual，
不是旧 V8 的单 MLP。没有教师、仿真、动作采样、训练入口或准入授权。
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from .contract import CONTRACT, SPEC_PATH, POSITION_SOURCE, install_actor_mask, sha256, validate_actor_mask


VERSION = 'r1_migration_v1_frozen_base_towing_residual'
INPUT_DIM = CONTRACT['policy_dim']
OUTPUT_DIM = CONTRACT['action_dim']
CONTEXT_START, CONTEXT_END = CONTRACT['layout']['task_context']['latest']
PAYLOAD_COLUMN, TOWED_COLUMN = range(CONTEXT_START, CONTEXT_END)
MASK_START, MASK_END = CONTRACT['ignored_policy_columns']


def _validate_network(network):
    """只接受已核验的有限 float32 ELU MLP，排除训练态会改变基线的层。"""
    widths = [INPUT_DIM, *CONTRACT['hidden_dims'], OUTPUT_DIM]
    if not isinstance(network, nn.Sequential) or len(network) != 7:
        raise ValueError('Expected the verified 300/512/256/128/16 ELU Sequential')
    for i, (left, right) in enumerate(zip(widths, widths[1:])):
        layer = network[2*i]
        if type(layer) is not nn.Linear or layer.in_features != left or layer.out_features != right or layer.bias is None:
            raise ValueError('Actor linear layer differs from the deployed architecture')
        if i < 3 and (type(network[2*i+1]) is not nn.ELU or network[2*i+1].alpha != 1.):
            raise ValueError('Actor activation must be the original ELU(alpha=1)')
    for value in network.state_dict().values():
        if value.dtype != torch.float32 or not torch.isfinite(value).all():
            raise ValueError('Actor state must be finite float32')


def tensor_fingerprint(network):
    """与设备无关的实际张量 SHA；不冒充尚未验证的 checkpoint 来源 SHA。"""
    digest = hashlib.sha256()
    for name, value in sorted(network.state_dict().items()):
        cpu = value.detach().cpu().contiguous()
        digest.update(json.dumps([name, list(cpu.shape), str(cpu.dtype)], separators=(',', ':')).encode())
        digest.update(cpu.numpy().tobytes())
    return digest.hexdigest()


def _zero_residual(base):
    # 同一结构便于复核原 mask 合同；不复制 base 权重作为新的非零残差。
    widths = [INPUT_DIM, *CONTRACT['hidden_dims'], OUTPUT_DIM]
    layers = []
    for i, (left, right) in enumerate(zip(widths, widths[1:])):
        layers.append(nn.Linear(left, right))
        if i < 3:
            layers.append(nn.ELU())
    first = next(base.parameters())
    result = nn.Sequential(*layers).to(device=first.device, dtype=first.dtype)
    with torch.no_grad():
        result[-1].weight.zero_()
        result[-1].bias.zero_()
    install_actor_mask(result)
    return result


class FrozenBaseTowingResidualActor(nn.Module):
    """可部署 300 维输入、16 维原始动作均值；只训练拖载修正支路。

    输入仍为原始部署观测，调用者不传接触力、坡度真值或教师阶段。
    nominal towed context > 0 激活修正；非拖载直接选择旧输出。
    base 必须先来自已确认的旧策略，且原80列 mask 已成立；本类不替调用者
    修改旧策略以制造一致性，也不加载或认证任何真实 checkpoint。
    """
    def __init__(self, base_actor):
        super().__init__()
        _validate_network(base_actor)
        validate_actor_mask(base_actor)
        self.base_actor = deepcopy(base_actor).eval()
        self.base_actor.requires_grad_(False)
        self.frozen_base_sha256 = tensor_fingerprint(self.base_actor)
        self.residual_actor = _zero_residual(self.base_actor)
        self.implementation_sha256 = sha256(Path(__file__))
        self.contract_source_sha256 = {
            'wheel_unified_v6/spec.py': sha256(SPEC_PATH),
            'reference_observations.py': sha256(POSITION_SOURCE),
            'migration_v1/contract.py': sha256(Path(__file__).with_name('contract.py')),
        }
        self.assert_invariants()

    def train(self, mode=True):
        super().train(mode)
        self.base_actor.eval()
        return self

    def forward(self, observations):
        if observations.ndim != 2 or observations.shape[-1] != INPUT_DIM:
            raise ValueError('Expected deployed observations with shape [N, 300]')
        if observations.dtype != torch.float32 or not torch.isfinite(observations).all():
            raise ValueError('Expected finite float32 deployed observations')
        payload = observations[:, PAYLOAD_COLUMN]
        towed = observations[:, TOWED_COLUMN]
        if ((payload < 0) | (payload > 1) | (towed < 0) | (towed > 1) | ((payload > 0) & (towed > 0))).any():
            raise ValueError('Nominal payload/towed context violates the mutually exclusive R1 contract')
        # 权重/梯度/Adam mask之外，部署也显式屏蔽80列，禁止反馈依赖教师动作历史。
        visible = observations.clone()
        visible[:, MASK_START:MASK_END] = 0.
        base = self.base_actor(visible)
        delta = self.residual_actor(visible)
        # 直接选择非拖载输出；不用 base + 0*delta，避免NaN乘0污染保护分支。
        result = torch.where((towed > 0).unsqueeze(-1), base+delta, base)
        if not torch.isfinite(result).all():
            raise ValueError('Active deployed Actor output is nonfinite')
        return result

    def trainable_parameters(self):
        """供未来独立拟合/PPO适配器使用；这里不创建优化器或执行更新。"""
        return (parameter for parameter in self.residual_actor.parameters() if parameter.requires_grad)

    def project_constraints(self, optimizer=None):
        """未来更新后恢复残差 mask/Adam；旧 base 一旦变化直接报错，不静默修补。"""
        install_actor_mask(self.residual_actor, optimizer)
        self.assert_invariants(optimizer)

    def assert_invariants(self, optimizer=None):
        _validate_network(self.base_actor)
        _validate_network(self.residual_actor)
        validate_actor_mask(self.base_actor)
        validate_actor_mask(self.residual_actor, optimizer)
        if self.base_actor.training or any(parameter.requires_grad for parameter in self.base_actor.parameters()):
            raise ValueError('Protected base must stay frozen and in evaluation mode')
        if tensor_fingerprint(self.base_actor) != self.frozen_base_sha256:
            raise ValueError('Protected base tensors changed')
        if optimizer is not None:
            protected = {id(parameter) for parameter in self.base_actor.parameters()}
            if any(id(parameter) in protected for group in optimizer.param_groups for parameter in group['params']):
                raise ValueError('Optimizer must exclude the protected base')

    def get_extra_state(self):
        return dict(version=VERSION, architecture='single_deployment_graph_frozen_base_plus_towing_residual',
            policy_dim=INPUT_DIM, action_dim=OUTPUT_DIM, ignored_policy_columns=[MASK_START, MASK_END],
            payload_column=PAYLOAD_COLUMN, towed_column=TOWED_COLUMN, gate='nominal_towed_context > 0',
            frozen_base_tensor_sha256=self.frozen_base_sha256,
            implementation_sha256=self.implementation_sha256,
            contract_source_sha256=dict(self.contract_source_sha256),
            checkpoint_provenance_verified=False, behavior_accepted=False, formal_ppo_authorized=False)

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError('Residual Actor metadata/source/base signature mismatch')

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if not strict or assign:
            raise ValueError('Residual Actor restore requires strict=True and assign=False')
        # 先验元数据，防止失败恢复在拒绝前污染已冻结的基线。
        self.set_extra_state(state_dict.get('_extra_state'))
        base_state = {key.removeprefix('base_actor.'): value for key, value in state_dict.items() if key.startswith('base_actor.')}
        current = self.base_actor.state_dict()
        if set(base_state) != set(current) or any(not torch.equal(base_state[key].to(current[key].device), current[key]) for key in current):
            raise ValueError('Restore attempted to replace protected base tensors')
        result = super().load_state_dict(state_dict, strict=True, assign=False)
        self.assert_invariants()
        return result
