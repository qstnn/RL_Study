"""学生独占动作接口/来源校验；不含环境worker，不可能触发教师接管。

这是纯策略接口，独立student_worker负责环境接线且尚未实际运行。仿真物理仍必须重建/核验；本模块
没有仿真启动命令，也不会将离线loss或checkpoint存在写成自主通过。
"""
import json
from pathlib import Path

import torch

from .bc import SCHEMA, load_baseline, source_signatures
from .contract import CONTRACT, sha256
from .dataset import EvidenceError, read_teacher_bundle
from .policy import FrozenBaseTowingResidualActor


class StudentActionInterface:
    """每次动作只取300维观测；没有env、time、teacher、fallback或额外动作偏置。"""
    def __init__(self, actor, *, inference_mode='row'):
        if type(actor) is not FrozenBaseTowingResidualActor:
            raise ValueError('Student interface only accepts the verified isolated residual Actor')
        if inference_mode not in ('batch', 'row'):
            raise ValueError('Inference mode must be explicit batch or row')
        actor.eval().assert_invariants()
        self.actor = actor
        self.inference_mode = inference_mode
        self.actor_calls = 0
        self.teacher_calls = 0

    @torch.inference_mode()
    def __call__(self, observations):
        if observations.ndim != 2 or observations.shape[1] != 300 or not len(observations):
            raise ValueError('Student action interface requires nonempty [N,300] observations')
        # 将实际执行方式固定并记录；不在失败时切换另一控制器或修改轮/腿动作。
        result = (self.actor(observations) if self.inference_mode == 'batch' else
                  torch.cat([self.actor(row[None]) for row in observations], dim=0))
        self.actor_calls += 1
        return result

    def execution_record(self):
        return dict(policy_dim=300, action_dim=16, inference_mode=self.inference_mode,
            actor_calls=self.actor_calls, teacher_calls=self.teacher_calls,
            actor_input='deployed_observation_only', autonomous_behavior_accepted=False,
            runtime_environment_verified=False, formal_ppo_authorized=False)


def load_student(checkpoint, *, device='cpu', inference_mode='row'):
    """以相邻manifest的SHA和完整真实教师出处重审加载；没有缺证据跳过选项。"""
    checkpoint = Path(checkpoint).resolve(strict=True)
    manifest = json.loads(checkpoint.with_name('manifest.json').read_text())
    expected = manifest.pop('student_checkpoint_sha256')
    if sha256(checkpoint) != expected:
        raise EvidenceError('Student checkpoint no longer matches its reviewed manifest SHA')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if saved.get('schema') != SCHEMA or saved.get('metadata') != manifest:
        raise EvidenceError('Student manifest and embedded provenance disagree')
    if source_signatures() != manifest.get('sources'):
        raise EvidenceError('Student execution/collection source signatures changed')
    if (manifest.get('artifact_kind') != 'offline_fitted_student_candidate'
            or manifest.get('policy_dim') != 300 or manifest.get('action_dim') != 16
            or manifest.get('teacher_runtime_required') is not False
            or manifest.get('critic_in_artifact') is not False
            or manifest.get('formal_ppo_authorized') is not False):
        raise EvidenceError('Artifact is not the isolated deployable student candidate')
    episodes = manifest['teacher_bundle']['episodes']
    _, current = read_teacher_bundle([episode['directory'] for episode in episodes])
    if current != manifest['teacher_bundle']:
        raise EvidenceError('Current complete teacher review differs from student provenance')
    baseline = manifest['baseline']
    # 加载模型不扰动调用者设置的随机流；真实执行仍为无环境/观测噪声。
    with torch.random.fork_rng(devices=[]):
        actor, verified_base = load_baseline(baseline['path'], baseline['checkpoint_sha256'])
    if verified_base != baseline:
        raise EvidenceError('Protected baseline contract or source no longer matches')
    actor.load_state_dict(saved['actor_state_dict'], strict=True)
    actor.to(device).eval().assert_invariants()
    return StudentActionInterface(actor, inference_mode=inference_mode), manifest


def environment_reconstruction_contract(manifest):
    """独立worker的必要接口；返回清单不会创建环境或证明物理一致。"""
    episodes = manifest['teacher_bundle']['episodes']
    profiles = [episode['physics_profile'] for episode in episodes]
    if len(profiles) < 2 or any(profile != profiles[0] for profile in profiles[1:]):
        raise EvidenceError('Student needs the same physical profile in both complete directions')
    return dict(version='r1_migration_v1_student_environment_implemented_unvalidated',
        physics_profile=profiles[0], deployed_contract=CONTRACT,
        initial_state='ordinary_default_pose_zero_velocity_no_preload_override',
        control='StudentActionInterface(observation) -> original native action manager',
        prohibited_runtime_paths=['teacher takeover', 'QP/analytic action blend',
            'wheel boost', 'teacher truth ready gate', 'reset to loaded posture'],
        command_protocol='explicit deployable preparation; actual move origin; recorded goal-distance brake trigger; 2s/5s',
        declared_task_cases=[episode['case'] for episode in episodes],
        minimum_accepted_net_distance_m=1., candidate_fixed_preparation_s=3.,
        preparation_cli_parameter='--preparation-seconds',
        requires_matching_actual_teacher_clocks=True,
        fixed_preparation_autonomous_verified=False,
        runtime_checks=['actual GPU16/4 on robot/cart/scene', 'mass/inertia/material/hitch readback',
            'policy300/action16 terms/order/scales/PD/clips', 'ordinary reset state',
            'actor-only call graph', 'zero observation noise and physical randomization',
            'native 1kHz contact/CoP and pre/post action records'],
        runtime_environment_verified=False, autonomous_behavior_accepted=False,
        formal_ppo_authorized=False)
