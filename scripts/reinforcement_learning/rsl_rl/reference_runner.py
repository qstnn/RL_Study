"""Correct-observation PPO reference: fresh initialization or strict same-task restore."""
from __future__ import annotations
from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import torch
from ppo_action_curriculum_runner import LightHWPPOActionCurriculumRunner
from ppo_grouped_noise_runner import LightHWGroupedNoiseRunner
from ppo_update_monitor import PPOUpdateMonitor
from lighthw_training_monitor import format_grouped_exploration, center_summary_rows
from reference_continuation import (INFO_KEY, SOURCE, SOURCE_UPDATES, validate_source,
    plan as continuation_plan, verify_restoration, validate_completion, tree_digest)
from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_lighthw.reference_observations import (
    OBSERVATION_VERSION, REFERENCE_PROFILE, observation_contract, validate_observation_cfg,
)


def state_digest(state):
    h=hashlib.sha256()
    for name,tensor in sorted(state.items()):
        h.update(name.encode());h.update(str(tuple(tensor.shape)).encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class LightHWReferenceRunner(LightHWGroupedNoiseRunner):
    _REFERENCE_KEY='lighthw_observation_reference'
    _REFERENCE_ROLES=('formal','validation','continuation')

    def __init__(self,env,train_cfg,log_dir=None,device='cpu',*,curriculum_training=True):
        if not train_cfg.get('lighthw_reference') or train_cfg.get('lighthw_observation_version')!=OBSERVATION_VERSION:
            raise ValueError('Reference runner requires its versioned independent task')
        validate_observation_cfg(env.unwrapped.cfg)
        self._reference_role=train_cfg.get('lighthw_reference_run_role')
        if self._reference_role not in self._REFERENCE_ROLES:
            raise ValueError('Reference run role must be formal, validation or continuation')
        self._continuation = None
        if self._reference_role == 'continuation':
            if not curriculum_training or not train_cfg.get('resume'):
                raise ValueError('Continuation entry requires full training-state resume')
            self._continuation = continuation_plan(train_cfg['max_iterations'],
                short_check=train_cfg.get('lighthw_reference_short_check',False))
            if env.num_envs != 1024 or train_cfg['seed'] != 42 or train_cfg['save_interval'] != 100:
                raise ValueError('Controlled continuation requires 1024 environments, seed42, save100')
        self._continuation_restored = False
        self._reference_log_dir=Path(log_dir) if log_dir else None
        self._reference_resume_source=None
        self._reference_algorithm_cfg=deepcopy(train_cfg['algorithm'])
        super().__init__(env,train_cfg,log_dir,device,curriculum_training=curriculum_training)
        mapping=self._curriculum_term.export_curriculum_state()['action_mapping']
        if (mapping.get('profile_id')!=REFERENCE_PROFILE
                or mapping.get('observation_contract')!=observation_contract()):
            raise ValueError('Reference runner requires the versioned reference action signature')
        # 新任务允许且要求随机初始化；旧GroupedNoiseRunner的迁移前置条件
        # 不适用于该独立profile。普通回放仍必须先严格load同任务checkpoint。
        self._curriculum_initialized=bool(curriculum_training)
        self._reference_initialization=dict(
            profile=REFERENCE_PROFILE,observation_contract=observation_contract(),
            origin='from_scratch',source_checkpoint=None,run_role=self._reference_role,
            initial_model_sha256=state_digest(self.alg.policy.state_dict()),
            initial_adam_states=len(self.alg.optimizer.state),
            initial_learning_rate=float(self.alg.learning_rate),
            initial_raw_std=self.alg.policy.log_std.detach().exp().cpu().tolist(),
            empirical_normalization=self.empirical_normalization,
            actor_normalizer=type(self.obs_normalizer).__name__,
            critic_normalizer=type(self.privileged_obs_normalizer).__name__,
        )
        if self._reference_initialization['initial_adam_states']!=0:
            raise ValueError('Fresh reference must start with empty Adam state')
        self._ppo_observer=PPOUpdateMonitor(self.alg) if curriculum_training else None
        self._exploration_cache={}
        if self._reference_log_dir:
            self._reference_log_dir.mkdir(parents=True,exist_ok=True)
            self._write_json('initialization.json',self._reference_initialization)
        print('[LightHW Reference] policy=60, critic=236, action=16; observation='+OBSERVATION_VERSION+
              (f'; allocated modules pending full continuation restore (no rollout yet)' if train_cfg.get('resume')
               else f'; new Actor/Critic/noise/Identity normalizers/Adam; initial LR={self.alg.learning_rate:g}'),flush=True)

    def _write_json(self,name,value):
        path=self._reference_log_dir/name
        temporary=path.with_suffix(path.suffix+'.tmp')
        temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
        temporary.replace(path)

    def _require_initialized_curriculum(self):
        if not self._curriculum_initialized:
            raise ValueError('Reference playback requires a checkpoint with the same observation version/profile')
        if self._reference_role == 'continuation' and not self._continuation_restored:
            raise ValueError('Continuation requires exact restoration from original formal model999 before sampling')

    def learn(self,num_learning_iterations,init_at_random_ep_len=False):
        if self._continuation is not None:
            if num_learning_iterations != self._continuation['additional_updates']:
                raise ValueError('Continuation update budget cannot change after provenance validation')
            if self._curriculum_term.schedule.total_updates != SOURCE_UPDATES:
                raise ValueError('This bounded continuation may be launched only once from original model999')
        return super().learn(num_learning_iterations,init_at_random_ep_len=init_at_random_ep_len)

    def load_route10_checkpoint(self,path):raise ValueError('Reference task forbids old-policy transfer')
    def load_route11_checkpoint(self,path):raise ValueError('Reference task forbids old-policy transfer')
    def load_route12_checkpoint(self,path):raise ValueError('Reference task forbids old-policy transfer')

    def save(self,path,infos=None):
        if infos is not None and not isinstance(infos,Mapping):raise TypeError('Checkpoint infos must be a mapping')
        if self._continuation is not None:
            appended=self._curriculum_term.schedule.total_updates-SOURCE_UPDATES
            # Native modulo uses a zero-based iteration. Continue saves at actual
            # +100/+200/+300 updates instead; final short-check save is retained.
            if appended % 100 and appended != self._continuation['additional_updates']: return
        record=dict(infos or {})
        record[self._REFERENCE_KEY]=deepcopy(self._reference_initialization)
        record[self._REFERENCE_KEY].update(algorithm=self._reference_algorithm_cfg,
            resumed_from=self._reference_resume_source,run_role=self._reference_role)
        record[self._NOISE_KEY]=self.alg.policy.noise_profile()
        if self._continuation is not None:
            record[INFO_KEY]=deepcopy(self._continuation)
            record[INFO_KEY].update(completed_total_updates=self._curriculum_term.schedule.total_updates,
                total_env_steps=self._curriculum_term.schedule.total_updates*24*self.env.num_envs)
        return LightHWPPOActionCurriculumRunner.save(self,path,infos=record)

    def load(self,path,load_optimizer=True):
        source = None
        if self._reference_role == 'continuation':
            if not load_optimizer: raise ValueError('Continuation cannot reset or skip Adam')
            source = validate_source(path)
        saved=torch.load(path,map_location='cpu',weights_only=False)
        if ('lighthw_contact_gate_ablation' in saved.get('infos',{})
                and not getattr(self,'_contact_gate_loading',False)):
            raise ValueError('Route17 checkpoint requires its explicit contact-reward contract; use route17 playback')
        if ('lighthw_leg_acc_weight_ablation' in saved.get('infos',{})
                and not getattr(self,'_leg_acc_weight_loading',False)):
            raise ValueError('Route19 checkpoint requires its explicit leg-acceleration weight contract')
        infos=saved.get('infos',{});record=infos.get(self._REFERENCE_KEY)
        if (not isinstance(record,Mapping) or record.get('profile')!=REFERENCE_PROFILE
                or record.get('observation_contract')!=observation_contract()
                or record.get('origin')!='from_scratch' or record.get('source_checkpoint') is not None
                or record.get('algorithm')!=self._reference_algorithm_cfg):
            raise ValueError('Checkpoint is not this corrected-observation PPO reference; old semantics cannot resume')
        self._curriculum_term.validate_curriculum_state(infos.get(self._CHECKPOINT_KEY))
        self._validate_training_checkpoint(saved)
        profile=self.alg.policy.noise_profile()
        if infos.get(self._NOISE_KEY)!=profile:raise ValueError('Reference exploration profile mismatch')
        noise=saved['model_state_dict']['log_std']
        lower=noise.new_tensor([math.log(profile['leg_std_min'])]*12+[math.log(profile['wheel_std_min'])]*4)
        upper=noise.new_tensor([math.log(profile['leg_std_max'])]*12+[math.log(profile['wheel_std_max'])]*4)
        if ((noise<lower)|(noise>upper)).any():raise ValueError('Reference checkpoint noise outside saved bounds')
        # All checks precede any network/Adam mutation. Base load restores exactly
        # the same action signature and synchronizes native PPO's adaptive LR.
        loaded=LightHWPPOActionCurriculumRunner.load(self,path,load_optimizer=load_optimizer)
        self._reference_initialization=deepcopy(dict(record))
        self._reference_resume_source=str(Path(path).resolve())
        if self._reference_role == 'continuation':
            restored=verify_restoration(self,source)
            self._continuation['restoration']=restored
            self._continuation_restored=True
            # Native RSL 3.0.1 stores the last iteration and would repeat its index.
            # Only book-keeping changes: next rollout/update is numbered 1000.
            self.current_learning_iteration=SOURCE_UPDATES
            self.tot_timesteps=24000*self.env.num_envs
            if self._reference_log_dir: self._write_json('continuation_restore.json',self._continuation)
            print('[LightHW continuation] exact Actor/Critic/log_std/Adam/Identity/LR restored; '
                  f'updates 1000 -> {self._continuation["target_total_updates"]}; '
                  f'Adam 20000 -> {self._continuation["expected_final_adam_step"]}; '
                  f'actual LR={self.alg.learning_rate:.12g}; fresh seeded simulator episodes',flush=True)
        elif INFO_KEY in infos:
            # Playback retains provenance without enabling training or resetting anything.
            self._loaded_continuation=deepcopy(infos[INFO_KEY])
        if self._reference_log_dir:
            self._write_json('restore.json',dict(checkpoint=self._reference_resume_source,
                checkpoint_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                optimizer_restored=load_optimizer,actual_learning_rate=self.alg.learning_rate,
                restored_model_sha256=state_digest(self.alg.policy.state_dict()),
                total_updates=self._curriculum_term.schedule.total_updates))
        return loaded

    @torch.no_grad()
    def _exploration(self,obs):
        policy=self.alg.policy;diagnostics=policy.diagnostics(obs)
        means=policy.actor(obs);base=policy._base_std()
        names=list(self._curriculum_term._joint_names)+list(self.env.unwrapped.action_manager.get_term('joint_vel')._joint_names)
        for index,name in enumerate(names):
            diagnostics[f'raw/{name}/mean']=means[:,index].mean()
            diagnostics[f'raw/{name}/std']=base[index]
        # Same-state conditional noise after the real wheel clip, not temporal
        # variation in target speeds; quadrature is deterministic and consumes no RNG.
        constants=policy._constants(means.double())
        nodes=constants['gh_nodes'];weights=constants['gh_weights']
        multiplier=torch.where(policy._stop_mask(obs),.4,1.).double()
        targets=((means[:,12:].double()[...,None]+base[12:].double()[None,:,None]
                  *multiplier[:,None,None]*nodes)*5).clamp(-37,37)
        center=(targets*weights).sum(-1,keepdim=True)
        std=((targets-center).square()*weights).sum(-1).sqrt()
        for j,name in enumerate(names[12:]):diagnostics[f'wheel/{name}/target_std_after_clip_rad_s']=std[:,j].mean()
        return diagnostics

    def log(self,locs,width=80,pad=35):
        observed,minibatches=self._ppo_observer.consume()
        obs=locs['obs'][:256]
        self._exploration_cache={k:float(v) for k,v in self._exploration(obs).items()}
        super().log(locs,width=width,pad=pad)
        for key,value in observed.items():self.writer.add_scalar('PPO_actual/'+key,value,locs['it'])
        for key,value in self._exploration_cache.items():self.writer.add_scalar('Exploration/'+key,value,locs['it'])
        for key,value in self._curriculum_term.transition_monitor.consume().items():
            self.writer.add_scalar('TrainingTransitions/'+key,value,locs['it'])
        chain=self._curriculum_term.wheel_monitor.consume()
        for key,value in chain.items():self.writer.add_scalar('ControlChain50Hz/'+key,value,locs['it'])
        for label,mask in (('moving',obs[:,9].abs()>.05),('stop',obs[:,9:12].norm(dim=-1)<=.05)):
            self.writer.add_scalar(f'Exploration/{label}/samples',int(mask.sum()),locs['it'])
            if mask.any():
                for key,value in self._exploration(obs[mask]).items():
                    self.writer.add_scalar(f'Exploration/{label}/{key}',float(value),locs['it'])
        if self._reference_log_dir:
            with (self._reference_log_dir/'ppo_minibatches.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(iteration=locs['it'],total_updates=self._curriculum_term.schedule.total_updates,
                    **observed,minibatches=minibatches),allow_nan=False)+'\n')
            with (self._reference_log_dir/'wheel_control_chain.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(iteration=locs['it'],sample_rate_hz=50,
                    wheel_order=['FR','FL','RR','RL'],values=chain),allow_nan=False)+'\n')
        if self._continuation is not None and self._curriculum_term.schedule.total_updates % 100 == 0:
            self.save(str(self._reference_log_dir/f'model_{locs["it"]}.pt'))
        if (locs['it']%self._console_interval==0 or locs['it']==locs['tot_iter']-1):
            print((f"PPO 实测 | KL均值 {observed['minibatch_kl_mean']:.5f} | KL最大 {observed['minibatch_kl_max']:.5f}"
                   f" | 实际LR {observed['learning_rate_used_last']:.3g}").center(112),flush=True)

    def _log_forward_summary(self,snapshot,context,values,locs):
        if self._continuation is not None:
            self._log_continuation_summary(snapshot,context,values,locs)
            print(format_grouped_exploration(self._exploration_cache),flush=True)
            return
        LightHWPPOActionCurriculumRunner._log_forward_summary(self,snapshot,context,values,locs)
        print(format_grouped_exploration(self._exploration_cache),flush=True)

    def _log_continuation_summary(self,snapshot,context,values,locs,*,record=None,source_updates=1000):
        # Original training gate remains computed and logged by the parent.
        # This separate read-only monitor drops only MOVING contact conditions.
        # Random training windows cannot replace fixed-protocol independent hold acceptance.
        from lighthw_training_monitor import quality_checks
        checks=quality_checks(snapshot,max_speed=context['max_speed'],
            parameters=context['gate_parameters'],min_samples=context['min_samples'])
        selected=[c for c in checks if c.key not in ('forward/four_wheel_support','fast/four_wheel_support')]
        for check in selected:
            if check.valid:self.writer.add_scalar('PhaseMonitor_v2/gap/'+check.key,check.gap,locs['it'])
        for key,value in snapshot.items():
            if math.isfinite(value):self.writer.add_scalar('PhaseMonitor_v2/window/'+key,value,locs['it'])
        failed=[c.key for c in selected if not c.valid or c.gap>0]
        if context['due']:
            self.writer.add_scalar('PhaseMonitor_v2/complete_window_met',not failed,locs['it'])
        record=self._continuation if record is None else record
        rows=[f"续训累计 {int(values['total_updates'])}/{record['target_total_updates']} | "
              f"本次追加 {int(values['total_updates'])-source_updates}/{record['additional_updates']} | PhaseMonitor_v2 只读"]
        for label,title in (('forward','前进'),('fast','上限组'),('stop','停车')):
            row={k.split('/',1)[1]:v for k,v in snapshot.items() if k.startswith(label+'/')}
            if row.get('samples',0):
                rows.append(f"[{title}] 命令 {row.get('command_vx',0):.3f} | 实速 {row.get('actual_vx',0):.3f} | "
                    f"误差 {row.get('speed_error',0):.3f} m/s | 样本 {row['samples']:.0f}")
                rows.append(f"[{title}接触/姿态] 四轮 {row.get('four_wheel_support',0):.1%} | "
                    f"非轮 {row.get('body_contact_fraction',0):.2%} | 倾角 {row.get('tilt_deg',0):.2f}° | "
                    f"偏航 {row.get('yaw_rate',0):.3f} rad/s")
            else:rows.append(f'[{title}] 稳态样本不足，不作验收判断')
        rows+=['v2监控缺口: '+(', '.join(failed) if failed else '本窗口未见缺口；仍需独立起停验收'),
               '移动支撑仅记录；原 Curriculum/window_passed 保留 v1 语义；预算结束即退出。']
        print(center_summary_rows(rows),flush=True)

    def export_action_mapping(self,directory,*,checkpoint_path):
        super().export_action_mapping(directory,checkpoint_path=checkpoint_path)
        (Path(directory)/'observation_contract.json').write_text(json.dumps(observation_contract(),indent=2)+'\n')
