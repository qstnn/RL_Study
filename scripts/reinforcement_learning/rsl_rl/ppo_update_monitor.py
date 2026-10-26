"""Read-only native PPO minibatch telemetry; no replacement update/loss/scheduler.

Capture the storage's old distribution and read the current distribution immediately
before each native optimizer step. The KL expression is identical to installed
RSL-RL 3.0.1, including its epsilon. Tests compare against actual native update locals
and verify identical weights, Adam, LR, losses and RNG with/without this observer.
"""
from __future__ import annotations
import torch


class PPOUpdateMonitor:
    def __init__(self, algorithm):
        if algorithm.policy.is_recurrent or algorithm.symmetry or algorithm.rnd or algorithm.is_multi_gpu:
            raise ValueError('Reference telemetry only supports this feed-forward single-GPU PPO configuration')
        self.alg=algorithm;self.records=[];self._old=None
        original=algorithm.storage.mini_batch_generator
        def generator(*args,**kwargs):
            for batch in original(*args,**kwargs):
                self._old=(batch[7],batch[8])
                yield batch
            self._old=None
        algorithm.storage.mini_batch_generator=generator
        self.hook=algorithm.optimizer.register_step_pre_hook(self._before_step)

    @torch.inference_mode()
    def _before_step(self, optimizer, args, kwargs):
        if self._old is None:raise RuntimeError('PPO telemetry did not observe the minibatch')
        old_mu,old_sigma=self._old
        mu=self.alg.policy.action_mean[:old_mu.shape[0]]
        sigma=self.alg.policy.action_std[:old_sigma.shape[0]]
        kl=torch.sum(torch.log(sigma/old_sigma+1e-5)
                     +(old_sigma.square()+(old_mu-mu).square())/(2*sigma.square())-.5,axis=-1)
        self.records.append((kl.mean().detach(),float(optimizer.param_groups[0]['lr'])))

    def consume(self):
        expected=self.alg.num_learning_epochs*self.alg.num_mini_batches
        if len(self.records)!=expected:
            raise RuntimeError(f'PPO telemetry expected {expected} minibatches, got {len(self.records)}')
        kl=torch.stack([row[0] for row in self.records]).cpu().tolist()
        lr=[row[1] for row in self.records]
        self.records.clear()
        return dict(minibatch_kl_mean=sum(kl)/len(kl),minibatch_kl_max=max(kl),
                    minibatch_kl_min=min(kl),minibatch_kl_last=kl[-1],
                    minibatch_learning_rate_min=min(lr),minibatch_learning_rate_max=max(lr),
                    learning_rate_used_last=lr[-1],optimizer_steps=expected),dict(kl=kl,learning_rate=lr)
