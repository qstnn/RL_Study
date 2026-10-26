"""R6真实RSL-RL保存/恢复与逐更新CSV，可独立进行CPU集成检查。"""
import csv
import json
import statistics
from rsl_rl.runners import OnPolicyRunner
from launch import spec


def make_runner_class(run, payload_kg):
    class PayloadRunner(OnPolicyRunner):
        def save(self, path, infos=None):
            return super().save(path, infos=dict(infos or {}, r6_contract=spec.CONTRACT,
                payload_kg=payload_kg, learning_rate=self.alg.learning_rate))

        @staticmethod
        def env_episode_means(infos):
            import torch
            keys = {key for info in infos for key in info}
            return {key:torch.cat([torch.as_tensor(info[key]).reshape(-1) for info in infos if key in info]).float().mean()
                    for key in keys}

        def log(self, locs, *pos, **kw):
            super().log(locs, *pos, **kw)
            samples, self.env.r6_samples = self.env.r6_samples, []
            row = {'iteration': locs['it'], 'payload_kg': payload_kg, 'learning_rate': self.alg.learning_rate}
            for key in ('tilt_deg','speed','height','four_wheel','torque_peak','torque_saturation','torque_clipped','terminated','timeout'):
                values = [s[key] for s in samples if key in s]
                row[key] = sum(values) if key in ('terminated','timeout') else sum(values)/len(values) if values else None
                if row[key] is not None:
                    self.writer.add_scalar('R6/'+key, row[key], locs['it'])
            # JSONL允许不同阶段的episode奖励/终止项出现或缺失，不丢掉原始TensorBoard。
            def scalar(value):
                return float(value.detach().float().mean().cpu()) if hasattr(value, 'detach') else float(value)
            update = {'iteration':locs['it'], 'learning_rate':self.alg.learning_rate,
                      'collection_time_s':locs['collection_time'], 'learning_time_s':locs['learn_time'],
                      'losses':{k:scalar(v) for k,v in locs['loss_dict'].items()},
                      'policy_noise_std':scalar(self.alg.policy.action_std),
                      'episode_reward_last100':float(statistics.mean(locs['rewbuffer'])) if locs['rewbuffer'] else None,
                      'episode_length_steps_last100':float(statistics.mean(locs['lenbuffer'])) if locs['lenbuffer'] else None,
                      'episode_terms':{k:scalar(v) for k,v in self.env_episode_means(locs['ep_infos']).items()},
                      'r6':row}
            with (run/'ppo_updates.jsonl').open('a') as stream:
                stream.write(json.dumps(update, allow_nan=False)+'\n')
            path = run/'training_metrics.csv'
            exists = path.exists()
            with path.open('a', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                if not exists: writer.writeheader()
                writer.writerow(row)

    return PayloadRunner


def resume_runner(runner, path, saved):
    """恢复模型/优化器/实际学习率，从下一更新开始；物理状态与RNG重新初始化。"""
    runner.load(path, load_optimizer=True)
    runner.current_learning_iteration = int(saved['iter'])+1
    runner.alg.learning_rate = saved['infos']['learning_rate']
    for group in runner.alg.optimizer.param_groups:
        group['lr'] = runner.alg.learning_rate
