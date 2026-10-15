"""R6真实RSL-RL保存/恢复与逐更新CSV，可独立进行CPU集成检查。"""
import csv
from rsl_rl.runners import OnPolicyRunner
from launch import spec


def make_runner_class(run, payload_kg):
    class PayloadRunner(OnPolicyRunner):
        def save(self, path, infos=None):
            return super().save(path, infos=dict(infos or {}, r6_contract=spec.CONTRACT,
                payload_kg=payload_kg, learning_rate=self.alg.learning_rate))

        def log(self, locs, *pos, **kw):
            super().log(locs, *pos, **kw)
            samples, self.env.r6_samples = self.env.r6_samples, []
            row = {'iteration': locs['it'], 'payload_kg': payload_kg, 'learning_rate': self.alg.learning_rate}
            for key in ('tilt_deg','speed','height','four_wheel','torque_peak','torque_saturation','terminated','timeout'):
                values = [s[key] for s in samples if key in s]
                row[key] = sum(values) if key in ('terminated','timeout') else sum(values)/len(values) if values else None
                if row[key] is not None:
                    self.writer.add_scalar('R6/'+key, row[key], locs['it'])
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
