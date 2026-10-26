"""Route 15：原生 PPO 的分组探索、受检恢复与联合课程日志。"""

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import torch
import rsl_rl.runners.on_policy_runner as native_runner
from rsl_rl.runners import OnPolicyRunner
from lighthw_grouped_noise_policy import LightHWGroupedNoiseActorCritic
from lighthw_training_monitor import compact_native_summary, center_summary_rows
from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_lighthw import roll_stop_core as core


class LightHWRollStopActorCritic(LightHWGroupedNoiseActorCritic):
    """沿用同一个 PPO 条件高斯分布；本路线动作是线性裁剪，不调用旧tanh诊断。"""

    @torch.no_grad()
    def diagnostics(self, observations):
        means = self.actor(observations)
        std = self._base_std()
        return dict(leg_raw_std=std[:12].mean().item(), wheel_raw_std=std[12:].mean().item(),
                    leg_clip_fraction=(means[:, :12].abs() >= 1).float().mean().item(),
                    wheel_clip_fraction=(means[:, 12:].abs() * 5 >= 37).float().mean().item())


def _json_value(value):
    if callable(value):
        return f"{value.__module__}:{value.__qualname__}"
    if isinstance(value, slice):
        return [value.start, value.stop, value.step]
    return str(value)


class LightHWRollStopRunner(OnPolicyRunner):
    KEY = "lighthw_roll_stop"

    def __init__(self, env, train_cfg, log_dir=None, device="cpu", *, curriculum_training=True):
        if (not train_cfg.get(self.KEY) or train_cfg["algorithm"]["class_name"] != "PPO"
                or train_cfg.get("empirical_normalization") or train_cfg.get("clip_actions") is not None):
            raise ValueError("RollStop requires its own PPO config, raw actions and unnormalized observations")
        if any(train_cfg["algorithm"].get(k) for k in ("symmetry_cfg", "rnd_cfg")):
            raise ValueError("RollStop baseline does not use auxiliary learning or symmetry augmentation")
        if curriculum_training and log_dir is None:
            raise ValueError("RollStop training requires logs for curriculum updates")
        native_runner.LightHWRollStopActorCritic = LightHWRollStopActorCritic
        self.training_curriculum = curriculum_training
        self.command = env.unwrapped.command_manager.get_term("base_velocity")
        self.total_updates = self.stage_updates = self.window_updates = self.passes = 0
        self.last_window = None
        self.loaded = curriculum_training
        obs, extra = env.get_observations()
        dims = (obs.shape[-1], extra["observations"]["critic"].shape[-1], env.num_actions)
        if dims != (60, 236, 16):
            raise ValueError(f"RollStop actual observation/action dimensions: {dims}")
        manager = env.unwrapped.observation_manager
        if manager.active_terms["policy"] != ["base_lin_vel", "base_ang_vel", "projected_gravity",
                "velocity_commands", "joint_pos", "joint_vel", "actions"]:
            raise ValueError("RollStop requires command at policy slice 9:12")
        super().__init__(env, deepcopy(train_cfg), log_dir, device)
        self._noise_hook = self.alg.optimizer.register_step_post_hook(
            lambda optimizer, args, kwargs: self.alg.policy.project_grouped_noise_())
        cfg = env.unwrapped.cfg
        leg = env.unwrapped.action_manager.get_term("joint_pos")
        wheel = env.unwrapped.action_manager.get_term("joint_vel")
        self.mapping = dict(profile=core.PROFILE, dimensions=list(dims), physics_dt=cfg.sim.dt,
            control_dt=env.unwrapped.step_dt, leg_joints=list(leg._joint_names),
            leg_offsets=leg._offset[0].cpu().tolist(), leg_limits=list(core.LEG_LIMITS),
            leg_mapping="default + clip(raw,-1,1)*limits; soft URDF safety clip",
            wheel_joints=list(wheel._joint_names), wheel_scale=5.0, wheel_limit=37.0,
            pose_height_target=None, noise=self.alg.policy.noise_profile())
        expected_legs = self.alg.policy.noise_profile()["leg_action_order"]
        if (self.mapping["leg_joints"] != expected_legs or self.mapping["wheel_joints"] !=
                self.alg.policy.noise_profile()["wheel_action_order"]):
            raise ValueError("Unexpected physical action order")
        # 允许play关闭观测噪声/固定命令，但不允许改变动作、动力学或奖励后冒充同路线resume。
        signature = dict(mapping=self.mapping, robot=cfg.scene.robot.to_dict(),
            rewards=cfg.rewards.to_dict(), actions=cfg.actions.to_dict(),
            algorithm=train_cfg["algorithm"], policy=train_cfg["policy"],
            speeds=core.SPEEDS, acceleration=core.ACCELERATION, cycle=core.CYCLE_SECONDS,
            observation_scales={group: {key: getattr(getattr(cfg.observations, group), key).scale
                for key in manager.active_terms[group]} for group in ("policy", "critic")})
        self.signature = hashlib.sha256(json.dumps(signature, sort_keys=True, default=_json_value).encode()).hexdigest()
        print("[LightHW RollStop verified] policy=60, critic=236, action=16; "
              "fresh PPO or strict same-route resume; physics=1000Hz, control=50Hz", flush=True)

    def log(self, locs, width=80, pad=35):
        with compact_native_summary(width):
            super().log(locs, width, pad)
        self.total_updates += 1
        self.stage_updates += 1
        self.window_updates += 1
        due = self.window_updates >= 100
        show = due or self.total_updates % 10 == 0 or self.total_updates == 1
        if not show:
            return
        snapshot = self.command.snapshot()
        checks = core.gate_results(snapshot, self.command.stage, self.env.num_envs)
        cap, step_probability = core.phase_spec(self.command.stage)
        def fmt(key, digits=3):
            value = snapshot.get(key)
            return "待采样" if value is None else f"{value:.{digits}f}"
        noise = self.alg.policy.diagnostics(self.env.get_observations()[0])
        gap = (None if snapshot["speed_mae"] is None else
               max(0, snapshot["speed_mae"] - core.thresholds(cap)["speed_mae"]))
        rows = [f"[课程] 更新 {self.total_updates} | 阶段 {self.command.stage}/15 | 上限 {cap:.1f} m/s | 阶跃周期 {step_probability:.0%}",
            f"[速度] 命令 {fmt('command_speed')} | 实际 {fmt('actual_speed')} | MAE {fmt('speed_mae')} | 超门槛 {gap if gap is None else round(gap, 3)} m/s",
            f"[接触] 四轮支撑 {fmt('support')} | 非轮接触率 {fmt('nonwheel', 5)} | 倾角>10° {fmt('tilt_bad', 5)} | 失败率 {fmt('failure', 5)}",
            f"[停车] 平面速度 {fmt('stop_speed')} m/s | 末2秒漂移 {fmt('stop_drift')} m | 刹停路程 {fmt('brake_distance')} m",
            f"[起停] 完整周期 {fmt('cycles', 0)} | 联合合格率 {fmt('cycle_success')} | 阶跃周期 {fmt('step_cycles', 0)} | 阶跃合格率 {fmt('step_success')}",
            f"[探索] 腿raw std {noise['leg_raw_std']:.3f} | 轮目标std {5*noise['wheel_raw_std']:.3f} rad/s | 腿均值裁剪 {noise['leg_clip_fraction']:.1%}",
            f"[门槛] {'完整' if due else '部分'}窗口 {self.window_updates}/100 | 未过：{', '.join(k for k,v in checks.items() if not v) or '全部通过'}"]
        print(center_summary_rows(rows), flush=True)
        for key, value in snapshot.items():
            if value is not None and self.writer is not None:
                self.writer.add_scalar(f"RollStop/{key}", value, self.total_updates)
        if self.writer is not None:
            self.writer.add_scalar("RollStop/stage", self.command.stage, self.total_updates)
        if due:
            self.last_window = dict(updates=self.total_updates, stage=self.command.stage,
                                    metrics=snapshot, checks=checks)
            with open(Path(self.log_dir) / "quality_windows.jsonl", "a", encoding="utf-8") as stream:
                stream.write(json.dumps(self.last_window, ensure_ascii=False, allow_nan=False) + "\n")
            stage, self.passes, advanced = core.next_stage(self.command.stage, self.passes, self.stage_updates, checks)
            self.command.reset_window()
            self.window_updates = 0
            if advanced:
                self.command.set_stage(stage)
                self.stage_updates = 0
                # 新阶段重新起停并重新收集整周期，不将旧上限的周期计为新阶段证据。
                self.command._resample_command(slice(None))
                print(f"[LightHW RollStop] Joint gates passed; stage -> {stage}: {core.phase_spec(stage)}", flush=True)

    def save(self, path, infos=None):
        data = dict(infos or {})
        data[self.KEY] = dict(profile=core.PROFILE, signature=self.signature, mapping=self.mapping,
            stage=self.command.stage, total_updates=self.total_updates, stage_updates=self.stage_updates,
            last_window=self.last_window, learning_rate=self.alg.learning_rate)
        super().save(path, data)

    def load(self, path, load_optimizer=True):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state = (checkpoint.get("infos") or {}).get(self.KEY)
        if not isinstance(state, dict) or state.get("signature") != self.signature:
            raise ValueError("Refusing foreign/changed RollStop checkpoint; start this independent task from scratch")
        core.phase_spec(state["stage"])
        for value in checkpoint["model_state_dict"].values():
            if not torch.isfinite(value).all():
                raise ValueError("Non-finite RollStop checkpoint")
        rate = state["learning_rate"]
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError("Invalid checkpoint learning rate")
        result = super().load(path, load_optimizer=load_optimizer)
        self.command.set_stage(state["stage"])
        self.command._resample_command(slice(None))
        self.command.reset_window()
        self.total_updates, self.stage_updates = state["total_updates"], state["stage_updates"]
        self.current_learning_iteration = checkpoint["iter"] + 1
        self.window_updates = self.passes = 0
        self.last_window = state["last_window"]
        if load_optimizer:
            self.alg.learning_rate = rate
        self.loaded = True
        print(f"[LightHW RollStop] checkpoint restored, stage={self.command.stage}, "
              f"true updates={self.total_updates}, curriculum {'active' if self.training_curriculum else 'frozen'}", flush=True)
        return result

    def get_inference_policy(self, device=None):
        if not self.loaded:
            raise ValueError("Load a RollStop checkpoint before playback")
        return super().get_inference_policy(device)

    def export_action_mapping(self, path, checkpoint_path=None):
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "roll_stop_action_mapping.json").write_text(json.dumps(
            dict(self.mapping, checkpoint=checkpoint_path, stage=self.command.stage), indent=2), encoding="utf-8")
