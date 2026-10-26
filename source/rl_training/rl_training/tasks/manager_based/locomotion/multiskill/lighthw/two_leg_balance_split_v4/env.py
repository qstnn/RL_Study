"""R4：12腿位置+4轮速，50Hz策略/1kHz事件验收，R4a后腿/R4b前腿使用独立Actor。"""
import torch
import isaaclab.sim as sim
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz
from .scene import BalanceSceneCfg
from .spec import LEG_JOINTS, WHEELS, WHEEL_BODIES, STAND, WEIGHTS, weights_for
from .rewards import task_rewards
from .core import BalanceState, targets, schedule, desired_gravity, support_geometry, masks


@configclass
class TwoLegBalanceCfg(DirectRLEnvCfg):
    scene = BalanceSceneCfg(num_envs=512, env_spacing=3., replicate_physics=True)
    sim = sim.SimulationCfg(dt=.001, render_interval=20, gravity=(0., 0., -9.81))
    decimation = 20
    episode_length_s = 14.
    is_finite_horizon = True
    action_space = 16
    observation_space = 305
    state_space = 88
    hold_seconds = 2.
    mode = "rear"
    reward_profile = "baseline_v4"
    target_pitches = (.9099663600406092, -.9794692378476809)
    observation_noise = False
    reset_jitter = 0.

    def __post_init__(self):
        super().__post_init__()
        self.viewer.origin_type = "world"
        self.viewer.eye = (2.5, -2.5, 1.5)
        self.viewer.lookat = (0., 0., .5)


class TwoLegBalanceEnv(DirectRLEnv):
    num_states = 88

    def __init__(self, cfg, render_mode=None, **kwargs):
        if cfg.mode not in ("front", "rear"):
            raise ValueError("R4 split task requires one fixed support mode")
        self.reward_weights = weights_for(cfg.reward_profile)
        self.physics_observer = None
        self.episode_sink = None
        super().__init__(cfg, render_mode, **kwargs)
        self.robot = self.scene["robot"]
        self.contacts = self.scene["contact_forces"]
        self.leg_ids = self.robot.find_joints(list(LEG_JOINTS), preserve_order=True)[0]
        self.wheel_ids = self.robot.find_joints(list(WHEELS), preserve_order=True)[0]
        self.joint_ids = self.leg_ids + self.wheel_ids
        self.wheel_body_ids = self.robot.find_bodies(list(WHEEL_BODIES), preserve_order=True)[0]
        self.contact_ids = self.contacts.find_bodies(list(WHEEL_BODIES), preserve_order=True)[0]
        self.other_ids = [i for i, n in enumerate(self.contacts.body_names) if n not in WHEEL_BODIES]
        self.masses = self.robot.root_physx_view.get_masses().to(self.device)
        self.mass = self.masses.sum(-1)
        # PhysX克隆后显式设置全部形状材料，再由worker读回；避免USD导入器默认0.5覆盖声明值。
        material = self.robot.root_physx_view.get_material_properties().clone()
        material[..., :2] = 1.
        material[..., 2] = 0.
        self.robot.root_physx_view.set_material_properties(
            material, torch.arange(self.num_envs, dtype=torch.int32, device="cpu"))
        self.actions = torch.zeros(self.num_envs, 16, device=self.device)
        self.previous_actions = torch.zeros_like(self.actions)
        self.joint_targets = targets(self.actions)
        self.mode = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.physics_steps = torch.zeros_like(self.mode)
        self.episode_count = torch.zeros_like(self.mode)
        self.reward_phase = torch.zeros_like(self.mode)
        self.state = BalanceState(self.num_envs, self.device, cfg.hold_seconds)
        self.history = torch.zeros(self.num_envs, 5, 61, device=self.device)
        self.history_reset = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._history_step = -1
        self._protocol_tick = self._sim_step_counter
        self._last_capture_tick = -1
        self.last_reports = []
        self.episode_rewards = {name: torch.zeros(self.num_envs, device=self.device) for name in self.reward_weights}

    def _setup_scene(self):
        self.scene.filter_collisions(global_prim_paths=["/World/ground"])

    def metrics(self):
        d = self.robot.data
        com = (d.body_com_pos_w * self.masses[:, :, None]).sum(1) / self.mass[:, None]
        wheels = d.body_link_pos_w[:, self.wheel_body_ids]
        return dict(
            com=com, com_vel=(d.body_com_lin_vel_w * self.masses[:, :, None]).sum(1) / self.mass[:, None],
            wheel_pos=wheels, force_z=self.contacts.data.net_forces_w[:, self.contact_ids, 2],
            # 球形外包半径给出保守净空，不把轮倾斜带来的中心升高误认作离地。
            clearance=wheels[:, :, 2] - .087501, gravity=d.projected_gravity_b,
            ang_vel=d.root_ang_vel_b, base_z=d.root_link_pos_w[:, 2],
            line_error=support_geometry(com, wheels, self.mode),
            body_force=self.contacts.data.net_forces_w[:, self.other_ids].norm(dim=-1).amax(-1),
            torque_peak=d.applied_torque.abs().amax(-1),
        )

    def command(self, t=None):
        if t is None:
            t = self.physics_steps * self.physics_dt
        phase, amplitude = schedule(self.state.clock(t), self.cfg.hold_seconds)
        pitch = self.actions.new_tensor(self.cfg.target_pitches)[self.mode] * amplitude
        return phase, desired_gravity(pitch)

    def _capture(self, tick):
        if self.physics_observer is not None and tick != self._last_capture_tick:
            self.physics_observer(self, tick)
            self._last_capture_tick = tick

    def _advance(self, tick):
        if tick == self._protocol_tick:
            return
        if tick != self._protocol_tick + 1:
            raise RuntimeError("R4 physics clock has a gap")
        self.physics_steps += 1
        t = self.physics_steps * self.physics_dt
        self.current_metrics = self.metrics()
        _, target = self.command(t)
        self.state.update(self.physics_dt, t, self.current_metrics, self.mode, target)
        self._protocol_tick = tick
        self._capture(tick)

    def _pre_physics_step(self, actions):
        if not torch.isfinite(actions).all():
            raise RuntimeError("R4 non-finite action")
        self.extras["log"] = {}
        self.state.hold_event.zero_()
        self.previous_actions.copy_(self.actions)
        self.actions.copy_(actions.clamp(-1., 1.))
        # 出生后0.2s消除落地瞬态；之后全部控制目标由Actor输出。
        self.actions[self.physics_steps < 200] = 0.
        self.joint_targets.copy_(targets(self.actions))

    def _apply_action(self):
        self._advance(self._sim_step_counter - 1)
        self.robot.set_joint_position_target(self.joint_targets, joint_ids=self.leg_ids)
        self.robot.set_joint_velocity_target(15. * self.actions[:, 12:], joint_ids=self.wheel_ids)

    def _get_dones(self):
        self._advance(self._sim_step_counter)
        if not torch.equal(self.physics_steps, self.episode_length_buf * self.cfg.decimation):
            raise RuntimeError("R4 reset/physics/episode clock mismatch")
        if (self.state.reason == 4).any():
            raise RuntimeError("R4 non-finite physics state")
        return self.state.failed.clone(), self.state.finished(self.physics_steps * self.physics_dt)

    def _get_rewards(self):
        """返回每环境标量。连续项乘dt；一次性双支撑/终末成功奖励不乘dt。

        转移/保持收益按最终双腿目标计算；返回收益必须先完成严格保持。
        接触和净空必须共同成立，禁止四轮站立、腾空、侧翻或滚走取得双支撑收益。
        动作变化/力矩/关节速度约束全程激活，不用全程四轮奖励抵消抬腿目标。
        """
        m, s, d = self.current_metrics, self.state, self.robot.data
        phase, target = self.command()
        v = masks(m, self.mode, target)
        full_target = desired_gravity(self.actions.new_tensor(self.cfg.target_pitches)[self.mode])
        missed = (phase >= 3) & (self.reward_phase < 3) & ~s.held & ~s.failed
        terms = task_rewards(
            m, s, phase, self.mode, full_target, self.mass, self.step_dt,
            d.joint_pos[:, self.leg_ids], d.applied_torque[:, self.leg_ids],
            d.joint_vel[:, self.leg_ids], d.joint_vel[:, self.wheel_ids],
            self.actions, self.previous_actions, self.reset_time_outs, missed,
            reward_profile=getattr(self.cfg, "reward_profile", "baseline_v4"),
        )
        self.reward_phase.copy_(phase)
        weights = getattr(self, "reward_weights", WEIGHTS)
        self.reward_terms = {k: value * weights[k] for k, value in terms.items()}
        for name, value in self.reward_terms.items():
            self.episode_rewards[name] += value
            self.extras["log"]["Reward/" + name] = value.mean()
        self.extras["log"]["R4/best_hold_s"] = s.best_hold.mean()
        self.extras["log"]["R4/pair_fraction"] = v["pair"].float().mean()
        self.extras["log"]["R4/entered_fraction"] = s.entered.float().mean()
        self.extras["log"]["R4/entry_best_s"] = s.entry_best.mean()
        reward = sum(self.reward_terms.values())
        if not torch.isfinite(reward).all():
            raise RuntimeError("R4 non-finite reward")
        return reward

    def _get_observations(self):
        d, s = self.robot.data, self.state
        phase, target = self.command()
        frame = torch.cat((
            d.root_ang_vel_b * .2, d.projected_gravity_b,
            d.joint_pos[:, self.leg_ids] - self.actions.new_tensor(STAND),
            d.joint_vel[:, self.joint_ids] * .05, self.actions,
            torch.nn.functional.one_hot(self.mode, 2).float(), target,
            torch.nn.functional.one_hot(phase, 5).float(),
            (self.physics_steps * self.physics_dt / self.cfg.episode_length_s)[:, None],
        ), -1)
        assert frame.shape[1] == 61
        # 同一步多次读取不可多推进历史，也不能改写PPO仍持有的旧obs。
        if self._history_step != self.common_step_counter or self.history_reset.any():
            noisy = frame.clone()
            if self.cfg.observation_noise:
                noisy[:, :3] += .002 * torch.randn_like(noisy[:, :3])
                noisy[:, 3:6] += .002 * torch.randn_like(noisy[:, 3:6])
                noisy[:, 6:18] += .002 * torch.randn_like(noisy[:, 6:18])
                noisy[:, 18:34] += .001 * torch.randn_like(noisy[:, 18:34])
            if self._history_step != self.common_step_counter:
                self.history[:, :-1] = self.history[:, 1:].clone()
                self.history[:, -1] = noisy
                self._history_step = self.common_step_counter
            self.history[self.history_reset] = noisy[self.history_reset, None, :]
            self.history_reset.zero_()
        m = self.metrics()
        extra = torch.cat((
            m["com"] - d.root_link_pos_w, m["com_vel"], d.root_lin_vel_b,
            m["force_z"] / (self.mass[:, None] * 9.81), m["clearance"],
            m["base_z"][:, None], m["line_error"][:, None], m["com"][:, :2] - s.origin,
            torch.stack((s.entry_time, s.hold_time, s.recovery_time), -1),
            torch.stack((s.entered, s.held, s.failed), -1).float(),
        ), -1)
        return {"policy": self.history.flatten(1).clone(), "critic": torch.cat((frame, extra), -1)}

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        ended = env_ids[self.episode_length_buf[env_ids] > 0]
        if ended.numel():
            data = {k: getattr(self.state, k)[ended].detach().cpu().tolist()
                    for k in self.state.FLOATS + self.state.BOOLS + ("reason", "start_time")}
            modes = self.mode[ended].cpu().tolist()
            durations = (self.physics_steps[ended] * self.physics_dt).cpu().tolist()
            rewards = {k: v[ended].cpu().tolist() for k, v in self.episode_rewards.items()}
            self.last_reports = [
                dict(env_id=i, mode=("front", "rear")[modes[j]], duration_s=durations[j],
                     timeout=bool(self.reset_time_outs[i]), **{k: v[j] for k, v in data.items()},
                     rewards={k: v[j] for k, v in rewards.items()})
                for j, i in enumerate(ended.cpu().tolist())]
            if self.episode_sink:
                self.episode_sink(self.last_reports)
            for key in ("entered", "held", "success", "best_hold", "best_recovery"):
                self.extras["log"]["Ended/" + key] = getattr(self.state, key)[ended].float().mean()
            self.episode_count[ended] += 1
        super()._reset_idx(env_ids)
        root = self.robot.data.default_root_state[env_ids].clone()
        root[:, :3] += self.scene.env_origins[env_ids]
        if self.cfg.reset_jitter:
            jitter = (torch.rand(len(env_ids), 3, device=self.device) * 2 - 1) * self.cfg.reset_jitter
            root[:, 3:7] = quat_from_euler_xyz(jitter[:, 0], jitter[:, 1], jitter[:, 2])
        # 固定条件基线保留默认单位四元数；关闭扰动时不消费随机数。
        root[:, 7:] = 0.
        q = self.robot.data.default_joint_pos[env_ids].clone()
        self.robot.write_root_pose_to_sim(root[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(root[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(q, torch.zeros_like(q), None, env_ids)
        self.actions[env_ids] = self.previous_actions[env_ids] = 0.
        self.joint_targets[env_ids] = q[:, self.leg_ids]
        self.physics_steps[env_ids] = 0
        self.reward_phase[env_ids] = 0
        self.history_reset[env_ids] = True
        self.mode[env_ids] = int(self.cfg.mode == "rear")
        self.state.reset(env_ids, root[:, :2])
        for value in self.episode_rewards.values():
            value[env_ids] = 0.
