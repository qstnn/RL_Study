"""R4b近90度前腿持续站立：原物理/动作、显式返回、重置前末状态价值观测。"""
import torch
import isaaclab.sim as sim
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz
from two_leg_balance_split_v4.scene import BalanceSceneCfg
from two_leg_balance_split_v4.spec import LEG_JOINTS, WHEELS, WHEEL_BODIES, STAND
from .settings import WEIGHTS, PROFILE
from .rewards import task_rewards
from .gates import targets, desired_gravity, support_geometry, masks
from .core import SustainState


@configclass
class SustainBalanceCfg(DirectRLEnvCfg):
    scene = BalanceSceneCfg(num_envs=512, env_spacing=3., replicate_physics=True)
    sim = sim.SimulationCfg(dt=.001, render_interval=20, gravity=(0., 0., -9.81))
    decimation = 20
    episode_length_s = 40.
    observation_hold_s = 10.
    stop_on_observation = False
    is_finite_horizon = False
    action_space = 16
    observation_space = 305
    state_space = 88
    mode = "front"
    reward_profile = PROFILE
    target_pitches = (1.5707963267948966, 1.5707963267948966)
    observation_noise = False
    reset_jitter = 0.

    def __post_init__(self):
        super().__post_init__()
        self.viewer.origin_type = "world"
        self.viewer.eye = (2.5, -2.5, 1.5)
        self.viewer.lookat = (0., 0., .5)


class SustainBalanceEnv(DirectRLEnv):
    num_states = 88

    def __init__(self, cfg, render_mode=None, **kwargs):
        if cfg.mode != "front" or cfg.reward_profile != PROFILE:
            raise ValueError("R4 split task requires one fixed support mode")
        self.reward_weights = dict(WEIGHTS)
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
        self.older_actions = torch.zeros_like(self.actions)
        self.joint_targets = targets(self.actions)
        self.mode = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.physics_steps = torch.zeros_like(self.mode)
        self.episode_count = torch.zeros_like(self.mode)
        self.reward_phase = torch.zeros_like(self.mode)
        self.state = SustainState(self.num_envs, self.device, cfg.observation_hold_s)
        self.terminal_critic = torch.zeros(self.num_envs, 88, device=self.device)
        self.terminal_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.history = torch.zeros(self.num_envs, 5, 61, device=self.device)
        self.history_reset = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.command_dirty = torch.zeros_like(self.history_reset)
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
            leg_position=d.joint_pos[:, self.leg_ids], wheel_velocity=d.joint_vel[:, self.wheel_ids],
        )

    def command(self, t=None):
        if t is None:
            t = self.physics_steps * self.physics_dt
        phase, amplitude = self.state.schedule(t)
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
        self.current_metrics['smooth_rms'] = self.state.jitter.rms.clone()
        self._protocol_tick = tick
        self._capture(tick)

    def _pre_physics_step(self, actions):
        if not torch.isfinite(actions).all():
            raise RuntimeError("R4 non-finite action")
        self.extras["log"] = {}
        self.state.hold_event.zero_()
        self.state.stable_step_s.zero_()
        self.state.recovery_step_s.zero_()
        self.terminal_valid.zero_()
        self.older_actions.copy_(self.previous_actions)
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
        truncated = (self.physics_steps >= self.max_episode_length * self.cfg.decimation) | self.state.boundary
        if self.cfg.stop_on_observation:
            truncated |= self.state.observation_complete & self.state.stand_request
        self.extras['physical_terminated'] = self.state.failed.clone()
        return self.state.failed.clone(), truncated & ~self.state.failed

    def _get_rewards(self):
        """目的：持续稳定质量按时间累加；返回由明确命令开启，无2秒或timeout大奖。

        激活：入场/起身/保持/显式返回分阶段；平衡积分来自每个1ms样本。
        返回：每环境标量，物理失败只在结束控制步结算一次。
        """
        m, s, d = self.current_metrics, self.state, self.robot.data
        phase, target = self.command()
        v = masks(m, self.mode, target)
        full_target = desired_gravity(self.actions.new_tensor(self.cfg.target_pitches)[self.mode])
        missed = torch.zeros_like(s.failed)  # v7没有旧定时返回处罚。
        terms = task_rewards(
            m, s, phase, self.mode, full_target, self.mass, self.step_dt,
            d.joint_pos[:, self.leg_ids], d.applied_torque[:, self.leg_ids],
            d.joint_vel[:, self.leg_ids], d.joint_vel[:, self.wheel_ids],
            self.actions, self.previous_actions, self.reset_time_outs, missed,
            older_actions=self.older_actions,
            reward_profile=self.cfg.reward_profile,
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
            s.stand_request.float()[:, None],
        ), -1)
        assert frame.shape[1] == 61
        if self._history_step == self.common_step_counter:
            self.history[self.command_dirty, -1] = frame[self.command_dirty]
        self.command_dirty.zero_()
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
            m["base_z"][:, None], m["line_error"][:, None], s.jitter.rms,
            torch.stack((s.entered, s.held, s.failed, s.stand_request), -1).float(),
        ), -1)
        return {"policy": self.history.flatten(1).clone(), "critic": torch.cat((frame, extra), -1)}

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        ended = env_ids[self.episode_length_buf[env_ids] > 0]
        if ended.numel():
            # DirectRLEnv在计算done/reward后自动reset；必须在此之前捕获末状态价值输入。
            terminal = self._get_observations()['critic']
            self.terminal_critic[ended] = terminal[ended]
            self.terminal_valid[ended] = True
            self.extras['terminal_critic_observation'] = self.terminal_critic
            self.extras['terminal_observation_valid'] = self.terminal_valid
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
        self.actions[env_ids] = self.previous_actions[env_ids] = self.older_actions[env_ids] = 0.
        self.joint_targets[env_ids] = q[:, self.leg_ids]
        self.physics_steps[env_ids] = 0
        self.reward_phase[env_ids] = 0
        self.history_reset[env_ids] = True
        self.mode[env_ids] = int(self.cfg.mode == "rear")
        self.state.reset(env_ids, root[:, :2])
        for value in self.episode_rewards.values():
            value[env_ids] = 0.

    def request_return(self, env_ids=None):
        """外部命令接口；不移动机器人，不重置历史/滤波器/时钟。"""
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        self.state.request_return(env_ids, self.physics_steps * self.physics_dt)
        # 命令变化需立即更新当前帧，不能多移位历史。
        self.command_dirty[env_ids] = True
