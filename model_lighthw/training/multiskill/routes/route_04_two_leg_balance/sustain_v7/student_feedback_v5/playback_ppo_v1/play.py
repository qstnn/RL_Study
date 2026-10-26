"""Visualize the reviewed R4a final PPO mean Actor without changing training."""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
import time
import traceback
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
V5 = HERE.parent
sys.path.insert(0, str(V5 / "evaluation_ppo_v1"))
from common_eval import now, read, write, sha, require  # noqa: E402
from common_feedback import parent  # noqa: E402

DEFAULT_REVIEW = V5 / "results/2026-10-23_formal_1000_review/review.json"


def reviewed_actor(review_path: Path):
    """Bind the existing 300 s independent review to the complete PPO Actor."""
    review_path = review_path.resolve(strict=True)
    review = read(review_path)
    if (review["kind"] != "r4a_completed_formal_training_and_frozen_long_stand_review_v1"
            or not review["training_technical_pass"]
            or not review["final_fixed_condition_continuous_stand_60s_300s_pass"]):
        raise ValueError("Playback requires the completed final R4a 60/300 s review")
    for filename, key in (("training_audit.json", "training_audit_sha256"),
                          ("evaluation_audit_results.json", "independent_crosscheck_sha256")):
        if sha(review_path.parent / filename) != review[key]:
            raise ValueError("Final review evidence changed: " + filename)
    binding = review["evidence_bindings"]["300"]
    evaluation = Path(binding["run"]).resolve(strict=True)
    if read(evaluation / "session.json")["status"] != "completed":
        raise ValueError("Reviewed 300 s evaluation is not complete")
    for name in ("request", "result", "verification"):
        if sha(evaluation / (name + ".json")) != binding[name + "_sha256"]:
            raise ValueError("Reviewed 300 s evidence changed: " + name)
    for name, digest in binding["trace_sha256"].items():
        if sha(evaluation / name) != digest:
            raise ValueError("Reviewed 300 s trace changed: " + name)
    verification = read(evaluation / "verification.json")
    if (verification["hold_seconds"] != 300. or verification["seeds"] != [43]
            or verification["passes"] != 2 or not verification["all_case_hold_gate"]
            or verification["teacher_constructed"]
            or not verification["deterministic_mean_actor_verified"]):
        raise ValueError("Reviewed evaluation is not two-slot teacher-off mean-Actor endurance")
    actor, metadata = require(evaluation)
    if (metadata["checkpoint_sha256"] != review["final_checkpoint_sha256"]
            or metadata["checkpoint_sha256"] != verification["checkpoint_sha256"]
            or metadata["checkpoint_updates"] != review["final_model_global_updates"]
            or Path(metadata["checkpoint"]) != Path(review["final_checkpoint"])):
        raise ValueError("The reviewed final checkpoint and actual loaded PPO Actor differ")
    return actor, metadata, evaluation


def prepare_run(run: Path, evaluation: Path, request: dict):
    run.mkdir(parents=True, exist_ok=False)
    write(run / "request.json", request)
    shutil.copy2(evaluation / "geometry.json", run / "geometry.json")
    shutil.copytree(evaluation / "assets", run / "assets")
    asset_hashes = {}
    for source in sorted((evaluation / "assets").rglob("*")):
        if source.is_file():
            relative = source.relative_to(evaluation)
            digest = sha(source)
            if sha(run / relative) != digest:
                raise ValueError("Copied reviewed asset changed: " + str(relative))
            asset_hashes[str(relative)] = digest
    write(run / "playback_asset_sources.json", asset_hashes)
    archive = run / "playback_source_snapshot"
    archive.mkdir()
    sources = {}
    for source in (HERE / "play.py", HERE / "play.sh"):
        shutil.copy2(source, archive / source.name)
        sources[source.name] = sha(source)
    write(run / "playback_sources.json", sources)


def setup(run: Path, args):
    """Same two-slot sustain physics as evaluation; only window/camera differ."""
    print("[R4 PLAY] Launching Isaac Sim " + ("headless" if args.headless else "GUI"), flush=True)
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=args.headless, device="cuda:0").app
    env = None
    try:
        import gymnasium as gym
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from two_leg_balance_sustain_v7.env import SustainBalanceCfg, SustainBalanceEnv

        cfg = SustainBalanceCfg()
        cfg.scene.num_envs, cfg.seed, cfg.sim.device = 2, 42, "cuda:0"
        cfg.observation_hold_s = float(args.hold_seconds)
        cfg.episode_length_s = float(args.hold_seconds) + 15.
        cfg.stop_on_observation = True
        cfg.target_pitches = tuple(row["pitch_rad"] for row in read(run / "geometry.json")["candidates"])
        cfg.scene.robot.spawn.asset_path = str(run / "assets/urdf/lighthw.urdf")
        cfg.scene.robot.spawn.usd_dir = str(run / "asset_cache")
        cfg.scene.robot.spawn.usd_file_name = "playback.usd"
        cfg.viewer.origin_type = "env"
        cfg.viewer.env_index = args.view_env
        cfg.viewer.eye = (2., -2., 1.5)
        cfg.viewer.lookat = (0., 0., .55)

        class Wrapper(RslRlVecEnvWrapper):
            def __init__(self, *values, **kwargs):
                super().__init__(*values, **kwargs)
                self.num_privileged_obs = gym.spaces.flatdim(
                    self.unwrapped.single_observation_space.spaces["critic"])

        print("[R4 PLAY] Creating original two-slot environment and a new USD cache", flush=True)
        env = Wrapper(SustainBalanceEnv(cfg=cfg), clip_actions=1.)
        print("[R4 PLAY] Checking actual URDF, mass, limits, fixed reset and observations", flush=True)
        spec = importlib.util.spec_from_file_location(
            "r4_playback_original_runtime", parent.ROUTE / "split_v4/scripts/worker.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.runtime(env, run)
        base = env.unwrapped
        if cfg.is_finite_horizon or not (base.scene.env_origins[:, :2].abs() < 190.).all():
            raise ValueError("Playback changed horizon semantics or scene-origin boundary")
        write(run / "playback_runtime.json", dict(raw_policy_dim=305, actor_input_dim=325,
            critic_dim=88, action_dim=16, physics_dt=base.physics_dt, control_dt=base.step_dt,
            scene_origins=base.scene.env_origins.cpu().tolist(), view_env=args.view_env,
            camera_origin_type="env", camera_eye=cfg.viewer.eye, camera_lookat=cfg.viewer.lookat,
            headless=args.headless, teacher_constructed=False, mean_action_only=True,
            no_automatic_return=True, stop_on_observation=True,
            observation_hold_s=cfg.observation_hold_s, technical_time_limit_s=cfg.episode_length_s,
            cfg_seed=42, rollout_seed=43, new_bc_updates=0, new_ppo_updates=0,
            full_1khz_trace_saved=False, independent_behavior_verification=False))
        env.reset()
        return app, env
    except BaseException:
        if env is not None:
            env.close()
        app.close()
        raise


def state_summary(base, active, steps):
    metrics = base.metrics()
    gravity = metrics["gravity"]
    return dict(control_steps=steps, physics_time_s=(base.physics_steps * base.physics_dt).cpu().tolist(),
        active=active.cpu().tolist(), root_position_m=base.robot.data.root_link_pos_w.cpu().tolist(),
        body_pitch_deg=torch.rad2deg(torch.atan2(gravity[:, 0], -gravity[:, 2])).cpu().tolist(),
        com_speed_m_s=metrics["com_vel"].norm(dim=-1).cpu().tolist(),
        capture_time_s=base.state.capture_time.cpu().tolist(),
        uninterrupted_post_capture_s=base.state.post_capture_s.cpu().tolist(),
        first_break_time_s=base.state.first_break_time.cpu().tolist(),
        failed=base.state.failed.cpu().tolist(), stand_request=base.state.stand_request.cpu().tolist())


def show_progress(run, state):
    write(run / "progress.json", dict(state, updated_at=now()))
    with (run / "progress.jsonl").open("a") as stream:
        import json
        stream.write(json.dumps(dict(state, at=now()), allow_nan=False) + "\n")
    positions = [[round(value, 4) for value in row] for row in state["root_position_m"]]
    print("[R4 PLAY] "
        f"step={state['control_steps']} sim_s={[round(v, 3) for v in state['physics_time_s']]} "
        f"pitch_deg={[round(v, 2) for v in state['body_pitch_deg']]} "
        f"hold_s={[round(v, 3) for v in state['uninterrupted_post_capture_s']]} "
        f"first_break={state['first_break_time_s']} active={state['active']} root_xyz={positions}", flush=True)


@torch.inference_mode()
def rollout(app, env, actor, args, run):
    from features_feedback import FeedbackFeatures
    base = env.unwrapped
    env.seed(43)
    raw, _ = env.reset()
    builder = FeedbackFeatures(actor.prefix, 2, base.device)
    obs = builder(raw, torch.ones(2, dtype=torch.bool, device=base.device))
    base.last_reports = []
    active = torch.ones(2, dtype=torch.bool, device=base.device)
    ended, terminals = {}, []
    steps, reason = 0, "technical_time_limit"
    latest_state = state_summary(base, active, steps)
    show_progress(run, latest_state)
    print("[R4 PLAY] Actor running: four-wheel start -> rear standing; no teacher or keyboard action", flush=True)
    try:
        for _ in range(base.max_episode_length):
            if not app.is_running():
                reason = "window_closed"
                break
            started = time.monotonic()
            if tuple(obs.shape) != (2, 325):
                raise ValueError("Playback must construct all 325 causal Actor inputs")
            action = actor(obs)
            if tuple(action.shape) != (2, 16) or not torch.isfinite(action).all():
                raise ValueError("Final PPO mean action is invalid")
            next_raw, _, done, extras = env.step(action)
            steps += 1
            for case in base.last_reports:
                slot = case["env_id"]
                if bool(active[slot]) and bool(done[slot]):
                    if not bool(extras["terminal_observation_valid"][slot]):
                        raise ValueError("Missing pre-reset terminal observation")
                    terminal = extras["terminal_critic_observation"][slot]
                    if not torch.isfinite(terminal).all():
                        raise ValueError("Nonfinite pre-reset terminal observation")
                    ended[slot] = dict(case)
                    terminals.append(dict(slot=slot, physical_failure=bool(extras["physical_terminated"][slot]),
                        timeout=bool(extras["time_outs"][slot]), terminal_critic_observation=terminal.cpu().tolist()))
                    print(f"[R4 PLAY] slot={slot} ended: success={case['success']} "
                          f"capture_s={case['capture_time']:.3f} hold_s={case['post_capture_s']:.3f} "
                          f"first_break={case['first_break_time']} failure_reason={case['reason']}", flush=True)
            active &= ~done.bool()
            if any(case["failed"] or not case["success"] for case in ended.values()):
                reason = "episode_failed"
                break
            if bool((base.state.interrupted & active).any()):
                reason = "continuous_stand_interrupted"
                print("[R4 PLAY] First standing interruption detected; ending this playback", flush=True)
                break
            if not bool(active.any()):
                reason = "observation_complete"
                break
            if steps % 50 == 0:
                latest_state = state_summary(base, active, steps)
                show_progress(run, latest_state)
            if args.max_steps is not None and steps >= args.max_steps:
                reason = "max_steps_reached"
                break
            raw = next_raw
            obs = builder(raw, done.bool())
            if not args.headless:
                time.sleep(max(0., base.step_dt - (time.monotonic() - started)))
    except KeyboardInterrupt:
        reason = "interrupted"
        print("[R4 PLAY] Ctrl+C received; closing this playback", flush=True)
    except Exception:
        if app.is_running():
            raise
        reason = "window_closed"
    final_state = (dict(latest_state, snapshot_before_user_exit=True)
                   if reason in ("window_closed", "interrupted") else state_summary(base, active, steps))
    show_progress(run, final_state)
    write(run / "terminal.json", terminals)
    result = dict(kind="reviewed_final_ppo_visual_playback", stop_reason=reason, control_steps=steps,
        completed_cases=[ended[key] for key in sorted(ended)], last_state=final_state,
        observation_passes=sum(bool(case["success"]) for case in ended.values()),
        all_observation_pass=reason == "observation_complete" and len(ended) == 2,
        new_bc_updates=0, new_ppo_updates=0, teacher_constructed=False,
        deterministic_mean_actor=True, robot_behavior_accepted=False,
        independent_behavior_verification=False, full_1khz_trace_saved=False,
        note="Visualization only; prior independent 300 s review is separate. No automatic return or replay loop.")
    write(run / "result.json", result)
    return result


def positive(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive number of control steps")
    return parsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--hold-seconds", type=int, choices=(10, 60, 300), default=300)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--max-steps", type=positive, help="Stop a short integration check before full standing acceptance")
    parser.add_argument("--check", action="store_true", help="CPU-only source/checkpoint/review preflight; no simulation")
    parser.add_argument("--view-env", type=int, choices=(0, 1), default=0)
    parser.add_argument("--output", type=Path, help="New output directory; never overwrite an existing run")
    args = parser.parse_args()
    torch.set_num_threads(4)
    print("[R4 PLAY] Verifying final review, 300 s evidence, frozen sources and PPO checkpoint", flush=True)
    actor, metadata, evaluation = reviewed_actor(args.review)
    print(f"[R4 PLAY] CPU PASS: global{metadata['checkpoint_updates']}, Actor325/Critic88/action16\n"
          f"[R4 PLAY] checkpoint={metadata['checkpoint']}\n"
          f"[R4 PLAY] sha256={metadata['checkpoint_sha256']}", flush=True)
    if args.check:
        print("[R4 PLAY] Check only: no simulator, GUI, training or new output directory", flush=True)
        return 0
    run = (args.output or HERE / "runs" / (now().replace(":", "-") + "_play")).resolve()
    request = dict(kind="reviewed_final_ppo_visual_playback", review=str(args.review.resolve()),
        review_sha256=sha(args.review), reviewed_evaluation_run=str(evaluation), ppo_source=metadata,
        hold_seconds=args.hold_seconds, headless=args.headless, max_steps=args.max_steps,
        view_env=args.view_env, teacher_off=True, mean_action_only=True, raw_policy_dim=305,
        actor_input_dim=325, output=str(run), full_1khz_trace=False)
    prepare_run(run, evaluation, request)
    started, app, env = now(), None, None
    write(run / "session.json", dict(status="starting", started_at=started))
    print(f"[R4 PLAY] output={run}", flush=True)
    try:
        app, env = setup(run, args)
        write(run / "session.json", dict(status="running", started_at=started))
        actor = actor.to(env.device).eval()
        result = rollout(app, env, actor, args, run)
        if any(sha(HERE / name) != digest for name, digest in read(run / "playback_sources.json").items()):
            raise ValueError("Playback executable changed while running")
        status = "completed" if result["stop_reason"] == "observation_complete" else result["stop_reason"]
        write(run / "session.json", dict(status=status, started_at=started, finished_at=now(),
            stop_reason=result["stop_reason"], observation_passes=result["observation_passes"]))
        print(f"[R4 PLAY] Finished: {result['stop_reason']}; "
              f"completed observation passes={result['observation_passes']}/2\n[R4 PLAY] Results: {run}", flush=True)
        if result["stop_reason"] == "interrupted":
            return 130
        return 2 if result["stop_reason"] in ("episode_failed", "continuous_stand_interrupted", "technical_time_limit") else 0
    except BaseException as error:
        write(run / "error.json", dict(error=str(error), traceback=traceback.format_exc(), at=now()))
        write(run / "session.json", dict(status="failed", started_at=started, finished_at=now(), error=str(error)))
        raise
    finally:
        if env is not None:
            env.close()
        if app is not None:
            app.close()


if __name__ == "__main__":
    raise SystemExit(main())
