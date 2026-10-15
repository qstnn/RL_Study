"""CPU content tests; none of these constitute an Isaac Sim rollout."""

from copy import deepcopy
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common
import protocol


def test_registration_is_explicit_and_preserves_old_registry():
    import gymnasium as gym
    before = dict(gym.registry)
    common.catalog.register_tasks()
    common.catalog.register_tasks()
    assert set(gym.registry) - set(before) <= {common.catalog.TASK}
    assert all(gym.registry[k] is value for k, value in before.items())
    for key in common.catalog.ROUTES:
        if key not in ("01", "06"):
            with pytest.raises(ValueError, match="no implemented"):
                common.catalog.implemented_route(key)


def test_urdf_and_meshes_match_action_order_and_limits():
    manifest = common.asset_manifest()
    assert manifest["joint_count"] == 16
    assert len(manifest["meshes"]) == 17
    assert 11 < manifest["mass_kg"] < 12
    assert common.catalog.CONTRACT["action_order"] == list(common.catalog.LEG_NAMES + common.catalog.WHEEL_NAMES)


def test_observation_named_mask_retains_wheel_velocity_semantics():
    path = common.SOURCE.parents[1] / "velocity/config/wheeled/deeprobotics_lighthw/reference_observations.py"
    observations = common.module_from_file("_multiskill_test_named_obs", path)
    physical = tuple(reversed(common.catalog.JOINT_NAMES))
    positions = torch.arange(16).float().reshape(1, 16)
    robot = SimpleNamespace(joint_names=physical, data=SimpleNamespace(joint_pos=positions, default_joint_pos=torch.zeros_like(positions)))
    env = SimpleNamespace(scene={"robot": robot})
    result = observations.joint_positions_v1(env, SimpleNamespace(name="robot", joint_names=common.catalog.JOINT_NAMES))
    expected = torch.tensor([[0. if "wheel" in name else float(physical.index(name)) for name in common.catalog.JOINT_NAMES]])
    torch.testing.assert_close(result, expected)
    assert sum(common.catalog.POLICY_TERMS.values()) == 57
    assert "base_lin_vel" not in common.catalog.POLICY_TERMS
    assert "height_scan" not in common.catalog.POLICY_TERMS


def test_standing_terms_are_zero_for_forward_lateral_and_yaw_commands():
    rewards = common.module_from_file("_multiskill_test_rewards", common.SOURCE / "rewards.py")
    commands = torch.tensor([[0.,0.,0.], [.3,0.,0.], [0.,.3,0.], [0.,0.,.3]])
    data = SimpleNamespace(root_lin_vel_b=torch.ones(4,3), root_ang_vel_b=torch.ones(4,3),
        joint_vel=torch.ones(4,16), joint_pos=torch.ones(4,16), default_joint_pos=torch.zeros(4,16))
    forces = torch.ones(4,4,3)*10
    env = SimpleNamespace(command_manager=SimpleNamespace(get_command=lambda _: commands),
        scene={"robot": SimpleNamespace(data=data), "contact_forces": SimpleNamespace(data=SimpleNamespace(net_forces_w=forces))})
    for value in [rewards.stand_motion(env),
                  rewards.stand_wheel_speed(env, SimpleNamespace(name="robot", joint_ids=[3,7,11,15])),
                  rewards.stand_leg_pose(env, SimpleNamespace(name="robot", joint_ids=[0,1,2])),
                  rewards.stand_support(env, SimpleNamespace(name="contact_forces", body_ids=[0,1,2,3]))]:
        assert value[0] > 0
        assert torch.count_nonzero(value[1:]) == 0


def test_output_modes_are_unique_and_planned_routes_create_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "ROOT", tmp_path)
    paths = [common.new_run("01", mode) for mode in ("train", "smoke", "play", "evaluate", "train")]
    assert len(set(paths)) == 5
    assert all(p.is_relative_to(tmp_path / "routes/route_01_wheel_locomotion") for p in paths)
    assert {p.parent.name for p in paths} == {"runs", "validation", "playbacks", "evaluations"}
    with pytest.raises(ValueError):
        common.new_run("05a", "train")
    assert not (tmp_path / "routes/route_05_acrobatics").exists()


def test_checkpoint_contract_rejects_legacy_and_nonfinite_without_writing(tmp_path):
    path = tmp_path / "model.pt"
    saved = {"model_state_dict": {"weight": torch.ones(2)}, "infos": None}
    torch.save(saved, path)
    before = common.sha256(path)
    with pytest.raises(ValueError, match="legacy transfer"):
        common.validate_checkpoint(path)
    assert common.sha256(path) == before
    saved["infos"] = {"multiskill_contract": deepcopy(common.catalog.CONTRACT)}
    torch.save(saved, path)
    common.validate_checkpoint(path)
    saved["model_state_dict"]["weight"][0] = float("nan")
    torch.save(saved, path)
    with pytest.raises(ValueError, match="Non-finite"):
        common.validate_checkpoint(path)


def perfect_rows():
    return [dict(t=i*protocol.DT, velocity_error=0., tilt_deg=1., speed=0., yaw_rate_abs=0.,
                 four_wheel=1., target_clip_fraction=0., torque_abs_max=2., torque_limit_fraction=0.,
                 x=0., y=0., yaw_rad=0., body_contact=0.) for i in range(protocol.STEPS)]


def test_evaluation_completion_is_not_final_acceptance():
    result = protocol.summarize(perfect_rows(), "zero", 43)
    assert result["success"] and result["complete"]
    assert result["final_acceptance"] == "not_evaluated"
    assert not protocol.summarize(perfect_rows()[:-1], "zero", 43)["success"]
    assert not protocol.summarize(perfect_rows(), "zero", 43, "terminated")["success"]


def test_evaluation_catches_tracking_support_collision_and_nan():
    for key, value in [("velocity_error", .4), ("four_wheel", 0.), ("body_contact", 1.), ("tilt_deg", float("nan"))]:
        rows = perfect_rows()
        for row in rows:
            row[key] = value
        assert not protocol.summarize(rows, "start_stop_06", 43)["success"]


def test_command_sequence_has_settle_cruise_and_hold():
    assert protocol.command_at("start_stop_06", 0) == (0,0,0)
    assert protocol.command_at("start_stop_06", 3) == (.6,0,0)
    assert protocol.command_at("start_stop_06", 11) == (0,0,0)
    assert all(protocol.command_at("zero", t) == (0,0,0) for t in (0,3,10,11,17))


def test_planned_cli_is_rejected_before_simulation_or_output_creation():
    result = subprocess.run([sys.executable, str(common.ROOT / "run.py"), "train", "--route", "03"], capture_output=True, text=True)
    assert result.returncode != 0 and "no implemented training task" in result.stderr
