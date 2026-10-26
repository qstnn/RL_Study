"""Run the unchanged R1 V7 physical probe with one isolated passive-trailer variant.

Example: --wheel-radius 0.20 changes wheel geometry only. Use worker.py options
for case, output, teacher pose, duration and physics device. No PPO is launched.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import worker
import research_common
from trailer_variant import replace_passive_wheels
from trailer_brake import install_reverse_brake


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--wheel-radius", type=float, default=0.14)
    parser.add_argument("--rolling-coefficient", type=float, default=0.02)
    parser.add_argument("--drawbar-damping", type=float, default=150.0)
    parser.add_argument("--anti-rollback-limit", type=float, default=0.0,
                        help="Optional passive one-way clutch torque limit, Nm per trailer wheel")
    parser.add_argument("--anti-rollback-speed-scale", type=float, default=0.03)
    parser.add_argument("--combined-research", action="store_true",
                        help="Explicitly allow multiple physical trailer changes after single-factor checks")
    parser.add_argument("--variant-control", action="store_true",
                        help="Run the identical wrapper with unchanged trailer parameters")
    selected, remainder = parser.parse_known_args()
    if not 0.14 <= selected.wheel_radius <= 0.22:
        parser.error("--wheel-radius must lie in [0.14, 0.22] m")
    if not 0.002 <= selected.rolling_coefficient <= 0.02:
        parser.error("--rolling-coefficient must lie in [0.002, 0.02]")
    if not 100.0 <= selected.drawbar_damping <= 300.0:
        parser.error("--drawbar-damping must lie in [100, 300] N s/m")
    if selected.anti_rollback_limit and not 4.0 <= selected.anti_rollback_limit <= 12.0:
        parser.error("--anti-rollback-limit must be 0 or [4, 12] Nm/wheel")
    if not 0.01 <= selected.anti_rollback_speed_scale <= 0.1:
        parser.error("--anti-rollback-speed-scale must lie in [0.01, 0.1] rad/s")
    varied = sum((selected.wheel_radius != 0.14,
                  selected.rolling_coefficient != 0.02,
                  selected.drawbar_damping != 150.0,
                  selected.anti_rollback_limit > 0.0))
    if selected.variant_control and varied != 0:
        parser.error("--variant-control requires unchanged trailer parameters")
    if not selected.variant_control and not selected.combined_research and varied != 1:
        parser.error("Change exactly one trailer property, or use --combined-research")
    if selected.combined_research and varied < 2:
        parser.error("--combined-research requires at least two changed properties")
    if selected.anti_rollback_limit:
        if "--cases" not in remainder:
            parser.error("Passive clutch probe requires an explicit --cases file")
        index = remainder.index("--cases")
        cases = json.loads(Path(remainder[index + 1]).read_text())
        if any(case["slope_deg"] != 25.0 or case["towed_kg"] != 40.0 for case in cases):
            parser.error("Passive clutch sign is validated only for +25 degree / 40 kg cases")

    source_write_cart = research_common.launch.CART.write_cart_urdf
    source_write_json = research_common.launch.write_json
    source_configure = worker.configure
    variant = {
        "wheel_radius_m": selected.wheel_radius,
        "rolling_coefficient": selected.rolling_coefficient,
        "drawbar_stiffness_N_per_m": 2000.0,
        "drawbar_damping_N_s_per_m": selected.drawbar_damping,
        "anti_rollback_limit_Nm_per_wheel": selected.anti_rollback_limit,
        "anti_rollback_speed_scale_rad_s": selected.anti_rollback_speed_scale,
        "added_hardware": "four passive one-way axle clutches" if selected.anti_rollback_limit else None,
        "one_factor_only": varied == 1,
    }

    def write_cart(path: Path) -> None:
        source_write_cart(path)
        variant["geometry"] = replace_passive_wheels(Path(path), selected.wheel_radius)
        source_write_json(Path(path).parent.parent / "trailer_variant.json", variant)

    def write_json(path: Path, content: dict) -> None:
        if Path(path).name == "request.json":
            content = dict(content)
            execution_physics = dict(content["execution_physics"])
            execution_physics["rolling_coefficient"] = selected.rolling_coefficient
            execution_physics["trailer_variant"] = dict(variant)
            content["execution_physics"] = execution_physics
        source_write_json(path, content)

    def configure(*args, **kwargs):
        cfg, agent = source_configure(*args, **kwargs)
        cfg.rolling_coefficient = selected.rolling_coefficient
        from rl_training.tasks.manager_based.locomotion.multiskill.lighthw.wheel_unified_v6 import cart
        from rl_training.tasks.manager_based.locomotion.multiskill.lighthw.wheel_unified_v6 import cart_asset
        cart.RADIUS = selected.wheel_radius
        cart_asset.RADIUS = selected.wheel_radius
        physical_drawbar = cart_asset.drawbar_force

        def damped_drawbar(robot_point, cart_point, robot_velocity, cart_velocity, active,
                           rest=1.0, stiffness=2000.0, damping=150.0):
            return physical_drawbar(robot_point, cart_point, robot_velocity, cart_velocity,
                                    active, rest=rest, stiffness=stiffness,
                                    damping=selected.drawbar_damping)

        cart.drawbar_force = damped_drawbar
        # The teacher imports the asset function at every step. Keep its towing
        # force prediction identical to the physical cart's 1 kHz force law.
        cart_asset.drawbar_force = damped_drawbar
        if selected.anti_rollback_limit:
            install_reverse_brake(cart, selected.anti_rollback_limit,
                                  selected.anti_rollback_speed_scale)
        return cfg, agent

    source_module_from_file = research_common.launch.module_from_file

    def module_from_file(name, path):
        module = source_module_from_file(name, path)
        if selected.anti_rollback_limit and name == "_r1_v7_physical_recorder":
            original_recorder = module.PhysicsRecorder

            class BrakeRecorder(original_recorder):
                def __init__(self, base, evaluation_worker):
                    super().__init__(base, evaluation_worker)
                    import torch
                    for key in ("cart_reverse_brake_torque_Nm", "cart_reverse_brake_joint_power_W",
                                "cart_reverse_brake_joint_velocity_pre_rad_s"):
                        self.buffers[key] = torch.zeros((base.num_envs, base.cfg.decimation, 4), device=base.device)
                        self.history[key] = []
                    key = "cart_reverse_brake_body_reaction_w_Nm"
                    self.buffers[key] = torch.zeros((base.num_envs, base.cfg.decimation, 3), device=base.device)
                    self.history[key] = []

                def sample(self):
                    index = self.cursor
                    super().sample()
                    import torch
                    base = self.base
                    zero = torch.zeros((base.num_envs, 4), device=base.device)
                    self.buffers["cart_reverse_brake_torque_Nm"][:, index] = getattr(
                        base, "r1_reverse_brake_torque_Nm", zero)
                    self.buffers["cart_reverse_brake_joint_power_W"][:, index] = getattr(
                        base, "r1_reverse_brake_joint_power_W", zero)
                    self.buffers["cart_reverse_brake_joint_velocity_pre_rad_s"][:, index] = getattr(
                        base, "r1_reverse_brake_joint_velocity_pre_rad_s", zero)
                    self.buffers["cart_reverse_brake_body_reaction_w_Nm"][:, index] = getattr(
                        base, "r1_reverse_brake_body_reaction_w_Nm",
                        torch.zeros((base.num_envs, 3), device=base.device))

            module.PhysicsRecorder = BrakeRecorder
        return module

    research_common.launch.CART.write_cart_urdf = write_cart
    research_common.launch.write_json = write_json
    research_common.launch.module_from_file = module_from_file
    worker.configure = configure
    sys.argv = [sys.argv[0], *remainder]
    worker.main()


if __name__ == "__main__":
    main()
