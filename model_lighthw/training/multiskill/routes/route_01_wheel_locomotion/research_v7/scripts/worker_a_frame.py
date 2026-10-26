"""Isolated, no-PPO R1 A-frame linkage comparison using the V7 physical worker.

The original V6 source, V7 worker, cart URDF, robot and motor limits remain
unchanged. This entry point replaces only the single virtual spring with two
passive, massless rod forces; it is not yet a manufacturable linkage model.
"""

from pathlib import Path

import worker
from linkage_variant import (
    FORCE_GUARD_N, HALF_WIDTH_M, MATCHED_LONGITUDINAL_DAMPING_NSPM,
    MATCHED_LONGITUDINAL_STIFFNESS_NPM, REST_LONGITUDINAL_M,
    matched_rod_parameters,
)


_configure = worker.configure
_module_from_file = worker.launch.module_from_file
_write_json = worker.launch.write_json


def variant_configure(out, n, seed, position, velocity, noise=False):
    cfg, agent = _configure(out, n, seed, position, velocity, noise)
    from rl_training.tasks.manager_based.locomotion.multiskill.lighthw.wheel_unified_v6 import cart as v6_cart
    from linkage_runtime import apply_a_frame_cart_forces
    from controller_a_frame import AFrameTeacher
    import controller

    # V6 TowingWheelAction resolves this global at every 1 ms physics step.
    # The V6 cart.py and controller.py files are never edited or copied over.
    v6_cart.apply_cart_forces = apply_a_frame_cart_forces
    controller.Teacher = AFrameTeacher
    rest, stiffness, damping = matched_rod_parameters()
    _write_json(out / "a_frame_linkage_contract.json", dict(
        variant="two_passive_axial_rods",
        robot_attachment="one original robot hitch point",
        cart_attachments="original cart front-center point +/- 0.20 m in cart local Y",
        longitudinal_rest_m=REST_LONGITUDINAL_M,
        half_width_m=HALF_WIDTH_M,
        rod_rest_length_m=rest,
        rod_stiffness_npm=stiffness,
        rod_damping_nspm=damping,
        original_matched_longitudinal_stiffness_npm=MATCHED_LONGITUDINAL_STIFFNESS_NPM,
        original_matched_longitudinal_damping_nspm=MATCHED_LONGITUDINAL_DAMPING_NSPM,
        axial_force_guard_n=FORCE_GUARD_N,
        attached_robot_and_cart_masses_unchanged=True,
        caveat="Virtual massless spring rods with no collision thickness; new hardware geometry requires fabrication and collision validation.",
    ))
    return cfg, agent


def variant_module_from_file(name, path):
    module = _module_from_file(name, path)
    if name != "_r1_v7_physical_recorder":
        return module
    parent = module.PhysicsRecorder

    class AFramePhysicsRecorder(parent):
        """Append exact, pre-integration two-rod force/torque/power at 1 kHz."""

        EXTRA_SHAPES = dict(
            a_frame_robot_force_w=(3,), a_frame_robot_torque_w=(3,),
            a_frame_cart_force_w=(3,), a_frame_cart_torque_w=(3,),
            a_frame_cart_rod_forces_w=(2, 3), a_frame_cart_points_w=(2, 3),
            a_frame_rod_tensions_n=(2,), a_frame_rod_lengths_m=(2,),
            a_frame_tip_clearance_m=(), a_frame_mechanical_power_w=(),
            a_frame_spring_energy_j=(), a_frame_spring_energy_rate_w=(),
            a_frame_damping_dissipation_w=(), a_frame_guard=(),
            a_frame_ground_guard=(), a_frame_sample_valid=(),
        )

        def __init__(self, base, evaluation_worker):
            import torch
            super().__init__(base, evaluation_worker)
            for key, shape in self.EXTRA_SHAPES.items():
                self.buffers[key] = torch.zeros((base.num_envs, base.cfg.decimation, *shape), device=base.device)
                self.history[key] = []
            original_interval_sample = base.r1_torque_recorder.sample

            def exact_drawbar_interval():
                before = base.r1_torque_recorder.cursor
                original_interval_sample()
                if base.r1_torque_recorder.cursor > before and hasattr(base, "r1_a_frame_last"):
                    self.buffers["applied_drawbar_force_w"][:, before] = base.r1_a_frame_last["robot_force_w"]

            base.r1_torque_recorder.sample = exact_drawbar_interval

        def sample(self):
            index = self.cursor
            super().sample()
            if not hasattr(self.base, "r1_a_frame_last"):
                return
            current = self.base.r1_a_frame_last
            fields = dict(
                a_frame_robot_force_w="robot_force_w",
                a_frame_robot_torque_w="robot_torque_w",
                a_frame_cart_force_w="cart_force_w",
                a_frame_cart_torque_w="cart_torque_w",
                a_frame_cart_rod_forces_w="cart_rod_forces_w",
                a_frame_cart_points_w="cart_points_w",
                a_frame_rod_tensions_n="raw_tensions_n",
                a_frame_rod_lengths_m="lengths_m",
                a_frame_tip_clearance_m="tip_clearance_m",
                a_frame_mechanical_power_w="mechanical_power_w",
                a_frame_spring_energy_j="spring_energy_j",
                a_frame_spring_energy_rate_w="spring_energy_rate_w",
                a_frame_damping_dissipation_w="damping_dissipation_w",
                a_frame_guard="guard",
                a_frame_ground_guard="ground_guard",
            )
            for target, source in fields.items():
                self.buffers[target][:, index] = current[source]
            self.buffers["a_frame_sample_valid"][:, index] = 1

    module.PhysicsRecorder = AFramePhysicsRecorder
    return module


def variant_write_json(path, value):
    if Path(path).name == "request.json":
        value = dict(value)
        value["linkage_variant"] = "two_passive_axial_rods"
        value["physical_drawbar_replacement"] = True
        value["teacher_external_force_uses_same_linkage"] = True
    return _write_json(path, value)


def main():
    worker.configure = variant_configure
    worker.launch.module_from_file = variant_module_from_file
    worker.launch.write_json = variant_write_json
    return worker.main()


if __name__ == "__main__":
    main()
