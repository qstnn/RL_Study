"""WBC teacher using the same two-rod force model as the physical substeps."""

import numpy as np
import torch

from controller import Teacher as SingleDrawbarTeacher
from linkage_runtime import current_linkage_state


class AFrameTeacher(SingleDrawbarTeacher):
    """Only replace the external drawbar force in the existing teacher."""

    def apply(self, baseline, step):
        b, r = self.base, self.robot
        view = r.root_physx_view
        linkage = current_linkage_state(b)
        cpu = lambda x: x.detach().cpu().numpy().astype(float)
        values = dict(
            M=cpu(view.get_generalized_mass_matrices()),
            g=cpu(view.get_gravity_compensation_forces()),
            c=cpu(view.get_coriolis_and_centrifugal_compensation_forces()),
            J=cpu(view.get_jacobians()), arm=cpu(view.get_dof_armatures()),
            q=cpu(r.data.joint_pos), qd=cpu(r.data.joint_vel),
            pos=cpu(r.data.root_pos_w), quat=cpu(r.data.root_quat_w),
            v=cpu(r.data.root_com_lin_vel_w), omega=cpu(r.data.root_ang_vel_w),
            body_com=cpu(r.data.body_com_pos_w),
            hitch=cpu(linkage["robot_point_w"]),
            external=cpu(linkage["robot_force_w"]),
            default=cpu(r.data.default_joint_pos),
            cmd=cpu(b.command_manager.get_command("base_velocity")[:, 0]),
            cart_v=cpu((b.scene["trailer"].data.root_com_lin_vel_w * b.r1_tangent).sum(-1)),
            towed_kg=cpu(b.r1_towed_kg), origin=cpu(b.scene.env_origins),
            normal=cpu(b.r1_normal), slope=cpu(b.r1_slope_rad),
            elapsed=cpu(b.episode_length_buf) * 0.02,
            contact=cpu(b.scene["contact_forces"].data.net_forces_w[:, b.r1_wheel_sensor_ids]),
            applied_tau=cpu(r.data.applied_torque),
            contact_history=cpu(self.physical.buffers["contact_force_w"]),
            friction_history=cpu(self.physical.buffers["friction_force_w"]),
            contact_cop=cpu(self.physical.buffers["native_contact_cop_w"]),
            contact_count=cpu(self.physical.buffers["native_contact_count"][:, -1]),
            contact_separation=cpu(self.physical.buffers["native_contact_min_separation_m"][:, -1]),
            contact_count_history=cpu(self.physical.buffers["native_contact_count"]),
            contact_separation_history=cpu(self.physical.buffers["native_contact_min_separation_m"]),
            wheel_center_history=cpu(self.physical.buffers["wheel_pos_w"]),
        )
        action = baseline.clone()
        self.last = {}
        for i in range(b.num_envs):
            if i in self.closed or i in self.failures or values["elapsed"][i] < 1.5:
                continue
            try:
                candidate, info = self.single(i, values)
                self.last[i] = info
                if not info["success"]:
                    self.failures[i] = dict(
                        step=step, time_s=float(values["elapsed"][i]),
                        **{k: v for k, v in info.items() if k not in ("x", "tau", "contact_forces", "acceleration", "rolling_velocity")},
                    )
                    continue
                action[i] = torch.as_tensor(candidate, device=action.device, dtype=action.dtype)
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                self.failures[i] = dict(
                    step=step, time_s=float(values["elapsed"][i]),
                    reason=type(exc).__name__ + ": " + str(exc),
                )
        return action
