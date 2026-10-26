"""从已存1kHz原始指标重新计算成功判据，独立于训练奖励和回合汇总。"""
import argparse
import numpy as np
import torch
from common import read, write
from two_leg_balance.core import BalanceState, support_geometry, schedule, desired_gravity, targets


def verify(folder):
    report = read(folder / "evaluation.json")
    hold = read(folder / "contract.json")["hold_seconds"]
    count = 0
    for case in report["cases"]:
        # np.load(NPZ)按键惰性解压；必须在物理步循环外加载一次。
        with np.load(folder / case["trace"]) as archive:
            data = {name: archive[name] for name in archive.files}
        i = case["trace_env"]
        indices = np.flatnonzero(data["active"][:, i])
        state = BalanceState(1, "cpu", hold)
        origin = data["origin"][indices[0], i] if "origin" in data else data["com"][indices[0], i, :2]
        state.reset(torch.tensor([0]), torch.tensor(origin[None]))
        previous = 0.
        for row in indices:
            time = float(data["t"][row, i])
            if abs((time - previous) - .001) > 2e-6:
                raise ValueError("R4 trace has a gap or reset within an active episode")
            previous = time
            names = ("com", "com_vel", "force_z", "clearance", "gravity", "ang_vel",
                     "base_z", "line_error", "body_force", "torque_peak")
            m = {k: torch.from_numpy(np.array(data[k][row, i:i + 1])) for k in names}
            mode = torch.tensor(data["mode"][row, i:i + 1])
            recomputed = support_geometry(m["com"], torch.tensor(data["wheel_pos"][row, i:i + 1]), mode)
            torch.testing.assert_close(m["line_error"], recomputed, atol=1e-5, rtol=1e-4)
            state.update(.001, torch.tensor([time]), m,
                         torch.tensor(data["mode"][row, i:i + 1]), torch.tensor(data["target_gravity"][row, i:i + 1]))
        for name in ("entered", "held", "failed", "success"):
            if bool(getattr(state, name)[0]) != case[name]:
                raise AssertionError(f"{case['trace']} env{i} {name} disagrees")
        for name in ("best_hold", "best_recovery"):
            if abs(float(getattr(state, name)[0]) - case[name]) > .002:
                raise AssertionError(f"{case['trace']} env{i} {name} disagrees")
        count += 1
    result = dict(cases=count, protocol_recomputed=True, support_geometry_recomputed=True, behavior_accepted=False)
    write(folder / "trace_verification.json", result)
    return result


if __name__ == "__main__":
    from pathlib import Path
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    print(verify(parser.parse_args().folder))
