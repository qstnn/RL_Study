"""执行真实奖励函数体，复现显式执行器上限与求解器上限混淆。"""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import torch

REPO = Path(__file__).resolve().parents[7]
SOURCE = REPO / 'source/rl_training/rl_training/tasks/manager_based/locomotion/multiskill/lighthw/wheel_unified_v5'


def test_effort_reward_uses_explicit_actuator_limits_in_joint_order():
    # 与真实GPU诊断同型：求解器1e9，PD轮24/腿48；执行器索引故意乱序。
    robot = SimpleNamespace(
        data=SimpleNamespace(applied_torque=torch.tensor([[24., 48., 24.]]),
                             joint_effort_limits=torch.full((1, 3), 1.e9)),
        actuators={
            'leg': SimpleNamespace(joint_indices=[2, 1], effort_limit=torch.tensor([[48., 48.]])),
            'wheel': SimpleNamespace(joint_indices=[0], effort_limit=torch.tensor([[24.]])),
        },
    )
    env = SimpleNamespace(scene={'robot': robot})
    tree = ast.parse((SOURCE / 'mdp.py').read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'effort_usage')
    namespace = {'torch': torch}
    path = SOURCE / 'torque.py'
    if path.exists():
        spec = importlib.util.spec_from_file_location('_regression_torque', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        namespace['torque'] = module
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE / 'mdp.py'), 'exec'), namespace)
    # (1^2+1^2+0.5^2)/3；不能因PhysX上限1e9得到近零惩罚。
    torch.testing.assert_close(namespace['effort_usage'](env), torch.tensor([.75]))
