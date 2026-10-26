"""只增加物理步力矩观测；动作/PD/仿真积分仍使用原Articulation实现。"""
from isaaclab.assets import Articulation


class TorqueObservedArticulation(Articulation):
    def write_data_to_sim(self):
        super().write_data_to_sim()
        recorder = getattr(self, 'r1_torque_recorder', None)
        if recorder is not None:
            recorder.sample()
