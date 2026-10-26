"""纯张量地形命令变换；不操作Actor动作或奖励。"""
import torch

VERSION = 'lighthw_terrain_command_distribution_v1'


def flat_at_position(position, origins, size, assigned_columns, flat_columns):
    """按实际tile分类；地图边框回退到原分配列，避免猜测边框地形能力。"""
    size = position.new_tensor(size)
    cell = torch.floor((position[:, :2] - (origins[0, 0, :2] - size / 2)) / size).long()
    inside = (cell >= 0).all(1) & (cell[:, 0] < origins.shape[0]) & (cell[:, 1] < origins.shape[1])
    column = torch.where(inside, cell[:, 1], assigned_columns)
    return torch.isin(column, torch.as_tensor(flat_columns, device=column.device)), inside


def terrain_command(base_command, flat, flat_cap=3., rough_cap=1., minimum=.08):
    """保留原零命令/锚点/随机分位，只把rough非零分布0.08..3映射成0.08..1。

    运动命令是前进速度而非轮速前馈；控制输出始终全部来自Actor。
    地形切换仅重映射已经采样的命令，不额外抽随机数，不改变8秒重采样周期。
    """
    if not 0 < minimum < rough_cap <= flat_cap:
        raise ValueError('Require0 < minimum < rough_cap <= flat_cap')
    if rough_cap == flat_cap:
        return base_command.clone()
    ratio = ((base_command - minimum) / (flat_cap - minimum)).clamp(0., 1.)
    rough = minimum + ratio * (rough_cap - minimum)
    return torch.where(base_command == 0., 0., torch.where(flat, base_command, rough))
