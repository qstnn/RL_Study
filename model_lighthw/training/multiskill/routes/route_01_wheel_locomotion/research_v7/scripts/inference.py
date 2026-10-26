"""固定研究评估的矩阵形状，避免把批大小造成的舍入差异当成物理槽位差异。"""

def infer_actor(actor, observation, mode='row'):
    """行推理保留模型/观测/动作，不量化、不裁剪；训练仍需单独验证批量和扰动鲁棒性。"""
    if mode == 'batch':
        return actor(observation)
    if mode != 'row' or observation.ndim != 2 or not len(observation):
        raise ValueError('Expected a nonempty observation matrix and row/batch mode')
    import torch
    return torch.cat([actor(observation[i:i+1]) for i in range(len(observation))], dim=0)
