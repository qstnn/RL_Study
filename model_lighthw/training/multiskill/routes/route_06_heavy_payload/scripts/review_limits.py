"""Read-only actuator limit diagnostics; never changes gains, rewards or physics."""

def actuator_effort_limits(robot):
    """实际名义限幅：显式执行器effort_limit与PhysX限幅的较小值，按关节原序返回。"""
    import torch
    limits=robot.data.joint_effort_limits.clone()
    for actuator in robot.actuators.values():
        ids=actuator.joint_indices
        limits[:,ids]=torch.minimum(limits[:,ids],actuator.effort_limit)
    return limits
