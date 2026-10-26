# R4独立配置

实际配置为源码包 `two_leg_balance/spec.py`（版本/动作/奖励/验收/PPO）和 `env.py:TwoLegBalanceCfg`、`scene.py:BalanceSceneCfg`。
Actor305/Critic88/Action16，前后模式共用Actor；正式入口保存每次有效env.yaml、agent.json及contract.json。
详细数值、规划及启动命令见[路线说明](../README.md)。不修改旧高速R4、R1、R5A或R6配置。
