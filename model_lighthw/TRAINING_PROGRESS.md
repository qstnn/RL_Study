# LightHW 技能分支训练进展

最后更新：**2026-10-15 16:23（Asia/Shanghai，UTC+08:00）**。

本分支为`lighthw-multiskill-heavy-payload-r6`，基于本地`lighthw`提交`62e4614`。
用户确认本机正在进行R1训练；R6在独立工作区实现，供另一台电脑拉取启动。

**修改内容：** 补齐R6环境、PPO、固定刚性载荷质量/质心/惯量生成、显式惯量导入、
58/237/16载荷条件观测/动作契约、训练/同路线续训/回放/评估入口、独立日志与启动结果核验。
默认空载，显式按0/5/10/15/25/40/65kg分级持载；每次run固定质量，不自动晋级。
补齐共享依赖和基础入口缺失的skills命名空间，17个规范网格通过Git LFS同步。

**验证与训练日期：** 2026-10-15，29项CPU检查、29个Python文件编译、
静态本地导入闭包和全部入口检查通过。真实RSL-RL3.0.1以CPU合成数据完成2次更新并恢复追加1次，
检查checkpoint、Adam、学习率、迭代号和CSV；该模型不是机器人策略。
原工作区96个选定源码/资产哈希保持。本轮R6 GPU实例化、短验与机器人PPO均未运行，承载行为尚未验收。

**当前阶段/下一步：** 另一台完整lab环境拉取分支与LFS后，先做32环境100步短验、2次PPO保存短验，
再用README命令进行空载1000次持载基线。检查固定持载评估后显式升负载并复测轻载保留。
单episode缓加载、载荷碰撞体、热/结构模型与实机65kg持续承载不在本轮已验证范围。

[跨电脑启动](training/multiskill/README.md) · [R6命令](training/multiskill/routes/route_06_heavy_payload/README.md) ·
[验证记录](training/multiskill/routes/route_06_heavy_payload/records/validation.json)。
