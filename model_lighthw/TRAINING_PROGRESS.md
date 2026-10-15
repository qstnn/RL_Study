# LightHW 技能分支训练进展

最后更新：**2026-10-15 19:23（Asia/Shanghai，UTC+08:00）**。

**2026-10-15 19:23 新增网页 ZIP 获取方式，R6 训练配置保持。** 用户另一台电脑 Git 拉取有问题，新增
`training/multiskill/prepare_zip.py`：仅用 Python 标准库，通过 HTTPS 按固定已发布版本补齐17个规范网格，
逐个 SHA256 校验、正确文件跳过、临时文件完成后再替换；`--check` 只读离线检查。
补齐 README 中 ZIP 下载/解压/资产准备/启动流程，资产缺失提示同时支持 ZIP 与 Git。
没有修改 R6 奖励、PPO、载荷、控制或路径配置，没有改动原工作区 R1 源码；无需 `.git` 目录。

**本轮验证与训练日期：** 2026-10-15，在无 `.git` 且名称含空格的解压副本中，0kg/65kg CPU资产检查及
既有29项CPU测试通过。LFS指针识别、真实HTTPS下载一个网格（禁用PATH中的Git）、全部17网格哈希通过；
其余16个网格由本地已校验资产填入副本，未重复进行全量HTTP下载。注入错误HTTP响应后，原文件不变且临时文件清理。
原工作区33个多技能源码哈希保持。R6 GPU短验、机器人PPO和承载行为本轮均未执行。
验证见[ZIP验证记录](training/multiskill/routes/route_06_heavy_payload/records/zip_validation.json)。

**当前阶段/下一步：** ZIP交付入口已完成本地验证；另一台电脑下载技能分支ZIP并运行资产准备命令后，
继续按下方原启动流程做GPU短验及空载基线。远端Archive设置没有修改，原Git拉取方式仍可用。

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
