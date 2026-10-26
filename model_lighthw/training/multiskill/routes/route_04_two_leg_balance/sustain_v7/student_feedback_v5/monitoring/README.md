# R4a 正式训练只读进度显示

更新：2026-10-23 15:24（Asia/Shanghai）；正式训练日期2026-10-23。

用户报告“停在启动界面”，实际是终端没有转发迭代进度。已检查主机进程、`session.json`、checkpoint和跨时间`updates.jsonl`，当前run持续从54推进到71、82、90，最近第107/1000次新增更新（global141）。初始化已完成，训练worker仍采样；父启动器等待worker结束是正常进程关系。

`run_ppo.py`把worker输出重定向到`console.log`，但worker每轮仅向`updates.jsonl`写进度。`console.log`停在网络结构输出不能说明训练停止。此处用独立stdlib监控读取已有JSONL，不更改冻结训练源、奖励、模型或训练文件，不启动、停止或重启训练。

在**另一个终端**运行（当前训练终端保持运行）：

```bash
cd /home/user/RL_lab/rl_training
R4_BASE=model_lighthw/training/multiskill/routes/route_04_two_leg_balance/sustain_v7/student_feedback_v5
/home/user/miniconda3/envs/lab/bin/python "$R4_BASE/monitoring/monitor.py" \
  --run "$R4_BASE/ppo_integration_v1/runs/2026-10-23_15-13-34_formal"
```

默认每5秒输出一条：session记录状态、本次新增更新/预算、模型global更新号、Critic Adam实际步数、日志最新更新时间间隔、最新已保存checkpoint。`--once`只读取一次；`--interval 2`改为2秒。监控中按Ctrl+C只退出监控，不会停止训练。进度行临时写到一半或日志尚未创建时显示等待；长时间没更新只显示日志年龄，不自动判定卡死。`running`是session记录状态，不替代进程活性证据。

当前run从已准入global34重新开始。第一次`15-06-07_formal`实际完成106次新增更新后在15:13:22收到KeyboardInterrupt；其第100次新增更新保存的`model_000134.pt`保留，后6次仅有更新日志。第二次`15-13-34_formal`并未从该134模型续接，两次local计数不能相加成当前模型global计数。

验证：2项半行/缺失文件CPU检查和当前真实run的CLI显示通过；完整冻结来源校验仍匹配准入快照。该监控是运行状态显示，不是中途冻结评估或训练成果验收。当前不需要再次执行train；训练完成后由用户通知进行模型审查。
