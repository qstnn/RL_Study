# Scripts

放这一条训练线专用的训练、播放和评估脚本。

现有脚本：

- `train_flat.sh`
- `train_rough.sh`
- `play_flat.sh`
- `play_rough.sh`
- `train_blind_rough.sh`
- `play_blind_rough.sh`

盲 rough 入口使用任务 `Rough-Deeprobotics-LightHW-Blind-v0`，默认 run_name 为 `lighthw_rough_blind`。它移除 policy 和 critic 的 height_scan，观测维度为 policy=60、critic=60、action=16。

这些脚本会先切到本 route 目录，再调用仓库根目录的 RSL-RL 启动入口，因此日志会自然落到本 route 的 `logs/` 下。

播放脚本额外支持 `--fixed_command VX VY WZ`，可直接用固定速度命令做回放检查；键盘回放的前进键也会跟随当前 `lin_vel_x` 上限。
