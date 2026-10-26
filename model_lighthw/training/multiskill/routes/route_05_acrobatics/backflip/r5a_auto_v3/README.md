> reference 文本快照：当前目录和入口见 [R5 路线索引](../../README.md)。下文保留原版本研究与离线交付说明；文中模型、数据及旧 ZIP 不随本分支上传，旧补丁安装步骤对应其原始目录布局。

# R5A v3：固定工况，连续训练，末尾统一评估，精简USB回收

先关闭初始姿态随机化、物理域随机化和额外观测噪声。PPO保留策略动作采样探索发力；评估/play使用确定性Actor。先学习固定平地空载工况，当前并未完成后空翻。

## 安装与唯一训练命令

把补丁ZIP和同名`.sha256`拷到另一台电脑并解压。进入原来能运行`model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/scripts/r5a_auto_v2.sh`的完整R5A项目根目录，假设补丁解压在Downloads：

```bash
python3 ~/Downloads/lighthw-r5a-autoflip-v3-fixed-patch-20261020/install.py "$PWD"
bash model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/scripts/r5a_auto_v3.sh check
bash model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/scripts/r5a_auto_v3.sh train
```

第一行改成补丁实际解压路径。安装只新增v3，先校验基础代码/URDF/网格/天空与补丁SHA；不覆盖旧任务、不联网、不需要Git。相同文件可重复安装，同名不同内容会拒绝。使用原完整工程的Python/Isaac Sim/依赖/资产；原有`R5A_PYTHON`、`R5A_ISAAC_SIM`设置继续生效。

**一条train命令持续完成全部流程：**

1. 单个训练进程连续完成默认512环境、2000次更新尝试。中途只保存本机检查点，不暂停评估、不分段重启。
2. 训练结束后，只进行一次统一候选评估阶段：比较最后最多3个已保存模型（包括最终模型），每个固定工况回放1次。选择依据依次为物理一致性、完整成功、合格单圈/恢复、有效转角等。候选更新数从检查点内部读取，不能用`model_文件名`当实际更新数。
3. `usb_review`只保留选中的1个模型、完整选中回放及精简训练证据。其他候选仅保留评分摘要。选择范围是最后几个候选，不宣称全训练历史最优。都没完成任务时标记`best_available_not_passed`，不会冒充成功。

训练期间仍保留非有限值/KL保护；连续5次更新被拒绝时会停止并导出故障摘要，这是异常保护，不是分段核查。物理动量筛查在结束后的统一评估进行。

## USB里有哪些资料

只拷终端最后打印的`[USB REVIEW ZIP]`和同名`.sha256`。默认位于`usb_review/route_05a/`。无需拷贝`offline_sessions`或整个项目。

精简ZIP包含：

- 1个选中检查点（包含Actor/Critic/Adam/计数，便于后续分析或同版本接续）；
- 候选比较、成功/物理判据、选中完整6秒1kHz轨迹、50Hz策略输入/动作、录像、图表；
- 全程PPO更新与训练曲线、所有结束回合的分窗口统计、最多20条代表回合；
- 单套源码/依赖快照、运行配置、机器人读回、迁移来源、日志尾部与文件SHA。

**不装入USB：**其他模型、全量回合JSONL、完整大日志、重复源码及USD缓存。这些仍保留在训练电脑打印的`[LOCAL FULL DATA]`目录（默认`offline_sessions/route_05a/`），必要时可追加提取。统计摘要不能替代被省略的全部原始训练回合；选中的冻结回放原始数据完整保留。

如果只想评价最终一个模型，可明确使用：

```bash
bash model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/scripts/r5a_auto_v3.sh train --candidates 1
```

默认3个末期候选更能避免最后一次更新偶然变差；仍只在训练结束后统一评估。`--save-every 200`是本机存档频率，不是评估频率。

## 起点、物理和奖励

内置已审查v2第1800次Actor和探索std，无需另找模型；新建Critic、Adam和v3计数。旧v2不能直接传给`--checkpoint`，该参数仅接受相同实现指纹的v3模型。正式训练不从本机短验模型开始。

1ms物理/20ms控制；速度求解迭代2→16，位置仍8；保留质量、腿/轮48/24Nm及12.57/37.7rad/s、轮armature 0.00243216、自碰撞与陀螺项。冻结2000 Actor的H漂移由0.07051降到0.01976kg·m²/s；1800 Actor降到0.01633。动量筛查收紧至`max(0.02, 0.05*|H起始|)`，不是零残差或硬件认证。

策略自主寻找预蹲、蹬伸、收腿和落地；无高度逐级课程、参考动作、助跳或根速度注入。真实有效腾空后才结算有界预蹲/后旋协同，紧凑奖励仅随新增有效后旋增加；提高旋转收益，降低单纯跳高收益；约100°阶段已有非零且有上界的恢复反馈。成功仍要求350–385°单圈、合法触地和终末四轮连续站稳1秒。跑满6秒、mean episode length=300或被选中都不等于成功。

## 回放与重打包

回放最新选中模型，World相机可自由移动，无自动跟随：

```bash
bash model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/scripts/r5a_auto_v3.sh play
```

若需从某个v3检查点继续，`--updates`表示新增总预算；仍连续训练到结束：

```bash
bash model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/scripts/r5a_auto_v3.sh train --checkpoint /实际路径/selected.pt --updates 2000
```

训练中断/打包失败，完整资料留在`[LOCAL FULL DATA]`。使用实际目录重新生成精简包：

```bash
bash model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/scripts/r5a_auto_v3.sh collect --folder /实际完整运行目录
```

若统一评估的候选均物理不合格，会标为`physics_rejected`并保留证据，不把它设为默认回放检查点。先带回精简ZIP审查，不修改容差强行继续。
