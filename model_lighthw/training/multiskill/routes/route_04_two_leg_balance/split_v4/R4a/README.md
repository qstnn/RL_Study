# R4a 后腿支撑站立

固定`rear`模式，独立Actor/Critic/Adam和`runs/`。当前优先审查分支。

`unload_diag_v1`小预算诊断已于2026-10-21完成；100/150/最终模型的冻结评估共0/18成功，暂停后续500次预算。详见[诊断审查](../results/2026-10-21_R4a_diag_200_review/README.md)。以下是已执行命令，仅供追溯，不建议原样重跑：

```bash
cd /home/user/RL_lab/rl_training
bash model_lighthw/training/multiskill/routes/route_04_two_leg_balance/split_v4/scripts/r4a.sh train \
  --reward_profile unload_diag_v1 --num_envs 512 --max_iterations 200 --save_every 50 --seed 42
```

环境、奖励档案、验收和后续升级条件见[拆分v4说明](../README.md)。完整目标尚未验收。
