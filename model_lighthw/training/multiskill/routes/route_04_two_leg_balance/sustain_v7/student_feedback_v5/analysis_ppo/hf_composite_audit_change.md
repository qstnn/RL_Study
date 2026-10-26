# 高频门独立审查逻辑修正记录

初版 `review_physics.py` SHA256 为 `6c01ab8cfa475373dbbeb2eb71fba9b5648722199d6f11785ef97cddae034c54`，其完整源码字节保存在 `review_physics_before_composite_audit.txt`；修正后为 `e7fecdfe14c28943c8b258a250f50771fa416686358c9c6eceec4a6206be9654`，可应用的完整差异保存在 `review_physics_composite_audit.patch`。初版报告保存在 `results/*_physics_review_before_composite_audit.json`，修正后的报告使用原 canonical 文件名。物理 trace、PPO 执行源和门槛均未更动。

逻辑差异（`review_physics.py`）：

```diff
- hf_limit_disagreement = count(recorded_each_hf_pass != independent_each_hf_pass)
- reconstruction_verified requires hf_limit_disagreement == 0
+ hf_limit_disagreement = count(recorded_each_hf_pass != independent_each_hf_pass)
+ hf_limit_disagreement_details = exact tick, time, channel, both RMS values, limit, phase and capture status
+ smooth_gate_disagreement = count(all(recorded_each_hf_pass) != all(independent_each_hf_pass))
+ reconstruction_verified requires smooth_gate_disagreement == 0
```

独立数值始终严格使用原 HF 阈值 `[0.02, 0.05, 0.003490658503988659, 0.5]`；没有给物理门增加容差。下表的四项顺序为 COM 速度、角速度、腿位置、轮速度；`T/F` 是各项严格 `<=` 门。三帧都还未捕获，`held=false`，且记录与离线的综合 `all(四项门)` 都是 false，故完整 qualified、hold、capture、first-break 状态机没有差异。

| Run / slot / tick / 时间 | Phase | 记录四项 HF RMS | 独立四项 HF RMS | 记录门 / 独立门 |
| --- | ---: | --- | --- | --- |
| `n16_resume_01` / 6 / 5177 / 5.178000450134277 s | 2 | `[0.011439737863838673, 0.057513754814863205, 0.0034905525390058756, 0.334244966506958]` | `[0.011439973404655227, 0.0575146377894477, 0.003490695408200124, 0.334244375122104]` | `[T,F,T,T]` / `[T,F,F,T]` |
| `n16_formal_smoke_01` / 6 / 2082 / 2.0830001831054688 s | 1 | `[0.014009865000844002, 0.09515322744846344, 0.0034905734937638044, 0.20791801810264587]` | `[0.014009784772805208, 0.09515229648193127, 0.003490689641343593, 0.20791674106349756]` | `[T,F,T,T]` / `[T,F,F,T]` |
| `n16_formal_smoke_01` / 14 / 2895 / 2.8960001468658447 s | 1 | `[0.013851713389158249, 0.08126243203878403, 0.0034906379878520966, 0.5000439286231995]` | `[0.013851808533200286, 0.08126262708786747, 0.003490669396891686, 0.5000431287631781]` | `[T,F,T,F]` / `[T,F,F,F]` |

初版 formal-smoke 汇总 `14/16`，原因仅是把上述两个单项比较位不等当作状态机不一致。修正后综合 smooth 门逐帧无差异，状态机及全部原始物理量复核通过，formal-smoke 为 `16/16`。报告仍逐帧披露这两处浮点边界差异，且捕获后的各 HF 峰值均低于原阈值。新增 CPU 反例测试验证：一项 HF 浮点跨线、另一项明确超线时，单项差异被记录，但综合门判定不改变；真正的综合 HF 违例仍使审查失败。
