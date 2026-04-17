# 代码规范补充 - Phase2.9

## 适用范围

本补充规范只覆盖 `Phase2.9` 新引入的：

- 长时序 `credit assignment`
- `credit_weight`
- `readiness_potential`
- `prior anneal`

## 规则

1. `GAMMA / LAMDA / POTENTIAL_GAMMA` 要一起检查，避免主折扣和势能折扣脱节。
2. `GAE` 必须显式使用 `done mask`，终局样本不能继续 bootstrap。
3. 如果引入 `credit_weight`，必须限制上界，并同步把平均值打进 `workflow` 日志和监控。
4. `legal action mask` 是硬约束，可以长期保留；`safe_prior / prep_prior` 是软引导，必须支持退火。
5. 退火过程至少要可观测到 `safe_prior_scale` 和 `prep_prior_scale`。
6. 如果状态势能已拆成 `safety/resource/flash/readiness`，则 `GAMEOVER` 日志和 monitor 都要同步更新。
7. 正式进入 `GRU/LSTM` 之前，先把非递归版 `Phase2` 指标尽量收稳，避免后续无法归因。
