# Phase2.9 开发补充

## 目标

本轮仍然停留在 `Phase2`，不正式进入 `Phase3/4`，但开始为后续 `PPO + GRU/LSTM` 做铺垫。

这轮重点不是继续堆 reward 系数，而是把下面三件事做扎实：

1. 长时序 `credit assignment`
2. 更彻底的 `potential-based shaping`
3. 保留硬 `mask`、逐步退火手工 `prior`

## 本轮改动

### 1. PPO / GAE 长时序调整

- `Config.GAMMA` 从 `0.99` 提高到 `0.995`
- `Config.LAMDA` 从 `0.95` 提高到 `0.97`
- `GAE` 显式使用 `done mask`
- PPO batch 内对 `advantage` 做归一化

目的：

- 让 `450-550` 这段关键准备期和加速期的收益，更稳定地传回前面的决策
- 减少长局/短局混合时 advantage 尺度不一致的问题

### 2. 长时序 credit weight

本轮新增 `credit_weight`，对以下关键阶段样本做轻量加权：

- `speedup_prep`
- `flash_preserve_window`
- `buff_ready_window`
- `post_speedup`

原则：

- 只做轻量加权，不做激进重采样
- 加权有上限，避免训练被少量关键阶段样本“绑架”

### 3. readiness potential

在原有：

- `safety_potential`
- `resource_potential`
- `flash_potential`

之外，新增：

- `readiness_potential`

它主要描述：

- `500` 前是否具备 `buff/flash` 备战能力
- `500` 后是否仍具备速度对抗能力
- 当前是否在向更合适的 speedup-ready 状态前进

主 shaping 继续采用：

```text
gamma * Phi(s') - Phi(s)
```

### 4. prior anneal

保留：

- `legal action mask`
- `safe_prior`
- `prep_prior`

但新增：

- `safe_prior_scale`
- `prep_prior_scale`

按累计观测步数线性退火。

目标：

- 前期继续用 hand-crafted prior 稳定样本质量
- 后期逐步把控制权还给 PPO 本身

## 新增日志与监控

本轮新增重点观测项：

- `readiness_potential`
- `credit_weight`
- `safe_prior_scale`
- `prep_prior_scale`

`[GAMEOVER]` 里重点看：

- `ready_pot`
- `ready_pot_avg`
- `credit_w_avg`
- `safe_scale_avg`
- `prep_scale_avg`

## 这一轮之后怎么看是否还能留在 Phase2

如果下面几项继续改善，说明 `Phase2` 还能再推进一轮：

- `550+` 占比继续上升
- `600+` 不再只是偶发
- `buff_ready_at_speedup` 稳定抬升
- `safe_prior_count` 下降时，分数还能稳住

如果这些指标开始停滞，再考虑把长期目标推进到：

- `PPO + GRU/LSTM`
