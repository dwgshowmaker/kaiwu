# 基于 DIY 的 PPO迭代开发方案

## 文档目标

本文档用于指导当前峡谷追猎智能体的 PPO 迭代开发。

实现策略调整为：

- `agent_ppo` 作为 **参考基线**，尽量保持不动
- `agent_diy` 作为 **实际开发载体**，所有新代码都落在这里

算法增强仍按以下顺序推进：

1. 先做 **特征 + 奖励** 升级
2. 再做 **16 动作闪现**
3. 然后做 **训练配置随机化 / 课程学习**
4. 最后再做 **模型升级**

该顺序的核心原则是：

- 先解决“信息不足、目标不对齐”的问题
- 再释放环境中最强的动作能力
- 再提升泛化
- 最后才增加模型复杂度

工程落地原则是：

- 先把 `agent_ppo` 的可运行 PPO 骨架迁入 `agent_diy`
- 后续所有迭代都在 `agent_diy` 内完成
- 保留 `agent_ppo` 作为对照组，方便回归与效果比较

---

## 一、现有 PPO 基线概述

当前基线是一个轻量 PPO Actor-Critic，实现链路完整，但仍属于“最简可运行版本”。

### 1. 当前代码职责划分

| 文件                                     | 职责                               |
| :--------------------------------------- | :--------------------------------- |
| `agent_ppo/feature/preprocessor.py`    | 观测预处理、奖励塑形               |
| `agent_ppo/conf/conf.py`               | 特征维度、动作维度、PPO 超参数     |
| `agent_ppo/feature/definition.py`      | `ObsData` / `SampleData` / GAE |
| `agent_ppo/model/model.py`             | Actor-Critic 网络                  |
| `agent_ppo/agent.py`                   | 推理、动作采样、模型保存加载       |
| `agent_ppo/workflow/train_workflow.py` | 训练交互、采样、回传样本           |
| `agent_ppo/conf/monitor_builder.py`    | 监控面板配置                       |

### 2. 开发载体选择

后续开发不建议直接在 `agent_ppo` 上堆改动，原因如下：

1. `agent_ppo` 已经是当前唯一可运行基线，适合作为对照组
2. `agent_diy` 本来就是给自定义算法准备的目录，更适合承接新版本
3. 把新代码放进 `agent_diy` 后，后续可以随时切回 `ppo` 对比，不会把基线污染掉

因此，本文后续提到的“改代码”，默认都指向 `agent_diy`。

### 3. 当前基线的实际情况

- 输入是 **40 维特征**
- 动作空间只有 **8 个移动动作**
- 奖励主要是 **存活奖励 + 远离怪物奖励**
- 局部地图只取了一个简化后的 **4x4 通行性**
- 没有显式建模 **宝箱、buff、闪现收益、逃生路线质量**

### 4. 当前基线的主要短板

1. 训练目标和比赛得分不完全对齐现在更像“尽量别死”，但比赛实际上是“活得久并尽量多拿宝箱”。
2. 观测表达力偏弱21x21 局部地图被压缩过度，宝箱和 buff 信息没有形成有效输入。
3. 动作能力不完整环境支持 16 动作，但当前策略无法使用闪现。
4. 泛化训练不足
   默认环境参数较固定，容易学成“固定套路”。

---

## 二、总体迭代方案

建议采用四阶段开发，每阶段都要求“先完成、再验证、再进入下一阶段”。

| 阶段    | 主题                  | 目标                                       | 是否建议立即做 |
| :------ | :-------------------- | :----------------------------------------- | :------------- |
| Phase 0 | DIY 骨架迁移          | 在 `agent_diy` 中复刻一个可运行 PPO 基线 | 是             |
| Phase 1 | 特征 + 奖励升级       | 让策略真正学会“保命 + 抢分”              | 是             |
| Phase 2 | 16 动作闪现           | 补齐动作空间上限                           | 是             |
| Phase 3 | 环境随机化 / 课程学习 | 提升隐藏图泛化与稳定性                     | 是             |
| Phase 4 | 模型升级              | 提升复杂策略表达能力                       | 放最后         |

---

## 三、Phase 0：将可运行 PPO 骨架迁移到 DIY

### 1. 阶段目标

当前 `agent_diy` 还是模板工程，`agent.py`、`algorithm.py`、`model.py`、`train_workflow.py` 等核心逻辑都没有完成。

因此在做任何策略增强前，必须先完成一件事：

- 以 `agent_ppo` 为参考
- 在 `agent_diy` 中复刻一份 **可运行、可训练、可评估** 的 PPO 基线

注意：

- 这是“迁移骨架”，不是一开始就做大改
- Phase 0 的目标是让 `diy` 版本先具备与 `ppo` 近似的行为和训练链路

### 2. 建议迁移的文件

- `agent_diy/agent.py`
- `agent_diy/algorithm/algorithm.py`
- `agent_diy/model/model.py`
- `agent_diy/feature/definition.py`
- `agent_diy/feature/preprocessor.py`（新建）
- `agent_diy/workflow/train_workflow.py`
- `agent_diy/conf/conf.py`
- `agent_diy/conf/monitor_builder.py`

### 3. 迁移原则

1. 先保持行为近似，再做增强

- 先把 `agent_ppo` 的 8 动作、基础 MLP、基础奖励链路搬到 `agent_diy`
- 不要在迁移时顺手加入大量新特性

2. 代码位置可以重构，但接口要兼容框架

- `Agent` 的 `predict / exploit / learn / save_model / load_model`
- `workflow` 的采样与发送样本逻辑
- `SampleData` 的字段定义

3. 先确保 `algorithm_name = "diy"` 可运行

- 只有 `diy` 版本真正跑起来，后面的 Phase 1 才有实施意义

### 4. Phase 0 验收标准

- `train_test.py` 切到 `algorithm_name = "diy"` 后可正常启动训练
- `agent_diy` 训练曲线与当前 `agent_ppo` 同量级
- 推理、保存模型、加载模型链路全部可用

---

## 四、Phase 1：特征 + 奖励升级

### 1. 阶段目标

在不改 PPO 主体损失的前提下，先让输入信息更完整、奖励更贴近真实目标。

这一阶段不建议改动 `algorithm.py` 的 PPO 损失公式，优先改：

- `agent_diy/feature/preprocessor.py`
- `agent_diy/conf/conf.py`
- `agent_diy/feature/definition.py`
- `agent_diy/conf/monitor_builder.py`

### 2. 特征改造方案

当前 40 维特征过于简化，建议升级为“结构化向量特征 + 更强局部地图摘要”。

推荐的第一版特征设计如下：

#### hero_self

建议保留并扩展为 8 维：

- `hero_x_norm`
- `hero_z_norm`
- `flash_cd_norm`
- `flash_ready`
- `buff_remain_norm`
- `has_buff`
- `step_norm`
- `remaining_step_norm`

#### monster_features

每只怪物建议扩展为 8 维，共 16 维：

- `exist_or_in_view`
- `rel_x_norm`
- `rel_z_norm`
- `dist_norm`
- `dir_x_sign`
- `dir_z_sign`
- `speed_norm`
- `is_close_threat`

说明：

- 不要只保留绝对坐标，优先给相对位置
- `is_close_threat` 可用距离阈值编码，例如距离小于 6 格为 1

#### treasure_features

建议加入最近 2 个宝箱的特征，共 10 维：

- `exists`
- `rel_x_norm`
- `rel_z_norm`
- `dist_norm`
- `is_same_quadrant_as_escape`

每个宝箱 5 维，选最近两个即可；多余宝箱忽略。

#### buff_features

建议加入最近 2 个 buff 的特征，共 10 维：

- `exists`
- `rel_x_norm`
- `rel_z_norm`
- `dist_norm`
- `is_worth_pick`

其中 `is_worth_pick` 可用启发式判断：

- 附近有怪物时 buff 更值钱
- 前期 buff 更值钱
- 已有 buff 时价值下降

#### local_map_features

建议不要再只取 4x4，而是改为以下两种方式之一：

方案 A：**21x21 下采样到 7x7**

- 共 49 维
- 每个格子记录该区域可通行比例

方案 B：**8 方向路线统计（已舍弃，作备选方案）**

- 共 24 维
- 每个方向统计 3 个量：
  - `step_1_passable`
  - `step_2_passable`
  - `corridor_length`

选择方案A。

#### progress_features

建议增加环境节奏特征，共 4 维：

- `monster2_eta_norm`
- `monster_speedup_eta_norm`
- `treasure_collected_ratio`
- `buff_collected_ratio`

### 3. Phase 1 实际落地特征规模

Phase 1 已采用 **105 维** 特征，仍保持轻量 MLP 可承受的规模，同时比 Phase 0 的 40 维提供更多任务信息。

实际落地版本：

- hero: 8
- monsters: 16
- treasures: 10
- buffs: 10
- local map: 49（21x21 视野下采样为 7x7）
- legal action: 8
- progress: 4

总计：**105 维**

### 4. 奖励改造方案

奖励必须从“只鼓励活着”升级成“鼓励高分生存”。

建议保留原有生存主线，但补上以下奖励项：

| 奖励项                   | 建议                      | 说明                         |
| :----------------------- | :------------------------ | :--------------------------- |
| `survive_reward`       | `+0.01`                 | 每步基础生存奖励             |
| `danger_escape_reward` | `0.05 * min_dist_delta` | 与最近怪物拉开距离时给正奖励 |
| `treasure_reward`      | `+1.0 ~ +1.5`           | 宝箱数量增量奖励             |
| `buff_reward`          | `+0.3 ~ +0.8`           | 收到 buff 时奖励，危险时更高 |
| `good_flash_reward`    | `+0.3 ~ +0.8`           | 闪现后危险显著下降时奖励     |
| `bad_flash_penalty`    | `-0.1 ~ -0.3`           | 浪费闪现时惩罚               |
| `stuck_penalty`        | `-0.02`                 | 连续原地踏步或撞墙时惩罚     |
| `fail_penalty`         | `-10.0`                 | 被抓到的终局惩罚             |
| `complete_bonus`       | `+8.0 ~ +10.0`          | 存活到结束的奖励             |

### 5. 奖励实现原则

1. 所有事件奖励尽量基于“计数差分”

- 宝箱奖励用 `treasure_collected_count` 的增量
- buff 奖励用 `collected_buff` 的增量
- 避免仅凭位置猜测是否吃到物件

2. 闪现奖励必须是“条件奖励”

- 闪现后最近怪物距离明显变大
- 或闪现路径上拿到了宝箱 / buff
- 否则不要给奖励

3. 奖励量级不要失衡

- 单次宝箱奖励可以明显高于一步生存奖励
- 但不能高到导致模型无视死亡风险

### 6. Phase 1 验收标准

建议至少观察以下指标：

- 平均存活步数是否上升
- 平均总分是否上升
- 平均宝箱数是否上升
- 失败局中是否更少出现“撞墙死”“原地抖动”
- reward 曲线是否比旧版更稳定

### 7. Phase 1 需要新增的监控项

建议在 `agent_diy/conf/monitor_builder.py` 里新增：

- `episode_steps`
- `total_score`
- `treasure_count`
- `buff_count`
- `flash_count`
- `fail_rate`
- `avg_min_monster_dist`
- `stuck_count`

---

## 五、Phase 2：16 动作闪现

### 1. 阶段目标

补齐动作空间，让策略能学会：

- 危险时闪现逃命
- 隔墙位移
- 顺路收集宝箱和 buff

### 2. 需要改动的文件

- `agent_diy/conf/conf.py`
- `agent_diy/feature/definition.py`
- `agent_diy/feature/preprocessor.py`
- `agent_diy/agent.py`
- `agent_diy/model/model.py`

### 3. 具体改动

#### 动作维度

- 将 `ACTION_NUM` 从 `8` 改为 `16`
- `SampleData.prob` 和 `legal_action` 维度同步改为 16
- Actor 输出维度同步改为 16

#### legal_action

- 不再只截取移动动作
- 直接使用环境提供的 16 维合法动作
- 闪现合法性由环境冷却控制

#### 闪现相关特征

建议至少保留：

- `flash_cd_norm`
- `flash_ready`
- `last_action_is_flash`
- `recent_flash_count`

#### 闪现奖励

建议先做“保守奖励”：

- 闪现后最近怪物距离增加明显时奖励
- 闪现路径收集到宝箱 / buff 时奖励
- 远离怪物不足、且未获得收益时，给轻微惩罚

### 4. 风险点

1. 动作空间变大后探索会更难前期训练曲线可能变差，这是正常现象。
2. 闪现很容易被学成“乱交技能”所以必须同时有 `good_flash_reward` 和 `bad_flash_penalty`。
3. 如果 Phase 1 特征不够好，Phase 2 效果会很有限
   因此不要跳过 Phase 1。

### 5. Phase 2 验收标准

- 危险局面下闪现使用率上升
- 被近身怪物贴脸击杀的比例下降
- 平均宝箱数不下降，最好提升
- 总分高于“只有移动动作”的最优模型

---

## 六、Phase 3：训练配置随机化 / 课程学习

### 1. 阶段目标

这一阶段的目标不是单图刷分，而是提升对隐藏图和不同节奏配置的泛化能力。

### 2. 改动范围

- `agent_diy/workflow/train_workflow.py`
- `agent_diy/conf/train_env_conf.toml`

### 3. 推荐做法

建议把 `train_env_conf.toml` 作为“默认上限配置”，在 `agent_diy/workflow/train_workflow.py` 中按 episode 动态调整训练难度。

### 4. 建议的课程学习三阶段

#### Stage A：生存入门

- `map = [1, 2, 3, 4]`
- `map_random = true`
- `treasure_count = 8`
- `buff_count = 2`
- `monster_interval = 500`
- `monster_speedup = 800`
- `max_step = 600`

目标：

- 先学会基础保命
- 减少训练前期的大量秒死样本

#### Stage B：标准训练

- `map = [1, 2, 3, 4, 5, 6, 7]`
- `map_random = true`
- `treasure_count = 10`
- `buff_count = 2`
- `monster_interval = 300`
- `monster_speedup = 500`
- `max_step = 800`

目标：

- 平衡保命与拿分
- 开始学习更多地图结构

#### Stage C：泛化冲刺

- `map = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]`
- `map_random = true`
- `treasure_count = 10`
- `buff_count = 2`
- `monster_interval = -1`
- `monster_speedup = -1`
- `max_step = 1000`

目标：

- 提前适应随机节奏
- 避免过拟合固定怪物出现时间

### 5. 进阶随机化建议

在课程学习跑通后，可进一步加入：

- 随机地图采样权重
- 随机 `buff_cooldown`
- 随机 `talent_cooldown`
- 随机 `max_step`

注意：

- 每次只加一种随机化
- 加完后先观察训练是否崩掉，再继续

### 6. Phase 3 验收标准

- 不同公开地图上的分数更均衡
- 训练后期平均分稳定，不再大幅抖动
- 切换到默认评测配置时，性能没有明显回退

---

## 七、Phase 4：模型升级

### 1. 阶段目标

只有在前 3 个阶段完成后，才建议升级模型。

如果前面三步没做好，直接换大模型通常只会：

- 训练更慢
- 调参更难
- 不一定涨分

### 2. 推荐升级顺序

#### 方案 1：更强的 MLP

在现有结构上升级为：

- `input -> 256 -> 128 -> 64 -> actor / critic`

并尝试加入：

- `LayerNorm`
- `SiLU` 或 `ReLU`

优点：

- 改动最小
- 容易和原模型对比

#### 方案 2：双分支模型

将输入拆成：

- 向量分支：hero / monster / treasure / buff / progress
- 地图分支：局部地图特征

最后融合后输出 actor / critic。

优点：

- 更符合任务结构
- 比单纯堆大 MLP 更合理

#### 方案 3：小型 CNN 地图编码

如果保留 21x21 或 7x7 栅格地图，可以对地图分支加轻量 CNN。

适用条件：

- 已确认地图特征是性能瓶颈
- 已有稳定训练基线

### 3. 不建议现在就做的事

- 一上来用 LSTM / Transformer
- 一上来做超大网络
- 在没有稳定监控的情况下盲目加层

### 4. Phase 4 验收标准

- 同等训练资源下，分数高于 Phase 3
- 曲线更稳，而不是只在个别 checkpoint 偶然更高
- 推理耗时仍可接受

---

## 八、建议的具体开发顺序

### 第 1 周：先完成 Phase 0 + Phase 1 起步

优先任务：

1. 在 `agent_diy` 里复刻可运行 PPO 骨架
2. 新建 `agent_diy/feature/preprocessor.py`
3. 确保 `algorithm_name = "diy"` 可以正常训练
4. 再开始重写特征提取和奖励

完成标准：

- `diy` 已可替代 `ppo` 跑通训练
- 再进入特征与奖励增强阶段

### 第 2 周：把 Phase 1 做扎实

优先任务：

1. 重写 `agent_diy/feature/preprocessor.py` 的特征提取
2. 增加宝箱 / buff / 卡墙 / 闪现相关奖励
3. 增加监控项
4. 在默认 8 动作下验证 reward 是否正确

完成标准：

- 训练能稳定跑
- 平均总分高于旧版
- 平均宝箱数有提升

### 第 3 周：做 Phase 2

优先任务：

1. 将动作维度扩到 16
2. 联调 legal action
3. 加入闪现奖励与监控
4. 观察模型是否学会“危险时闪现”

完成标准：

- 闪现动作被真实使用
- 不出现大量无效闪现
- 总分高于 8 动作版本

### 第 4 周：做 Phase 3

优先任务：

1. 在训练工作流中加入课程学习
2. 地图随机化
3. 怪物出现时间和加速时间随机化
4. 固定评测配置做阶段对比

完成标准：

- 不同地图表现更加均衡
- 训练后期不易崩盘

### 第 5 周及以后：再做 Phase 4

优先任务：

1. 保留最优 Phase 3 方案
2. 只替换模型，不同时大改奖励和课程
3. 每次只验证一个模型改动

完成标准：

- 明确确认“模型结构升级”本身带来增益

---

## 九、实施时的关键注意事项

### 1. 不要同时改太多层

每个阶段只改一类问题，否则无法判断收益来源。

### 2. 奖励和指标一起改

只改奖励、不加监控，后面很难定位问题。

### 3. 先追求稳定提升，再追求极限分数

稳定比偶发高分更重要，尤其是隐藏图评测场景。

### 4. PPO 主体先别乱动

当前 `algorithm.py` 的 PPO clip、value loss、entropy loss 可以先保留。
在特征、奖励、动作、训练策略都成型之前，不建议优先折腾 PPO 损失。

### 5. 每次改动后都要记录 Git

每完成一轮可独立验证的代码或文档改动，都必须执行一次 Git 记录，避免后续无法追踪改动来源。

固定流程如下：

```bash
git status
git add .
git commit -m "填写本次改动说明"
git push
```

提交信息建议简短说明本次改动目标，例如：

```bash
git commit -m "add diy ppo development plan"
git commit -m "implement diy ppo baseline"
git commit -m "improve diy feature and reward shaping"
```

注意：

- 提交前先看 `git status`，确认没有混入无关文件
- 一次 commit 只记录一类改动，方便回滚和对比
- 如果 `git push` 失败，先记录失败原因，再确认远端仓库或权限配置

### 6. 每次改动后及时更新代码规范

每轮代码、奖励、特征、训练流程或监控项发生变化后，都要同步检查并更新 `峡谷追猎/开发指南/代码规范.md`。

必须同步更新的典型场景：

- 特征维度或拼接顺序变化
- 奖励项、奖励权重或跨步缓存变化
- 动作空间变化
- 训练流程、课程学习或评估流程变化
- 新增监控指标
- Git 或验证流程变化

如果代码和规范不一致，以最新代码为准，并在同一轮提交中修正规范文档。

---

## 十、第一阶段建议直接开工的文件清单

第一批建议先改以下文件：

- `agent_diy/agent.py`
- `agent_diy/algorithm/algorithm.py`
- `agent_diy/model/model.py`
- `agent_diy/feature/definition.py`
- `agent_diy/feature/preprocessor.py`（新建）
- `agent_diy/workflow/train_workflow.py`
- `agent_diy/conf/conf.py`
- `agent_diy/conf/monitor_builder.py`

现阶段先不要动：

- `agent_ppo/algorithm/algorithm.py`
- `agent_ppo/model/model.py`
- `agent_ppo/workflow/train_workflow.py`

原因是：

- 保留 `agent_ppo` 作为稳定基线
- 避免把实验版本和基线版本混在一起
- 后续可以直接比较 `ppo` 与 `diy` 的收益差异

---

## 十一、结论

这套 PPO 的最佳改进路径，不是直接换更复杂的网络，而是：

1. 先把可运行 PPO 骨架迁到 `agent_diy`
2. 让输入更懂环境
3. 让奖励更贴近比赛目标
4. 再开放闪现动作
5. 再做训练泛化
6. 最后才考虑模型升级

实际开发目录选择 `agent_diy`，`agent_ppo` 保持为参考基线。
如果按这个顺序推进，开发风险最低，收益也最稳定。
