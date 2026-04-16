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
| `survive_reward`       | `+0.02`                 | 每步基础生存奖励             |
| `danger_escape_reward` | 危险时 `0.08 * min_dist_delta`，非危险时 `0.02 * min_dist_delta` | 与最近怪物拉开距离时给正奖励 |
| `treasure_reward`      | `+1.0 ~ +1.5`           | 宝箱数量增量奖励             |
| `buff_reward`          | `+0.3 ~ +0.8`           | 收到 buff 时奖励，危险时更高 |
| `good_flash_reward`    | `+0.3 ~ +0.8`           | 闪现后危险显著下降时奖励     |
| `bad_flash_penalty`    | `-0.1 ~ -0.3`           | 浪费闪现时惩罚               |
| `stuck_penalty`        | `-0.08 * stuck_steps`   | 连续原地踏步或撞墙时惩罚     |
| `blocked_action_penalty` | 危险时 `-0.14`，非危险时 `-0.06` | 最近动作导致原地不动时惩罚 |
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

### 6. Phase 1 日志诊断与修正

对 `D:\kaiwu\kaiwusong\train\log` 的训练日志排查结论：

- 训练链路本身没有明显报错，learner、actor、样本发送和模型同步都在运行
- 失败主要发生在策略层面，早期随机策略大量在 30 步内被抓
- 当前样本的平均存活步数偏低，宝箱和 buff 获取率也偏低，说明“先活下来”仍是 Phase 1 的第一优先级

因此 Phase 1 增加一个轻量安全动作先验，并在 Phase 1.1 升级为短视野逃生规划：

- `Preprocessor` 根据最近怪物相对位置、合法动作和相邻格通行性计算 `safe_action`
- Phase 1.1 中，`safe_action` 不再只看一步方向，而是同时评估后续 3 格通路长度、目标格局部开阔度、走后与最近怪物的距离变化，以及最近撞墙动作的冷却惩罚
- `danger_level` 只在最近怪物距离进入危险阈值时大于 0
- `Agent.predict()` 仅在 `danger_level >= 0.25` 时把策略概率与 `safe_action` 先验混合
- 先验最大权重为 `0.6`，目的是帮 PPO 度过早期大量秒死阶段，而不是替代模型决策

同时 `[GAMEOVER]` 日志和监控新增 `danger_level`、`blocked_count`、`danger_steps`、`near_death_count`、`safe_prior_count` 和 `safe_action_count`，方便判断后续失败是否仍集中在贴脸危险、撞墙卡住或安全先验未命中的状态。

### 7. Phase 1 验收标准

建议至少观察以下指标：

- 平均存活步数是否上升
- 平均总分是否上升
- 平均宝箱数是否上升
- 失败局中是否更少出现“撞墙死”“原地抖动”
- reward 曲线是否比旧版更稳定

### 8. Phase 1 需要新增的监控项

建议在 `agent_diy/conf/monitor_builder.py` 里新增：

- `episode_steps`
- `total_score`
- `treasure_count`
- `buff_count`
- `flash_count`
- `fail_rate`
- `avg_min_monster_dist`
- `stuck_count`
- `danger_level`
- `blocked_count`
- `danger_steps`
- `near_death_count`
- `safe_prior_count`
- `safe_action_count`

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

### 4. Phase 2 当前实现策略

本轮 Phase 2 采用“保守闪现”实现，而不是一上来就鼓励大量交技能：

- `Config.FEATURES` 中 legal action 从 `8` 扩展为 `16`，总特征维度变为 **113 维**
- `Preprocessor` 直接使用环境返回的 16 维合法动作掩码
- `ACTION_TO_VECTOR` 同时包含 8 个移动动作和 8 个闪现动作，其中直线闪现距离为 10，斜向闪现距离为 8
- `safe_action` 允许在 16 动作内搜索，但闪现只有在高危状态下才更容易被选中
- 闪现评分更强调目标格是否可通行、闪现后的局部开阔度，以及与最近怪物距离是否显著增加
- 对“无效闪现”增加惩罚，对“明显脱险闪现”或“顺路拿资源闪现”增加奖励

### 5. Phase 2 工程兼容性

从 8 动作模型切到 16 动作模型后，旧的 Phase 1 checkpoint 很可能不兼容。

因此实现中允许：

- `Agent.load_model()` 遇到旧维度 checkpoint 时跳过加载并保留当前权重
- 通过 warning 明确记录“模型文件存在，但维度不兼容”

这样可以避免 Phase 2 启动时被旧模型直接卡死。

### 6. Phase 2 数值稳定性修复

Phase 2 初版引入 16 动作和安全先验后，训练启动阶段可能出现：

- `predict() Exception sum(pvals[:-1].astype(np.float64)) > 1.0`
- `workflow() Exception 'NoneType' object is not subscriptable`

根因是：

- 安全先验混合后的概率分布在浮点误差下可能不再满足采样接口的严格要求
- `predict()` 抛异常后，框架侧可能返回空结果，进而在 `workflow` 中触发连锁下标异常

本轮修复策略：

- 在 `Agent.predict()` 中对模型输出概率和安全先验混合后的概率都做显式归一化
- 采样逻辑改为累计概率采样，避免依赖 `np.random.multinomial` 的严格概率和检查
- `workflow` 中对空的 `predict()` 返回值增加防御分支，避免直接把线程打崩
- 保留日志信息，方便后续确认是否还有上层业务异常

### 7. Phase 2.1 终局与闪现质量优化

从训练日志看，Phase 2 已经能显著提升中后期生存，但仍存在：

- 大量对局能活到 `400+` 步却仍然失败
- `good_flash_count` 已经不低，但 `bad_flash_count` 仍持续出现
- 很多对局在后半段依旧长时间处于高危追逃状态

因此 Phase 2.1 增加两类优化：

1. 更保守的闪现先验选择

- `safe_action` 不再只比较“所有动作里的最高分”
- 而是分成 `best_move` 和 `best_flash` 两路评分
- 只有在高危且闪现明显更优，或闪现分数显著超过移动时，才允许优先选择闪现

2. 闪现后的 3 步随访奖励

- 当发生闪现后，记录闪现前最近怪物距离
- 接下来 3 步内，如果仍出现 `near_death` 或 `blocked`，记为 `flash_trap_count`
- 如果 3 步内持续脱险并保持明显增距，记为 `flash_hold_count`
- 对 `flash_trap_count` 给追加惩罚，对 `flash_hold_count` 给追加奖励

同时加入阶段性生存奖励：

- 存活达到 `200/400/600/800` 步分别给予一次性 milestone reward
- 用于把策略从“能逃很久”进一步推向“尽量活到终局”

### 7.1 Phase 2.2 继续优化方向

基于 `2026-04-16 20` 点这批新日志，当前已经没有明显程序报错，主要瓶颈变成：

- 后半段经常能打到 `500+` 步，但仍然 `0 WIN`
- `good_flash_count` 已经不低，但 `flash_trap_count` 仍然偏高
- `safe_prior_count` 偏高，说明高危时仍然较依赖安全先验兜底

因此继续留在 Phase 2 做一版更保守的终局闪现优化：

1. 更保守的闪现先验混合

- 如果 `safe_action` 是闪现，不再只看危险值是否足够高
- 还要同时看闪现评分相对普通移动的领先幅度、落点后续通路长度，以及落点陷阱风险
- 闪现型 `safe_action` 的先验混合权重低于普通移动，避免先验把策略强行带进高风险闪现

2. 更强的闪现落点评分

- `Preprocessor._score_escape_action()` 对闪现动作新增更强的死路惩罚、边缘惩罚和低开阔度惩罚
- 闪现不再只看“目的地是否可达”，还会继续看落点之后沿该方向的短走廊长度
- 如果闪现虽然瞬时增距，但落点过靠边、后续通路太短或局部开阔度太差，会显著拉低评分

3. 更严格的闪现后持续脱险判定

- `flash_review` 阶段除了看 `blocked` / `near_death`，还会记录 review 窗口内的最低危险等级和持续脱险步数
- 如果闪现后始终没真正脱离高危，仍记为 `flash_trap_count`
- 如果闪现后持续几步都保持安全并维持明显增距，再记为 `flash_hold_count`

4. 更贴近终局目标的 late-game shaping

- `400` 步后，如果仍能保持较低危险并与怪物拉开安全距离，给轻量正奖励
- late-game 的坏闪现和 trap 闪现惩罚更强，促使策略把闪现留给真正关键的高压时刻

### 7.2 Phase 2.3 回调型修正

在 `2026-04-16 21:04` 左右的新日志中，出现了一个很明确的信号：

- 平均分数、后半段平均分数和平均宝箱数都比上一轮下降
- `good_flash_count` 下降，`bad_flash_count` 和 `flash_trap_count` 上升
- `blocked_count`、`stuck_count` 也同步上升

这说明上一轮 Phase 2.2 的方向没有错，但“闪现约束 + trap 判定”收得太紧了，导致策略重新变得保守和犹豫。

因此 Phase 2.3 不再继续加码惩罚，而是做一版回调型修正：

1. 放松高危闪现先验

- 高危险状态下，允许闪现相对 `best_move` 只有小幅优势时也能进入先验混合
- 对闪现的 `safe_path_len` 和 `safe_trap_risk` 阈值放宽，避免把本来能救命的紧急闪现直接压掉

2. 放松 trap 判定

- `flash_review` 不再因为“没有明显连续安全步”就直接记成 `flash_trap_count`
- 更强调真正的坏信号：`blocked`、`near_death`、明显增距不足，以及持续高危但没拉开距离

3. 下调 late-game 坏闪现惩罚

- late-game 仍然保留坏闪现惩罚，但强度低于 Phase 2.2
- 目标不是鼓励乱闪，而是避免策略因为怕惩罚而再次学成“不敢闪”

4. 新增闪现诊断指标

- `safe_flash_step_count`
- `safe_flash_prior_count`
- `safe_action_margin`
- `safe_trap_risk`

这样下一轮看日志时，可以直接判断：

- 安全先验到底有多少步在推荐闪现
- 安全闪现的评分优势是否太小
- 当前策略是否仍然被高估的 `trap_risk` 压住

### 8. 风险点

1. 动作空间变大后探索会更难前期训练曲线可能变差，这是正常现象。
2. 闪现很容易被学成“乱交技能”所以必须同时有 `good_flash_reward` 和 `bad_flash_penalty`。
3. 如果 Phase 1 特征不够好，Phase 2 效果会很有限
   因此不要跳过 Phase 1。

### 9. Phase 2 验收标准

- 危险局面下闪现使用率上升
- 被近身怪物贴脸击杀的比例下降
- 平均宝箱数不下降，最好提升
- 总分高于“只有移动动作”的最优模型

建议新增关注以下指标：

- `good_flash_count`
- `bad_flash_count`
- `flash_escape_gain`
- `flash_count`
- `flash_trap_count`
- `flash_hold_count`
- `late_game_steps`
- `safe_flash_step_count`
- `safe_flash_prior_count`
- `safe_action_margin`
- `safe_trap_risk`

本轮继续优化后，额外重点看：

- `flash_trap_count` 是否下降
- `bad_flash_count` 是否下降
- `late_game_steps` 是否继续上升
- `500+` 步对局数是否增加
- 是否开始出现首批 `WIN`
- `safe_flash_prior_count / safe_flash_step_count` 是否回到更合理区间

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

---

## 十二、Phase 1/2 势能化 Shaping 补充

继续只在 Phase 1 / Phase 2 范围内迭代时，不能一直靠新增固定系数奖励去“堆规则”，
否则很容易出现 reward 项彼此打架、局部指标刚升整体分数又回落的问题。

因此本轮开始把主 shaping 改成更原则化的“状态势能增量”：

1. Phase 1：状态势能

- 在 `Preprocessor` 中计算 `state_potential`
- 状态势能拆成 `safety_potential`、`resource_potential`、`flash_potential`
- 每步主 shaping 改为接近 `0.99 * Phi(s') - Phi(s)` 的势能差分

2. Phase 2：闪现随访看势能

- 闪现是否“真的救命”，不再只看单步距离变化
- `flash_review` 同时看闪现后几步内的峰值势能提升
- 如果距离和势能都没明显改善，才更倾向记为 `flash_trap_count`
- 如果势能明显改善，即使还没完全脱离追击，也允许记为有效闪现

3. 新增诊断项

- `state_potential`
- `safety_potential`
- `resource_potential`
- `flash_potential`

这样下一轮看日志时，可以更快回答三个问题：

- 当前是不是安全势能太低，说明还在乱跑
- 当前是不是资源势能太低，说明安全时不会主动拿宝箱 / buff
- 当前是不是闪现势能判断失真，说明闪现相关 shaping 还要再调

### 7.4 根据 `2026-04-16 21` 点日志的后续修正

新一轮日志里，虽然平均分数相对上一轮有回升，但仍存在两个明确问题：

- `blocked_count`、`stuck_count` 仍然偏高，说明策略仍会绕墙和回环
- 势能日志如果只看终局最后一帧，`resource_potential` 很容易长期显示为 `0`，不利于判断整局 reward 是否有效

因此继续在 Phase 1 / 2 范围内补两类修正：

1. 反回环修正

- `safe_action` 打分时，如果候选落点反复接近最近几步走过的位置，会加额外惩罚
- reward 中也会对“近期重复经过同一区域”的行为给轻量惩罚
- 新增 `loop_count` 统计，专门区分“不是撞墙，但在绕圈”的问题

2. 势能诊断改为整局平均

- `[GAMEOVER]` 日志除了保留终局时刻的 `state_pot/safety_pot/resource_pot/flash_pot`
- 还新增 `state_pot_avg/safety_pot_avg/resource_pot_avg/flash_pot_avg`
- monitor 上报也改为按整局平均势能记录，避免只看终局瞬间造成误判

### 7.5 Phase2.5 加速阶段优化

根据 `2026-04-16 21/22` 两轮环境日志，当前瓶颈已经很明确：

- 进入怪物加速阶段的局有 `119` 局
- 其中 `102` 局死在 `500-549` 步
- 只有 `17` 局能撑过 `550+`
- `buff>0` 的局撑过 `550` 的比例，明显高于 `buff=0`

这说明当前 Phase 1 / Phase 2 已经把“前 500 步基础生存”做起来了，但还没有把“500 步前准备、500 步后切换生存模式”做出来。

因此继续留在 `agent_diy` 的 Phase 1 / Phase 2 范围内，新增一个 `Phase2.5` 子阶段：

1. 500 步前准备

- 在怪物加速前约 `80` 步进入 `speedup_prep` 阶段
- 提高 buff 势能、buff 奖励和 buff 工具位标记
- 同时下调这段时间的宝箱偏好
- 对低危险、无收益的闪现增加额外惩罚，减少把闪现浪费在 500 步前

2. 500 步后生存

- 当怪物速度进入 `2` 格/步后，进入 `post_speedup` 阶段
- 如果没有 buff，则进一步降低宝箱吸引力，强化开阔区、走廊长度、边缘风险相关的安全偏好
- 如果既没有 buff 又没有可用闪现，则增加额外惩罚，明确告诉 PPO 这是“未准备好”的坏状态

3. 新增诊断指标

- `speedup_prep_steps`
- `post_speedup_steps`
- `post_speedup_buffless_steps`
- `post_speedup_unready_steps`
- `buff_ready_at_speedup`
- `flash_ready_at_speedup`

这样下一轮看日志时，可以直接判断：

- 模型是否在 500 步前主动做准备
- 500 步后到底是“无 buff 死亡”还是“有 buff 但路线选择差”
- 闪现到底是被乱用掉了，还是关键时刻还保留着

实际开发目录选择 `agent_diy`，`agent_ppo` 保持为参考基线。
如果按这个顺序推进，开发风险最低，收益也最稳定。

### 7.6 Phase2.6 临近 500 步的关键备战优化

根据 `2026-04-17 00` 点日志，这一轮已经能明显突破 `500` 步，但出现了新的明确瓶颈：

- `34/77` 局进入了 `500+`
- `11/77` 局进入了 `550+`
- `4/77` 局进入了 `600+`
- `1/77` 局进入了 `700+`
- 但 `buff_ready_at_speedup = 0/77`
- `flash_ready_at_speedup = 0/77`

这说明当前策略已经不再是“500 步一到就死”，而是“能熬到 500 步，但几乎从不带着关键资源进入加速阶段”。

因此继续留在 `Phase 1 / Phase 2` 范围内，再补一层 `Phase2.6`：

1. 拆分两段关键备战窗口

- `500` 步前最后 `100` 步视为 `flash preserve window`
- `500` 步前最后 `50` 步视为 `buff ready window`
- 前者重点是少乱交闪现，后者重点是尽量带着 buff 进入加速

2. 增强临近加速的 buff shaping

- 在 `speedup_prep` 且未持有 buff 时，增加“朝最近 buff 靠近”的增量奖励
- 进入最后 `50` 步后，进一步提高 buff 奖励和 buff 势能
- 同时继续压低这段时间对宝箱的偏好，避免模型在关键窗口去贪宝箱

3. 加强临近加速的闪现保留约束

- 在最后 `100` 步内，如果危险不高、也没有拿到 buff/宝箱收益，则更强地惩罚闪现
- `safe_action` 对闪现先验也要更保守，避免在备战期把闪现提前浪费掉

4. 调整 500 步切换时的奖励

- 如果进入加速阶段时有 buff，给更高的一次性正奖励
- 如果没有 buff 但保留了闪现，也给中等奖励
- 如果二者都没有，则给更明确的负奖励

5. 修正 loop 指标

- 原来的 `loop_count` 已经出现“走久了就不断累加”的失真
- 需要改成真正的“短窗口回环次数”，只在重复落点达到阈值时计数
- 这样下一轮日志里，`loop_count` 才能继续作为有效诊断指标

6. 新增或重点关注的日志指标

- `prep_flash_ready_steps`
- `prep_buff_active_steps`
- `buff_ready_at_speedup`
- `flash_ready_at_speedup`

这样下一轮分析时，就能更直接地区分三种情况：

- 模型是否已经在最后 `100` 步学会保闪
- 模型是否已经在最后 `50` 步学会主动带 buff 进 `500`
- 如果仍然死在 `500-550`，到底是“没准备好”，还是“准备好了但后期路线还不够强”

### 7.7 Phase2.7 备战 Buff 先验与更强的 450-500 窗口引导

根据 `2026-04-17 01` 点日志，上一轮 `Phase2.6` 已经出现了一个很明确的中间结果：

- `prep_flash_ready > 0` 的局有明显增加，说明“最后一段尽量留闪现”开始生效
- 但 `prep_buff_active = 0`
- `buff_ready_at_speedup = 0`

这说明模型已经开始意识到“500 前别乱交闪”，但仍然没有学会“450-500 主动去拿 buff”。

因此继续留在 `Phase 1 / Phase 2` 范围内，再补一轮 `Phase2.7`：

1. 新增轻量备战 buff 先验

- 只在最后 `100` 步的备战窗口内考虑
- 只在危险较低时启用，不和高危逃生先验抢控制权
- 只使用普通移动，不鼓励为了拿 buff 提前交闪现

2. 最后 `50` 步更强地拉高 buff 目标

- 对接近最近 buff 的行为给更强的增量奖励
- 对仍然离 buff 很远的状态给轻惩罚
- 同时进一步压低这段时间的宝箱偏好

3. 500 步切换奖励重新拉开档位

- 有 buff 进入 `500`：高奖励
- 只有闪现进入 `500`：低于有 buff 的奖励
- 两者都没有：更明确的惩罚

4. 新增日志与监控

- `prep_prior_count`
- `prep_action_count`

这样下一轮就能判断：

- 模型到底有没有真的触发“去拿 buff”的行为引导
- 是“先验根本没用上”，还是“用了但奖励仍不够强”
