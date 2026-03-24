# plan2.md

## 1. 新阶段定位
本计划不再以“补齐 diffusion 基础能力”为主目标，而是聚焦一个更窄、更论文导向的问题：

**让 `DiffusionLatentStudent` 在当前 dexterous-hand screw-driving 任务中，稳定追平或逼近 `ProprioAdapt` baseline，并明确差距来源与有效改进路径。**

本阶段默认不再扩展新 student 大类，不再把 `action-chunk diffusion` 作为主攻方向，也不再优先做大范围框架调整。

---

## 2. 当前已知前提

### 2.1 已冻结的主线
- canonical path 保持：
  - `XHandHoraScrewDriver -> PPO teacher -> student distillation -> unified evaluation`
- 当前 diffusion 主分支固定为：
  - `DiffusionLatentStudent`
- 当前强 baseline 固定为：
  - `ProprioAdapt`

### 2.2 当前阶段结论
- `action-chunk diffusion` 已完成必要验证，但当前不适合作为主线候选。
- `latent_recon05` 是当前最强 diffusion-latent 代表。
- 目前 latent 距离 `adapt` 不是“完全不可用”，而是：
  - nominal / hard 已有竞争力
  - 某些轻扰动与 seed 下仍存在稳定性和 reward alignment 问题
- 最近几轮最小试探表明：
  - 单纯增大 teacher BC 无效
  - 粗粒度 base-action anchor 只有局部 seed 收益，整体不足以升级主线

### 2.3 当前阶段核心判断
后续不应再靠“盲扫超参”推进，而应围绕：
- gap 定位
- reward alignment
- torque / pose penalty 对齐
- 中段行为稳定性

做更小、更可验证的结构化优化。

---

## 3. 本阶段核心目标

### 目标 A：冻结 latent diffusion 主代表
默认主代表固定为：
- `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_recon05_seed42_15min/stage2_diffusion_nn/model_best.ckpt`

除非后续新版本经过统一口径验收明确更优，否则不替换主代表。

### 目标 B：把比较对象收缩到 `adapt`
后续 diffusion 主线的核心问题不再是：
- “能不能模仿 teacher”

而是：
- “能不能追平或逼近当前 `ProprioAdapt`”
- “在哪些场景下没有追平”
- “差距来自训练、推理、reward alignment、还是鲁棒性机制”

### 目标 C：把 gap 定位做成可执行闭环
必须能回答：
- latent diffusion 比 adapt 差在哪里
- 差距发生在 nominal、light、hard 的哪一档
- 差距发生在 rollout 的哪个阶段
- 差距更像动作过激、恢复不足、还是姿态/力矩代价失衡

### 目标 D：只做 thesis-safe 的小改动
后续只允许做：
- 小 loss 项
- 小正则项
- 小型 selector / diagnostics 增强
- 已有 latent 路线上的小结构增强

默认不做：
- 第五套 student 算法
- 大模型替换
- 大规模数据平台重构
- 与当前 teacher-student 主链不兼容的改动

---

## 4. 本阶段研究问题

### Q1
为什么 `latent diffusion` 已经能在 nominal / hard 取得一定收益，但仍未稳定追平 `adapt`？

### Q2
当前 latent 的主要差距是：
- 表达能力不足
- 训练目标与部署目标不一致
- reward shaping 对齐不足
- 还是特定扰动条件下的行为稳定性不足

### Q3
哪些最小修改能够带来“整体均值稳定提升”，而不是只改善单个 seed？

---

## 5. 本阶段执行主线

### W1：参考基线冻结
完成条件：
- 明确当前 `adapt` 代表 ckpt
- 明确当前 `latent_recon05` 代表 ckpt
- 统一使用固定评测口径：
  - nominal
  - `light_v2`
  - hard
- 统一使用 multiseed 作为升级依据，而不是单 seed 或训练峰值

### W2：gap 诊断主线
完成条件：
- 完成 `latent_recon05 vs adapt` 的统一口径对比
- 至少完成一轮 rollout diagnostics 对比
- 能明确 gap 的主要阶段位置：
  - early
  - mid
  - late
- 能明确至少两项关键差异指标：
  - `rotation_reward`
  - `pose_diff_penalty`
  - `torques`
  - `reward_per_step`

### W3：最小结构增强
只允许推进以下类型：
- reward-aligned regularizer
- torque-aware / action-magnitude regularizer
- 更贴近部署的轻量选择指标
- latent 路线中的小型结构增强

每次只做一个最小改动，并遵守：
1. 先 smoke
2. 再完整 15min
3. 再做统一口径评测
4. 无明确收益则停止该分支

### W4：鲁棒性收敛
完成条件：
- 新 latent 候选至少在一个主要口径上明确优于 `latent_recon05`
- 若要升级主代表，必须至少满足：
  - 不是只赢单 seed
  - 不靠训练峰值叙事
  - 不以显著伤害另一个主要口径为代价

### W5：论文可写性收敛
完成条件：
- 能清楚写出：
  - 当前 latent 的优点
  - 当前 latent 与 adapt 的差距
  - 已尝试过哪些修复
  - 哪些修复无效
  - 哪类修复最有前景

---

## 6. 当前建议 milestone

### N0：主代表冻结
完成条件：
- `latent_recon05` 作为当前 diffusion-latent 主代表写入所有后续比较语境
- 未通过验收的新版本不得替换主代表

### N1：adapt-gap 定位
完成条件：
- 完成 `latent_recon05 vs adapt` 的最新统一口径对照
- 明确当前最主要差距是在：
  - nominal
  - `light_v2`
  - hard
  - 或 rollout 中段行为

### N2：中段稳态增强
完成条件：
- 至少验证 1-2 个 reward-aligned 最小改动
- 明确哪类改动只是“改善个别 seed”，哪类改动有整体收益潜力

### N3：升级判定
完成条件：
- 若出现新的 latent 候选，必须通过 multiseed + 三档口径验收后，才允许升级主代表

### N4：阶段总结
完成条件：
- 形成一版足够写论文实验章节的结论草稿
- 明确下一阶段是否继续追平 adapt，还是转向效率/采样成本优化

---

## 7. 验收规则

### 7.1 升级主代表规则
一个新 latent 候选只有在以下条件同时满足时，才可升级主代表：

1. 完成完整 15min 训练
2. 至少完成 `nominal + light_v2 + hard` 中相关主要口径评测
3. 至少完成 3 seed 聚合中的一个关键口径
4. 均值提升不是来自单个 seed 偶然抬升
5. 不以明显破坏另一个关键口径为代价

### 7.2 失败分支归档规则
若一个新改动满足以下任一条件，应立即归档而非继续扫参：
- 两个以上已验收 seed 同时低于当前主代表
- 训练峰值升高但统一评测明显回退
- 只改善单个 seed，聚合均值仍差
- 没有提供新的失败模式信息

### 7.3 决策优先级
后续所有判断优先级固定为：
1. multiseed deploy 指标
2. 三档扰动口径
3. rollout 阶段诊断
4. 训练峰值

训练峰值不得单独作为“升级依据”。

---

## 8. 暂不纳入本阶段
- 新的 diffusion student 大类
- 恢复 `action-chunk` 为主线
- 大规模数据集平台
- Pasini / 外部 repo 接入
- sim-to-real 方向扩展
- 大模型替换或复杂 transformer 化

---

## 9. 立即执行建议

### Step 1
用 rollout diagnostics 对比：
- `latent_recon05`
- `baseanchor03`

在以下两个设置上的中段行为：
- `seed42 / light_v2`
- `seed44 / light_v2`

目标：
- 确认 `seed44` 的改善是否真的来自 `pose_diff_penalty / torques` 回落
- 确认 `seed42` 的退化具体发生在哪个阶段

### Step 2
如果 Step 1 证实“降低 torque / pose penalty 失配”确实有效，则实现一个比 `baseanchor03` 更直接、粒度更细的最小正则：
- torque-aware regularizer
或
- action-magnitude regularizer

### Step 3
若新正则仍不能形成整体提升，则阶段性停止结构试探，回到：
- `latent_recon05` 作为论文主 diffusion 代表
- 把当前主要价值转写为：
  - nominal / hard 竞争力
  - 失败模式定位
  - 与 adapt 的差距解释

---

## 10. 本阶段退出条件
满足以下条件之一即可结束本阶段：

### 退出条件 A：成功收敛
- 出现一个明确优于 `latent_recon05` 的 latent 候选
- 且它在主要评测口径上更接近或追平 `adapt`

### 退出条件 B：研究型收敛
- 没有再找到稳定提升的最小改动
- 但已清楚定位 latent 与 adapt 的主要差距
- 已足够支撑论文中的方法比较、失败分析与后续优化方向

### 退出条件 C：方向切换
- latent 主线进入明显收益递减
- 需要由你更新下一版计划，正式切到：
  - 采样效率
  - 推理成本
  - 或更高层的 thesis contribution 轴
