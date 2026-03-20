# AGENTS.md

## 1. 工作流定位
本仓库服务于本科毕业设计的长期推进，不是一次性实验脚本集合。
治理目标只有三个：

- 持续推进：始终维持“可运行主线”与“下一步最小闭环”
- 持续收敛：优先减少分支扩散、减少重复总结、减少无效重构
- 持续复盘：每次有效实验都要沉淀结论、失败原因与下一步决策条件

GPT 负责治理、方向纠偏、阶段规划与 issue 仲裁。
Codex 负责局部实现、复现、调参、脚本、评测接线与工程闭环。

---

## 2. 工作流规则

### 2.1 单一事实基线
研究方向以 `research_broadmap.md` 为准。
仓库实验脉络以 `repo_strategy_map.md` 为准。
若两者存在张力，默认采用“研究目标不变、工程路径保守”的解释，不自行脑补新路线。

### 2.2 单一主路径优先
当前默认主路径是：

`XHandHoraScrewDriver -> PPO teacher -> ProprioAdapt/PAdapt baseline -> diffusion student -> 统一评测`

未经过 GPT 更新治理文件前，不默认把 Pasini 分支、外部 real-world repo、或多任务扩展视为当前主线。

### 2.3 不重复做无用总结
除非用于 issue 升级或治理更新，Codex 不应反复总结整个项目。
默认只维护以下最小必要产物：

- 当前改动目标
- 受影响模块
- 实验命令
- 结果摘要
- 下一步动作

### 2.4 改动必须绑定实验目的
任何代码、配置或脚本修改都必须能回答以下问题之一：

- 它是在修复主线闭环吗？
- 它是在补齐基线对比吗？
- 它是在推进 diffusion student 主线吗？
- 它是在提升评测可信度吗？
- 它是在降低部署/导出/复现实验风险吗？

无法回答时，默认不做。

### 2.5 先闭环，后扩展
默认顺序：

1. 锁定 canonical path
2. 锁定评测与日志
3. 锁定 teacher 与非 diffusion baseline
4. 跑通一个 diffusion student
5. 再做效率/鲁棒性/泛化扩展

### 2.6 变更最小化
优先选择：
- 局部替换
- 可回滚改动
- 兼容现有 `train.py` / Hydra / stage1-stage2 目录结构
- 不破坏 teacher-student 输入兼容关系

避免无明确收益的大规模目录迁移、接口重写和风格性重构。

### 2.7 记录要围绕决策，而不是围绕过程
实验记录不追求“全量流水账”，只记录：
- 改了什么
- 为什么改
- 结果如何
- 是否支持当前路线
- 是否触发升级

---

## 3. 实验评价规则

### 3.1 评价优先级
实验评价必须优先回答以下四件事：

1. 主线是否仍然可运行
2. 新方法是否优于或至少接近当前可靠 baseline
3. 改动是否增加了额外工程风险
4. 该结果是否改变当前阶段优先级

### 3.2 统一比较原则
所有 student 侧方法比较，默认至少与以下对象对齐：

- PPO teacher
- 当前 PAdapt / ProprioAdapt
- BC baseline
- 如实现成本低，再加入 KL distillation 或 action-chunk BC

不能只看单个 reward 曲线判断方法优劣。

### 3.3 统一指标束
默认指标束包括：

- episode reward
- episode length
- rotation reward
- screw angular velocity
- screw angular position
- positive velocity ratio
- 失败 reset 触发情况
- 接触保持/稳定性相关现象
- student 训练损失（如 latent loss / BC loss / total loss）
- 若涉及部署：推理延迟、导出可用性、接口兼容性

### 3.4 评价以“可验证结论”为目标
每轮实验至少产出一种结论：

- 支持继续推进
- 不支持继续推进
- 结论不足，需要补最小验证

没有结论的实验，视为未完成。

### 3.5 主线方法的评价门槛
任何 diffusion 方案若要进入“当前主线有效推进”，至少要满足：

- 能在 canonical path 上稳定训练/评测
- 不破坏 teacher-student 主接口
- 能与 baseline 使用同一评测口径比较
- 不是只在单次可视化中“看起来有效”
- 明确知道它比 baseline 好在哪里，或差在哪里

### 3.6 本科毕设约束优先
当“研究新颖性”和“按期完成”冲突时，优先保证：

- 路径清晰
- 工作量可控
- 实验可验证
- 论文可写

---

## 4. Codex 可自行处理的事项

以下事项，Codex 可直接处理，不必升级 GPT：

### 4.1 局部实现与接线
- 在既有主路径内新增/修改 student 模块
- 训练脚本、评测脚本、导出脚本的接线
- Hydra 配置补齐与参数透传
- logging、checkpoint、输出目录整理
- baseline 复现所需的轻量代码补全

### 4.2 局部 bug 修复
- 维度不匹配
- checkpoint 加载问题
- 配置缺项
- 路径错误
- 评测脚本断裂
- 导出脚本小修
- 明确不会改变治理边界的兼容性修复

### 4.3 复现与对比
- teacher / PAdapt / BC 等既定 baseline 复现
- 在既定指标束下补跑对比实验
- 整理结果摘要与最小结论

### 4.4 小范围实验迭代
- 学习率、batch size、horizon、loss 权重等局部调参
- 不改变研究问题定义的轻量 ablation
- 不改变主线假设的采样步数试验、数据过滤试验、日志补强

### 4.5 文档维护
- 更新与当前实现严格对应的 README 局部说明
- 维护实验命令、运行方式、输出位置
- 生成局部实验记录与 issue 草稿

---

## 5. 必须升级回 GPT 的事项

出现以下任一情况，Codex 必须生成 `codeagent_issue.md` 并升级：

### 5.1 路线级分歧
- 需要在 action diffusion、latent diffusion、residual diffusion、offline dataset 路线间重新排序
- 当前主线明显跑不通，需切换研究主叙事
- 发现 repo 实际结构与 roadmap 假设不一致，足以影响阶段目标

### 5.2 治理边界变化
- 需要改 milestone
- 需要改验收标准
- 需要新增/删除 baseline
- 需要改变“Codex 可自行处理”的边界
- 需要修改 issue 升级机制本身

### 5.3 关键事实冲突
- Hora 主路径无法继续作为 canonical path
- Pasini 必须提前纳入当前阶段
- teacher-student 兼容约束无法满足
- 当前 student 不再适合被视为 latent imitation + BC 结构
- 外部 repo 成为当前阶段不可回避依赖

### 5.4 实验结论冲击主计划
- diffusion 主线连续失败且失败原因具共性
- baseline 已足够强，主线 diffusion 难以形成有效贡献
- 推理延迟 / TorchScript 导出 / 接口改造成本超出本科阶段可控范围
- 鲁棒性、效率或样本效率的结果与原路线判断显著相反

### 5.5 高成本决策
- 需要引入大规模 dataset 化训练流程
- 需要重写 student interface
- 需要改动环境主逻辑而不是 student 层
- 需要引入 critic / offline RL / multimodal / sim-to-real 新子系统

---

## 6. issue 回馈与治理修订流程

### 6.1 `codeagent_issue.md` 必须包含
- 背景：当前任务与所属 milestone
- 现象：出现了什么问题
- 证据：日志、指标、报错、实验对比
- 已尝试方案：最多列 3 个
- 局部结论：目前能确认什么，不能确认什么
- 候选决策：建议 A / B / C
- 推荐动作：Codex 认为最保守可执行的下一步

### 6.2 GPT 收到 issue 后的处理顺序
GPT 按以下顺序判断：

1. 这是临时执行问题，还是治理问题？
2. 是否影响当前阶段目标？
3. 是否影响长期稳定规则？

### 6.3 GPT 的三种处理方式
#### A. 只给临时建议，不改治理文件
适用于：
- 一次性调参建议
- 局部实现选择
- 不改变阶段目标的小问题

#### B. 更新 `PLANS.md`
适用于：
- 当前阶段目标、优先级、milestone、验收标准变化
- 主线不变，但推进顺序需要调整
- 需要正式启用保底路线或缩减当前范围

#### C. 更新 `AGENTS.md`
适用于：
- 工作流规则变化
- 评价规则变化
- 升级边界变化
- GPT/Codex 分工变化
- issue 治理流程变化

### 6.4 何时同时更新两者
只有当 issue 同时改变：
- 长期稳定规则
- 当前阶段计划

才同时修改 `AGENTS.md` 与 `PLANS.md`。

---

## 7. 默认研究治理立场
- 主研究问题在 student 阶段，不在重写 teacher
- 主工程风险在评测闭环、接口兼容、训练稳定性与推理成本
- 主对照对象是 PAdapt/BC/teacher，而不是无限扩展的新方法集合
- 主目标是形成“本科阶段可完成、可验证、可写论文”的 diffusion student 路线
- 在没有 GPT 明确更新治理前，默认不扩展到 Pasini stage-2、offline RL、multimodal generalist policy、外部 real-world repo 深度联动