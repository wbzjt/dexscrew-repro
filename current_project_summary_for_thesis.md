# 当前毕设项目情况总结

> 本文件面向毕业论文收尾与论文框架设计整理，重点总结“已经有证据支撑的项目事实”。  
> 整理时间：2026-05-08。  
> 主要依据：开题报告、项目规划文档、阶段验收总结、鲁棒性总结、云端实验 handoff、当前 CoDriveThesis fixed-step/episode eval 汇总。  
> 注意：本文档不描述代码实现细节，不逐文件解释代码，不把开源框架已有能力写成本人原创。

## 1. 项目定位与研究对象

本项目基于一个已有的灵巧手强化学习与 teacher-student 蒸馏开源框架。原始框架面向 Isaac Gym 中的高自由度灵巧手操作任务，核心流程是先训练具有更多状态信息的 privileged teacher，再将 teacher 的策略能力蒸馏到只依赖较少观测信息的 student 中。

当前研究对象是灵巧手旋拧、screwdriver、rotation 类接触操作任务，具体实验集中在 Isaac Gym 仿真环境中的两指 CoDriveThesis 旋拧任务。该任务具有高维动作空间、接触丰富、长时序反馈稀疏或不稳定等特点，适合用于研究生成式 student 在灵巧手控制中的适用边界。

需要明确的是：当前项目不是从零实现整个灵巧手强化学习框架，也不是从零构建 Isaac Gym 环境。原始环境、teacher-student 训练范式、PPO teacher 路径和 ProprioAdapt-style student 基础来自已有框架。本项目主要工作聚焦于 teacher-student 框架中的 student 蒸馏阶段。

核心研究问题可以概括为：在高维、接触丰富的灵巧手控制任务中，diffusion-style 生成式 student 能否替代或改进原始 ProprioAdapt-style student，并进一步分析不同 diffusion 变体在建模空间、采样步数、控制响应性、鲁棒性和推理成本方面的适用边界。

## 2. 开题报告原始目标对应关系

| 开题目标 | 当前完成情况 | 是否适合写入论文正文 | 备注 |
| ---- | ------ | ---------- | -- |
| 灵巧手旋转类接触操作 | 已完成 | 适合 | 当前任务集中于 Isaac Gym 中的灵巧手旋拧 / rotation / lightbulb-style 接触操作。 |
| 强化学习 teacher | 已完成 | 适合 | 当前论文主表使用 PPO teacher 作为 upper-bound 和 demonstration source。需要说明 teacher 主要来自已有框架或已有 checkpoint，不是本文创新点。 |
| diffusion student | 已完成 | 适合 | 已覆盖 Diffusion Latent、Consistency Diffusion、Flow Matching、Action-Chunk Diffusion 等 student 变体。 |
| 行为克隆 / RL baseline 对比 | 已完成 | 适合 | 已包含 ProprioAdapt-style student、Pure BC、BC/LatentBC、DAgger、DOTPG 等对比对象；DOTPG 当前更适合作为 reference baseline。 |
| 鲁棒性分析 | 部分完成 | 适合，但需谨慎 | 历史文档已有 XHandHoraScrewDriver 口径鲁棒性结果；当前 CoDriveThesis formal robustness 仍需最终汇总。 |
| 采样效率或实时性分析 | 部分完成 | 适合，但需补充最终表 | 当前已规划并运行 NFE / latency eval；完整最终表仍待补充。 |
| Sim-to-Sim 或部署验证 | 证据不足，待补充 | 不建议作为主要结果 | 文档中有 smoke / export 相关记录，但未找到足以支撑当前 CoDriveThesis 主实验的完整 Sim-to-Sim 或部署结果。建议放入不足或展望。 |
| 论文撰写与答辩 | 部分完成 | 适合 | 当前进入实验收尾与论文初稿设计阶段。 |

## 3. 原始开源框架概述

从论文角度看，原始框架是一个 staged teacher-student pipeline。

Stage 1 使用 PPO 训练 privileged teacher。Teacher 可以访问更完整的仿真信息，例如 privileged information、点云信息或 simulator state，因此 teacher 更适合作为性能上界和 demonstration source，而不是最终部署策略。

Stage 2 使用 ProprioAdapt-style student 进行蒸馏。Student 通常只能使用 proprioceptive history 等更受限的输入，通过模仿 teacher 的 latent 表征和动作输出获得控制能力。因此，原始 student 不是第二阶段强化学习，而是 imitation-distillation。

原始 student 的主要组成包括：

- latent distillation：让 student 从较少观测中预测或对齐 teacher latent；
- action behavior cloning：让 student 动作接近 teacher 动作；
- adapter-based adaptation：通过适配模块增强 student 在受限观测下的泛化能力。

因此，ProprioAdapt-style student 是本文的重要 strong baseline。论文中应将其作为开源框架原本已有的强对照方法，而不是本文原创方法。

## 4. 本项目实际完成的工作范围

当前项目已经建立了 teacher-student canonical path：以 PPO teacher 作为上界和数据来源，将 student 蒸馏作为主要研究对象，并围绕不同 student 形式建立了训练、评测和对比流程。

PPO teacher 方面，当前 CoDriveThesis 主实验使用已有 teacher checkpoint 作为统一基准。该 teacher 是所有 student 的 demonstration source 和 upper-bound，不应作为本文核心创新。

ProprioAdapt-style student 已作为原始 strong baseline 纳入比较。Pure BC baseline 已用于判断 latent distillation 和 adapter 机制相对于简单行为克隆的价值。diffusion-style student 已覆盖 latent diffusion、consistency diffusion、flow matching 和 action-chunk diffusion 等多种建模选择。

项目已经实现 teacher rollout / demonstration 数据接口、student 蒸馏训练流程、fixed-step 统一评测、多 seed 评测，以及面向论文主表的结果汇总流程。当前云端正式实验已生成一批 CoDriveThesis main eval 结果，能够支撑主实验初步结论。

仍存在 artifact gap：部分历史 diffusion 结论只有 summary 文档或旧任务口径结果；当前 CoDriveThesis formal robustness、NFE / latency、action-chunk len=1 representation ablation 仍处于执行或待汇总状态；部分训练使用了 timeout 后的 train-best checkpoint 作为 eval checkpoint，需要在论文中透明说明。

## 5. 算法对比对象总结

### 5.1 PPO Teacher

- 算法定位：teacher、upper-bound、demonstration source。
- 输入输出：输入为 privileged 或更完整的仿真观测，输出为灵巧手控制动作。
- 替代关系：不替代 student，是 student 蒸馏的目标策略。
- 训练目标或损失：通过 PPO 强化学习最大化环境 reward。
- 预期优势：可利用更多状态信息，性能通常高于只使用 proprioception 的 student。
- 当前实验表现：当前 CoDriveThesis main eval 中，PPO Teacher 固定步 reward 约为 5.479，是当前主表最高水平。
- 当前主要问题：并非最终可部署 student，也不是本文主要创新。
- 是否适合作为论文主结果：适合作为 upper-bound 和 teacher reference，不适合作为本人核心贡献。

### 5.2 ProprioAdapt-style Student

- 算法定位：原始开源框架中的 strong student baseline。
- 输入输出：输入为受限 proprioceptive history 等 student 可用信息，输出为控制动作。
- 替代关系：是原始 Stage 2 student，不是本文替代对象之外的弱 baseline。
- 训练目标或损失：结合 latent distillation、action behavior cloning 和 adapter adaptation。
- 预期优势：在受限观测下保留 teacher 的关键策略信息，通常强于简单 BC。
- 当前实验表现：当前 CoDriveThesis main eval 固定步 reward 约为 4.880，低于 Consistency 和 Flow，但仍强于 Diffusion Latent。
- 当前主要问题：性能稳定但不一定达到 teacher；在当前主评测中被部分 diffusion-style latent 方法超过。
- 是否适合作为论文主结果：适合作为强 baseline，是论文必须保留的核心对照。

### 5.3 Pure BC Baseline

- 算法定位：基础行为克隆对照。
- 输入输出：输入为 student 观测，输出为模仿 teacher 的动作。
- 替代关系：用于去除 latent distillation / adapter 机制，检验简单动作模仿的能力边界。
- 训练目标或损失：以 teacher action 为目标进行行为克隆。
- 预期优势：结构简单，便于判断复杂 student 机制是否真正有效。
- 当前实验表现：当前 CoDriveThesis main eval 固定步 reward 约为 4.817，略低于 ProprioAdapt-style student。
- 当前主要问题：缺少 latent 对齐和适配机制，可能难以稳定捕捉隐含环境因素。
- 是否适合作为论文主结果：适合作为 baseline 对比，帮助说明原始 student 机制和 diffusion student 的价值。

### 5.4 Diffusion Latent Student

- 算法定位：在 latent 表征层进行 diffusion-style 生成建模的 student。
- 输入输出：输入为 student 观测或历史信息，输出为用于控制的 latent 表征，再间接影响动作。
- 替代关系：主要替代或改进原始 deterministic latent prediction 部分。
- 训练目标或损失：通过 diffusion-style 去噪目标学习 teacher latent 分布。
- 预期优势：相比确定性 latent 预测，理论上更能表达多模态或不确定 latent 分布。
- 当前实验表现：当前 CoDriveThesis main eval 固定步 reward 约为 4.569，低于 PAdapt 和 Pure BC。
- 当前主要问题：普通 diffusion latent 未显示出稳定优势，可能受采样步骤、噪声建模、控制响应延迟和 teacher latent 分布特性影响。
- 是否适合作为论文主结果：适合作为 diffusion 变体之一，但不适合作为最佳方法主张。

### 5.5 Consistency Diffusion Student

- 算法定位：基于 consistency / few-step generation 思路的 latent 生成式 student。
- 输入输出：输入为 student 观测或历史信息，输出为 latent 表征或 latent 条件控制信号。
- 替代关系：替代原始 deterministic latent prediction，并尝试降低 diffusion 多步采样成本。
- 训练目标或损失：通过 consistency-style 目标提升不同噪声水平或采样步之间输出的一致性。
- 预期优势：few-step 生成更适合实时控制，可能比普通 diffusion latent 更稳定。
- 当前实验表现：当前 CoDriveThesis main eval 固定步 reward 约为 5.228，是当前 student 中最强；episode return 也高于 PAdapt 和 Pure BC。
- 当前主要问题：仍低于 teacher；NFE / latency 和鲁棒性最终表仍待补充；历史旧任务中 hard robustness 未稳定超过 PAdapt。
- 是否适合作为论文主结果：适合作为当前 diffusion 主方法和主要正结果。

### 5.6 Flow Matching Diffusion Student

- 算法定位：基于 flow matching / continuous flow 思路的生成式 latent student。
- 输入输出：输入为 student 观测或历史信息，输出为 latent 表征或生成式控制中间量。
- 替代关系：作为传统 diffusion 逐步去噪过程的替代或生成式 student 变体。
- 训练目标或损失：学习从噪声或简单分布到目标 latent 分布的连续流或速度场。
- 预期优势：可能具有更平滑的生成路径和更好的采样效率。
- 当前实验表现：当前 CoDriveThesis main eval 固定步 reward 约为 5.102，低于 Consistency 但高于 PAdapt 和 Pure BC。
- 当前主要问题：历史旧任务中 flow 曾表现较弱，说明其效果对任务、配置和评测口径敏感；当前 latency / NFE 数据仍待最终汇总。
- 是否适合作为论文主结果：适合作为重要 diffusion 变体和次优正结果。

### 5.7 Diffusion Action-Chunk Student

- 算法定位：直接在 action-space 生成短时序动作片段的 diffusion student。
- 输入输出：输入为 student 观测或历史信息，输出为一段动作序列或动作 chunk。
- 替代关系：不只替代 latent prediction，而是尝试直接生成动作序列。
- 训练目标或损失：通过 action-chunk diffusion 目标学习 teacher 的短时序动作分布。
- 预期优势：理论上可建模短时序动作相关性，减少单步动作噪声。
- 当前实验表现：当前 CoDriveThesis 主表未找到完整 action-chunk eval 数据；历史旧任务中 action-chunk diffusion 多次表现较差或不稳定。
- 当前主要问题：高维动作空间和接触控制对动作生成误差非常敏感；chunk 维度可能放大分布偏移和响应延迟。
- 是否适合作为论文主结果：更适合作为 representation / action-space ablation，用于说明 action-space diffusion 的困难，不适合作为主要正结果。

## 6. 已实现功能总结

| 功能 | 当前状态 | 论文中作用 | 证据来源 / 待补充 |
| -- | ---- | ----- | ---------- |
| teacher 训练或加载 | 已完成 | 提供 upper-bound 和 demonstration source | 当前 CoDriveThesis teacher checkpoint 已用于统一评测；teacher 训练过程不作为本文创新。 |
| teacher rollout 数据采集 | 已完成 | 为 student 蒸馏提供监督数据 | 项目总结文档确认已有 teacher rollout / demonstration 接口。 |
| student 蒸馏训练 | 已完成 | 论文核心实验流程 | 多个 student 已完成训练或已有 checkpoint。 |
| ProprioAdapt-style student | 已完成 | 原始 strong baseline | 当前 main eval 已有 PAdapt 结果。 |
| Pure BC baseline | 已完成 | 基础模仿学习对照 | 当前 main eval 已有 Pure BC 结果。 |
| Diffusion Latent | 已完成 | diffusion latent 基础变体 | 当前 main eval 已有结果。 |
| Consistency Diffusion | 已完成 | 当前最强 diffusion student | 当前 main eval 已有结果。 |
| Flow Matching | 已完成 | 重要 diffusion 变体 | 当前 main eval 已有结果。 |
| Action-Chunk Diffusion | 部分完成 | action-space ablation | 历史结果存在；当前 CoDriveThesis 主表完整 eval 待补充。 |
| 统一评测 | 已完成 | 支撑公平比较 | 当前 fixed-step + episode/progress eval 已产出主结果。 |
| 多 seed 评测 | 已完成 | 支撑稳定性分析 | formal methods 当前有 3 train seeds × 3 eval seeds；部分 reference baseline 为 single-train-seed。 |
| robustness 评测 | 部分完成 | 支撑适用边界分析 | 历史任务有结果；当前 CoDriveThesis formal robustness 待最终汇总。 |
| fixed-step 评测 | 已完成 | 主实验定量指标 | 当前主表使用 2048-step fixed eval。 |
| 训练日志 | 部分完成 | 支撑可复现性 | 云端输出目录有日志；部分历史 artifact 不完整。 |
| 指标汇总 | 已完成 / 部分完成 | 论文表格来源 | main eval 已汇总；NFE、latency、robustness 仍待最终表。 |
| 可视化 | 部分完成 | 论文图表 | 图表生成已规划；最终可用图需待补充。 |
| export / deployment | 证据不足，待补充 | 可作为工程延伸 | 当前不建议写成主要结果。 |
| Sim-to-Sim | 证据不足，待补充 | 可作为展望 | 未找到完整最终结果。 |

## 7. 实验指标体系

| 指标 | 是否已有数据 | 用于评价什么 | 是否建议写入论文 |
| -- | ------ | ------ | -------- |
| episode reward / fixed-step reward | 已有 | 总体控制性能 | 建议作为主指标之一。 |
| episode length | 已有 | 策略稳定持续时间与 reset 情况 | 建议写入。 |
| rotation progress | 已有 | 旋拧任务核心进展 | 强烈建议写入。 |
| screw angular velocity | 部分已有 | 旋转速度与动作有效性 | 建议写入，若最终表缺失则作为辅助指标。 |
| screw angular position | 部分已有 | 旋转位移与任务完成度 | 建议写入。 |
| positive velocity ratio | 已有 | 正向旋转比例，反映控制方向性 | 建议写入。 |
| failure reset / done rate | 已有 | 失败率与稳定性 | 建议写入。 |
| success rate / success_2pi_rate | 已有但数值普遍接近 0 | 完成指定旋转阈值的比例 | 可写入，但需说明当前阈值较严格或任务设置导致成功率低。 |
| latent loss | 部分已有 | student latent 对齐质量 | 可作为训练分析指标。 |
| behavior cloning loss | 部分已有 | 动作模仿质量 | 可作为训练分析指标。 |
| diffusion loss | 部分已有 | diffusion latent 学习质量 | 可作为方法分析指标。 |
| consistency loss | 部分已有 | consistency student 训练质量 | 若日志完整，建议写入附录或训练分析。 |
| flow matching loss | 部分已有 | flow matching student 训练质量 | 若日志完整，建议写入附录或训练分析。 |
| robustness under perturbation | 部分完成 | 环境扰动下稳定性 | 建议写入，但当前 CoDriveThesis 最终表待补充。 |
| sampling steps / NFE | 进行中 | 生成式 student 的采样成本 | 建议写入，待最终数据。 |
| inference latency / sampling cost | 进行中 | 实时控制可行性 | 建议写入，待最终数据。 |

## 8. 实验结果汇总

### 数据口径说明

当前建议论文主表采用 CoDriveThesis formal eval 口径：2048 fixed steps、numEnvs=48、eval seeds 42/43/44。formal student 方法包含 3 个 train seeds × 3 个 eval seeds；Teacher 为单 checkpoint × 3 eval seeds；BC/DAgger/DOTPG 为 single-train-seed reference baseline。

历史文档中还存在 XHandHoraScrewDriver 256-step 口径结果。该口径与当前 CoDriveThesis 的任务、teacher checkpoint、评测 horizon 和 checkpoint selection 均不同，不能直接混入当前主表。

### 8.1 主实验结果表

以下表格使用当前 CoDriveThesis main eval 聚合结果。Reward 为 fixed-step reward mean；Failure Reset 使用 done_rate 近似表示。

| 方法 | Reward | Episode Length | Rotation Progress | Positive Velocity Ratio | Failure Reset | 备注 |
| ----------------------- | -----: | -------------: | ----------------: | ----------------------: | ------------: | -- |
| PPO Teacher | 5.479 | 686.5 | 2.355 | 0.817 | 0.00104 | teacher upper-bound，单 checkpoint × 3 eval seeds。 |
| ProprioAdapt Student | 4.880 | 673.8 | 2.182 | 0.799 | 0.00107 | 原始 strong baseline，3 train seeds × 3 eval seeds。 |
| Pure BC | 4.817 | 668.8 | 2.207 | 0.799 | 0.00109 | 基础行为克隆 baseline，略低于 PAdapt。 |
| Diffusion Latent | 4.569 | 642.1 | 1.816 | 0.815 | 0.00116 | 普通 latent diffusion 当前未超过 PAdapt / Pure BC。 |
| Consistency Diffusion | 5.228 | 678.8 | 2.358 | 0.811 | 0.00105 | 当前最强 student，接近 teacher。 |
| Flow Matching Diffusion | 5.102 | 672.7 | 2.265 | 0.814 | 0.00107 | 当前第二强 diffusion student，高于 PAdapt / Pure BC。 |
| Action-Chunk Diffusion | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 当前 CoDriveThesis 主表未找到完整 action-chunk eval；历史旧任务结果显示该方向较不稳定。 |

补充 reference baseline：

| 方法 | Reward | Episode Length | Rotation Progress | Positive Velocity Ratio | Failure Reset | 备注 |
| -- | --: | --: | --: | --: | --: | -- |
| BC / LatentBC | 4.086 | 654.9 | 2.132 | 0.760 | 0.00110 | single-train-seed reference baseline。 |
| DAgger | 4.768 | 685.1 | 2.138 | 0.802 | 0.00107 | single-train-seed reference baseline。 |
| DOTPG | 3.446 | 683.3 | 2.241 | 0.702 | 0.00104 | 当前作为 reference baseline；不是本论文 diffusion 主线。 |

### 8.2 Robustness 结果表

当前 CoDriveThesis formal robustness 最终表尚未找到完整汇总，因此不建议把下表作为论文主表。以下只保留历史 XHandHoraScrewDriver 256-step 口径参考，用于说明项目曾做过鲁棒性分析。

| 方法 | Nominal | Light Perturbation | Hard Perturbation | 结论 |
| -- | ------: | -----------------: | ----------------: | -- |
| PPO Teacher | 3.056 | 2.915 | 2.763 | 历史 teacher upper-bound；非当前 CoDriveThesis 主表口径。 |
| ProprioAdapt Student | 2.168 | 2.079 | 1.838 | 历史口径下仍是较强 baseline。 |
| Pure BC | 1.883 | 2.191 | 1.850 | 历史结果显示 Pure BC 并非总是弱，但稳定解释需结合任务口径。 |
| Diffusion Latent | 2.063 | 1.789 | 1.572 | 历史口径下未稳定超过 PAdapt。 |
| Consistency Diffusion | 2.336 | 2.045 | 1.714 | 历史口径下 nominal 较强，hard 不稳定超过 PAdapt。 |
| Flow Matching Diffusion | 1.783 | 1.588 | 1.367 | 历史口径下表现较弱。 |
| Action-Chunk Diffusion | 待补充 | 待补充 | 待补充 | 历史文档多处显示 action-chunk 较不稳定，但主表数据不完整。 |

当前 CoDriveThesis robustness：待补充。建议等待 formal robustness eval 完成后，以当前任务口径重建该表。

### 8.3 多 seed 结果表

以下为当前 CoDriveThesis fixed-step reward 聚合。formal student 的 n=9 表示 3 train seeds × 3 eval seeds；Teacher 和部分 reference baseline 为 n=3。

| 方法 | Seeds | Mean | Std | 结论 |
| -- | ----: | ---: | --: | -- |
| PPO Teacher | 3 | 5.479 | 0.014 | 当前 upper-bound。 |
| ProprioAdapt Student | 9 | 4.880 | 0.139 | 强 baseline，但低于 Consistency / Flow。 |
| Pure BC | 9 | 4.817 | 0.167 | 接近 PAdapt，说明简单动作模仿已有较强能力。 |
| Diffusion Latent | 9 | 4.569 | 0.115 | 普通 latent diffusion 当前未体现优势。 |
| Consistency Diffusion | 9 | 5.228 | 0.062 | 当前最稳定且最强的 student。 |
| Flow Matching Diffusion | 9 | 5.102 | 0.155 | 高于 PAdapt / Pure BC，但波动略大于 Consistency。 |
| BC / LatentBC | 3 | 4.086 | 0.047 | reference baseline，明显弱于 formal main methods。 |
| DAgger | 3 | 4.768 | 0.021 | 接近 Pure BC / PAdapt，但不是本文主线。 |
| DOTPG | 3 | 3.446 | 0.021 | 当前 reward 较低，适合作为 reference，不适合展开为本文核心。 |
| Action-Chunk Diffusion | 待补充 | 待补充 | 待补充 | 当前 CoDriveThesis 完整数据待补充。 |

### 8.4 Diffusion 变体对比表

| 方法 | 建模空间 | 采样方式 | 主要优势 | 主要问题 | 当前结论 |
| ----------------------- | ---- | ---- | ---- | ---- | ---- |
| Diffusion Latent | latent 表征空间 | 多步或固定步去噪 | 能表达 latent 分布不确定性 | 当前 reward 低于 PAdapt / Pure BC，采样成本可能影响控制响应 | 适合作为基础 diffusion 对照，不是最佳结果。 |
| Consistency Diffusion | latent 表征空间 | few-step / consistency generation | 性能强，采样步数潜在更少，当前稳定性最好 | 仍需最终 latency / NFE / robustness 数据支撑 | 当前最适合作为论文主 diffusion 方法。 |
| Flow Matching Diffusion | latent 表征空间 | flow / continuous generation | 当前主评测高于 PAdapt / Pure BC，具有生成路径平滑的潜力 | 历史旧任务结果较弱，说明配置敏感 | 适合作为重要正结果和机制对比。 |
| Action-Chunk Diffusion | action-space / action chunk | 生成动作片段 | 可探索短时序动作相关性 | 高维动作 chunk 容易放大误差和延迟；当前主表缺完整结果 | 适合作为 ablation，说明 action-space diffusion 的困难。 |

### 结果一致性问题

当前文档中存在明显的结果口径差异。历史总结中曾认为 PAdapt 是主线强 baseline，Consistency 是可接受但不稳定的 diffusion reference，Flow 较弱；而当前 CoDriveThesis main eval 显示 Consistency 和 Flow 均超过 PAdapt / Pure BC。

该差异不应简单解释为文档矛盾，主要来源包括：

- 任务不同：历史结果多来自 XHandHoraScrewDriver，当前主表来自 Dexh13HoraLightbulbSim2RealTwoFingerCoDriveThesis；
- teacher checkpoint 不同；
- 评测 horizon 不同：历史多为 256-step，当前为 2048-step fixed eval；
- checkpoint selection 不同：当前 formal 训练因 5h timeout 未达到 eval-select interval，部分 run 使用 train-best symlink 作为 deploy checkpoint；
- diffusion 实现和配置经历多轮迭代。

建议论文采用当前 CoDriveThesis formal eval 作为主表；历史 XHandHoraScrewDriver 结果只作为研究迭代、附录或方法调试背景。

## 9. 当前阶段主要结论

1. 原始 ProprioAdapt-style student 仍是强 baseline。它在当前 CoDriveThesis 中固定步 reward 约为 4.880，显著强于 Diffusion Latent 和 BC/LatentBC，并且历史鲁棒性结果也显示其稳定性较好。

2. Pure BC 与 ProprioAdapt 的差别说明，简单行为克隆在当前任务中已经可以获得较强性能，但 latent distillation / adapter 机制仍带来一定提升。当前 Pure BC reward 约为 4.817，略低于 PAdapt。

3. 四种 diffusion-style student 中，Consistency Diffusion 当前结果最稳定、最强；Flow Matching 当前也表现出明显正结果；普通 Diffusion Latent 未超过经典 baseline；Action-Chunk Diffusion 缺少当前主表完整数据，历史结果显示不稳定。

4. action-space / action-chunk diffusion 尚未表现出明显优势。高维动作片段直接生成可能受到动作维度、控制频率、接触反馈敏感性和误差累积影响，更适合作为失败或边界分析。

5. latent diffusion / consistency / flow matching 相比 action-space diffusion 更适合当前任务。尤其是 Consistency 和 Flow 在当前 CoDriveThesis 主评测中超过 PAdapt / Pure BC，说明在 latent 表征层进行生成建模更符合 teacher-student 蒸馏结构。

6. 如果 diffusion 没有明显超过 baseline，不应解释为“diffusion 不适合灵巧手”。更稳妥的表述是：在当前任务和实现条件下，diffusion 的优势受到建模空间、采样机制、控制响应性、teacher 动作分布、训练预算和 checkpoint selection 的共同影响。

7. 高维灵巧手接触任务对 diffusion 的主要挑战包括：动作空间高维、接触动力学不连续、长期进展指标难以通过短期 imitation loss 完全捕捉、多步采样可能引入控制延迟，以及 teacher 行为分布可能并非明显多模态。

8. 当前结果支持“diffusion 在高维接触任务中具有条件性优势，而不是必然优于强 baseline”这一结论。Consistency 和 Flow 在当前主评测中有优势，但普通 latent diffusion 和 action-chunk diffusion 并未稳定胜出。

## 10. 当前项目不足与证据缺口

| 问题 | 严重程度 | 对论文影响 | 建议处理方式 |
| -- | ---- | ----- | ------ |
| 部分历史 outputs / checkpoint / logs 不完整 | 中 | 影响复现实验链条 | 主论文采用当前 formal run；历史结果只作补充。 |
| 当前 CoDriveThesis robustness 最终表未汇总 | 高 | 影响鲁棒性章节可信度 | 等待云端正式结果完成后补表；若未完成，降级为展望或历史参考。 |
| NFE / latency 数据仍在进行中 | 高 | 影响实时性与采样效率分析 | 最终必须补充 reward-latency / NFE 表，否则只写为待完成分析。 |
| Action-Chunk 当前主表数据缺失 | 中 | 影响 diffusion 变体完整比较 | 若最终仍无完整数据，将其作为 ablation 的未完成项或失败案例。 |
| 不同文档中结果口径不一致 | 高 | 容易导致论文表述混乱 | 明确区分历史 XHand 口径与当前 CoDriveThesis 口径。 |
| diffusion rollout diagnostics 语义存在不确定 | 中 | 影响对 failure mode 的精确解释 | 只使用可解释的 reward、progress、done、latency 等指标；复杂 diagnostics 谨慎写。 |
| diffusion export / deployment 不完整 | 中 | 不能支撑部署贡献 | 不写成主要贡献，放入不足或展望。 |
| Sim-to-Sim 没有明确最终结果 | 高 | 不能支撑开题中的部署验证目标 | 若无最终数据，不写成实验结果。 |
| checkpoint selection 存在 train-best symlink deviation | 中 | 影响实验规范性 | 在实验设置中透明说明：formal 训练 5h timeout 未触发 eval-select，采用 train-best 替代 deploy-best。 |
| DOTPG reward 较低且非论文主线 | 低 | 可能分散论文重点 | 只作为 reference baseline，不展开为主要方法。 |

## 11. 本科毕设论文可写贡献点建议

1. 贡献点表述：基于已有 teacher-student 灵巧手强化学习框架，完成了面向 student 蒸馏阶段的 diffusion-style 生成式替换与实验验证。  
   支撑证据：当前项目已在 CoDriveThesis 任务上比较 PPO Teacher、ProprioAdapt、Pure BC 和多种 diffusion student。  
   适合章节：方法章节、实验设置章节。

2. 贡献点表述：实现并比较了 Diffusion Latent、Consistency Diffusion、Flow Matching、Action-Chunk Diffusion 等多种 student 建模方式。  
   支撑证据：当前 main eval 中 Consistency 和 Flow 具有明确正结果，Diffusion Latent 和 Action-Chunk 提供边界分析。  
   适合章节：方法章节、消融实验章节。

3. 贡献点表述：建立了 ProprioAdapt / Pure BC / diffusion variants 的统一 fixed-step 与多 seed 评测对比流程。  
   支撑证据：当前 CoDriveThesis main eval 已形成多方法、多 seed 的聚合表。  
   适合章节：实验设置章节、主实验结果章节。

4. 贡献点表述：分析了 diffusion 在高维、接触丰富灵巧手控制中的适用边界。  
   支撑证据：当前结果显示 Consistency / Flow 在 latent space 中有优势，而普通 Diffusion Latent 和 Action-Chunk 并未稳定胜出。  
   适合章节：结果分析章节、讨论章节。

## 12. 建议论文主线

本文基于已有的灵巧手 teacher-student 强化学习框架，聚焦高维接触操作任务中的 student 蒸馏阶段，设计并实现多种 diffusion-style 生成式 student，包括 latent diffusion、consistency diffusion、flow matching 和 action-chunk diffusion；通过与 PPO teacher、原始 ProprioAdapt-style student、Pure BC 以及其他 imitation / RL reference baseline 的统一评测比较，分析生成式 student 在灵巧手旋拧任务中的性能、稳定性、采样效率和适用边界。

## 13. 建议放入正文 / 展望 / 不写的内容

| 内容 | 建议位置 | 原因 |
| ---------------------- | ----------- | ------------ |
| PPO teacher-student 框架 | 背景 / 方法基础 | 开源框架基础，不应写成本人原创。 |
| PPO Teacher | 实验 upper-bound | 提供 teacher reference 和 demonstration source。 |
| ProprioAdapt student | baseline | 原始 strong baseline，必须保留。 |
| Pure BC | baseline 对比 | 说明蒸馏机制相对简单 BC 的价值。 |
| 四种 diffusion student | 核心方法章节 | 本人主要工作集中在 student 替换与比较。 |
| Consistency / Flow main result | 主实验结果 | 当前 CoDriveThesis 结果最有支撑。 |
| Diffusion Latent 负结果 | 消融 / 讨论 | 说明普通 diffusion latent 未必优于 strong baseline。 |
| Action-Chunk Diffusion | 消融 / 边界分析 | 当前更适合说明 action-space diffusion 的困难。 |
| robustness 评测 | 实验章节或补充实验 | 当前 CoDriveThesis 最终表待补充；完成后可支撑适用边界分析。 |
| NFE / latency | 实验章节或补充实验 | 若最终数据完整，可支撑实时性分析。 |
| Sim-to-Sim | 展望或不足 | 若未完成，不要硬写成结果。 |
| deployment/export | 视完成情况 | 没有完整证据则放展望。 |
| 真实机器人部署 | 不写或展望 | 若未完成，不要写成贡献。 |
| DOTPG | reference baseline 或未来工作 | 当前 reward 较低，且属于下一阶段研究方向。 |

## 14. 需要用户补充的信息清单

为完成最终论文，还需要补充或确认以下信息：

1. 最终采用的 teacher checkpoint 名称、训练来源和是否完全属于当前 CoDriveThesis 基准。
2. 每个算法最终 checkpoint 的来源，尤其是 formal 训练中 train-best 与 eval-select best 的关系。
3. 当前 CoDriveThesis formal robustness 的最终表格。
4. 当前 NFE / latency / reward-latency 的最终表格和图。
5. Action-Chunk Diffusion 在当前 CoDriveThesis 口径下的最终 eval 结果。
6. 每个实验最终采用的 train seed 和 eval seed 数量。
7. 是否保留 BC/LatentBC、DAgger、DOTPG 在主表中，还是放入 appendix/reference baseline。
8. 是否有最终训练曲线、reward vs wall-clock、reward vs teacher-query budget 图。
9. 是否有失败案例截图、rollout 可视化或旋拧进展曲线。
10. robustness stressor 的最终设置和是否与论文文字完全一致。
11. success rate 阈值是否采用 2π，或是否需要补充更适合当前任务的 progress-based success 指标。
12. 最终论文中 diffusion 方法命名：建议统一为 Diffusion Latent、Consistency Latent、Flow Matching Latent、Action-Chunk Diffusion。
13. 是否有 Sim-to-Sim 或 export / deployment 的最终证据；若没有，应统一写入不足与展望。
14. 历史 XHandHoraScrewDriver 结果是否进入附录；若进入，必须明确标注为历史任务口径，不能与当前 CoDriveThesis 主表混合。
