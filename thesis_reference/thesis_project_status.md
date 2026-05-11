# 毕业设计项目现状总结

## 0. 生成说明

本文档生成时间：2026-05-08。  
最终实验同步更新：2026-05-09。  
本文档用途：作为后续 GPT 综合“开题报告 + 当前项目实际进展 + 实验数据”设计毕业论文框架的项目事实档案。本文档不是论文初稿，也不是代码说明文档。

当前本地仓库状态：

- 本地分支：`diffusion`，跟踪 `cloud/diffusion`。
- 当前提交：本地与云端均核实到 `21e8eff`。
- 本地工作区：存在多处已同步的未提交改动，包括算法文件、任务配置、评测脚本和论文实验脚本。本文档不回滚、不修改这些代码文件。
- 本地 artifact gap 已部分补齐：旧历史实验仍缺本地原始 artifact，但当前 CoDriveThesis formal suite 的最终 CSV/JSON/log/status/manifest 已同步到 `thesis_reference/cloud_final_results_20260506_210723/`。
- 云端 artifact：已只读核实云端 `/root/code/dexscrew-repro/outputs/paper_codrive_thesis_full_20260506_210723/` 存在，并已同步主要结果到本地。
- 云端最终阶段：`done`。`main_eval`、`nfe_latency`、`representation_eval`、`robustness`、`validation` 均已完成；robustness 为 `108/108` rows，异常日志命中为 `0`。
- 最终实验设计和结论另有专门文件记录：`thesis_reference/final_experiment_design_and_conclusions.md`。

主要信息来源：

| 类型 | 路径 / 来源 | 作用 |
| -- | -- | -- |
| 开题报告 | `thesis_reference/王炳彰本科毕业设计（论文）开题报告(diffusion).md` | 原始毕设目标、研究范围、计划进度。 |
| 原项目说明 | `README.md` | 开源框架四阶段方法说明。 |
| 仓库规则 | `AGENTS.md` | 明确 canonical path、current student 定位、cloud handoff 要求。 |
| 项目策略图 | `instruction_docs/repo_strategy_map.md` | teacher-student 框架、任务、指标和研究切入点概述。 |
| baseline 说明 | `docs/baseline.md` | 明确 current student 不是 Pure BC，而是 ProprioAdapt-style imitation-distillation。 |
| 阶段总结 | `plan_summary_v1.md` | 说明已完成 pipeline、baseline、diffusion、rollout、robustness workflow，同时指出 artifact gap。 |
| 历史验收 | `docs/stage_acceptance_summary.md` | 旧 XHandHoraScrewDriver 口径的阶段验收结果与 artifact 路径记录。 |
| 鲁棒性总结 | `docs/robustness_eval_summary.md` | 旧 XHandHoraScrewDriver nominal/light/hard 结果和 action-chunk/latent 分析。 |
| diffusion 总览 | `docs/diffusion_algorithm.md` | diffusion、consistency、flow、action-chunk 的历史定位与旧任务结论。 |
| consistency 结论 | `docs/plansv5_5_final_verdict.md` | 旧 XHand 口径下 Consistency V5.5 accepted 结果。 |
| flow 结论 | `docs/plansv8_final_verdict.md` | 旧 XHand 口径下 Flow Matching recovery sprint 结果。 |
| DexH13 smoke | `docs/plansv9_final_verdict.md` | DexH13 / XHand lightbulb smoke 级接入结果。 |
| 当前实验脚本 | `scripts/paper_codrive_thesis_full.sh`、`scripts/paper_codrive_eval.py`、`scripts/paper_codrive_summarize.py` | 当前 CoDriveThesis formal suite、eval、summary 流程。 |
| 当前云端结果 | `thesis_reference/cloud_final_results_20260506_210723/aggregate_csv/main_aggregate.csv` | 当前 CoDriveThesis main eval 聚合结果，已同步到本地。 |
| 最终实验设计与结论 | `thesis_reference/final_experiment_design_and_conclusions.md` | 专门记录 formal suite 设计、最终结果和论文结论建议。 |

证据等级定义：

- A. artifact-verified：有 outputs/log/checkpoint/eval CSV 或明确实验输出支撑，并在本次检查中能看到 artifact。
- B. doc-reported：summary Markdown 中报告了结果，但当前本地没有原始 artifact。
- C. code-verified：代码、脚本或配置存在，但没有对应结果。
- D. user-claimed：只有用户描述，项目文件中未找到证据。
- E. not-found：未找到证据。

本文档为事实整理，不是最终论文结论。后续论文应以最终冻结的 CSV、图表和实验口径为准。

## 1. 项目一句话概述

本项目基于已有的 Isaac Gym 灵巧手 teacher-student 强化学习框架，围绕灵巧手旋拧 / rotation 类接触操作任务，研究在 teacher-student 蒸馏阶段引入 diffusion-style 生成式 student 是否能改进或替代原始 ProprioAdapt-style student，并与 Pure BC、PAdapt、DAgger、DOTPG 等 baseline 进行统一评测。

需要区分两个任务口径：

- 仓库历史 canonical path 是 `XHandHoraScrewDriver -> PPO teacher -> current student -> evaluation/export`，这一点在 `AGENTS.md` 和历史 summary 中明确。
- 当前毕业论文 formal cloud suite 的主任务不是旧 `XHandHoraScrewDriver`，而是 `Dexh13HoraLightbulbSim2RealTwoFingerCoDriveThesis`。因此旧 XHand 结果只能作为历史迭代或附录参考，当前论文主表建议以 CoDriveThesis formal eval 为准。

基础框架是 Isaac Gym 灵巧手 teacher-student 强化学习框架。Teacher 是 PPO privileged teacher，可以访问 privileged information、point cloud 或 simulator state 等更完整信息。Current student 是 ProprioAdapt-style imitation-distillation student，不是第二阶段 RL，也不是 Pure BC。本人新增重点应写作 student 蒸馏阶段的 diffusion-style 替换、baseline 对比和统一评测，而不是从零实现 PPO teacher 或整个 teacher-student 框架。

## 2. 项目边界与个人贡献边界

| 模块/内容 | 来源/性质 | 当前状态 | 是否作为个人主要贡献 | 证据来源 | 论文中建议写法 |
| -- | -- | -- | -- | -- | -- |
| Isaac Gym 灵巧手仿真框架 | 开源/已有框架内容 | 可用 | 否 | `README.md`、`instruction_docs/repo_strategy_map.md` | 写作实验平台和复现基础。 |
| PPO privileged teacher | 开源/已有框架核心阶段 | 当前 CoDriveThesis 使用已有 teacher checkpoint | 否 | `README.md`、`AGENTS.md`、云端 `sim2real/codrive_thesis/best_reward_3655.17.pth` | 写作 teacher upper-bound / demonstration source。 |
| ProprioAdapt-style current student | 原始框架 Stage 2 student | 已作为 strong baseline 训练和评测 | 否，作为强 baseline | `docs/baseline.md`、云端 `main_aggregate.csv` | 写作原始 strong baseline，不要写成本人提出。 |
| Pure BC | 本项目 baseline 对照 | 已实现并在当前 formal suite 评测 | 是，可作为 baseline 整理工作 | `dexscrew/algo/ppo/pure_bc.py`、云端 `main_aggregate.csv` | 写作用于隔离 latent distillation / adapter 价值的基础对照。 |
| Diffusion Latent Student | 本项目 diffusion-style student | 已实现并在当前 formal suite 评测 | 是 | `dexscrew/algo/ppo/diffusion_latent_student.py`、云端 `main_aggregate.csv` | 写作 latent-space diffusion student 变体。 |
| Consistency Diffusion Student | 本项目 diffusion-style student | 已实现并在当前 formal suite 评测 | 是 | `dexscrew/algo/ppo/consistency_latent_student.py`、`docs/plansv5_5_final_verdict.md`、云端 `main_aggregate.csv` | 写作 few-step / consistency-style latent student，当前主结果最强。 |
| Flow Matching Student | 本项目 diffusion-style student | 已实现并在当前 formal suite 评测 | 是 | `dexscrew/algo/ppo/flow_matching_latent_student.py`、`docs/plansv8_final_verdict.md`、云端 `main_aggregate.csv` | 写作 flow-style latent student 变体；需说明旧 XHand 与当前 CoDriveThesis 结果口径不同。 |
| Diffusion Action-Chunk Student | 本项目 exploratory diffusion 分支 | 已实现；当前 CoDriveThesis representation / NFE eval 已完成，但不纳入主方法 | 是，但更适合 ablation | `thesis_reference/cloud_final_results_20260506_210723/aggregate_csv/representation_aggregate.csv`、`nfe*_aggregate.csv` | 写作 action-space / representation ablation，不建议作为主正结果。 |
| teacher rollout / demonstration 接口 | 本项目复现与扩展内容 | 已有接口与脚本 | 可作为工程支撑 | `plan_summary_v1.md`、`scripts/collect_screwdriver_teacher_rollout.sh` | 写作数据接口与蒸馏流程支撑。 |
| 统一 fixed-step / episode eval | 本项目整理与实现 | 当前 formal main eval 已完成 | 是 | `scripts/paper_codrive_eval.py`、云端 `main_aggregate.csv` | 写作统一评测流程。 |
| robustness / multiseed 汇总 | 本项目整理与实验 | 历史 XHand 完成；当前 CoDriveThesis formal robustness 也已完成并同步 | 是 | `thesis_reference/cloud_final_results_20260506_210723/aggregate_csv/robust_*_aggregate.csv` | 当前论文可使用 CoDriveThesis robustness；旧 XHand 只作历史参考。 |
| Sim-to-Sim / 真实部署 | 开题目标之一 | 未找到完整最终证据 | 否 | `README.md` 只说明部署路径；当前无最终 artifact | 没证据不写成完成结果，放展望或不足。 |

## 3. 原始 Teacher-Student 框架概述

原始框架是 staged teacher-student pipeline。

PPO teacher 的作用是利用仿真中的 privileged information、point cloud 或 simulator state 训练高性能 oracle policy。Teacher 在训练和评估中可以访问比最终部署策略更多的信息，因此它更适合作为性能上界和示范来源，而不是最终部署策略。

Student 不能直接访问 privileged information，因为真实部署或受限观测设置中通常无法获得完整仿真状态。ProprioAdapt-style student 的核心思想是用 proprioceptive history 预测 teacher latent，从而在缺少 privileged information 的情况下近似 teacher 的控制行为。

Current student 不是第二阶段 RL。它是在环境中在线收集 teacher/student 对齐数据，通过 imitation-distillation 训练 student。其高层关系为：

- action BC：让 student 输出动作接近 teacher 动作；
- latent distillation：让 student latent 接近 teacher privileged latent；
- adapter-based adaptation：通过 proprioceptive history 适配不可见的环境或接触状态；
- teacher policy backbone 多数保持冻结或复用，student 主要学习从受限观测到 latent/action 的映射。

因此，论文不能把 current student 简化成 Pure BC。Pure BC 只是去掉 latent distillation 或 adapter 机制后的基础对照。

## 4. 任务环境与实验对象

任务名称：

- 历史 canonical 任务：`XHandHoraScrewDriver`。
- 当前论文 formal suite 主任务：`Dexh13HoraLightbulbSim2RealTwoFingerCoDriveThesis`。

任务目标：让灵巧手通过接触操作持续驱动物体旋转，关注 screw / lightbulb / rotation progress。当前 CoDriveThesis 任务更贴近论文收尾阶段统一评测所用基准。

灵巧手动作空间的高层含义：策略输出不是直接写成“真实关节力矩”，而是控制灵巧手目标或增量动作，底层控制器再执行。论文中可表述为高维连续控制动作，不应展开到具体实现细节。

主要观测组：

- `obs`：非 privileged 的主要策略观测；
- `proprio_hist`：student 用于推断 latent 的 proprioceptive history；
- `privileged information`：teacher 训练和 student 蒸馏监督中使用的仿真状态；
- `point cloud / object state`：在部分 teacher/student 设定中作为额外信息或 teacher latent 的组成来源。

Reward / reset / domain randomization 高层说明：

- Reward 主要围绕旋转进展、接触稳定、姿态偏差、能耗或控制惩罚等构成。
- Reset 与最大 episode length、手指距离、停滞、失去接触、旋转上限等有关。
- Domain randomization 覆盖观测噪声、摩擦、质量、COM、初始位姿、外力或控制扰动等。

该任务属于高维、接触丰富、长时序、不稳定控制任务，原因是：

- 灵巧手自由度高，动作空间维度大；
- 手指-物体接触具有强非线性和不连续性；
- 短期 imitation loss 不一定能反映长期旋转进展；
- 生成式 student 的采样误差可能在闭环控制中被快速放大；
- 任务成功依赖持续接触和方向一致的旋转，而不仅是单步动作拟合。

论文中适合使用的任务指标包括 reward、episode length、rotation progress / screw angular position、screw angular velocity、positive velocity ratio、done rate / failure reset、success threshold、robustness degradation、sampling steps 和 inference latency。

## 5. 算法对比矩阵

| 方法 | 类型 | 替换/对比位置 | 高层输入 | 高层输出 | 当前状态 | 是否已训练 | 是否已评测 | 证据等级 | 证据来源 | 论文中建议角色 |
| -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| PPO Teacher | RL privileged teacher | teacher / upper-bound | obs + privileged information / point cloud | 控制动作、teacher latent | 完整实现并评测 | 是 | 是 | A | `README.md`、`scripts/paper_codrive_thesis_full.sh`、云端 `main_aggregate.csv` | upper-bound 和 demonstration source。 |
| ProprioAdapt-style Student / PAdapt | imitation-distillation student | 原始 Stage 2 student baseline | obs + proprio history | student latent + 控制动作 | 完整实现并评测 | 是 | 是 | A | `docs/baseline.md`、`dexscrew/algo/ppo/padapt.py`、云端 `main_aggregate.csv` | strong baseline。 |
| Pure BC | imitation baseline | 去除 latent distillation 的对照 | obs + proprio history | 控制动作 | 完整实现并评测 | 是 | 是 | A | `dexscrew/algo/ppo/pure_bc.py`、云端 `main_aggregate.csv` | 基础 BC baseline。 |
| Diffusion Latent | latent-space diffusion student | 替换 deterministic latent prediction | obs/proprio history + latent noise | generated latent + 控制动作 | 完整实现并评测 | 是 | 是 | A | `dexscrew/algo/ppo/diffusion_latent_student.py`、云端 `main_aggregate.csv` | diffusion 基础变体。 |
| Consistency Diffusion | consistency-style latent student | 替换 deterministic latent prediction | obs/proprio history + consistency noise level | generated latent + 控制动作 | 完整实现并评测 | 是 | 是 | A | `dexscrew/algo/ppo/consistency_latent_student.py`、`docs/plansv5_5_final_verdict.md`、云端 `main_aggregate.csv` | 当前最强 diffusion 主结果。 |
| Flow Matching Diffusion | flow-style latent student | 替换 deterministic latent prediction | obs/proprio history + flow state | generated latent + 控制动作 | 完整实现并评测 | 是 | 是 | A | `dexscrew/algo/ppo/flow_matching_latent_student.py`、`docs/plansv8_final_verdict.md`、云端 `main_aggregate.csv` | 重要 diffusion 变体。 |
| Diffusion Action-Chunk | action-space diffusion student | 直接生成动作片段 | obs/proprio history + action noise | action chunk | 已实现并完成 representation / NFE eval，但不纳入主方法 | 是 | 是 | A | `thesis_reference/cloud_final_results_20260506_210723/aggregate_csv/representation_aggregate.csv`、`nfe*_aggregate.csv` | representation / action-space ablation。 |
| BC / LatentBC | imitation baseline | reference baseline | student observation | 控制动作 | 完整实现并评测 | 是 | 是 | A | `scripts/paper_codrive_thesis_full.sh`、云端 `main_aggregate.csv` | single-train-seed reference。 |
| DAgger | imitation / dataset aggregation baseline | reference baseline | student observation + teacher correction | 控制动作 | 完整实现并评测 | 是 | 是 | A | `scripts/paper_codrive_thesis_full.sh`、云端 `main_aggregate.csv` | single-train-seed reference。 |
| DOTPG | 对偶优化 / next-paper baseline | reference baseline | student state/action相关输入 | 控制动作 | 完整实现并评测 | 是 | 是 | A | `docs/dotpg_env.md`、`thesis_reference/DOTPG-draft.md`、云端 `main_aggregate.csv` | 当前论文只建议作为 reference。 |
| Residual Diffusion | latent residual variant | diffusion latent ablation | base latent + residual target | corrected latent | 仅设计/文档提及 + code path 存在 | 历史训练报告 | 历史评测报告 | B/C | `docs/plansv2_m4_residual_*.md`、`dexscrew/algo/ppo/diffusion_latent_student.py` | 附录或负结果。 |
| Latent-only / decode-only | ablation | latent 或 decoder 机制拆分 | student latent | action / recon | 仅文档报告 | 历史报告 | 历史评测报告 | B | `docs/robustness_eval_summary.md` | 不建议主表。 |
| Adapter range ablation | PAdapt trainable range ablation | current student 内部消融 | proprio history | latent/action | 仅文档报告 | 历史报告 | 历史报告 | B | `plan_summary_v1.md`、`docs/stage_acceptance_summary.md` | 可作为工程过程，不建议正文重点。 |
| Action-chunk BC / first-action BC | action-chunk 内部目标 | action-chunk 辅助监督 | obs/proprio history | action chunk / first action | code-verified + doc-reported | 历史报告 | 历史报告 | B/C | `dexscrew/algo/ppo/diffusion_action_chunk_student.py`、`docs/robustness_eval_summary.md` | action-chunk failure analysis。 |
| KL distillation | student KL 蒸馏 | 未确认 | 未找到 | 未找到 | 未找到证据 | 未找到 | 未找到 | E | 仅找到 PPO KL scheduler，不是 student KL distillation | 不写。 |

## 6. 已实现功能总结

### 6.1 训练流程

| 功能 | 当前状态 | 证据来源 | 备注 |
| -- | -- | -- | -- |
| teacher 训练 | 原始框架支持；当前 CoDriveThesis 使用已有 teacher checkpoint | `README.md`、`scripts/screwdriver_teacher.sh`、`scripts/paper_codrive_thesis_full.sh` | 当前论文不要写成从零训练 teacher 的原创贡献。 |
| current student / PAdapt 训练 | 已支持，当前 formal suite 已训练 3 seeds | `dexscrew/algo/ppo/padapt.py`、云端 train status `124`、云端 `main_aggregate.csv` | `124` 是 timeout 到时结束，不等同异常失败。 |
| Pure BC 训练 | 已支持，当前 formal suite 已训练 3 seeds | `dexscrew/algo/ppo/pure_bc.py`、云端 `main_aggregate.csv` | baseline 对照。 |
| Diffusion Latent 训练 | 已支持，当前 formal suite 已训练 3 seeds | `dexscrew/algo/ppo/diffusion_latent_student.py`、云端 `main_aggregate.csv` | 当前表现低于 PAdapt / Pure BC。 |
| Consistency Diffusion 训练 | 已支持，当前 formal suite 已训练 3 seeds | `dexscrew/algo/ppo/consistency_latent_student.py`、云端 `main_aggregate.csv` | 当前 CoDriveThesis student 中最强。 |
| Flow Matching 训练 | 已支持，当前 formal suite 已训练 3 seeds | `dexscrew/algo/ppo/flow_matching_latent_student.py`、云端 `main_aggregate.csv` | 当前 CoDriveThesis 表现强于 PAdapt。 |
| Action-Chunk Diffusion 训练 | 已支持；当前 formal representation 训练存在 len1 checkpoint | `dexscrew/algo/ppo/diffusion_action_chunk_student.py`、云端 train_outputs | 当前主 eval aggregate 未包含 action-chunk 主表结果。 |

### 6.2 数据与 rollout

| 功能 | 当前状态 | 证据来源 | 备注 |
| -- | -- | -- | -- |
| teacher rollout 采集 | 已有接口和脚本 | `plan_summary_v1.md`、`scripts/collect_screwdriver_teacher_rollout.sh` | 文档报告，当前本地无 rollout artifact。 |
| diffusion student 使用数据字段 | code-verified | diffusion/consistency/flow/action-chunk 算法文件 | 只确认存在，不在论文中展开代码细节。 |
| offline dataset | 未找到完整证据 | 开题报告有设想，当前项目主流程偏 online teacher distillation | 论文不要写成完整离线 dataset 消融。 |
| rollout pretrain / action chunk 数据 | 历史文档报告 + code/script 存在 | `plan_summary_v1.md`、`scripts/screwdriver_student_diffusion_action_chunk_rollout_pretrain.sh` | 适合写作探索性工程，不建议主结论。 |
| 数据接口完整性 | 部分完整 | `plan_summary_v1.md` | 当前满足蒸馏实验，但 artifact gap 仍存在。 |

### 6.3 评测流程

| 功能 | 当前状态 | 证据来源 | 备注 |
| -- | -- | -- | -- |
| nominal eval | 已有 | `scripts/paper_codrive_eval.py`、历史 robustness scripts | 当前 CoDriveThesis main eval 已完成。 |
| fixed-step eval | 已有 | 云端 `main_aggregate.csv` | 当前主表使用 2048 steps。 |
| multi-seed eval | 已有 | 云端 `main_aggregate.csv` | formal methods 为 3 train seeds × 3 eval seeds。 |
| robustness eval | 历史 XHand 完成；当前 CoDriveThesis formal robustness 已完成 | `thesis_reference/cloud_final_results_20260506_210723/aggregate_csv/robust_*_aggregate.csv` | 当前论文应使用 CoDriveThesis robustness；旧 XHand robustness 只作历史参考。 |
| summary doc | 已有 | `docs/stage_acceptance_summary.md`、`docs/robustness_eval_summary.md`、当前本文档 | 旧文档有口径冲突。 |
| 统一指标束 | 已完成 | `thesis_reference/cloud_final_results_20260506_210723/aggregate_csv/` | main、NFE、latency、representation、robustness 均已有最终聚合表。 |

### 6.4 可视化、导出与部署

| 功能 | 当前状态 | 证据来源 | 备注 |
| -- | -- | -- | -- |
| 可视化 | 脚本存在 | `scripts/vis_*.sh` | 未核实当前 CoDriveThesis 可视化产物。 |
| current student export | 原框架支持 | `README.md`、`student_eval.py`、`scripts/convert_student_jit.sh` | 主要面向 ProprioAdapt-style student。 |
| diffusion student export | 风险 / 不完整 | `plan_summary_v1.md` | 文档明确 diffusion export 不支持或不完整。 |
| evaluation/export parity | 存在风险 | `plan_summary_v1.md` | 不应把 diffusion deployment 写成完成。 |
| Sim-to-Sim / real deployment | 未找到证据 | `README.md` 仅指向部署路径和外部 repo | 若未补充 artifact，建议放展望。 |

## 7. 实验协议与指标

### 7.1 当前 CoDriveThesis formal eval 协议

| 项 | 当前记录 |
| -- | -- |
| 任务 | `Dexh13HoraLightbulbSim2RealTwoFingerCoDriveThesis` |
| 平台 | Isaac Gym |
| teacher checkpoint | `sim2real/codrive_thesis/best_reward_3655.17.pth` |
| formal methods | PAdapt、PureBC、DiffusionLatent、ConsistencyLatent、FlowMatching |
| reference methods | PPO Teacher、BC/LatentBC、DAgger、DOTPG |
| train seeds | formal methods: 42, 43, 44；reference baselines 多数为 single train seed |
| eval seeds | 42, 43, 44 |
| fixed-step | 2048 steps |
| numEnvs | 48 |
| headless | 是 |
| 当前主结果来源 | 云端 `outputs/paper_codrive_thesis_full_20260506_210723/aggregate_csv/main_aggregate.csv` |
| checkpoint selection deviation | `model_best_deploy.ckpt` 因 5h timeout 未触发 eval-select，由 hotfix 指向 `model_best_train.ckpt`；必须在论文实验设置中说明。 |

### 7.2 历史 XHandHoraScrewDriver robustness 协议

| 项 | 当前记录 |
| -- | -- |
| 任务 | `XHandHoraScrewDriver` |
| 评测 steps | 多数为 256 或 512 steps |
| 条件 | nominal、light_v2、hard |
| seeds | 多处文档报告 seeds 42/43/44 |
| 结果来源 | `docs/stage_acceptance_summary.md`、`docs/robustness_eval_summary.md`、`docs/plansv5_5_final_verdict.md`、`docs/plansv8_final_verdict.md` |
| 是否可作为当前论文主表 | 不建议直接作为主表；可作为历史迭代或附录。 |

### 7.3 指标可用性

| 指标 | 当前数据状态 | 主要作用 | 证据来源 |
| -- | -- | -- | -- |
| reward / fixed-step reward | 已有 | 主性能指标 | 云端 `main_aggregate.csv`、历史 docs |
| episode length | 已有 | 稳定持续时间 | 云端 `main_aggregate.csv` |
| rotation progress / screw angular position | 已有 progress 聚合 | 任务核心进展 | 云端 `main_aggregate.csv` |
| screw angular velocity | 历史 docs 提及，当前主表未单列 | 旋转速度 | `instruction_docs/repo_strategy_map.md` |
| positive velocity ratio | 已有 | 旋转方向一致性 | 云端 `main_aggregate.csv` |
| failure reset / done rate | 已有 | 失败率 / reset 风险 | 云端 `main_aggregate.csv`、历史 docs |
| reset reason | 当前主表未找到完整汇总 | failure mode 分析 | `scripts/paper_codrive_eval.py` 支持 info 聚合，但最终表待确认 |
| latent loss | 历史部分有 | latent 对齐 | `docs/robustness_eval_summary.md` |
| behavior cloning loss | 历史部分有 | action imitation | `docs/robustness_eval_summary.md` |
| diffusion loss | 未找到最终主表 | diffusion 训练质量 | 算法文件 code-verified，最终数值待补充 |
| consistency loss | 未找到最终主表 | consistency 训练质量 | 算法文件 code-verified，最终数值待补充 |
| flow matching loss | 未找到最终主表 | flow 训练质量 | 算法文件 code-verified，最终数值待补充 |
| sampling steps / NFE | 已有最终数据 | 推理成本与性能权衡 | `thesis_reference/cloud_final_results_20260506_210723/aggregate_csv/nfe*_aggregate.csv` |
| chunk length | code/script 有 | action-chunk ablation | `scripts/paper_codrive_thesis_full.sh` |
| inference latency / sampling cost | 已有最终数据 | 实时控制可行性 | `thesis_reference/cloud_final_results_20260506_210723/aggregate_csv/latency_nfe*_aggregate.csv` |

## 8. 实验结果汇总

### 8.1 主实验结果表

以下结果来自当前 CoDriveThesis cloud formal main eval。Reward 为 fixed-step reward mean；Failure Reset 使用 done_rate mean。证据等级为 A，因为本次只读检查确认云端 CSV artifact 存在。

| 方法 | Run ID / Checkpoint | Seeds | Episodes | Reward | Episode Length | Rotation Progress | Positive Velocity Ratio | Failure Reset | 证据等级 | 证据来源 |
| -- | -- | --: | --: | --: | --: | --: | --: | --: | -- | -- |
| PPO Teacher | `sim2real/codrive_thesis/best_reward_3655.17.pth` | eval 42/43/44 | episode target 256 / eval seed | 5.479 | 686.5 | 2.355 | 0.817 | 0.00104 | A | 云端 `aggregate_csv/main_aggregate.csv` |
| ProprioAdapt Student | `train_outputs/padapt_s*/stage2_nn/model_best_deploy.ckpt` | 3 train × 3 eval | 同上 | 4.880 | 673.8 | 2.182 | 0.799 | 0.00107 | A | 云端 `aggregate_csv/main_aggregate.csv` |
| Pure BC | `train_outputs/purebc_s*/stage2_bc_nn/model_best_deploy.ckpt` | 3 train × 3 eval | 同上 | 4.817 | 668.8 | 2.207 | 0.799 | 0.00109 | A | 云端 `aggregate_csv/main_aggregate.csv` |
| Diffusion Latent | `train_outputs/diffusion_latent_s*/stage2_diffusion_nn/model_best_deploy.ckpt` | 3 train × 3 eval | 同上 | 4.569 | 642.1 | 1.816 | 0.815 | 0.00116 | A | 云端 `aggregate_csv/main_aggregate.csv` |
| Consistency Diffusion | `train_outputs/consistency_latent_s*/stage2_consistency_nn/model_best_deploy.ckpt` | 3 train × 3 eval | 同上 | 5.228 | 678.8 | 2.358 | 0.811 | 0.00105 | A | 云端 `aggregate_csv/main_aggregate.csv` |
| Flow Matching Diffusion | `train_outputs/flow_matching_s*/stage2_flow_nn/model_best_deploy.ckpt` | 3 train × 3 eval | 同上 | 5.102 | 672.7 | 2.265 | 0.814 | 0.00107 | A | 云端 `aggregate_csv/main_aggregate.csv` |
| Action-Chunk Diffusion | representation / NFE ablation | eval seed 42/43/44 | fixed-step | len=8: -1.232；len=1: 0.141 | 不适用 | 不适用 | 不适用 | 不适用 | A | `thesis_reference/cloud_final_results_20260506_210723/aggregate_csv/representation_aggregate.csv`、`nfe*_aggregate.csv` |
| BC / LatentBC | `outputs/Dexh13HoraLightbulb_student_bc_codrive_thesis/.../model_best_student_eval.ckpt` | train seed 42 × eval 42/43/44 | 同上 | 4.086 | 654.9 | 2.132 | 0.760 | 0.00110 | A | 云端 `aggregate_csv/main_aggregate.csv` |
| DAgger | `outputs/Dexh13HoraLightbulb_student_dagger_codrive_thesis/.../model_best_student_eval.ckpt` | train seed 42 × eval 42/43/44 | 同上 | 4.768 | 685.1 | 2.138 | 0.802 | 0.00107 | A | 云端 `aggregate_csv/main_aggregate.csv` |
| DOTPG | `outputs/Dexh13HoraLightbulb_student_dotpg_codrive_thesis/.../model_best.ckpt` | train seed 42 × eval 42/43/44 | 同上 | 3.446 | 683.3 | 2.241 | 0.702 | 0.00104 | A | 云端 `aggregate_csv/main_aggregate.csv` |

注意：formal methods 的 `model_best_deploy.ckpt` 在当前 run 中是 hotfix symlink，指向 `model_best_train.ckpt`。证据路径：云端 `status/checkpoint_selection_hotfix.txt`、`status/checkpoint_aliases.tsv`。

### 8.2 鲁棒性实验结果表

当前 CoDriveThesis formal robustness 已完成并同步。下表仍保留历史 `XHandHoraScrewDriver` 256-step nominal/light_v2/hard 口径结果作为历史参考；当前论文主表应优先使用 `thesis_reference/cloud_final_results_20260506_210723/aggregate_csv/robust_*_aggregate.csv`。

| 方法 | Nominal | Light Perturbation | Hard Perturbation | 主要变化 | 证据等级 | 证据来源 |
| -- | --: | --: | --: | -- | -- | -- |
| PPO Teacher | 3.056 | 2.915 | 2.763 | 历史 upper-bound，扰动下下降但仍最高 | B | `docs/stage_acceptance_summary.md` |
| ProprioAdapt Student | 2.168 | 2.079 | 1.838 | 历史 hard 条件较稳 | B | `docs/robustness_eval_summary.md`、`docs/stage_acceptance_summary.md` |
| Pure BC | 1.883 | 2.191 | 1.850 | light/hard 不弱，nominal 波动较大 | B | `docs/robustness_eval_summary.md` |
| Diffusion Latent | 2.063 | 1.789 | 1.572 | 有正回报但 hard 明显低于 PAdapt | B | `docs/robustness_eval_summary.md` |
| Consistency Diffusion | 2.336 | 2.045 | 1.714 | nominal 强，hard 未超过 PAdapt | B | `docs/plansv5_5_final_verdict.md` |
| Flow Matching Diffusion | 1.783 | 1.588 | 1.367 | 旧 XHand 口径较弱 | B | `docs/plansv8_final_verdict.md` |
| Action-Chunk Diffusion | 有冲突 | 有冲突 | 有冲突 | 多轮从负回报到局部改善，但不稳定；旧文档结论不统一 | B | `docs/robustness_eval_summary.md` |
| 当前 CoDriveThesis robustness | 见最终 CSV | 见最终 CSV | 见最终 CSV | 已完成 108/108 rows；Consistency / Flow 多数扰动下优于 PAdapt / PureBC | A | `thesis_reference/cloud_final_results_20260506_210723/aggregate_csv/robust_*_aggregate.csv` |

### 8.3 Diffusion 方法机制对比表

| 方法 | Sampling Steps | Chunk Length | Latent/Action 表示 | 推理成本 | 稳定性现象 | 证据来源 |
| -- | -- | -- | -- | -- | -- | -- |
| Diffusion Latent | 当前 formal 默认训练/推理 steps 约为 10；NFE 已完成 | 不适用 | latent 表示 | 中等，需多步采样 | 当前 CoDriveThesis reward 低于 PAdapt/PureBC；历史 hard 也较弱 | `thesis_reference/cloud_final_results_20260506_210723/aggregate_csv/main_aggregate.csv`、`nfe*_aggregate.csv` |
| Consistency Diffusion | formal 默认 infer steps=1；NFE 已完成 | 不适用 | latent 表示 | 低 NFE 即可保持高 reward | 当前 CoDriveThesis 最强 student 之一；旧 XHand hard 未稳定超过 PAdapt | `thesis_reference/cloud_final_results_20260506_210723/aggregate_csv/main_aggregate.csv`、`nfe*_aggregate.csv` |
| Flow Matching Diffusion | formal 默认 infer steps=1；NFE 已完成 | 不适用 | latent 表示 | 低 NFE 即可保持高 reward，NFE=1 仍强 | 当前 CoDriveThesis 高于 PAdapt；旧 XHand V8 失败，存在口径冲突 | `thesis_reference/cloud_final_results_20260506_210723/aggregate_csv/main_aggregate.csv`、`nfe*_aggregate.csv` |
| Diffusion Action-Chunk | 默认 diffusion steps 10；NFE 已完成 | len=8；另有 len=1 representation eval | action chunk | 高，动作维度和 chunk 维度放大采样成本 | 当前 formal 结果明显弱于 latent-space 方法 | `thesis_reference/cloud_final_results_20260506_210723/aggregate_csv/representation_aggregate.csv`、`nfe*_aggregate.csv` |

### 8.4 蒸馏/训练损失表

当前 CoDriveThesis `main_aggregate.csv` 主要汇总 reward/progress/done/episode 指标，没有找到完整的最终训练损失总表。以下仅记录已在文档中明确出现的损失类信息。

| 方法 | Latent Loss | BC Loss | Diffusion Loss | 收敛现象 | 证据来源 |
| -- | -- | -- | -- | -- | -- |
| ProprioAdapt | 未找到最终表 | 未找到最终表 | 不适用 | 文档称为 latent distillation + action BC + adapter-based adaptation | `docs/baseline.md` |
| Pure BC | 不适用或未找到 | 未找到最终表 | 不适用 | 仅行为克隆对照，当前 CoDriveThesis reward 接近 PAdapt | `docs/baseline.md`、云端 `main_aggregate.csv` |
| Diffusion Latent | 历史 M5 报告 nominal latent_mse 约 0.080 | 历史 M5 报告 action_mse 约 0.146 | 未找到最终表 | 历史有正回报但 robust 不足；当前主表低于 PAdapt | `docs/robustness_eval_summary.md`、云端 `main_aggregate.csv` |
| Consistency Diffusion | 未找到最终表 | 未找到最终表 | consistency loss 未找到最终表 | 当前 CoDriveThesis 表现最好；旧 XHand V5.5 accepted | `docs/plansv5_5_final_verdict.md`、云端 `main_aggregate.csv` |
| Flow Matching Diffusion | 未找到最终表 | 未找到最终表 | flow loss 未找到最终表 | 当前 CoDriveThesis 强；旧 XHand V8 failed | `docs/plansv8_final_verdict.md`、云端 `main_aggregate.csv` |
| Action-Chunk Diffusion | 不适用 | 历史含 first-action / chunk BC 相关记录，但最终表未找到 | 未找到最终表 | 历史训练分数与部署评测存在错位 | `docs/robustness_eval_summary.md` |

### 8.5 多 seed 统计表

以下为当前 CoDriveThesis fixed-step reward 聚合。

| 方法 | Metric | Mean | Std | Seeds | Episodes | 证据来源 |
| -- | -- | --: | --: | -- | -- | -- |
| PPO Teacher | fixed_step_reward | 5.479 | 0.014 | eval 42/43/44 | episode target 256 / seed | 云端 `main_aggregate.csv` |
| ProprioAdapt Student | fixed_step_reward | 4.880 | 0.139 | 3 train × 3 eval | 同上 | 云端 `main_aggregate.csv` |
| Pure BC | fixed_step_reward | 4.817 | 0.167 | 3 train × 3 eval | 同上 | 云端 `main_aggregate.csv` |
| Diffusion Latent | fixed_step_reward | 4.569 | 0.115 | 3 train × 3 eval | 同上 | 云端 `main_aggregate.csv` |
| Consistency Diffusion | fixed_step_reward | 5.228 | 0.062 | 3 train × 3 eval | 同上 | 云端 `main_aggregate.csv` |
| Flow Matching Diffusion | fixed_step_reward | 5.102 | 0.155 | 3 train × 3 eval | 同上 | 云端 `main_aggregate.csv` |
| BC / LatentBC | fixed_step_reward | 4.086 | 0.047 | train seed 42 × eval 42/43/44 | 同上 | 云端 `main_aggregate.csv` |
| DAgger | fixed_step_reward | 4.768 | 0.021 | train seed 42 × eval 42/43/44 | 同上 | 云端 `main_aggregate.csv` |
| DOTPG | fixed_step_reward | 3.446 | 0.021 | train seed 42 × eval 42/43/44 | 同上 | 云端 `main_aggregate.csv` |
| Action-Chunk Diffusion | fixed_step_reward | 未找到 | 未找到 | 未找到 | 未找到 | 当前主表缺失 |

## 9. 证据等级与 artifact 追踪

| 结论/结果 | 证据等级 | artifact 或文档路径 | 是否可复现 | 风险说明 |
| -- | -- | -- | -- | -- |
| 原项目是 Isaac Gym teacher-student 框架 | B/C | `README.md`、`instruction_docs/repo_strategy_map.md` | 可复现依赖环境 | 属于开源框架基础，不是个人原创。 |
| current student 是 ProprioAdapt-style imitation-distillation，不是 Pure BC | B/C | `AGENTS.md`、`docs/baseline.md` | 可通过代码确认 | 论文必须避免简化错误。 |
| 当前 CoDriveThesis main eval 已完成 | A | 云端 `outputs/paper_codrive_thesis_full_20260506_210723/aggregate_csv/main_aggregate.csv` | 云端可复现，需保留 run dir | 本地没有 artifact，需同步或引用云端路径。 |
| Consistency 当前 CoDriveThesis 最强 student | A | 云端 `main_aggregate.csv` | 可复现需同 checkpoint/seed | 只能在当前 CoDriveThesis 口径下成立。 |
| Flow 当前 CoDriveThesis 高于 PAdapt | A | 云端 `main_aggregate.csv` | 可复现需同 checkpoint/seed | 与旧 XHand V8 结论冲突，必须标注口径差异。 |
| Diffusion Latent 当前低于 PAdapt/PureBC | A | 云端 `main_aggregate.csv` | 可复现需同 checkpoint/seed | 不代表所有 diffusion 失败，只代表该变体当前结果。 |
| Action-Chunk 历史不稳定 | B | `docs/robustness_eval_summary.md` | 本地无 artifact | 当前 CoDriveThesis 主表缺失，不宜主张最终结论。 |
| 历史 XHand robustness 表 | B | `docs/stage_acceptance_summary.md`、`docs/robustness_eval_summary.md` | 本地无 outputs | 不可与当前 CoDriveThesis 主表合并。 |
| NFE / latency 正在进行 | A partial | 云端 `status/phase.txt`、`raw_csv/nfe*_append.csv` | 待完成 | 不能提前写最终效率结论。 |
| checkpoint selection 使用 train-best symlink | A | 云端 `status/checkpoint_selection_hotfix.txt`、`status/checkpoint_aliases.tsv` | 可核实 | 论文实验设置必须透明说明。 |
| Sim-to-Sim / real deployment 完成 | E | 未找到最终 artifact | 不可复现 | 不应写成完成贡献。 |
| diffusion export 完整支持 | B/C negative | `plan_summary_v1.md`、`student_eval.py` | 需进一步验证 | 当前存在 export parity 风险。 |

本地 artifact gap：当前本地仓库没有 `outputs/`、checkpoint、eval CSV、训练 log。这意味着除云端已核实的 formal run 外，许多历史结论只能算 doc-reported。建议论文收尾前把云端 final CSV、manifest、commands.log、关键 checkpoint SHA、最终图表同步到本地或单独归档。

## 10. 当前实验现象与初步判断

1. ProprioAdapt-style student 是否是强 baseline？  
   是。历史 XHand 口径和当前 CoDriveThesis 口径均显示 PAdapt 是强 baseline。当前 CoDriveThesis fixed-step reward 为 4.880，高于 Diffusion Latent 和 BC/LatentBC，但低于 Consistency 和 Flow。

2. Pure BC 与 ProprioAdapt 的差异是否有数据支撑？  
   有。当前 CoDriveThesis 中 Pure BC 为 4.817，PAdapt 为 4.880，差距不大但 PAdapt 略高。历史文档中 Pure BC 有时在 light/hard 上不弱，说明简单 action imitation 已经很强，不能把 baseline 设得过弱。

3. Diffusion Latent 是否表现更稳定或更有潜力？  
   当前证据显示普通 Diffusion Latent 并未超过 PAdapt / Pure BC。它适合作为基础 diffusion 对照，而不是最佳方法。其潜力应写为“有研究价值但受建模和采样影响”，不能写成稳定优于 baseline。

4. Action-Chunk Diffusion 是否存在稳定性、响应性、采样成本或部署问题？  
   是。历史文档多次记录 action-chunk 的训练分数与纯 student 评测不一致、负回报、selector 与部署不对齐、chunk length 和 deterministic 推理影响等问题。当前 CoDriveThesis representation / NFE 结果也显示 action-chunk 明显弱于 latent-space 方法，因此更适合作为失败模式或 representation ablation。

5. Consistency Diffusion 是否有完整训练和评测证据？  
   有。代码存在，旧 XHand 有 V5.5 accepted 文档，当前 CoDriveThesis formal suite 有 3 train seeds × 3 eval seeds 聚合结果。证据等级可按当前 cloud main eval 记为 A。

6. Flow Matching Diffusion 是否有完整训练和评测证据？  
   有当前 CoDriveThesis 训练和评测证据；但旧 XHand 文档中 Flow V8 结论较弱。论文中必须说明这是不同任务/协议造成的结果差异，不能把旧结论和新结论混为同一实验。

7. diffusion 是否可以写成“全面优于 baseline”？  
   不能。更稳妥的结论是：在当前 CoDriveThesis formal eval 中，Consistency 和 Flow 两种 latent generative student 超过了 PAdapt / Pure BC，但普通 Diffusion Latent 和 Action-Chunk 并未稳定胜出。Diffusion 的优势具有条件性，依赖建模空间、采样机制、控制响应性和 checkpoint selection。

8. 当前最适合毕设论文的主线是什么？  
   建议主线为“diffusion-style generative student 在高维接触灵巧手蒸馏中的适用边界分析”，其中 Consistency / Flow 作为主要正结果，Diffusion Latent 和 Action-Chunk 作为边界与消融分析。不要把论文主线写成单纯 action diffusion，也不要写成 diffusion 全面胜利。

## 11. 开题目标与实际完成情况映射

| 开题/计划目标 | 当前完成情况 | 证据来源 | 毕设中建议处理方式 |
| -- | -- | -- | -- |
| 灵巧手仿真平台 | 已完成 / 基于开源框架 | `README.md`、`instruction_docs/repo_strategy_map.md` | 写作实验基础，不写成从零实现。 |
| RL teacher | 已完成 / 使用已有 checkpoint | `README.md`、云端 teacher checkpoint | 写作 teacher upper-bound。 |
| teacher rollout / 示范数据 | 部分完成 | `plan_summary_v1.md`、rollout scripts | 写作蒸馏数据接口；若无 artifact，少写离线 dataset。 |
| diffusion student | 已完成 | diffusion/consistency/flow/action-chunk 文件、云端 results | 写作本人主要工作。 |
| BC baseline | 已完成 | `pure_bc.py`、云端 results | 写作 baseline 对照。 |
| ProprioAdapt/current student baseline | 已完成 | `docs/baseline.md`、云端 results | 写作原始 strong baseline。 |
| 动作时序建模/action chunk | 部分完成 | `diffusion_action_chunk_student.py`、历史 docs | 写作 exploratory ablation，不作为主结果。 |
| 采样效率/consistency/flow | 部分完成 | consistency/flow 代码与主评测；NFE/latency 进行中 | 结果完整后写入效率分析，否则标注待补充。 |
| 鲁棒性评测 | 部分完成 | 历史 docs；当前 CoDriveThesis final robustness 未找到 | 若最终云端完成则写，否则放历史参考/不足。 |
| Sim-to-Sim | 未找到证据 | 开题报告有目标，仓库未见最终结果 | 放展望或不足。 |
| 部署/export | ProprioAdapt 路径存在；diffusion 不完整 | `README.md`、`plan_summary_v1.md` | 不写成 diffusion 部署完成。 |

## 12. 论文可用材料与不可夸大内容

### 12.1 论文中可以稳妥写的内容

- 基于已有 Isaac Gym 灵巧手 teacher-student 框架，开展 student 蒸馏阶段研究。
- 使用 PPO privileged teacher 作为 upper-bound 和 demonstration source。
- 将原始 ProprioAdapt-style student 作为 strong baseline，而不是弱 BC baseline。
- 实现并比较 Pure BC、Diffusion Latent、Consistency Diffusion、Flow Matching、Action-Chunk Diffusion 等 student 变体。
- 建立当前 CoDriveThesis formal main eval，包含 fixed-step reward、episode length、rotation progress、positive velocity ratio、done rate 等指标。
- 当前 CoDriveThesis 结果支持：Consistency 和 Flow 在该口径下优于 PAdapt / Pure BC；Diffusion Latent 和 Action-Chunk 则体现出边界和困难。
- 分析 diffusion 在高维接触任务中的条件性优势，而不是绝对优势。

### 12.2 论文中不应夸大的内容

- 不应声称从零构建完整灵巧手 RL 框架。
- 不应声称 PPO teacher、ProprioAdapt-style student 是本人从零提出。
- 不应把 current student 简化成 Pure BC。
- 不应声称 diffusion 全面优于 baseline。
- 不应把旧 XHand 口径结果和当前 CoDriveThesis 口径结果合并成一个无差别主表。
- 不应声称 Action-Chunk Diffusion 已经是有效主方法，除非后续补齐当前 CoDriveThesis 结果。
- 不应声称完成真实机器人部署、Sim-to-Sim 或 diffusion export，除非补充明确 artifact。
- 不应忽略 `model_best_deploy.ckpt -> model_best_train.ckpt` 的 checkpoint selection deviation。

## 13. 当前风险、不一致与待补充项

| 风险/缺口 | 影响 | 当前证据 | 建议处理 |
| -- | -- | -- | -- |
| 本地 artifact gap | 历史结果无法本地 artifact-verify | 本地无 `outputs/` 等目录 | 同步云端 final artifact 或在论文中明确 artifact 来源。 |
| 当前 CoDriveThesis 与旧 XHand 口径不一致 | 容易误用结果 | 旧 docs 与云端 main eval 结论不同 | 主表只用 CoDriveThesis；旧结果放附录/历史。 |
| Flow 旧结论弱、当前结论强 | 可能造成论文逻辑冲突 | `docs/plansv8_final_verdict.md` vs 云端 `main_aggregate.csv` | 明确任务、checkpoint、horizon 和配置差异。 |
| Consistency/Flow 状态需最终命名统一 | 影响论文方法章节 | 多处文档命名不同 | 统一命名为 Consistency Latent / Flow Matching Latent。 |
| Action-Chunk 不纳入主表 | 主表聚焦 latent-space 方法；action-space 作为 ablation | `representation_aggregate.csv` 显示 len=8 为 -1.232、len=1 为 0.141 | 写作 action-space diffusion 边界分析。 |
| NFE / latency 已完成但需转成论文图表 | 支撑实时性结论 | `latency_nfe*_aggregate.csv`、`nfe*_aggregate.csv` | 可用于 reward-latency / NFE 消融图。 |
| Robustness CoDriveThesis 已完成但需转成论文表图 | 支撑鲁棒性主结论 | `robust_*_aggregate.csv` | 可用于 robustness degradation 表和图。 |
| checkpoint selection hotfix | 影响实验规范性 | 云端 `checkpoint_selection_hotfix.txt` | 在实验设置中透明说明。 |
| diffusion diagnostics 语义不确定 | 影响 failure mode 解释 | `plan_summary_v1.md` 指出 rollout diagnostics 风险 | 不用不确定 diagnostics 作强结论。 |
| diffusion export 不支持 | 影响部署章节 | `plan_summary_v1.md` | 不写成完成部署。 |
| 多 seed 对 reference baseline 不统一 | 主表统计公平性风险 | BC/DAgger/DOTPG 为 single-train-seed reference | 主表标注 train seed 数；或将其放 reference table。 |
| success rate 接近 0 | 可能影响读者理解 | 云端 `success_2pi_rate` 多数为 0 | 解释 2π 阈值严格，主指标用 reward/progress/done。 |
| 训练损失最终表缺失 | 方法分析不完整 | 当前 main aggregate 不含 loss | 从 logs/TensorBoard 追加或不在正文强写。 |
| 图表未最终归档 | 论文材料不完整 | 当前未见 final figures | 等 `paper_codrive_build_artifacts.py` 或手动生成最终图。 |

## 14. 给 GPT 的后续分析提示

- 最终论文最可能的主线：基于已有 teacher-student 框架，研究 diffusion-style latent generative student 在高维灵巧手接触控制蒸馏中的适用边界。
- 最可靠的实验数据：当前云端 CoDriveThesis `main_aggregate.csv`，尤其 PPO Teacher、PAdapt、PureBC、DiffusionLatent、ConsistencyLatent、FlowMatching 的 2048-step fixed eval。
- 最需要人工确认的问题：NFE/latency 是否完成、CoDriveThesis robustness 是否完成、Action-Chunk len1/len8 representation eval 是否完成、最终是否同步云端 artifacts。
- 适合放主实验的算法：PPO Teacher、ProprioAdapt Student、Pure BC、Diffusion Latent、Consistency Latent、Flow Matching Latent。
- 适合放扩展实验或附录的算法：Action-Chunk Diffusion、BC/LatentBC、DAgger、DOTPG、Residual Diffusion、adapter range ablation。
- 只能放展望或不足的内容：Sim-to-Sim、真实机器人部署、diffusion export、完整离线 dataset 消融、KL distillation。
- 写论文时最重要的边界：PPO teacher / ProprioAdapt 原始框架不是个人原创；本人贡献集中于 student 蒸馏阶段的 diffusion-style 替换、baseline 对比、统一评测与适用边界分析。
