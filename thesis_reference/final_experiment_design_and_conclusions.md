# CoDrive Thesis 最终实验设计与结论记录

> 生成时间：2026-05-09  
> 用途：毕业论文收尾阶段的实验设计、结果和结论依据记录。  
> 结论性质：这是基于当前云端 formal suite 的项目事实整理，不是论文最终文本。

## 1. 本文件回答的问题

本项目的实验设计和结论不应只保存在 Codex 对话记忆里。当前已经固化到本地的主要记录如下：

| 类型 | 本地路径 | 作用 |
| -- | -- | -- |
| 项目事实档案 | `thesis_reference/thesis_project_status.md` | 面向论文框架设计的总项目状态说明。 |
| 最终实验设计与结论 | `thesis_reference/final_experiment_design_and_conclusions.md` | 本文件，专门记录 formal suite 的实验设计、主结果和论文结论。 |
| 云端最终 artifact 同步包 | `thesis_reference/cloud_final_results_20260506_210723/` | 从云端同步的 CSV、JSON、log、manifest、commands、status、脚本和配置快照。 |
| 云端 handoff 快照 | `thesis_reference/cloud_final_results_20260506_210723/docs/cloud_session_handoff.md` | 云端实验过程和 handoff 记录。 |
| 当前论文综述草案 | `current_project_summary_for_thesis.md` | 更偏论文叙事和项目综述。 |

因此，后续写论文时应以本地文件和同步 artifact 为准，而不是依赖对话上下文。

## 2. Artifact 同步状态

云端最终实验目录：

`/root/code/dexscrew-repro/outputs/paper_codrive_thesis_full_20260506_210723/`

本地同步目录：

`thesis_reference/cloud_final_results_20260506_210723/`

已同步内容：

| 内容 | 本地路径 |
| -- | -- |
| 聚合 CSV | `thesis_reference/cloud_final_results_20260506_210723/aggregate_csv/` |
| 原始 CSV | `thesis_reference/cloud_final_results_20260506_210723/raw_csv/` |
| 原始 JSON | `thesis_reference/cloud_final_results_20260506_210723/raw_json/` |
| 运行状态 | `thesis_reference/cloud_final_results_20260506_210723/status/` |
| 日志 | `thesis_reference/cloud_final_results_20260506_210723/logs/` |
| manifest | `thesis_reference/cloud_final_results_20260506_210723/manifest.txt` |
| commands | `thesis_reference/cloud_final_results_20260506_210723/commands.log` |
| jobs | `thesis_reference/cloud_final_results_20260506_210723/jobs.tsv` |
| 实验脚本快照 | `thesis_reference/cloud_final_results_20260506_210723/scripts/` |
| 任务/训练配置快照 | `thesis_reference/cloud_final_results_20260506_210723/configs/` |

未同步内容：

- 大模型 checkpoint 未整体同步到本地，避免占用过多空间。
- 需要 checkpoint 时，应回到云端 run 目录或单独按需同步。

## 3. Final Suite 完成状态

最终云端状态：

| 阶段 | 状态 | 时间记录 |
| -- | -- | -- |
| preflight | done | `jobs.tsv` |
| formal_training | done | `jobs.tsv` |
| representation_train | done | `jobs.tsv` |
| main_eval | done | `jobs.tsv` |
| nfe_latency | done | `jobs.tsv` |
| representation_eval | done | `jobs.tsv` |
| robustness | done | `jobs.tsv` |
| validation | done | `jobs.tsv` |

验证状态：

- `phase=done`
- robustness：`108 / 108` rows 完成
- bad log pattern：`0`
- GPU 已空闲

## 4. 实验设计

### 4.1 主任务

| 项 | 设置 |
| -- | -- |
| task | `Dexh13HoraLightbulbSim2RealTwoFingerCoDriveThesis` |
| 平台 | Isaac Gym |
| teacher checkpoint | `sim2real/codrive_thesis/best_reward_3655.17.pth` |
| fixed-step horizon | `2048` steps |
| numEnvs | `48` |
| eval seeds | `42, 43, 44` |
| formal train seeds | `42, 43, 44` |
| headless | True |

### 4.2 主实验方法

| 方法 | 角色 |
| -- | -- |
| PPO Teacher | privileged teacher / upper bound / demonstration source |
| ProprioAdapt / PAdapt | 原始开源框架 strong student baseline |
| PureBC | 基础行为克隆 baseline |
| Diffusion Latent | 普通 latent diffusion student |
| Consistency Latent | few-step consistency-style latent student |
| Flow Matching Latent | flow matching / continuous-flow latent student |
| Action-Chunk Diffusion | action-space / representation ablation |
| BC/LatentBC, DAgger, DOTPG | reference baselines |

### 4.3 NFE / Latency 设计

| 项 | 设置 |
| -- | -- |
| 方法 | Diffusion Latent, Consistency Latent, Flow Matching, Action-Chunk |
| NFE grid | `1, 2, 4, 8, 10` |
| latency batch | `1, 48, 256` |
| 指标 | fixed-step reward, policy latency mean/p95 |

### 4.4 Robustness 设计

| stressor | 含义 |
| -- | -- |
| nominal | 默认评测条件 |
| obs2x | 观测噪声加倍 |
| obs4x | 强观测噪声 |
| friction_wide | 摩擦范围扩大 |
| masscom_wide | 质量 / COM 扰动扩大 |
| initpos_noise | 初始物体位置噪声 |

Robustness 方法集合：

Teacher, PAdapt, PureBC, Diffusion Latent, Consistency Latent, Flow Matching。

## 5. 实验口径注意事项

1. PPO teacher 和 ProprioAdapt-style student 是开源 teacher-student 框架的重要组成部分，不应写成本人从零提出。
2. 本人主要工作应表述为 student 蒸馏阶段的 diffusion-style 替换、baseline 对比和统一评测。
3. 当前 formal run 中存在 checkpoint selection deviation：
   - 原计划使用 `model_best_deploy.ckpt`；
   - 由于 5h timeout 未触发 eval-select interval；
   - 已通过 hotfix 将 `model_best_deploy.ckpt` 指向 `model_best_train.ckpt`；
   - 论文中应透明说明。
4. 不应混用旧 `XHandHoraScrewDriver` 历史结果和当前 `CoDriveThesis` formal suite 结果。

## 6. 主实验结果

数据来源：

`thesis_reference/cloud_final_results_20260506_210723/aggregate_csv/main_aggregate.csv`

| 方法 | Fixed Reward | Std | Episode Return | Progress | Done Rate | 结论 |
| -- | --: | --: | --: | --: | --: | -- |
| Teacher PPO | 5.479 | 0.014 | 3764.2 | 2.355 | 0.00104 | teacher upper bound |
| Consistency Latent | 5.228 | 0.062 | 3538.9 | 2.358 | 0.00105 | 最强 student 之一 |
| Flow Matching | 5.102 | 0.155 | 3428.3 | 2.265 | 0.00107 | 最强 student 之一 |
| PAdapt | 4.880 | 0.139 | 3275.3 | 2.182 | 0.00107 | 原始强 baseline |
| PureBC | 4.817 | 0.167 | 3192.5 | 2.207 | 0.00109 | 简单但很强的 baseline |
| DAgger | 4.768 | 0.021 | 3283.3 | 2.138 | 0.00107 | reference baseline |
| Diffusion Latent | 4.569 | 0.115 | 2934.3 | 1.816 | 0.00116 | 普通 diffusion latent 不够强 |
| BC / LatentBC | 4.086 | 0.047 | 2684.8 | 2.132 | 0.00110 | 偏弱 baseline |
| DOTPG | 3.446 | 0.021 | 2330.4 | 2.241 | 0.00104 | 当前不适合作为本文主线 |

主实验结论：

- Consistency Latent 和 Flow Matching 均超过 PAdapt / PureBC。
- Teacher 仍是上界。
- 普通 Diffusion Latent 没有超过强 baseline。
- DOTPG 在当前任务中 reward 偏低，更适合保留为 reference，不建议展开为本文核心。

## 7. NFE / Latency 结果

数据来源：

`thesis_reference/cloud_final_results_20260506_210723/aggregate_csv/nfe*_aggregate.csv`

| 方法 | NFE=1 | NFE=2 | NFE=4 | NFE=8 | NFE=10 | 结论 |
| -- | --: | --: | --: | --: | --: | -- |
| Flow Matching | 5.257 | 5.278 | 5.240 | 5.237 | 5.264 | 低 NFE 已经很强，效率优势明显 |
| Consistency Latent | 5.190 | 5.198 | 5.111 | 5.129 | 5.089 | 全 NFE 稳定 |
| Diffusion Latent | 1.721 | 3.353 | 3.527 | 4.210 | 4.513 | 需要更多采样步，但仍弱于 Flow/Consistency |
| Action-Chunk | -0.647 | -0.622 | -0.775 | -1.113 | -1.232 | 不适合作为主方法 |

Latency 聚合结论：

- NFE=1 时各方法大约 `0.4-0.6 ms`。
- NFE=10 时大多约 `2.2-2.6 ms`。
- Flow / Consistency 在低 NFE 下即可达到高 reward，因此比普通 diffusion 更适合实时控制。

## 8. Representation / Action-Chunk 结果

数据来源：

`thesis_reference/cloud_final_results_20260506_210723/aggregate_csv/representation_aggregate.csv`

| 方法 | Reward | 结论 |
| -- | --: | -- |
| Action-Chunk len=8 | -1.232 | 高维 action chunk 失败明显 |
| Action-Chunk len=1 | 0.141 | 缩短 chunk 有改善，但仍远弱于 latent-space 方法 |

结论：

- action-space diffusion 不适合作为本文主方法。
- 该结果适合作为“为什么选择 latent generative student”的消融证据。

## 9. Robustness 结果

数据来源：

`thesis_reference/cloud_final_results_20260506_210723/aggregate_csv/robust_*_aggregate.csv`

| 条件 | Teacher | Consistency | Flow | PAdapt | PureBC | Diffusion Latent |
| -- | --: | --: | --: | --: | --: | --: |
| nominal | 5.479 | 5.190 | 5.257 | 4.825 | 4.610 | 4.513 |
| obs2x | 4.459 | 4.119 | 4.274 | 3.850 | 3.650 | 3.834 |
| obs4x | 2.500 | 2.092 | 1.996 | 1.597 | 1.453 | 1.967 |
| friction_wide | 5.266 | 4.839 | 4.969 | 4.641 | 4.332 | 4.185 |
| masscom_wide | 5.244 | 4.917 | 5.022 | 4.729 | 4.424 | 4.314 |
| initpos_noise | 3.626 | 5.069 | 4.889 | 4.703 | 4.507 | 4.032 |

Robustness 结论：

- Consistency 和 Flow 在多数扰动下优于 PAdapt / PureBC。
- 强观测噪声 `obs4x` 对所有方法都有明显伤害，但 diffusion latent family 仍整体优于 PAdapt / PureBC。
- `initpos_noise` 下 Teacher 反而弱于 student，这一点需要谨慎解释，可能体现 teacher 对该扰动口径更敏感，或 student 蒸馏策略更保守稳定。不能夸大为 student 全面超过 teacher。

## 10. 是否符合预期

总体判断：符合预期，并且主线结果好于预期。

符合预期的部分：

- Action-Chunk 表现差，说明高维 action-space diffusion 在该任务中难度大。
- 普通 Diffusion Latent 没有超过强 baseline，说明 diffusion 并非天然有效。
- PAdapt / PureBC 是强 baseline，不是弱对照。

好于预期的部分：

- Consistency Latent 超过 PAdapt / PureBC，接近 Teacher。
- Flow Matching 也超过 PAdapt / PureBC，并且在 NFE=1 时就很强。
- Robustness 结果整体支持 Consistency / Flow 的优势。

## 11. 建议论文核心结论

建议论文主结论写成：

> 在高维、接触丰富的灵巧手旋拧蒸馏任务中，diffusion-style student 并非天然优于强 baseline；其效果高度依赖建模空间和采样机制。实验表明，将生成建模放在 latent space，并采用 Consistency 或 Flow Matching 等 few-step 机制时，可以在 fixed-step reward、鲁棒性和推理效率上超过原始 ProprioAdapt-style student；而普通 diffusion latent 和 action-space action-chunk diffusion 则表现出明显局限。

不建议写成：

- “diffusion 全面优于所有 baseline”；
- “本文从零实现了完整 teacher-student 强化学习框架”；
- “Action diffusion 是当前任务最优方法”；
- “已经完成真实机器人部署或 Sim-to-Sim”。

## 12. 后续论文写作使用建议

主实验章节建议使用：

- Teacher PPO
- PAdapt
- PureBC
- Diffusion Latent
- Consistency Latent
- Flow Matching

消融章节建议使用：

- NFE / latency
- Action-Chunk len=8 vs len=1
- Robustness stressors

附录或 reference baseline 建议使用：

- BC / LatentBC
- DAgger
- DOTPG
- 历史 XHandHoraScrewDriver 结果

展望或不足建议写：

- Sim-to-Sim 未形成最终 artifact；
- diffusion export / deployment 不完整；
- checkpoint selection 使用 train-best hotfix；
- Action-Chunk 仍需更强的 action-space 建模或 rollout-aware objective。
