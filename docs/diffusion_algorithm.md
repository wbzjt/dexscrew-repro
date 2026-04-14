# Diffusion 技术路线与实现细节（Plan v2）

本文档面向当前仓库实现，描述正在使用的 diffusion 路线、算法结构和关键代码落点。

## 0. 一页看懂：这个项目在做什么，Diffusion 在哪里

这部分先用“非代码语言”回答核心问题，后续章节再给实现细节。

### 0.1 这个项目的工作内容是什么？

一句话：
这是一个**机械手拧螺丝（XHandHoraScrewDriver）任务**的 teacher-student 蒸馏工程，目标是在保持可训练、可复现、可评测的前提下，把 teacher 能力迁移到 student。

具体工作分三层：

1. **基线层（必须稳定）**  
   - PPO teacher 训练/加载  
   - current student（`ProprioAdapt` / `PureBC`）训练与评测  
   - 统一评测协议（`nominal/light_v2/hard`）
2. **生成增强层（Diffusion）**  
   - 在不破坏现有 student 主链路的前提下，尝试用 diffusion 提升表达能力与鲁棒性  
   - 主要实现是 `DiffusionLatentStudent`
3. **证据层（治理要求）**  
   - 每个结论都要有可追溯 artifact（run、log、ckpt、指标）  
   - 是否“接受一个新候选”由统一 gate 决定，不靠单次感觉

### 0.2 这个项目的技术路线是什么？

当前执行路线（Plan v2）是：

`evidence hardening -> latent gap closing -> latent diffusion -> (必要时 residual fallback)`

对应到工程含义：

1. **先把比较体系做硬**  
   - baseline、评测协议、日志口径统一
2. **先验证 latent 表达是否能承接 teacher 行为**  
   - 再决定要不要继续加大 diffusion 投入
3. **主推进用 latent diffusion，不是 action diffusion**  
   - action diffusion 目前是对照/附录分支
4. **如果 latent 主线不给收益，就走 residual/corrective diffusion**  
   - 作为保底主线，而不是无限扩张新分支

### 0.3 Diffusion 和这个项目的关系是什么？

关键点：**Diffusion 不是这个项目的全部，也不是独立替代路线；它是 teacher-student 主流程上的“可选增强模块”。**

在本仓库里，Diffusion 的角色是：

1. **位置上**：挂在 student 端，输入仍是 `proprio_hist`，输出仍回到 actor 解码动作  
2. **目标上**：尝试学习更丰富的 latent/action 分布结构，提升难条件下表现  
3. **治理上**：必须和当前强基线同口径比较；如果 robust 条件无增益，就不能升格为主结论

所以可以把关系理解成：

`主工程（teacher-student蒸馏）` 是主体，`diffusion` 是在主体上做的“受控增强实验轴”。

## 1. 当前技术路线（治理层）

依据 `AGENTS.md` 与 `PLANS_v2.md`，当前 diffusion 相关路线是：

1. 主线 A：`evidence hardening -> latent gap closing -> latent diffusion`
2. 保底主线 B：若 latent 路线 gate 失败，则切换 `residual/corrective diffusion`
3. `action diffusion` 当前是 exploratory/baseline 分支，不是主推进项

当前 canonical pipeline：

`XHandHoraScrewDriver -> PPO teacher -> current student -> diffusion student -> unified evaluation`

其中当前 diffusion student 的核心实现是 latent diffusion student（不是直接 action diffusion 主线）。

## 2. 实现入口与代码映射

- 训练入口脚本：
  - `scripts/screwdriver_student_diffusion_latent_robust_light.sh`
- 评测入口脚本（统一 nominal/light_v2/hard 协议）：
  - `scripts/eval_screwdriver_student_robustness.sh`
- 主算法实现：
  - `dexscrew/algo/ppo/diffusion_latent_student.py`
- 继承的 student 基类（ProprioAdapt）：
  - `dexscrew/algo/ppo/padapt.py`
- actor-critic 与 latent 编码器：
  - `dexscrew/algo/models/models.py`
  - `dexscrew/algo/models/block.py` (`TemporalConv`)

## 3. 算法核心思想

### 3.1 条件 latent diffusion（主实现）

这条路线的核心不是“直接生成动作”，而是“先生成 teacher 风格 latent，再解码为动作”。

#### 3.1.1 输入是什么（在本项目里分别对应什么）

训练时主要输入/监督量有 4 个：

1. `proprio_hist`  
   - 含义：学生可见的时序本体感觉历史（手部状态历史窗口）。  
   - 作用：作为 diffusion 的**条件信息**，告诉模型“当前动作上下文是什么”。
2. `obs`  
   - 含义：当前时刻学生策略输入观测。  
   - 作用：最后解码动作时，与 latent 拼接输入 actor。
3. `priv_info`（只用于 teacher 监督分支）  
   - 含义：teacher 可见的特权信息。  
   - 作用：通过 teacher 分支得到监督 latent `e_gt`。
4. `teacher_mu`（由 teacher 分支得到）  
   - 含义：teacher 在当前状态下的动作输出。  
   - 作用：作为 BC 监督，约束 student 动作不要偏离 teacher 行为。

#### 3.1.2 处理流程：输入经过什么变成什么输出

按一次训练迭代看，关键流程是：

1. 先构造 teacher latent 监督目标  
   - `e_gt = tanh(env_mlp(priv_info))`（通过 `_actor_critic` 路径得到）。
2. 在 latent 空间做前向加噪（Diffusion 正向过程）  
   - 从随机步 `t` 采样，构造 `x_t = sqrt(alpha_bar_t) * x0 + sqrt(1-alpha_bar_t) * noise`。  
   - 其中 `x0` 默认就是 `e_gt`（residual 模式除外）。
3. 条件去噪网络预测噪声  
   - `eps_pred = LatentDiffusionHead(proprio_hist, x_t, t)`。  
   - 这里 `proprio_hist` 是条件，`x_t` 是当前噪声 latent，`t` 是时间步嵌入。
4. 从噪声预测反推出净 latent  
   - `x0_pred = predict_x0(x_t, t, eps_pred)`，再 `pred_latent = tanh(x0_pred)`。
5. 把 `pred_latent` 解码成 student 动作  
   - `mu_student = actor_mlp([obs, pred_latent]) -> mu`。
6. 与 teacher 动作/latent 做联合监督  
   - diffusion 噪声损失 + BC 动作损失 + latent 重建损失（以及可选 anchor/tail 等）。

推理（test/eval）时流程类似，但不再用 teacher 监督更新参数，而是执行反向扩散采样得到 `latent`，再解码 `mu` 和环境交互。

#### 3.1.3 为什么这样做（设计动机）

1. 不直接扩散动作，而先扩散 latent  
   - 动作空间对接触任务很敏感，直接生成动作更容易不稳定；  
   - latent 空间更“抽象”，建模分布时更容易保持结构。
2. 保持 actor 解码链路不变  
   - `actor_mlp([obs, latent])` 这条 student 主链路继续沿用；  
   - diffusion 作为上游模块接入，工程改动小、兼容性强。
3. 同时保留 BC 监督  
   - diffusion 提供分布表达能力，BC 提供收敛锚点；  
   - 组合后比“纯生成”更稳，特别适合当前 thesis 的可复现推进。

#### 3.1.4 这体现了 diffusion 的什么思想

这套实现对应的是标准条件扩散思想在 latent 空间的落地：

1. 正向过程：逐步加噪（把目标 latent 扰动成不同噪声级别样本）。  
2. 反向过程：条件去噪（给定 `proprio_hist` 和时间步 `t` 预测噪声）。  
3. 随机时间步训练：让一个网络学会“任意噪声级别都能恢复”。  
4. 采样时逐步反推：从噪声 latent（或零初始化）迭代得到可用 latent。

#### 3.1.5 在项目里的输入输出闭环（最直观版）

`proprio_hist + obs`（学生可见信息）  
-> 条件 latent diffusion 生成 `pred_latent`  
-> actor 解码出 `mu_student`  
-> 环境执行并产出 reward/done  
-> 与 teacher 的 `e_gt / teacher_mu` 比较形成训练信号。

这样做的直接好处是：

1. student 端仍是可部署的标准动作输出接口；  
2. diffusion 的收益体现在“latent 表达能力增强”，而不是重写整条控制链；  
3. 能在统一评测协议下与 current student 做公平比较。

### 3.2 residual 模式（保底实现已接入）

可选 `diffusion_residual_base=True` 时：

1. base latent：`z_base = tanh(adapt_tconv(proprio_hist))`
2. 扩散目标切换为 residual：
   - `target_x0 = (e_gt - z_base) * residual_target_scale`
3. 推理还原：
   - `pred_latent = tanh((x / residual_target_scale) + z_base)`

该模式用于 Plan v2 的 residual/corrective fallback 主线。

## 4. 网络结构细节

`LatentDiffusionHead` 结构（`diffusion_latent_student.py`）：

1. `hist_encoder`：
   - flatten 后 MLP（ELU）编码 `proprio_hist`
2. `t_embed`：
   - 离散扩散步 `t` 的 embedding
3. `denoiser`：
   - concat(`hist_feat`, `x_t`, `t_feat`) 后 MLP 输出 `eps_pred`

扩散时间表：

- `betas = linspace(beta_start, beta_end, diffusion_steps)`
- `alphas = 1 - betas`
- `alpha_bars = cumprod(alphas)`

默认训练脚本通常给定：

- `diffusion_steps=10`
- `diffusion_steps_infer=10`

## 5. 训练目标（损失函数）

在 `DiffusionLatentStudent.train()` 中，总损失为加权和：

`L = w_diff * L_diff + w_bc * L_bc + w_recon * L_recon + w_anchor * L_anchor + w_l2 * L_l2 + w_tail * L_tail`

对应实现项：

1. `L_diff`（扩散噪声预测）  
   - `MSE(eps_pred, noise)`
2. `L_bc`（行为蒸馏）  
   - `MSE(mu_student, mu_teacher)`
3. `L_recon`（latent 重建）  
   - `MSE(pred_latent, e_gt)`
   - 支持 `latent_recon_coef` 的 schedule（start/end/steps）
4. `L_anchor`（base action 锚定）  
   - 可选，约束 student action 不偏离 base student action
5. `L_l2`（action 幅值正则）  
   - `mean(mu_student^2)`
6. `L_tail`（teacher delta tail）  
   - 对超过阈值的 action 误差尾部惩罚
   - 支持 selective 与 mid_only 窗口掩码

## 6. 参数冻结与可训练部分

初始化时会先冻结 `self.model` 全部参数，再训练：

1. `diffusion_model`（必训）
2. 额外可训练子模块（可选）：
   - 由 `diffusion_student_trainable_param_patterns` 匹配
   - 常见为 `adapt_tconv` 相关参数

优化器：

- `Adam(diffusion_model + extra_trainable_params, lr=diffusion_lr)`

## 7. 推理与评测机制

### 7.1 latent 采样（`sample_latent`）

反向扩散从 `t=diffusion_steps_infer-1` 到 `0`：

1. 默认 deterministic（`stochastic_infer=False`）时以 `x=0` 开始
2. 每步用 `eps_pred` 计算均值更新
3. 若 stochastic 模式且 `t>0`，再加噪声项
4. residual 模式下再做 base latent 回加
5. 最终 `tanh` 限幅输出 latent

### 7.2 测试输出（`test()`）

固定步评测（配合 `+test_num_steps`）会输出：

1. `EvalSummary`：`avg_reward`、`avg_done_rate`
2. `EvalReconSummary`：`latent_mse`、`latent_l1`、`action_mse_to_teacher`
3. residual 模式额外 `EvalResidualSummary`（residual 幅值/比例与 action correction 统计）

并支持：

- `diffusion_eval_decode_only=True`：直接用 base latent 解码，用于 gap-closing 检查

## 8. 训练与产物目录约定

diffusion latent 训练产物默认落在：

- checkpoint：`<run>/stage2_diffusion_nn/`
- tensorboard：`<run>/stage2_diffusion_tb/`

`model_best.ckpt` 由在线 `mean_eps_reward` 最优触发保存。

## 9. 当前实现的工程要点

1. 与原 `ProprioAdapt` 路径兼容，不破坏既有 actor 解码链路
2. diffusion 与 base student 权重/日志目录隔离（`stage2_diffusion_*`）
3. 评测协议统一到 fixed-step 输出，便于与 baseline 做可追溯比较
4. 已内置 residual 模式接口，支持 Plan v2 的 fallback 方向

## 10. 关键配置键（常用）

核心训练键（`train.ppo.*`）：

- `diffusion_steps`, `diffusion_steps_infer`
- `diffusion_lr`, `diffusion_loss_coef`
- `bc_loss_coef`
- `diffusion_latent_recon_coef`（及 start/end/schedule_steps）
- `diffusion_base_action_anchor_coef`
- `diffusion_action_l2_coef`
- `diffusion_teacher_delta_tail_coef`
- `diffusion_teacher_delta_tail_threshold`
- `diffusion_teacher_delta_tail_selective`
- `diffusion_teacher_delta_tail_mid_only`
- `diffusion_residual_base`
- `diffusion_residual_target_scale`
- `diffusion_stochastic_infer`
- `diffusion_eval_decode_only`
- `diffusion_eval_report_recon`
