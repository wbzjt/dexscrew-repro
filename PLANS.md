# PLANS.md

## 1. 当前阶段定义
当前阶段不是“全面铺开 diffusion 研究”，而是：

**在 canonical path 上锁定可复现实验闭环，准确刻画 current student baseline，并跑通一个可比较的 diffusion student 主实现。**

当前阶段以本科毕设可控推进为第一原则，以后续论文写作为第二原则，以长期扩展性为第三原则。

---

## 2. 当前阶段目标

### 目标 A：锁定 canonical path
锁定当前默认实验主路径为：

`XHandHoraScrewDriver -> PPO teacher -> current ProprioAdapt-style student -> diffusion student -> 统一评测`

本阶段不把 Pasini stage-2 视为必做项。

### 目标 B：锁定 current student 的准确方法学定位
在论文和实验治理中，明确当前 student 的正式定位为：

- action BC
- latent distillation
- adapter-only adaptation
- privileged teacher -> proprio-history student 的 imitation-distillation 路线

当前阶段必须避免继续把它简化写成“纯 BC baseline”。

### 目标 C：锁定统一评测口径
把 teacher、current student、pure BC、diffusion student 放到同一指标束下比较，避免后续反复返工补评测。

### 目标 D：锁定可靠 baseline 组
至少形成以下 baseline：

- PPO teacher
- current ProprioAdapt-style student
- pure BC baseline

若实现成本低且不扰乱主线，可补：
- latent-only student
- BC-only adapter student
- KL distillation
- action-chunk BC

### 目标 E：跑通一个 diffusion student
本阶段 diffusion student 的主目标是“能训练、能评测、能比较”，不是一次性把所有扩展都做完。

默认研究主目标仍是：
- action-chunk diffusion student

但当前工程落地采用保守顺序：
1. action-chunk diffusion
2. 若阻塞明显，退到 latent diffusion
3. 若仍不稳，退到 residual diffusion
4. 最坏情况下，以强 baseline + 受控扩展完成阶段收敛

### 目标 F：把 diffusion 的贡献点写清楚
当前阶段不接受以下空泛表述：

- diffusion 也能模仿 teacher
- diffusion 也能做 imitation

当前阶段要求至少明确一种新增价值：

- 更好建模 multimodal action distribution
- 更好建模 contact-rich temporal consistency
- 更强的恢复动作生成
- 更低的 failure reset 率
- 更好的鲁棒性 / 扰动恢复
- 在可接受采样成本下形成更稳定的动作序列

### 目标 G：预留一个 thesis contribution 轴
只有在 diffusion 主实现稳定后，才进入以下二选一贡献轴：

- 采样效率：去噪步数压缩、快速 sampler、低频生成 + 高频跟踪
- 鲁棒性：噪声、随机化、扰动、恢复片段、数据过滤/混合

---

## 3. 当前 milestone

### M0：主路径冻结
完成条件：
- Hora 主路径命令明确
- teacher/student 输出目录规范
- 关键配置入口明确
- 评测脚本入口明确
- 不再继续做无关目录/结构整理

### M1：baseline 定位与闭环
完成条件：
- PPO teacher 可复现
- current ProprioAdapt-style student 可复现
- pure BC baseline 可训练/评测
- 三者采用统一指标束
- 当前 student 的方法学描述在文档中已写准

### M2：student 机制拆分对照
完成条件：
- 至少完成一项 current student 结构拆分对照：
  - latent-only
  - BC-only
  - adapter training range ablation
- 能回答当前 baseline 强在哪里
- 能回答 diffusion 后续应该替代什么、补什么缺口

### M3：teacher 数据接口最小可用
完成条件：
- teacher rollout 可被稳定采集
- student 训练需要的最小字段明确
- 至少能支持一个 diffusion student 的训练输入
- 不要求此时就做完整通用 dataset 平台

### M4：diffusion student 首次跑通
完成条件：
- 一个 diffusion 方案在 canonical path 上完成训练与评测
- 可以与 current student 做同口径比较
- 知道主要瓶颈在训练、接口、推理还是评测哪一侧
- 至少能指出 diffusion 的一个明确收益点或失败点

### M5：阶段收敛
完成条件：
- 明确当前主线是否继续保持 action diffusion
- 若不适合，则正式切换到 latent / residual 保底主线
- 明确下一阶段贡献轴是“效率”还是“鲁棒性”
- 当前阶段结论已足够写入论文实验设计部分

---

## 4. 验收标准

### 4.1 工程验收
- `XHandHoraScrewDriver` 路径可稳定运行
- teacher 与 student 的训练/评测命令可复现
- 关键配置不会因轻微改动而整体失效
- 输出目录、日志、checkpoint、评测结果可追溯

### 4.2 baseline 验收
- 至少有 teacher、current student、pure BC 三组结果
- 三者采用同一评测口径
- 至少能回答：
  - 当前 student 比纯 BC 多了什么
  - 当前 student 强在哪里
  - diffusion 到底要超过谁、补什么缺口

### 4.3 diffusion 验收
以下至少满足前 4 条：

1. 跑通一个 diffusion student 方案
2. 能与 current student 做定量比较
3. 结果可复现，不依赖单次偶然可视化
4. 明确 diffusion 的新增价值或明确失败模式
5. 明确下一步该继续优化还是切换保底路线

### 4.4 论文验收
当前阶段结束时，至少能够清楚写出：

- 当前任务设定
- teacher-student 结构
- current student 的准确方法学定位
- baseline 组成
- diffusion 介入点
- 指标束
- 当前阶段结论
- 下一阶段优化方向

---

## 5. 暂不处理的内容

以下内容默认不纳入当前阶段交付：

- Pasini stage-2 正式支持
- 多任务/多物体扩展
- multimodal diffusion transformer
- offline RL critic / Diffusion-QL
- 真实机器人深度联调
- 外部 repo 的后续 real-world fusion 阶段
- 大规模通用 dataset 平台重构
- 以“美化代码结构”为目标的重构
- 没有统一评测口径支撑的新方法堆叠

---

## 6. 当前阶段优先顺序

### P0
锁定 Hora canonical path、训练入口、评测入口、日志出口

### P1
复现 teacher + current student + pure BC，形成统一 baseline 对比面

### P2
补一组 current student 机制拆分对照，明确它的强项来源：
- BC 成分
- latent distillation 成分
- adapter-only adaptation 成分

### P3
建立 teacher rollout 的最小数据接口，服务 diffusion student 研究

### P4
实现并跑通 diffusion student 首版
- 首选：action-chunk diffusion
- 阻塞时：latent diffusion
- 再阻塞时：residual diffusion

### P5
在 diffusion 首版稳定后，二选一进入：
- 采样效率优化
- 鲁棒性优化

### P6
只有当前五步稳定后，才考虑：
- sim-to-sim
- Pasini
- 更高阶 diffusion 扩展

---

## 7. 当前阶段的保底策略
如果 action diffusion 在本科阶段的接口、训练稳定性、推理成本或导出链路上持续受阻，则当前阶段不强行维持“完全替代 student”的叙事，允许退回以下保底顺序：

1. latent diffusion 替代 deterministic latent prediction
2. residual diffusion 作为 current student 上的修正模块
3. 保持 current student 为强基线，并把贡献轴转向评测、数据机制或鲁棒性优化

保底不意味着放弃 diffusion，而是保证阶段收敛与论文可完成。

---

## 8. 当前阶段退出条件
当满足以下条件时，当前阶段可视为结束并进入下一阶段：

- canonical path 稳定
- baseline 完整
- current student 的定位已写准
- 一个 diffusion student 已形成可比较结果
- 已明确主线继续方案或保底切换方案
- 已明确下一阶段只聚焦一个贡献轴