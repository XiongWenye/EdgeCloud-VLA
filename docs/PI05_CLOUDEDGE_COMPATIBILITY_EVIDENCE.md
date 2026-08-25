# π0.5 与 CloudEdgeVLA 方法兼容性：实验依据与判定标准

更新时间：2026-08-25

研究状态：进行中；本文档区分“已观测事实”“待验证假设”和“可下结论的边界”。

## 1. 研究问题

目标不是验证 π0.5 是否能处理 LIBERO，而是回答更窄、更可证伪的问题：

> Peng et al. (2026) 为 OpenVLA-OFT 设计的 CloudEdgeVLA 训练与边缘补偿机制，是否能在保持官方 π0.5 LIBERO 基线性能的前提下，直接迁移到 flow-matching π0.5？

论文：*Latency-Tolerant Cloud-Edge Collaborative Vision-Language-Action Models via Emergent Representational Specialization*，arXiv:2608.00569v1。

需要避免过度外推。即使下面的 V2/V3 都失败，也只能支持“该方法不能直接迁移到当前 π0.5 实现与实验协议”，不能支持“所有 flow-matching VLA 均不适合云边协同”。

## 2. 已对齐的 π0.5 基线

官方 OpenPI checkpoint、OpenPI evaluator、观测预处理、归一化、flow-matching 推理和 action chunk 执行已经逐项对齐到 LeRobot 转换。官方文档报告：

| Suite | Success |
|---|---:|
| Spatial | 98.8% |
| Object | 98.2% |
| Goal | 98.0% |
| LIBERO-10 | 92.4% |
| Average | 96.85% |

因此后续 CloudEdge 实验必须从该对齐 checkpoint 出发，不能把未对齐转换产生的低成功率误判为算法问题。

## 3. 论文实际公开的方法

论文的核心训练对象是 OpenVLA-OFT 7B，动作由连续 L1 回归头直接输出。公开细节包括：

- 云端 backbone：OpenVLA-OFT；LoRA 微调。
- 边缘视觉编码器：冻结的 SigLIP-Base。
- 边缘特征映射到动作 token hidden-state 空间。
- fresh cloud observation 与 stale cloud observation 使用配对 L1 监督。
- 当前边缘 RGB 在 stale 分支仍可用；proprioception 也是当前状态。
- 历史窗口 W=21；训练 120k steps；动作 chunk 形状为 8×7。
- 评估延迟从 1 到 dmax 均匀采样。因此 dmax=40 的平均观测年龄为 20.5 steps，而不是固定延迟 40。
- Appendix E 报告当前边缘信号的 rescue 仅约 0.03%，特征 cosine 约 0.030；论文自己的分析认为主要增益来自 backbone stability 和动作头对延迟特征的衰减。

论文没有充分公开以下关键复现参数：

- learning rate、effective batch size；
- LoRA rank 与具体 target modules；
- stale loss 的最终权重及 warmup；
- action chunk 中每次实际执行多少步；
- 延迟队列与 episode 边界的完整实现。

这意味着“与论文完全一致”在严格意义上不可验证，只能实现与公开描述一致并明确记录选择。

## 4. 为什么直接迁移到 π0.5 存在结构性风险

| 机制 | OpenVLA-OFT / 论文 | π0.5 | 迁移风险 |
|---|---|---|---|
| 动作生成 | 单次连续 L1 回归 | 多步 flow ODE 去噪 | residual 可能在每个 vector-field step 累积 |
| 动作表征 | 明确的 action-token hidden state | time-conditioned action-expert hidden state | hidden state 随 flow time/noise 改变，不一定是稳定语义空间 |
| 边缘注入位置 | 动作头前的 hidden representation | 可注入每次 velocity prediction，或 flow 完成后 | 两种注入不等价 |
| 执行语义 | 论文未完整公开 | 官方 chunk=10，通常一次执行多步 | chunk queue 会让当前边缘图像在后续环境步失效 |
| 训练目标 | 直接 L1 action loss | flow velocity MSE | paired stale L1 不能原样替换 flow objective |
| 基线敏感性 | LoRA 后仍由相同回归头解码 | flow sampler、action expert、projection 共同决定轨迹 | 很少训练也可能破坏 d0 基线 |

## 5. 已完成实验及事实

### 5.1 V1：边缘分支没有有效接入

机制诊断发现 V1 checkpoint 的边缘分支对动作没有形成有效因果路径。该结果说明 V1 不能用于判断 CloudEdge 方法有效性，也不能用于判断 π0.5 是否适配。

### 5.2 V2：在每个 flow velocity step 内注入 residual

V2 的改动：

- 冻结云端和边缘视觉编码器；
- LoRA 更新语言模型与 action expert；
- edge residual 注入每一次 flow velocity prediction；
- chunk=10，每次执行 5 步；
- fresh/stale 配对训练，W=21。

H200 smoke 已验证：

- 对齐 base 在零初始化 residual 下数值等价，最大误差为 0；
- checkpoint reload 等价，最大误差为 0；
- edge 分支梯度路径有效；
- trainable parameters 为 31.63M，占总参数约 0.845%。

候选筛选结果（LIBERO Spatial，seed 7，每点 100 episodes）：

| Model | Step | d0 | d10 |
|---|---:|---:|---:|
| aligned base | — | 98.4%（500 episodes） | 30.0%（500 episodes） |
| V2 lr=5e-6 | 2,500 | 97.0% | 45.0% |
| V2 checkpoints | 5,000 | 72.0% | — |
| V2 checkpoints | 10,000 | 51.0% | — |

事实性结论：

1. V2 在 2,500 steps 时提高了 d10，同时基本保持 d0。
2. 继续训练导致 d0 严重退化，因此不能照搬论文的 120k steps。
3. 这与“flow 内 residual 反复累积”及“π0.5 对 action expert 微调敏感”相符，但目前仍是机制假设，不是已证明的唯一因果解释。

### 5.3 V2 全量与 edge-zero 消融（运行中）

- Full: 4 suites × dmax={0,10,20,40} × seeds={7,8,9} × 500 episodes，job array 41750775。
- Edge-zero: 4 suites × 4 delays × seed 7 × 500 episodes，job array 41750776。
- Aggregation: job 41750777，依赖上述任务成功完成。

edge-zero 消融用于区分：

- “当前边缘图像确实贡献了增益”；
- “增益主要来自 stale training 对 cloud backbone/head 的正则化”。

若 edge-on 与 edge-zero 接近，则不能把延迟鲁棒性归因于在线边缘视觉纠偏。

## 6. V3：针对 π0.5 的最小结构改动

V3 不再把 edge residual 放入 flow vector field，而采用 cloud plan + post-flow edge correction：

1. 云端 π0.5 完成 10-step flow sampling，输出 base action chunk 与最后一步 action-expert features。
2. 当前边缘图像由冻结 SigLIP-Base 编码。
3. 轻量 correction head 使用 cloud features、base actions 和 current edge features，输出 additive action residual。
4. correction head 的最后一层零初始化，确保 bootstrap 时与官方 base 数值等价。
5. 每个环境步只执行第一个 action，并使用新的 current edge observation 重新推理/纠偏；n_action_steps 固定为 1。
6. LoRA rank=8，仅作用于 language model 和 action expert；vision backbones 与 action/time projections 冻结。
7. 训练使用相同 noise 和 flow time 的 fresh/stale paired branches：
   - flow velocity MSE；
   - 从 x_t 和预测 velocity 得到 action estimate；
   - post-flow correction 使用 direct L1 action loss。
8. 候选仅训练 2,500 steps，保存 625/1,250/2,500 checkpoints，避免已观测到的长训坍塌。

V3 是针对 π0.5 的方法改造，不应被表述为论文原方法的逐字复现。

## 7. 明确的可证伪假设

### H1：flow-time-conditioned hidden state 不是稳定的动作语义空间

测试：固定 observation/action，改变 noise 与 flow time，测量 action-expert hidden representation 的 cosine、范数与 correction 输出方差。

支持 H1 的现象：同一真实动作下 representation 的组内方差接近或超过 fresh/stale observation 造成的组间差异。

### H2：flow 内 residual 的反复累积导致训练不稳定

测试：比较 V2（每个 denoising step 注入）与 V3（flow 后单次注入）的 d0 retention 曲线和 residual/action norm。

支持 H2 的现象：相同或更低学习率下，V3 的 d0 随训练明显更稳定，而 V2 residual norm 随 flow steps 或训练 steps 增长。

### H3：action queue 稀释了当前边缘图像的价值

测试：同一 checkpoint 比较 execute horizon=5 与 horizon=1，并记录每个实际动作对应的 edge observation age。

支持 H3 的现象：horizon=1 在相同 cloud delay 下显著优于 horizon=5，且 edge-zero 差距扩大。

### H4：鲁棒性主要来自 stale training，而非在线 edge rescue

测试：current-edge、stale-edge、zero-edge、shuffled-edge 四种消融，cloud input 与随机种子保持一致。

支持 H4 的现象：current-edge 与 zero/shuffled-edge 成功率接近，而 cloud stale training checkpoint 明显优于未训练 base。

### H5：post-flow correction 能保留基线并提供可用的在线纠偏

V3 初筛准入标准（Spatial，seed 7，每点 100 episodes）：

- d0 >= 94%；
- d10 >= 46%，即严格超过 V2 winner 的 45%；
- bootstrap 与官方 base 最大绝对误差 <= 1e-5；
- vision backbones 无梯度；
- correction head 对两个不同 current edge observations 产生不同输出；
- 保存/重载最大绝对误差 <= 1e-5。

只有通过准入门槛才提交 V3 全量 4-suite、4-delay、3-seed 评估。

## 8. “π0.5 不直接适用论文方法”的判定边界

在以下条件全部满足后，可以形成较强的负面证据：

1. 官方 π0.5 baseline 和 evaluator 已对齐并复现；
2. V2、V3 均通过零初始化等价、梯度、reload 和 observation-age 测试；
3. 至少两个 learning rates、多个 early checkpoints 均评估；
4. current/stale/zero/shuffled edge 消融完成；
5. d0 retention 与 d10 robustness 无法同时达到预先登记门槛；
6. 失败不是 OOM、checkpoint 错误、预处理错误、归一化错误或 evaluator 不一致造成；
7. 多个 seeds 的置信区间支持结论。

满足后推荐的论文表述：

> CloudEdgeVLA 的 OpenVLA-OFT action-token specialization 机制不能直接迁移到当前 π0.5 flow-matching architecture。主要障碍是 flow-time-conditioned action representation、vector-field residual 的多步累积，以及 action-chunk execution 与 per-step current-edge grounding 的时间语义冲突。Post-flow correction 缓解其中一部分问题，但在我们的预注册门槛下仍未同时保持官方无延迟性能并达到目标延迟鲁棒性。

不应使用的表述：

> π0.5 不适合云边协同，或 flow matching 天然无法容忍延迟。

后者需要更多架构、数据集、机器人平台与替代分割位置的证据。

## 9. 可复现性记录

- Repository: XiongWenye/EdgeCloud-VLA
- Branch: codex/pi05-cloudedge-v2
- V2 full/ablation commit: 78c887e659b8b0416d54113c0197aff284fcc826
- V2 winner checkpoint:
  outputs/cloudedge_pi05_v2_lr5e-6/checkpoints/002500/pretrained_model
- V2 full jobs: 41750775, 41750776, 41750777
- V3 implementation commit: f96ddc2a4eb85b78b6b40ed447b67d27825b44d5
- V3 immutable snapshot:
  /n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA-code-snapshots/f96ddc2
- V3 jobs: bootstrap 41752244; smoke 41752247; train array 41752249;
  candidate evaluation array 41752251; aggregation 41752252
- Cluster: Harvard FAS RC / Kempner H200
- SLURM: partition=kempner_h200, account=kempner_ydu_lab

最终论文必须同时保存：代码 commit、完整 config、SLURM script、checkpoint hash、eval_info.json、每 task 成功率、seeds、episode 数、延迟采样实现和失败日志。
