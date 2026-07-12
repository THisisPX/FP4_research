# Kimi 调研 Prompt：RL Rollout 加速方向研究选题

## 背景

我是 CS PhD 学生，研究方向是 **RL rollout 加速 for LLM post-training**。当前 RL 训练（GRPO/PPO 等）中，rollout（生成）阶段消耗 65-90% 的总训练时间，是核心瓶颈。我关注的方法方向包括：推测解码（Speculative Decoding）、多 Token 预测（MTP）、低精度量化、KV Cache 优化、PD 分离、异步 RL 系统等。

## 已有调研覆盖

我已经用 Search + 文献阅读覆盖了以下方向，不需要重复搜索：

### 已确认不适合的方向（原因附）

1. **把 Sol-RL 的 "FP4 Explore / BF16 Train" 从扩散图像迁移到 LLM**：我做了三个 benchmark 的实验（GSM8K/MATH-500/AIME），结论是在 LLM 文本上 FP4 确实保持了一定程度的奖励排序（Top-8 overlap ~91% on MATH-500），但这个 idea 本质就是 A→B 迁移，创新性不够。

2. **量化 Rollout 的 Token 膨胀对 RL 的双刃剑效应**：2026 年 6 月论文发现量化模型生成更长 CoT。但这属于"发现现象 + 分析"的套路。

3. **多精度集成 Rollout for GRPO**：不同精度生成 completion 混合使用，偏工程 trick。

4. **自回归文本中精度-可靠性相变**：有意思但感觉可以做得更深。

5. **量化噪声 × GRPO 优势估计的交互偏差**：理论分析方向，但担心做不深。

### 核心痛点

目前看到的大多数量化 + RL 的工作（QuRL / QeRL / AIS / Jet-RL / Unified FP8 / INT4 QAT / ReQAT）都在问同一个问题：
> "如何让低精度推理和全精度训练之间的 gap 尽可能小？"

**但没有人问过：RL 训练本身到底需要多高的精度？**

所有现有工作都有一个隐含前提——"训练需要高精度，推理可以用低精度"。这个前提可能对 pretraining 成立（需要精确的 next-token prediction），但对 RL 训练不一定成立——因为 RL 的梯度来自采样的 rollout + Monte Carlo 优势估计，本身就有很大的统计噪声。如果采样噪声的量级本身就大于量化噪声，那么"训练需要高精度"这个前提就不成立。

## 我的核心问题

> RL 训练的精度需求到底是多少？为什么？

这个问题有两个可能的答案，**无论哪一种都是新知识**：

- **如果 RL 训练可以容忍 FP4**：整个 LLM post-training 的经济学需要重写。训练不该用 BF16 守着，推理用 FP4 —— 而是可以端到端 FP4，省 4× 显存和 2× 速度。
- **如果 RL 训练不能容忍 FP4**：到底是什么在训练中破碎了？梯度估计偏差？策略崩溃？数值下溢？答案将揭示 RL 训练优化的深层数学特性。

**这和已有工作的本质区别**：Jet-RL / Unified FP8 问的是"FP8 训练能不能做到？"（工程适配）。我问的是"RL 训练需要多少精度？为什么？"（科学问题）。

## 我有独特的基础设施

| 资源 | 说明 |
|------|------|
| **4× NVIDIA B300** | 288GB HBM3e/卡，共 1.15TB 显存；原生 NVFP4 算力 15 PFLOPS；同时支持 FP4 / FP8 / BF16 |
| **veRL 框架** | D:\learning\verl_cambricon（NPU 适配版本）|
| **vLLM** | D:\learning\vllm（新版，已有 NVFP4 量化支持 + 多精度推理 Python API）|

B300 的独占能力是：**原生 FP4 可以在有意义的模型规模（30B+）上跑完整的 RL 训练**——这在此前任何硬件上都不可行。H100/H200 没有原生 FP4，B200 有 FP4 但只有 192GB/卡。

## 我需要你做什么

请帮我对以下核心问题进行深度文献调研，目标是**确认是不是真的没有人做通过这个问题**，以及如果有相关但不同的工作，具体区别是什么：

### 核心问题

> **用 FP4 精度进行 RL 训练（不仅仅是推理/rollout）的可行性与限制**

具体子问题：
1. 有没有任何工作在 **LLM RL 训练的 optimizer/gradient 步骤中使用 FP4**？（注意：QeRL 用 NVFP4 权重 + LoRA + BF16 优化器，训练仍然是 BF16。Jet-RL 是 FP8 不是 FP4。QuRL 和 AIS 的 training 都是 BF16。）
2. 有没有理论工作分析过 **RL 策略梯度估计对模型精度下降的容忍度**？例如：PPO clip 比例、GRPO 组归一化、importance sampling ratio 在低精度下的行为。
3. 在传统 RL（非 LLM）领域，有没有人研究过"低精度训练"或"量化对策略梯度的影响"？
4. 有没有工作把 **量化噪声量级** 和 **RL 的 Monte Carlo 采样噪声量级** 做过直接比较？这会直接影响"RL 训练是否天然更能容忍低精度"这个论点。
5. FP4 原生素数计算（B300）是否能执行 RL 训练中的**所有操作**？特别关注：LayerNorm 的小批量统计、Softmax 的数值稳定性、KL 散度计算（k1/k2/k3）、GRPO 的 group 内归一化等。

### 搜索策略建议

请特别注意：
- 搜索 pre-2026 的传统 RL 文献（可能有低精度 RL 训练的早期尝试）
- 搜索 NVIDIA 的技术博客和文档（NVFP4 训练栈的完整操作覆盖范围）
- 搜索未发表的 arXiv 论文和 workshop 论文（2026 年 6-7 月可能有新工作）
- 交叉搜索关键词："low precision training" + "policy gradient" / "reinforcement learning" / "PPO" / "GRPO"；"FP4 training" + "RL" / "post-training"；"quantization error" + "Monte Carlo gradient estimation"

## 约束与偏好

- 拒绝纯"A+B"组合式工作（把一个方法从领域 X 搬到领域 Y，不加新东西）
- 希望有理论深度、能回答"为什么"的问题
- 希望依赖 B300 的独占能力（原生 FP4 + 大显存），这样有硬件差异化的护城河
- 倾向于 simple-but-deep 的实验设计

## 附：已有实验先导结果

在 MATH-500 上用 Qwen3-8B 做了 BF16/FP8/FP4 三精度对比（各 100 prompts × 16 completions）：

- FP4 vs BF16 Top-8 overlap: 91%（FP4 能可靠筛出 top 候选）
- FP4 vs BF16 Best match rate: 86%（两个精度选出的最优一致）
- FP4 mean_reward: 0.613 vs BF16 0.633（基本持平）
- Throughput: BF16 0.93s/completion, FP4 0.73s/completion（1.27× 加速）

这些结果表明 FP4 rollout 是可行的代理。但它们没有回答 FP4 **训练** 是否可行的问题。
