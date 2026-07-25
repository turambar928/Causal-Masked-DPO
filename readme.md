# Causal-Masked DPO: 面向多步推理的因果掩码偏好优化

执行计划见：[cmdpo_execution_plan.md](cmdpo_execution_plan.md)。

## 1. 核心想法

传统 DPO 在训练多步推理模型时，通常把一整段回答视为一个完整的 preference unit。只要最终答案错误，整条 reasoning trajectory 就会被当作 rejected response 进行惩罚。

这个处理在数学推理、代码生成、符号推导等任务中会带来明显问题：一个错误答案并不意味着所有中间步骤都错了。很多失败样本只是某一个关键步骤发生错误，后续推导则是在错误前提上继续展开。

例如一个 5 步数学解答：

```text
s1: 正确理解题意
s2: 正确列式
s3: 计算出错
s4: 基于错误结果继续推导
s5: 得到错误答案
```

Vanilla DPO 会把 `s1` 到 `s5` 全部作为 rejected response 惩罚。这会导致两个问题：

- **Over-penalty**：正确的前序步骤也被降低概率。
- **Reasoning degradation**：模型可能学到“少推理更安全”，从而损害长链条推理能力。

本文提出 **Causal-Masked DPO (CM-DPO)**：不重新构造 step-level preference pairs，也不额外训练复杂 PRM，而是在原始 trajectory-level DPO 上引入因果掩码权重，让 DPO 的惩罚集中在真正导致错误的步骤附近。

## 2. 问题定义

给定 prompt `x`，DPO 数据包含一个 preferred response `y_w` 和一个 rejected response `y_l`：

```text
(x, y_w, y_l)
```

对于 rejected response，我们将其切分为多步推理步骤：

```text
y_l = {s_1^l, s_2^l, ..., s_K^l}
```

假设可以通过轻量验证器、rollout 或执行反馈定位第一个错误步骤：

```text
s_m^l
```

其中：

- `s_{<m}^l` 是错误发生前的正确前缀。
- `s_m^l` 是第一个导致最终失败的关键错误步骤。
- `s_{>m}^l` 是在错误前提上继续展开的后续步骤。

CM-DPO 的核心假设是：**错误责任不是在 rejected response 中均匀分布的，而是具有因果结构。**

## 3. 方法

### 3.1 Step Segmentation

首先将 response 切分为语义步骤。对于数学 CoT，可以使用规则化分隔：

- 换行
- 编号步骤
- 句号或分号
- `Therefore`, `So`, `Thus` 等推理连接词

对于代码生成，可以使用更自然的结构单位：

- 函数块
- 语句块
- 单行语句
- 测试失败对应的 traceback 位置

切分后得到：

```text
y_l = {s_1^l, ..., s_K^l}
```

### 3.2 First-Error Localization

本文不依赖额外训练的 Process Reward Model，而使用轻量级定位方法估计第一个错误步骤 `m`。

可选定位策略包括：

1. **Monte Carlo Rollout**

   对每个前缀 `s_{\le k}` 继续采样多个 completion，观察能否恢复到正确答案。如果从某个步骤开始正确率显著下降，则该步骤可视为 first-error candidate。

2. **Verifier-based Localization**

   在数学任务中使用答案验证器、符号计算器或规则检查器判断中间状态是否仍可导向正确答案。

3. **Execution-based Localization**

   在代码任务中使用单元测试、编译器、解释器报错、traceback 或静态检查定位 first failing line/block。

4. **Hybrid Localization**

   将 rollout success rate、verifier score 和模型自身 confidence 结合，得到更鲁棒的错误步骤估计。

### 3.3 Causal Masking Weights

定位到第一个错误步骤 `m` 后，对 rejected trajectory 中不同步骤赋予不同权重：

```latex
w_k =
\begin{cases}
0, & k < m \\
1, & k = m \\
\gamma^{k-m}, & k > m
\end{cases}
```

其中 `\gamma \in [0, 1]` 是后续错误衰减系数。

这个设计表达三层含义：

- **Prefix Protection**：错误前的正确步骤不应作为 negative evidence。
- **First-error Concentration**：第一个错误步骤承担主要惩罚。
- **Post-error Decay**：错误后的步骤虽然也不理想，但它们部分受错误前提驱动，责任应递减。

## 4. Causal-Masked DPO Loss

标准 DPO 的隐式 reward 可写为：

```latex
R_\theta(y \mid x) =
\log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}
```

Vanilla DPO loss 为：

```latex
L_{\text{DPO}}(\theta)
=
- \mathbb{E}
\left[
\log \sigma
\left(
\beta
\left(
R_\theta(y_w \mid x)
-
R_\theta(y_l \mid x)
\right)
\right)
\right]
```

CM-DPO 保留 preferred response 的完整 reward，但对 rejected response 引入 step-level causal weights：

```latex
R_\theta^{\text{CM}}(y_l \mid x)
=
\sum_{k=1}^{K}
w_k
\left[
\log \pi_\theta(s_k^l \mid x, s_{<k}^l)
-
\log \pi_{\text{ref}}(s_k^l \mid x, s_{<k}^l)
\right]
```

最终损失为：

```latex
L_{\text{CM-DPO}}(\theta)
=
- \mathbb{E}
\left[
\log \sigma
\left(
\beta
\left(
R_\theta(y_w \mid x)
-
R_\theta^{\text{CM}}(y_l \mid x)
\right)
\right)
\right]
```

与 vanilla DPO 相比，CM-DPO 并没有改变 preference pair 的数据形态，仍然使用 `(x, y_w, y_l)`。变化只发生在 rejected response 的 likelihood aggregation 上。

## 5. 可选增强：正向前缀保护

除了在 rejected loss 中 mask 掉正确前缀，还可以显式加入一个轻量 prefix preservation regularizer：

```latex
L_{\text{prefix}}(\theta)
=
-
\lambda
\sum_{k < m}
\log \pi_\theta(s_k^l \mid x, s_{<k}^l)
```

总损失为：

```latex
L(\theta)
=
L_{\text{CM-DPO}}(\theta)
+
L_{\text{prefix}}(\theta)
```

这个项的作用是避免模型在偏好优化过程中遗忘 rejected response 中实际正确的推理前缀。实际实验中，`λ` 应设置得较小，避免模型过度模仿 rejected trajectory。

## 6. 与相关工作的差异

### 6.1 与 Step-DPO 的区别

Step-DPO 通常需要从完整推理轨迹中抽取或构造 step-level preference pairs，例如围绕 first-error step 构造局部的 chosen/rejected step pair。

CM-DPO 的不同点是：

- 不需要显式重构 step-level preference dataset。
- 不需要为每个错误步骤采样替代正确步骤。
- 直接在 trajectory-level DPO loss 中进行因果掩码。
- 保留完整回答的上下文结构，避免局部 step pair 脱离整体推理语境。

### 6.2 与 Process-DPO / PRM-guided Alignment 的区别

Process-DPO 或 PRM-guided 方法通常依赖过程奖励模型，对每个推理步骤进行显式打分。

CM-DPO 的目标是更轻量：

- 不要求人工 step-level reward 标注。
- 不要求额外训练 PRM。
- 可以使用 rollout、verifier、执行器等弱监督信号定位错误。
- 更适合低成本复现和快速扩展到新任务。

### 6.3 与 Token-level DPO 的区别

Token-level DPO 关注 token 粒度的 KL 或 preference weighting。它的粒度更细，但通常不显式建模推理步骤之间的因果责任。

CM-DPO 的重点不是 token importance，而是 semantic step-level causal credit assignment：

- 错误前缀应保护。
- 首个错误步骤应重点惩罚。
- 错误后续步骤应衰减惩罚。

## 7. 论文贡献点

可以将论文贡献组织为三点：

1. **Causal Credit Assignment for DPO**

   提出一种面向多步推理的因果掩码 DPO 目标，缓解 vanilla DPO 对 partially-correct rejected reasoning 的过度惩罚。

2. **PRM-free Error Localization**

   设计不依赖额外 PRM 训练的错误步骤定位机制，可以基于 rollout、verifier 或 execution feedback 自动估计 first-error step。

3. **Trajectory-level Preference Optimization with Step-level Masking**

   在不改变原始 preference pair 形态的前提下，将 step-level causal structure 注入 DPO loss，降低数据构造成本。

## 8. 实验设计

### 8.1 任务与数据集

数学推理：

- GSM8K
- MATH
- PRM800K 中带 step annotation 的子集
- Math-Shepherd 或其他过程监督数据

代码生成：

- HumanEval
- MBPP
- APPS 子集
- 带单元测试或 execution trace 的代码修复数据

### 8.2 模型

建议从中等规模开源模型开始：

- Qwen2.5-7B / Qwen2.5-Math-7B
- Llama-3-8B
- DeepSeekMath-7B

训练方式：

- LoRA / QLoRA 作为主实验
- Full fine-tuning 作为可选增强

### 8.3 Baselines

必须包含：

- SFT
- Vanilla DPO
- Step-DPO
- Token-weighted DPO / TDPO 类方法
- PRM-guided DPO，如果资源允许

### 8.4 Ablation Studies

关键消融：

- `CM-DPO w/o prefix mask`：不保护正确前缀。
- `CM-DPO first-error only`：只惩罚第一个错误步骤。
- `CM-DPO no decay`：错误后续步骤全额惩罚。
- `CM-DPO hard drop suffix`：错误后续步骤全部丢弃。
- 不同 `\gamma`：例如 `0, 0.25, 0.5, 0.75, 1.0`。
- 不同 first-error localization 准确率下的性能变化。
- 是否加入 `L_prefix`。

### 8.5 Evaluation Metrics

最终任务表现：

- GSM8K accuracy
- MATH accuracy
- HumanEval pass@1
- MBPP pass@1

推理质量分析：

- 正确前缀 token likelihood 是否下降。
- 错误步骤 likelihood 是否被有效压低。
- 平均 reasoning length 是否异常缩短。
- self-consistency accuracy 是否提升。
- first-error step 后的恢复能力是否增强。

## 9. 预期结论

如果假设成立，CM-DPO 应该表现出以下现象：

- 相比 vanilla DPO，最终准确率更高。
- 正确推理前缀的 likelihood 保持更好。
- 模型不会明显缩短 CoT 长度来规避错误。
- 相比 Step-DPO，数据构造更简单，训练成本更低。
- 在 first-error localization 不完美时仍保持一定鲁棒性。

## 10. 风险与应对

### 风险 1：first-error localization 不够准

应对：

- 使用 rollout success rate 而不是单次判断。
- 对定位不确定的样本降低权重。
- 加入 soft mask，而不是 hard mask。

### 风险 2：mask 后 rejected reward 尺度变化

应对：

- 使用 step/token 数量归一化。
- 对 chosen 和 rejected 的 reward 使用长度校正。
- 在 ablation 中比较 sum aggregation 与 mean aggregation。

### 风险 3：正确前缀并非真的正确

应对：

- 只在 verifier confidence 高时启用 prefix protection。
- 对数学题使用中间变量检查。
- 对代码题使用静态分析和单元测试定位。

### 风险 4：贡献被认为只是 weighted DPO

应对：

- 强调权重不是普通 token importance，而是由 first-error causal structure 决定。
- 做针对性的 likelihood analysis，证明方法确实保护了正确前缀。
- 与 Step-DPO、TDPO、PRM-guided 方法进行清晰对比。

## 11. 推荐论文标题

可选标题：

- Causal-Masked DPO for Multi-step Reasoning Alignment
- Protecting Correct Reasoning Prefixes in Direct Preference Optimization
- Causal Credit Assignment in Trajectory-level Preference Optimization
- Less Punishment, Better Reasoning: Causal Masking for DPO

## 12. 一句话总结

CM-DPO 的核心不是简单地把 DPO 做到 step level，而是解决 trajectory-level preference optimization 中的因果归因问题：当一个 rejected reasoning 只有局部步骤出错时，模型应该重点惩罚真正导致失败的步骤，而不是压低整条推理链。
