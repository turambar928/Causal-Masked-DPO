# CM-DPO 执行文档

本文档目标是把 **Causal-Masked DPO (CM-DPO)** 从论文想法拆解成可实现、可实验、可投稿的研究工程计划。

## 1. 研究目标

CM-DPO 要解决的问题是：在多步推理任务中，vanilla DPO 会把整条错误回答作为 rejected response 惩罚，但 rejected response 中常常包含正确前缀。这样会导致正确推理步骤被错误压低，造成 over-penalty 和 reasoning ability degradation。

本文的核心目标：

```text
在不构造 step-level preference pairs、不额外训练 PRM 的前提下，
通过 first-error localization 和 causal masking，
让 DPO 主要惩罚真正导致失败的步骤，同时保护错误前的正确推理前缀。
```

## 2. 方法定义

### 2.1 输入数据格式

每条训练样本包含：

```json
{
  "prompt": "...",
  "chosen": "...",
  "rejected": "...",
  "answer": "...",
  "metadata": {}
}
```

其中：

- `prompt`：题目或任务输入。
- `chosen`：正确或更优回答。
- `rejected`：错误或较差回答。
- `answer`：标准答案，用于 verifier 或 rollout 判断。
- `metadata`：可选字段，保存数据集来源、难度、题型等信息。

CM-DPO 需要额外生成：

```json
{
  "rejected_steps": ["s1", "s2", "..."],
  "first_error_step": 2,
  "step_weights": [0.0, 0.0, 1.0, 0.5, 0.25],
  "localization_confidence": 0.83
}
```

`first_error_step` 使用 0-based index 或 1-based index 都可以，但代码中必须统一。建议实现中使用 0-based index，论文中使用 1-based index。

### 2.2 Step Segmentation

数学 CoT 先用规则切分，避免一开始引入复杂 parser。

优先级：

1. 按换行切分。
2. 如果没有换行，按编号模式切分，例如 `Step 1`, `1.`, `(1)`。
3. 如果仍然太长，按句号、分号、`Therefore`, `Thus`, `So` 等连接词切分。
4. 合并过短片段，避免一个公式或孤立变量成为单独 step。

建议规则：

```text
min_step_chars = 8
max_step_chars = 500
```

如果一个 response 切分后少于 2 个 step，则该样本不用于 CM-DPO，可退化为 vanilla DPO 或直接过滤。

### 2.3 First-Error Localization

第一阶段先做可落地版本：**rollout-based localization + final answer verifier**。

对 rejected response 的每个前缀：

```text
p_k = prompt + s_1 + ... + s_k
```

从当前模型采样 `N` 个 completion，并检查 completion 的最终答案是否等于标准答案。

定义前缀成功率：

```latex
q_k = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[\operatorname{Verify}(c_i, a)=1]
```

first-error step 选择为第一个成功率明显下降的位置：

```latex
m = \min \{ k \mid q_{k-1} - q_k > \tau \}
```

如果没有明显下降，则使用保守策略：

- 如果所有 `q_k` 都较低，设 `m = 0`。
- 如果所有 `q_k` 都较高，过滤该 rejected 样本，因为它可能并非真正错误。
- 如果判断不确定，降低 `localization_confidence`，训练时减小该样本权重。

推荐默认参数：

```text
N = 4 or 8
temperature = 0.7
top_p = 0.95
tau = 0.3
min_confidence = 0.5
```

为了节省成本，第一版可以只对训练集子集做 localization，例如 10k 到 30k 条样本。

### 2.4 Step Weights

定位到第一个错误步骤 `m` 后：

```latex
w_k =
\begin{cases}
0, & k < m \\
1, & k = m \\
\gamma^{k-m}, & k > m
\end{cases}
```

实现中推荐：

```python
def build_cm_weights(num_steps: int, first_error: int, gamma: float) -> list[float]:
    weights = []
    for k in range(num_steps):
        if k < first_error:
            weights.append(0.0)
        elif k == first_error:
            weights.append(1.0)
        else:
            weights.append(gamma ** (k - first_error))
    return weights
```

默认 `gamma = 0.5`。实验中需要 sweep：

```text
gamma in {0.0, 0.25, 0.5, 0.75, 1.0}
```

其中：

- `gamma = 0.0`：只惩罚 first-error step。
- `gamma = 1.0`：错误后续步骤全额惩罚，但仍保护 prefix。
- `0 < gamma < 1`：post-error decay。

## 3. Loss 实现

### 3.1 Vanilla DPO

标准 DPO：

```latex
L_{\text{DPO}}(\theta)
=
-\log \sigma
\left(
\beta
\left[
R_\theta(y_w \mid x) - R_\theta(y_l \mid x)
\right]
\right)
```

其中：

```latex
R_\theta(y \mid x)
=
\log \pi_\theta(y \mid x)
-
\log \pi_{\text{ref}}(y \mid x)
```

### 3.2 CM-DPO

CM-DPO 保留 chosen response 的完整 reward：

```latex
R_\theta(y_w \mid x)
=
\log \pi_\theta(y_w \mid x)
-
\log \pi_{\text{ref}}(y_w \mid x)
```

对 rejected response 使用 step-level weighted reward：

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

最终：

```latex
L_{\text{CM-DPO}}(\theta)
=
-\log \sigma
\left(
\beta
\left[
R_\theta(y_w \mid x)
-
R_\theta^{\text{CM}}(y_l \mid x)
\right]
\right)
```

### 3.3 Token Mask 实现方式

实际训练时不需要逐 step 单独 forward。更高效的做法是构造 rejected response 的 token-level loss mask。

流程：

1. 对 rejected response 做 step segmentation。
2. tokenization 时记录每个 step 对应的 token span。
3. 将 step weight 展开为 token weight。
4. 计算 rejected token logprob。
5. 使用 token weight 聚合 rejected logprob ratio。

伪代码：

```python
chosen_logps = get_sequence_logps(model, prompt, chosen)
chosen_ref_logps = get_sequence_logps(ref_model, prompt, chosen)

rejected_token_logps = get_token_logps(model, prompt, rejected)
rejected_ref_token_logps = get_token_logps(ref_model, prompt, rejected)

chosen_reward = chosen_logps.sum() - chosen_ref_logps.sum()
rejected_reward = ((rejected_token_logps - rejected_ref_token_logps) * token_weights).sum()

loss = -F.logsigmoid(beta * (chosen_reward - rejected_reward))
```

### 3.4 长度归一化

CM-DPO mask 后 rejected reward 的尺度会变化。必须做长度归一化消融。

至少实现两种 aggregation：

1. **sum aggregation**

   ```python
   rejected_reward = (ratio * token_weights).sum()
   ```

2. **mean aggregation**

   ```python
   rejected_reward = (ratio * token_weights).sum() / token_weights.sum().clamp_min(1.0)
   ```

主实验建议先用 sum aggregation，因为它最接近 DPO 原始形式；附录报告 mean aggregation。

## 4. 工程实现计划

建议仓库结构：

```text
stepDPO/
  readme.md
  cmdpo_execution_plan.md
  requirements.txt
  scripts/
    prepare_data.py
    segment_steps.py
    localize_errors.py
    train_cmdpo.py
    evaluate_math.py
    analyze_likelihood.py
  cmdpo/
    data.py
    segmentation.py
    verifier.py
    localization.py
    collator.py
    loss.py
    trainer.py
    metrics.py
  configs/
    qwen2_5_7b_gsm8k_cmdpo.yaml
    qwen2_5_7b_gsm8k_dpo.yaml
    ablation_gamma.yaml
  outputs/
```

### 4.1 数据准备

第一阶段只做数学推理，避免代码生成引入执行环境复杂度。

推荐数据路径：

1. 使用 GSM8K 或 MATH 的 prompt-answer 数据。
2. 用 SFT model 采样多个 candidate responses。
3. 使用 final answer verifier 标记正确/错误。
4. 对每个 prompt 构造 `(chosen, rejected)` pair。
5. 对 rejected response 做 step segmentation 和 first-error localization。

如果想加快进度，可以先用现成 preference 数据集，但必须确认它包含：

- prompt
- chosen
- rejected
- final answer 或可验证标签

### 4.2 Verifier

数学 verifier 第一版只做最终答案抽取与归一化：

```text
"#### 42"
"\boxed{42}"
"The answer is 42"
```

归一化规则：

- 去掉空格和逗号。
- 分数统一成 `a/b`。
- 小数允许一定误差。
- LaTeX `\boxed{}` 中内容优先。

第二版可引入 `math-verify`、SymPy 或数据集官方 evaluator。

### 4.3 Localization 缓存

rollout localization 成本高，必须缓存。

每条样本保存：

```json
{
  "sample_id": "...",
  "steps": ["...", "..."],
  "prefix_success_rates": [0.75, 0.75, 0.0, 0.0],
  "first_error_step": 2,
  "confidence": 0.75,
  "weights": [0.0, 0.0, 1.0, 0.5]
}
```

缓存文件建议：

```text
data/processed/gsm8k_cmdpo_localized.jsonl
```

### 4.4 Trainer

推荐基于 Hugging Face Transformers + TRL 的 DPOTrainer 改造。

最小实现路径：

1. 先跑通 vanilla DPO。
2. 复制 DPO loss，加入 rejected token weights。
3. 自定义 data collator，返回：

```python
{
    "prompt_input_ids": ...,
    "chosen_input_ids": ...,
    "rejected_input_ids": ...,
    "rejected_token_weights": ...
}
```

4. 在 loss 中计算 weighted rejected reward。

需要注意：

- prompt 部分 token 不参与 response logprob。
- padding token 不参与 loss。
- rejected token weights 只作用于 rejected response，不作用于 prompt。
- chosen response 暂时不加 step mask。

## 5. 实验矩阵

### 5.1 主实验

模型：

```text
Qwen2.5-Math-7B-Instruct 或 Qwen2.5-7B-Instruct
```

数据：

```text
GSM8K train subset: 10k-30k preference pairs
GSM8K test: full test set
MATH subset: optional transfer evaluation
```

方法：

| Method | 说明 |
| --- | --- |
| SFT | 只做 supervised fine-tuning |
| Vanilla DPO | 标准 trajectory-level DPO |
| Prefix-Masked DPO | `w_k=0` for `k<m`, `w_k=1` for `k>=m` |
| First-Error DPO | 只惩罚 first-error step |
| CM-DPO | prefix mask + first-error full penalty + suffix decay |

主指标：

```text
GSM8K accuracy
MATH accuracy
average reasoning length
correct-prefix likelihood retention
```

### 5.2 消融实验

必须做：

| Ablation | 目的 |
| --- | --- |
| gamma sweep | 验证 post-error decay 是否必要 |
| no prefix mask | 验证 prefix protection 是否必要 |
| no suffix decay | 验证衰减是否优于全额惩罚 |
| first-error only | 验证只惩罚错误点是否过窄 |
| noisy localization | 验证定位错误时方法是否鲁棒 |
| sum vs mean aggregation | 排除 reward scale 影响 |

noisy localization 可以人工扰动 `m`：

```text
m' = m + delta, delta in {-1, 0, +1}
```

或者随机替换一定比例的 first-error labels：

```text
noise rate in {10%, 20%, 30%}
```

### 5.3 分析实验

论文里必须证明方法真的解决了 over-penalty，而不是只看最终 accuracy。

建议分析：

1. **Correct Prefix Likelihood Retention**

   统计训练前后 rejected response 中 `s_{<m}` 的 log-likelihood 变化。

   预期：

   ```text
   Vanilla DPO: 明显下降
   CM-DPO: 保持或下降更少
   ```

2. **First-Error Suppression**

   统计 `s_m` 的 log-likelihood ratio 是否被有效压低。

3. **Reasoning Length**

   检查 DPO 后模型是否倾向于输出更短 CoT。

4. **Step-level Recovery**

   给模型提供 rejected 的正确前缀 `s_{<m}`，让它继续生成，看能否恢复正确答案。

   预期 CM-DPO 的恢复率更高。

## 6. 推荐超参数

LoRA：

```text
r = 16
alpha = 32
dropout = 0.05
target_modules = q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
```

DPO：

```text
beta = 0.1
learning_rate = 5e-6 to 1e-5
batch_size = 64 effective
max_prompt_length = 512
max_response_length = 1024
epochs = 1 to 3
gamma = 0.5
aggregation = sum
```

Rollout localization：

```text
num_rollouts = 4
temperature = 0.7
top_p = 0.95
max_new_tokens = 512
tau = 0.3
```

## 7. 里程碑

### Week 1: 最小数据管线

目标：

- 跑通 GSM8K 数据读取。
- 实现 answer extraction 和 verifier。
- 实现 step segmentation。
- 构造少量 chosen/rejected pair。

交付：

```text
data/processed/gsm8k_pairs_small.jsonl
scripts/segment_steps.py
cmdpo/verifier.py
```

### Week 2: Error Localization

目标：

- 实现 rollout-based first-error localization。
- 生成 localized preference dataset。
- 完成 localization cache。

交付：

```text
data/processed/gsm8k_cmdpo_localized.jsonl
scripts/localize_errors.py
cmdpo/localization.py
```

### Week 3: CM-DPO Trainer

目标：

- 跑通 vanilla DPO baseline。
- 实现 token-level rejected weights。
- 跑通 CM-DPO 小规模训练。

交付：

```text
cmdpo/loss.py
cmdpo/trainer.py
scripts/train_cmdpo.py
```

### Week 4: 主实验

目标：

- 完成 SFT / DPO / Prefix-Masked / First-Error / CM-DPO。
- 在 GSM8K test 上评估。
- 初步跑 MATH transfer。

交付：

```text
outputs/main_results.csv
outputs/eval_logs/
```

### Week 5: 消融与分析

目标：

- gamma sweep。
- noisy localization。
- likelihood retention analysis。
- reasoning length analysis。

交付：

```text
outputs/ablation_gamma.csv
outputs/likelihood_analysis.csv
outputs/figures/
```

### Week 6: 写作

目标：

- 完成论文初稿。
- 补 related work。
- 整理图表。
- 准备 appendix。

交付：

```text
paper/draft.tex
paper/figures/
```

## 8. 最小可发表版本

如果资源有限，最小版本只需要完成：

1. GSM8K 上的 DPO vs CM-DPO。
2. 至少 3 个 ablation：

   ```text
   first-error only
   prefix mask without decay
   gamma sweep
   ```

3. 一个关键分析：

   ```text
   correct-prefix likelihood retention
   ```

4. 一个 transfer evaluation：

   ```text
   MATH subset 或 GSM8K-hard subset
   ```

这已经可以支撑 workshop 或 arXiv 技术报告。如果结果显著，再扩展到 MATH full、HumanEval 或更多模型。

## 9. 论文写作结构

建议结构：

```text
1. Introduction
   - DPO 在多步推理中的 over-penalty 问题
   - partially-correct rejected reasoning 的普遍性
   - CM-DPO 的核心直觉和贡献

2. Related Work
   - Direct Preference Optimization
   - Step-level preference optimization
   - Process reward models
   - Credit assignment in reasoning

3. Method
   - Problem formulation
   - Step segmentation
   - First-error localization
   - Causal-masked DPO objective
   - Complexity analysis

4. Experiments
   - Datasets
   - Baselines
   - Main results
   - Ablations
   - Analysis

5. Discussion
   - Localization noise
   - Limitations
   - Extension to code generation and agents

6. Conclusion
```

## 10. 成败判断标准

这篇论文是否值得继续推进，取决于三类结果。

必须看到：

```text
CM-DPO > Vanilla DPO on GSM8K accuracy
CM-DPO 保留 correct-prefix likelihood 明显优于 Vanilla DPO
gamma=0.25/0.5/0.75 优于 gamma=1.0 或 first-error only
```

最好看到：

```text
CM-DPO 在 MATH subset 上有 transfer gain
CM-DPO 的 average reasoning length 不明显缩短
CM-DPO 对 10%-20% localization noise 仍鲁棒
```

如果只看到 accuracy 小幅提升，但 prefix likelihood analysis 不成立，这篇会很难讲成因果归因论文，只能退化成 weighted DPO trick。

## 11. 下一步执行顺序

建议立即按这个顺序开工：

1. 先实现 verifier 和 step segmentation。
2. 用一个小模型或现成生成结果构造 500 条 toy preference pairs。
3. 离线跑 first-error localization，检查人工抽样准确率。
4. 实现 CM-DPO loss，先在 500 条数据上 overfit debug。
5. 扩到 10k pairs，跑 DPO vs CM-DPO。
6. 只有主结果有信号后，再做完整 ablation。

## 12. 当前代码入口

本仓库已经实现了 CM-DPO 的最小工程骨架。

安装依赖：

```bash
pip install -r requirements.txt
```

构造 GSM8K preference pairs：

```bash
python3 scripts/build_gsm8k_pairs.py \
  --model Qwen/Qwen2.5-Math-7B-Instruct \
  --limit 1000 \
  --num-candidates 4 \
  --output data/processed/gsm8k_pairs.jsonl
```

切分 rejected steps：

```bash
python3 scripts/segment_steps.py \
  --input data/processed/gsm8k_pairs.jsonl \
  --output data/processed/gsm8k_pairs_segmented.jsonl
```

生成 first-error step 和 causal weights：

```bash
python3 scripts/localize_errors.py \
  --input data/processed/gsm8k_pairs_segmented.jsonl \
  --output data/processed/gsm8k_cmdpo_localized.jsonl \
  --gamma 0.5
```

如果要使用 rollout-based localization，加入模型参数：

```bash
python3 scripts/localize_errors.py \
  --input data/processed/gsm8k_pairs_segmented.jsonl \
  --output data/processed/gsm8k_cmdpo_localized.jsonl \
  --gamma 0.5 \
  --model Qwen/Qwen2.5-Math-7B-Instruct \
  --num-rollouts 4
```

训练 CM-DPO：

```bash
python3 scripts/train_cmdpo.py \
  --model Qwen/Qwen2.5-Math-7B-Instruct \
  --data data/processed/gsm8k_cmdpo_localized.jsonl \
  --output-dir outputs/qwen2_5_7b_gsm8k_cmdpo \
  --use-lora \
  --beta 0.1
```

运行核心测试：

```bash
python3 -m unittest discover -s tests
```

当前实现包含：

- `cmdpo/segmentation.py`：CoT step segmentation。
- `cmdpo/verifier.py`：数学最终答案抽取与验证。
- `cmdpo/localization.py`：first-error localization 和 causal weights。
- `cmdpo/collator.py`：step weights 到 token weights 的展开。
- `cmdpo/loss.py`：CM-DPO loss。
- `cmdpo/trainer.py`：基于 Transformers Trainer 的 CM-DPO trainer。
- `scripts/build_gsm8k_pairs.py`：从 GSM8K 采样构造 preference pairs。
- `scripts/segment_steps.py`：离线步骤切分。
- `scripts/localize_errors.py`：离线错误定位与权重生成。
- `scripts/train_cmdpo.py`：训练入口。
- `scripts/analyze_likelihood.py`：weighted rejected likelihood 分析入口。
- `scripts/smoke_train_synthetic.py`：无需下载模型的合成 CM-DPO 机制验证。

## 13. 当前 smoke experiment 结果

由于当前环境无法连接 Hugging Face 下载 Qwen 模型，且 GPU 显存已有进程占用，先完成了一个无需外部模型的 synthetic smoke experiment。

运行命令：

```bash
python3 scripts/smoke_train_synthetic.py > outputs/synthetic_smoke_results.csv
```

结果：

```text
variant,prefix_delta,error_delta
vanilla_dpo,-6.3808,-8.7894
cm_dpo,-1.4826,-8.9184
first_error_only,-1.6152,-6.2248
```

解释：

- `prefix_delta` 表示 rejected response 中“正确但不同于 chosen 的前缀”训练前后的 log-likelihood 变化。
- `error_delta` 表示 rejected response 中错误部分的 log-likelihood 变化。
- vanilla DPO 明显压低正确前缀：`-6.3808`。
- CM-DPO 仍然强力压低错误部分：`-8.9184`，但对正确前缀的伤害小很多：`-1.4826`。
- first-error-only 对前缀保护也有效，但错误后续惩罚偏弱：`error_delta=-6.2248`。

这个结果支持 CM-DPO 的机制假设：相比 vanilla DPO，causal mask 可以更好地保护 rejected trajectory 中的正确前缀，同时保留对错误步骤的惩罚。

注意：这只是合成机制验证，不是 GSM8K accuracy 实验。真实效果仍需要在可下载/可访问模型后运行 GSM8K 小样本实验。

## 14. API 小样本数据实验

当前环境无法直接下载 Hugging Face 模型，因此使用 `api.txt` 中的 OpenAI-compatible API 做了 3 条 arithmetic preference pair 的数据构造实验。

生成命令：

```bash
python3 scripts/build_api_math_pairs.py \
  --model gpt-5.4-mini \
  --limit 3 \
  --output data/processed/api_math_pairs_3.jsonl \
  --max-tokens 256
```

切分与本地 verifier-based localization：

```bash
python3 scripts/segment_steps.py \
  --input data/processed/api_math_pairs_3.jsonl \
  --output data/processed/api_math_pairs_3_segmented.jsonl

python3 scripts/localize_errors.py \
  --input data/processed/api_math_pairs_3_segmented.jsonl \
  --output data/processed/api_math_pairs_3_localized.jsonl \
  --gamma 0.5
```

观察结果：

```text
row 1: first_error=3, weights=[0.0, 0.0, 0.0, 1.0, 0.5]
row 2: first_error=3, weights=[0.0, 0.0, 0.0, 1.0, 0.5]
row 3: first_error=3, weights=[0.0, 0.0, 0.0, 1.0, 0.5]
```

这 3 条 rejected solution 的错误都发生在第 4 个 step，例如：

```text
36 + 2 = 39
52 + 4 = 56
70 + 4 = 76
```

localization 正确把前缀步骤 mask 为 `0.0`，把 first-error step 设为 `1.0`，把最终错误答案 step 衰减为 `0.5`。

同时验证了 API rollout localization 路径：

```bash
python3 scripts/localize_errors_api.py \
  --model gpt-5.4-mini \
  --input <one_sample_jsonl> \
  --output data/processed/api_math_pairs_1_api_localized.jsonl \
  --gamma 0.5 \
  --num-rollouts 1 \
  --max-tokens 128
```

结果：

```text
first_error=3
prefix_success_rates=[1.0, 1.0, 1.0, 0.0, 1.0]
weights=[0.0, 0.0, 0.0, 1.0, 0.5]
```

说明 API rollout 和 verifier-based localization 在这条样本上给出一致 first-error step。

## 15. API 20 条 tiny overfit 结果

使用 `gpt-5.4-mini` 生成了 20 条 arithmetic preference pairs：

```text
data/processed/api_math_pairs_20.jsonl
data/processed/api_math_pairs_20_segmented.jsonl
data/processed/api_math_pairs_20_localized.jsonl
```

数据质量统计：

```text
num_rows: 20
chosen_correct: 20 / 20
rejected_wrong: 20 / 20
step_count_dist: {5: 11, 6: 9}
first_error_dist: {1: 4, 3: 14, 4: 1, 5: 1}
confidence_dist: {0.8: 2, 1.0: 18}
```

在这 20 条 API 文本上，用随机初始化 tiny GPT-2 做 overfit debug：

```bash
python3 scripts/smoke_train_jsonl_tiny.py \
  --data data/processed/api_math_pairs_20_localized.jsonl \
  --steps 120 \
  --gamma 0.5 \
  > outputs/api_math_20_tiny_overfit.csv
```

结果：

```text
variant,prefix_delta,error_delta,suffix_delta
vanilla,-27.2375,-4.9609,-14.5668
cm,-5.0811,-7.9746,-16.5690
first_error_only,-1.4722,-8.9599,-3.8313
```

解释：

- `vanilla` 对正确前缀伤害最大：`prefix_delta=-27.2375`。
- `cm` 显著保护正确前缀：`prefix_delta=-5.0811`，同时有效压低 first-error step：`error_delta=-7.9746`。
- `first_error_only` 最保护前缀，也强压 first-error step，但几乎不处理错误后缀：`suffix_delta=-3.8313`。
- `cm` 的优势是折中：保护前缀，同时通过 decay penalty 惩罚错误后续：`suffix_delta=-16.5690`。

这说明 CM-DPO 的三个机制在真实 API 生成文本上也能观测到：

```text
Prefix Protection: 成立
First-error Concentration: 成立
Post-error Decay: 成立
```
