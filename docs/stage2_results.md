# 第二阶段结果：总权重匹配后，当前位置加权仍无优势

完成日期：2026-09-30。全部使用本地 Qwen2.5-0.5B-Instruct，没有调用外部模型 API，没有使用 GPT/Claude API。

## 结论

本轮已完成训练、全量生成、验收、预先固定的统计检验和绘图。在这一冻结设置下，CM-DPO 不仅没有显示位置收益，而且在 ID、OOD 上均低于两个新对照；两个主要 OOD 差值的调整后区间均完全低于 0。这是针对当前配置的负向证据，不是对所有位置加权方法的普遍否定。

因此，不应继续以“准确定位并衰减后缀必然优于整体降权”为论文主结论，也不建议仅扩大模型就期待解决问题。应先修改机制叙事，并区分正向监督、总权重与位置安排的作用。

## 已完成范围

- 两个新对照各训练 seed 1/2/3，共 6 个新适配器；复用第一阶段已验证的 15 个适配器。
- 同一批 2,000 条训练 pair，1 epoch、250 个优化步；lr=1e-5，beta=.1，gamma=.25，LoRA r16/alpha32/dropout=.05。没有超参数搜索。
- 新建 ID/OOD 各 500 题，排除第一阶段 train/id/ood 的全部 3,000 个 prompt；新 split 内及 split 间也无重复。ID 四种已见模板，OOD 两种未见模板，仍为合成算术任务。
- Base 加 21 个适配器，各评估两个 split：44 组生成、22,000 条预测。统一 chat 格式、greedy、512 新 token 上限、batch32。
- 配置、数据、关键源码、模型/适配器及产物哈希通过验收；逐题顺序、答案与重新判分一致。6 个新适配器参数均有限，同 seed 的两个新对照参数确有差异。
- 31 项离线测试通过；覆盖真实 mask 总量、确定性置换、自定义 mask 的缓存/实时 loss 和梯度一致性、共同前缀抵消及新题去重。`git diff --check` 通过。
- 本轮没有新增 likelihood probe，也没有启动 GSM8K、1.5B 或额外调参；第一阶段结果及历史验证记录保留。

完整设计见 [实验协议](stage2_experiments.md)。

## 新对照到底控制了什么

`mass_matched_uniform`：每条 rejected 回答全部 token（含 EOS）使用 M/N，其中 M 是该条 CM mask 的总权重，N 是有效回答 token 数。平均系数约 .32181，范围 .24121–.37689；它不是旧的常数 .25 对照。实际 float32 mask 的最大总量误差为 3.8147e-6，满足冻结容差。

`prefix_preserving_shuffle`：正确前缀仍为 0，末回答 token 与 EOS 权重固定，其余非零 token 权重做确定性置换；精确保留权重多重集合及总量。2,000 条 mask 均实际改变，三个训练 seed 复用同一份置换。

同时修复 reference cache 忽略显式 `rejected_token_weights` 的问题，使 policy 与 reference 采用同一解析规则。第一阶段仅用 step weights，不受这一修复影响。

## 准确率

单位为百分比；“±”为三个训练 seed 的样本标准差，不是置信区间。Base 只有一次固定评估。所有数字来自本轮新测试题，不与旧表混用。

| 方法 | ID：均值 ± seed SD | OOD：均值 ± seed SD |
|---|---:|---:|
| Base | 58.00 | 48.60 |
| Vanilla DPO | 53.67 ± 2.01 | 43.80 ± 0.53 |
| Prefix-masked | 57.67 ± 4.27 | 42.73 ± 2.80 |
| CM-DPO，gamma=.25 | 34.00 ± 1.06 | 47.13 ± 0.76 |
| SFT | 65.53 ± 0.81 | 67.33 ± 8.43 |
| Uniform rejected weight=.25 | 57.73 ± 0.70 | 75.07 ± 0.70 |
| Mass-matched uniform | 57.20 ± 0.40 | 72.40 ± 3.27 |
| Prefix-preserving shuffle | 55.33 ± 2.20 | 65.73 ± 3.11 |

SFT 的 ID 均值最高，常数 .25 的整体降权 OOD 均值最高；这里的排序不是所有方法间的显著性检验。尤其 SFT 的 OOD 跨 seed 波动较大。

![第二阶段准确率：柱为均值，误差线为 seed SD，点为各 seed](../outputs/stage2_v1/eval512/accuracy.png)

## 预先固定的配对统计

下表为 CM-DPO 减去对照，单位为百分点。Bootstrap 同时重采样训练 seed 和共享题目，10,000 次，随机种子 20260930。两项主要 OOD 比较各用 97.5% 边际区间，进行 Bonferroni 家族调整；ID 为次要分析，使用 95% 区间。只有三个训练 seed，区间仍是近似推断。

| 对照 | OOD 差值 [97.5% CI]，主要 | ID 差值 [95% CI]，次要 |
|---|---:|---:|
| Mass-matched uniform | -25.27 [-32.27, -18.27] | -23.20 [-28.27, -18.00] |
| Prefix-preserving shuffle | -18.60 [-25.80, -11.20] | -21.33 [-26.87, -15.80] |

两个主要区间都完全低于 0，并非只是“区间跨 0、没有显著提升”。每项主要比较的三个 seed 中，CM-DPO 的 OOD 准确率也都更低；六项逐 seed exact McNemar 检验经 Holm 校正后最大 p 约 4.01e-6。这些是逐 seed 检验，不冒充单一跨 seed 总体检验。

程序按冻结判定规则输出 `insufficient_advantage_evidence`；结合差值方向，应解释为本设置下存在 CM-DPO 劣于两个新对照的证据，而不是等价。

## 模板与长度诊断

OOD 总分中存在明显的模板取舍，不能写成两个对照在每种模板上均更好。下表仅为描述性拆分，没有新增模板级显著性声明。

| 方法 | Classroom：三 seed 均值 | Recipe：三 seed 均值 |
|---|---:|---:|
| CM-DPO | 90.93 | 3.33 |
| SFT | 85.73 | 48.93 |
| Uniform .25 | 73.07 | 77.07 |
| Mass-matched uniform | 69.20 | 75.60 |
| Prefix-preserving shuffle | 63.47 | 68.00 |

CM-DPO 在 classroom 上较高，但在 recipe 上几乎失败；两个新对照改善 recipe，同时牺牲 classroom。这不是均匀的泛化改进。

CM-DPO 的 ID/OOD 平均输出为 80.8/78.6 token，mass-matched 为 71.8/69.2，shuffle 为 76.0/73.7；三者均无长度截断。主要差距不能归因于 512-token 上限。Base 的 ID/OOD 截断率为 .2%/1.8%，vanilla 为 .07%/1.13%，比较这些方法时需保留该背景。

## 机制边界

1. **总量相同不等于梯度相同。** 两个新对照说明当前真实位置安排没有带来预期优势，但不能把全部差距归因于单一因果机制，也不能声称已匹配梯度范数或优化动力学。
2. **对照仍含 oracle 信息。** Uniform 的 M 来自 CM mask，shuffle 保留正确前缀边界；均不是完全无定位信息的对照。置换是在 token 级进行，可能改变步内一致性；不代表一切随机 step 定位策略。
3. **共同正确前缀会直接抵消。** 训练 pair 共享完全相同的正确前缀。确定性 vanilla DPO 中其直接分数项抵消；仅 mask rejected 前缀会保留 chosen 的正向前缀监督。因果 GRU 的 autograd 测试验证这一点；实际训练存在独立 LoRA dropout=.05，不能宣称每次 forward 严格抵消，也不能宣称前缀相关模型梯度整体为零。
4. **泛化范围有限。** 一种 0.5B 模型、三 seed、六个固定算术模板；训练/ID 错误在第 2 步，OOD classroom 在第 2 步、recipe 在第 1 步。模板与错误位置没有完全解耦。一个固定 shuffle 实现跨 seed 复用，未估计置换 seed 的变异。
5. **没有真实任务或真实定位器证据。** 所有标签来自生成器；本轮不能替代 GSM8K 全测试集、人工定位核验或真实模型错误轨迹。更不能外推为 Qwen 大模型上的已证实结论。

## 对论文及下一步的建议

当前可以报告的可靠贡献是：在冻结的合成设置中，严格总量匹配和保留前缀的置换对照揭示了原方法的位置优势不成立。它是一项有价值的机制诊断，但不能自动视为已经具备 NAACL 主会论文的充分贡献。

建议下一阶段先立项一个小型机制实验，而不是直接扩规模：构造语义相当但不逐 token 相同的正确前缀，并在训练内独立改变 first-error 位置；将共同前缀是否相同、正向监督强度、rejected 总量与位置安排作为分离因素。先固定训练/验证/测试划分和判定规则，调参只用验证集，使用多份 shuffle seed。若机制证据成立，再考虑真实 Qwen 错误轨迹与 GSM8K。以上仅为建议，本轮没有擅自执行。

论文方法动机也应先修正：“共享正确前缀的净直接惩罚”不能作为本数据上的已证实机制；训练与测试数字应按协议区分，不能择优混用旧结果。本轮没有自动改写论文表格。

## 结果与复现入口

- [冻结配置](../configs/stage2_v1.json)、[执行脚本](../scripts/stage2.py)、[统计验收脚本](../scripts/analyze_stage2.py)、[回归测试](../tests/test_stage2_controls.py)。
- [全量验收记录](../outputs/stage2_v1/eval512/validation.json)、[配对统计](../outputs/stage2_v1/eval512/paired_statistics.json)。
- [总体准确率及长度](../outputs/stage2_v1/eval512/accuracy.csv)、[逐 seed 结果](../outputs/stage2_v1/eval512/seed_accuracy.csv)、[逐模板结果](../outputs/stage2_v1/eval512/template_accuracy.csv)。
- [PNG 图](../outputs/stage2_v1/eval512/accuracy.png)、[PDF 图](../outputs/stage2_v1/eval512/accuracy.pdf)、[自动生成简报](../outputs/stage2_v1/eval512/report.md)。
- [运行溯源](../outputs/stage2_v1/provenance.json)、[训练 manifest](../outputs/stage2_v1/training_manifest.json)、[GPU 1 评估 manifest](../outputs/stage2_v1/eval512/evaluation_gpu1_manifest.json)、[GPU 3 评估 manifest](../outputs/stage2_v1/eval512/evaluation_gpu3_manifest.json)。逐题预测位于同一 `eval512` 目录的 `*_details.jsonl`。
- 冻结源码及统计快照位于 `outputs/stage2_v1/frozen_source.tar.gz`、`frozen_analysis.tar.gz`。历史第一阶段源码哈希应使用其历史快照重验，不覆盖原验证记录。

```bash
OMP_NUM_THREADS=4 .venv/bin/python -m pytest -q
.venv/bin/python scripts/analyze_stage2.py
git diff --check
```
