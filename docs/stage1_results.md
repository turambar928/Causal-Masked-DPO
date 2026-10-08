# 第一阶段实验结果与论文影响

完成日期：2026-09-18。本轮续跑只使用本地 Qwen，无外部 API 调用；历史 GPT 定位审计不参与训练或本报告的证据。

## 结论

本阶段已完整完成，但结果不支持“CM-DPO 优于基线”。在冻结的 0.5B 算术实验中，SFT 的 ID 准确率均值最高，uniform-downweight 的 OOD 均值最高；CM-DPO 明显低于这两个对照。不能再把旧协议下的正向结果直接作为新论文主结论。

这不是对所有 CM-DPO 配置的否定：本轮只有一个模型、固定超参数和六种算术模板。它说明当前具体实现与设置没有形成足够的优势证据，下一步应优先检查机制与目标尺度，而不是直接扩大模型规模。

## 完成范围与有效性

- Qwen2.5-0.5B-Instruct，九个训练设置 × 三个 seed，共 27 个适配器。
- 每组同样的 2,000 条训练 preference，1 epoch、250 个优化步；LoRA r=16，alpha=32，dropout=.05，lr=1e-5，beta=.1。
- ID 500 题：四种已见模板、未见问题；OOD 500 题：两种未见模板。三个 split 的问题相互去重。
- base 加 27 个适配器各评估两个 split，共 56 组生成、28,000 条逐题预测。
- 每个适配器各做两组 held-out probe，共 54 组；每个 split 取前 128 题，按非空 span 计算 per-example per-token log probability。
- 统一 Qwen chat 格式、greedy、max_new_tokens=512、generation batch=32。训练 GPU 3；评估 GPU 1/3，均为 RTX 3090。
- 全量验收通过：配置、关键计算源码、模型与数据哈希、训练步数、目标函数标志、适配器配置、产物哈希、逐题问题/答案顺序及重新判分均一致。25 项离线测试通过。

修复后的 process-positive policy forward 保留梯度；本轮该对照有效，不能与此前 no-grad 实现的旧结果混用。完整协议和旧结果失效清单见 [实验协议](stage1_experiments.md)。

## 最终准确率

单位为百分比；“±”是三个训练 seed 的样本标准差，不是置信区间。base 只有一次固定评估。

| 方法 | ID：均值 ± seed SD | OOD：均值 ± seed SD |
|---|---:|---:|
| Base | 59.20 | 49.20 |
| Vanilla DPO | 53.00 ± 0.80 | 43.67 ± 2.16 |
| Prefix-masked | 60.60 ± 4.50 | 42.47 ± 3.95 |
| First-error-only | 36.47 ± 1.30 | 46.60 ± 0.20 |
| CM-DPO，gamma=.25 | 38.13 ± 0.61 | 45.93 ± 1.29 |
| SFT | 65.27 ± 1.03 | 65.27 ± 7.15 |
| DPO + chosen-NLL，.2 | 46.80 ± 1.56 | 18.33 ± 3.40 |
| CM-DPO + process-positive，.2 | 33.27 ± 1.17 | 60.07 ± 5.11 |
| Normalized CM-DPO | 43.27 ± 1.17 | 45.20 ± 1.78 |
| Uniform rejected weight=.25 | 58.07 ± 0.70 | 72.13 ± 0.99 |

“均值最高”只描述本轮观测排序，不等于对所有方法证明了统计优势；尤其没有据此声称 uniform 显著优于 SFT。

![三 seed 准确率与波动](../outputs/stage1_v1/eval512/accuracy.png)

## 配对统计支持什么

同时重采样训练 seed 和共享测试题的 paired bootstrap，10,000 次。下表为 CM-DPO 减去对照的准确率差，单位为百分点；95% CI 是边际、探索性区间，不是同时置信区间。

| 对照 | ID 差值 [95% CI] | OOD 差值 [95% CI] |
|---|---:|---:|
| Vanilla | -14.87 [-20.00, -9.80] | +2.27 [-1.00, +5.40] |
| Prefix-masked | -22.47 [-29.87, -15.13] | +3.47 [-2.60, +9.80] |
| First-error-only | +1.67 [-1.13, +4.40] | -0.67 [-2.53, +1.27] |
| SFT | -27.13 [-32.40, -21.80] | -19.33 [-26.80, -11.60] |
| Uniform .25 | -19.93 [-25.13, -14.73] | -26.20 [-32.00, -20.40] |

- CM-DPO 相对 vanilla 的 OOD 提升区间跨过 0，不能称为可靠提升；ID 则明显下降。
- 相对 prefix-masked，suffix decay 没有展现一致收益；相对 first-error-only，两种 split 的区间都跨过 0。
- CM-DPO 相对 SFT、uniform 的劣势在两个 split 的三个 seed 中，逐 seed exact McNemar 经 Holm 校正后均为 p<.05。
- process-positive 相对 CM-DPO 的 OOD 均值提高 14.13 个百分点，但 ID 均值下降 4.87 个百分点，不能称为无代价改进。
- SFT 相对 base 的 ID 提升，其联合 bootstrap CI 仍跨过 0；不要把较高均值一概写成显著提升。

全部 24 个 split/方法对比见统计 JSON。每个 split 的 12 个对比 × 3 个 seed，共 36 个 McNemar 检验统一做 Holm 校正。相同 base 预测只是与每个 seed 配对，不被当作三份独立 base 数据。比较清单在看到部分运行结果后完善，不声称事先注册；只有三个 seed，不确定性估计仍有限。

## 机制解释与主要限制

### 1. 共同前缀抵消是必须正面处理的问题

2,000/2,000 条训练 pair 的 chosen/rejected 共享完全相同的正确前缀。在确定性 log-probability 下，共同 token 前缀在 vanilla DPO 的 score difference 中抵消；mask rejected 前缀会使 chosen 端的正向前缀项保留下来。

因此，本轮不能证明“vanilla 对共同正确前缀有净直接惩罚，而 CM-DPO 消除了这种惩罚”。共享参数更新仍可能降低前缀 likelihood，独立 dropout 也会扰动逐次 forward 的抵消，但这些现象与净直接前缀惩罚不是同一件事。

### 2. Likelihood 变化不等于生成准确率提升

三 seed 的 ID per-token probe 均值：vanilla 的 prefix delta 为 -1.3621，CM-DPO 为 +1.3181；CM-DPO 的 error delta 为 -1.9410。然而 CM-DPO 的 ID 准确率仍远低于 SFT。该结果支持“目标函数改变了给定轨迹的条件 likelihood”，不支持“这种变化必然改善自由生成”。

OOD 的 prefix probe 只有 64 条 classroom 非空前缀；recipe 的错误在第 1 步，空前缀被排除。不能把该 probe 解释成覆盖全部 OOD。Suffix likelihood 同样条件于提供的 rejected 上下文，不是模型实际生成错误后缀的无条件概率。

### 3. 总体 OOD 分数掩盖模板差异

| 方法 | Classroom，三 seed 均值 | Recipe，三 seed 均值 |
|---|---:|---:|
| Base | 86.40 | 12.00 |
| Vanilla | 83.47 | 3.87 |
| CM-DPO | 89.87 | 2.00 |
| SFT | 83.87 | 46.67 |
| Uniform .25 | 67.20 | 77.07 |

CM-DPO 的 OOD 结果并非两种模板上的均匀泛化。OOD 仅是这两个固定未见模板，不等同于真实数学任务或任意推理结构泛化。

### 4. 目标尺度尚未被严格匹配

训练样本的平均 chosen token mask mass 为 71.774；rejected 平均 mask mass 分别为 vanilla 71.923、prefix-masked 55.733、first-error-only 15.133、CM-DPO 22.562、uniform .25 为 17.981。

Normalized 对照额外把 rejected sum 除以自身 mask mass，而 chosen 保持序列和；它不是对 chosen/rejected 两端同时归一化。Process-positive 的 .2 作用于前缀序列和，DPO+NLL 的 .2 作用于 chosen per-token NLL，两者数值尺度也不同。本轮没有超参数搜索，不能据此断言某类目标普遍无效。

### 5. 其他边界

- 训练和 ID first-error 均固定在第 2 步；OOD classroom 在第 2 步、recipe 在第 1 步。没有独立识别模板变化与错误位置变化的影响，也没有证明真实定位器的能力。
- 所有错误标签来自合成生成器；没有完成真实数据的人工定位核验。
- Base-only 256-token 预检暴露截断后，正式协议统一提高至 512；旧 256 输出仅保留为诊断。正式 CM-DPO/SFT 等无长度截断，vanilla OOD 截断率为 1.47%，base ID/OOD 为 0.2%/1.0%。CM-DPO 对 SFT 的主要差距不能归因于长度上限。
- 只评估一个 0.5B 模型；未重跑 1.5B，未完成 GSM8K preference 训练或全测试评估。已经下载 GSM8K test 不代表已做完该实验。
- 论文旧结果表、ACL 模板迁移及完整 related work 尚未在此阶段完成。本报告不是“论文可以直接投稿”的验收。

## 建议的下一阶段（尚未执行）

1. 用独立验证集和新测试集重建机制实验：区分完全共同前缀与语义正确但不同措辞的前缀，均衡 first-error 位置，避免当前测试集继续承担调参职责。
2. 加入 rejected 总权重匹配的对照，比较双侧归一化，并在验证集上选择 beta/gamma/辅助系数；保留 SFT 和 uniform 这两个必要对照。
3. 机制假设经检验后，再用本地 Qwen 构造风格匹配的真实 GSM8K preference，人工核验定位标签，按三个 seed 和完整 1,319 题测试协议评估。API 调用不属于本报告自动续跑的范围。

## 产物与复核

- [自动结果表与全部 probe](../outputs/stage1_v1/eval512/report.md)
- [机器可读准确率](../outputs/stage1_v1/eval512/accuracy.csv)；[逐 seed / 模板结果](../outputs/stage1_v1/eval512/template_accuracy.csv)
- [配对统计与校正 p 值](../outputs/stage1_v1/eval512/paired_statistics.json)
- [完整性验收与产物哈希](../outputs/stage1_v1/eval512/validation.json)
- [图表 PDF](../outputs/stage1_v1/eval512/accuracy.pdf)；[mask mass 诊断](../outputs/stage1_v1/eval512/mask_mass_summary.json)
- 逐题预测：`outputs/stage1_v1/eval512/*_details.jsonl`；适配器：`outputs/stage1_v1/adapters/`。
- 训练 manifest：`outputs/stage1_v1/manifest.json`；评估 manifest：`eval512/evaluation_manifest.json` 与 `eval512/evaluation_gpu3_manifest.json`，验证器合并检查两份记录。

```bash
.venv/bin/python scripts/summarize_stage1.py
.venv/bin/python scripts/validate_stage1_results.py
.venv/bin/python scripts/plot_stage1_results.py
.venv/bin/python -m pytest -q
```

源代码、冻结配置、合成数据和 tokenizer 配置快照：`outputs/stage1_v1/eval512/reproducibility_source.tar.gz`。不包含 API 密钥、历史 API 审计数据或模型/适配器权重；权重保留在现有模型与适配器目录，哈希记录在 manifest/validation 中。未提交或推送 Git。
