# 第一阶段：实验有效性重建（2026-09-18）

本阶段重新训练 0.5B 受控实验，检查修复后机制是否成立。它不替代真实 GSM8K 偏好训练、人工定位核验或多模型验证。

完成状态：27 组训练、56 组生成评估（28,000 条预测）、54 组 held-out probe 已完成并通过全量验收；最终结果及论文影响见 [结果报告](stage1_results.md)。

## 当前执行约束

- 用户随后已授权完成本阶段：续跑本地训练和评估，复用校验通过的 7 个完整适配器，重启被中断的 normalized seed 1；不调用 API。
- 本任务仅使用 Qwen：本地训练为 Qwen2.5-0.5B-Instruct；以后如需要真实数据 judge，默认路由为最后验证过的 `Qwen3.8-27B-no-thinking`，不自动回退 GPT/Claude 或其他模型。
- API 请求必须使用 `stream=True` 并完整消费响应；默认单并发，认证/封禁错误停止后续审计请求。代码修改只通过离线 mock 验证，未重新调用 API；流式本身不能解除既有账户封禁。
- 合成算术题已有 oracle 错误标签，无需再调用 API judge。此前 GPT 审计结果仅保留为历史记录，不纳入后续 Qwen-only 实验，也没有用于训练本轮 Qwen。

## 冻结协议

- 配置：`configs/stage1_v1.json`；数据 manifest：`data/processed/stage1_v1/manifest.json`。
- Qwen2.5-0.5B-Instruct；LoRA r=16, alpha=32, dropout=.05；七类投影层。
- 每组相同的 2,000 条 preference、1 epoch、lr=1e-5、beta=.1、有效 batch=8、seed=1/2/3。
- 共九个训练组：SFT、vanilla DPO、DPO+chosen-NLL(.2)、prefix-masked、first-error-only、CM-DPO(gamma=.25)、CM-DPO+process-positive(.2)、normalized CM-DPO、uniform rejected weight=.25。
- process-positive 保持与 CM-DPO 完全相同的样本，不再过滤缺少前缀的样本；辅助项是前缀序列的 summed log-ratio，DPO+NLL 是 chosen per-token NLL。这两个系数的尺度不同，均未做超参数搜索。
- 同一个 Qwen chat prompt 函数用于训练、生成和 probe；完整回答附加 EOS，rejected EOS 继承末步权重；辅助前缀不附加 EOS。步骤分隔符归前一步，vanilla 覆盖所有 response token。
- 严格禁止静默截断；prompt 不参与 response loss；累计 log probability 使用 float32。
- 冻结 reference 的每个 token log probability 可缓存；以精确输入 ID 哈希匹配，不能跨协议复用。缓存/实时两条路径有 loss 和梯度一致性回归测试。

## 数据及评估

- 训练：inventory / tickets / garden / shipping 四类各 500 题。
- ID test：相同四类，共 500 题，与训练问题完全去重。
- OOD test：classroom / recipe 两个未训练模板，各 250 题。
- 保留旧模板文本与人工注入错误方式，但各参数独立均匀采样，消除单整数取模造成的周期和重复限制。因数据协议改变，不与旧表直接作差。
- 该固定模板划分的训练/ID first-error 均在第 2 步，OOD 为第 1 或第 2 步；这是本阶段的限制，不能用它证明任意错误位置上的定位泛化，ID 上恒猜第 2 步也可达到 100%。后续真实数据及变错误位置实验必须补上。
- 2,000/2,000 训练 pair 的 chosen/rejected 共享完全相同的正确前缀。在确定性 log-probability 下，该共享前缀项在 vanilla DPO 的 score difference 中精确抵消；prefix masking 则使 chosen 端的正向前缀项保留下来。因此不能把这组实验解释成 vanilla 对共同前缀存在净直接惩罚。共享参数更新仍可能改变前缀 likelihood，训练时独立 dropout 也会扰动抵消；机制讨论必须区分这些现象。SFT 和 DPO+NLL 对照用于检查正向监督/格式学习的替代解释，但不构成全部解释的排除。
- 三个 split 相互无重复，ID/OOD 不参与调参；OOD 只表示未见文本模板，不等同于任意数学任务或运算结构泛化。
- 正式生成：greedy，max_new_tokens=512；保存逐题答案、正确性、实际输出 token 数、EOS/长度截断标志及样本 ID。256-token base 预检发现大量回答贴近上限后，将正式评估提高到 512；预检结果不与正式结果混合。
- 机制 probe：ID/OOD 各前 128 题，计算非空 span 的 per-example per-token 均值；空前缀不计作 0。不是训练集 probe。
- 已下载 GSM8K 官方 test 全部 1,319 题；本阶段算术训练在其上的评估只能称跨数据集迁移，不能称 GSM8K 偏好训练结果。
- API 定位审计：100 条 ID oracle 题，gpt-5.6-sol，直连、低速限流；失败请求单独记录。它不替代真实数据人工核验。

API 运行状态：初始并发请求出现大量 429，改为单并发和退避后，共取得 72/100 条有效定位，72 条均匹配 oracle。但 ID oracle 错误位置恒为第 2 步，因此不能将该结果解释为真实数据定位能力。随后网关明确返回 HTTP 403 / `User has been banned`，已停止所有 API 请求，等待账户恢复；没有尝试绕过服务端账户封禁。本地 GPU 实验继续。

## 旧证据的处理

| 旧产物 | 状态 | 原因与后续处理 |
|---|---|---|
| process-positive checkpoint、表格与 probe | 对照无效 | policy 辅助 forward 在 no_grad 内；修复后重跑 |
| 原 0.5B generation 表 | 历史结果，退出新协议比较 | raw 训练与 chat 评估混用；分隔符 mask 与新协议不同 |
| 原 normalized 对照 | 需重跑 | 公式与实现分母不同；chosen/rejected 相对尺度未联合调参 |
| 原 arithmetic train/eval | 仅历史模板内结果 | 模板重用且 seed 取模导致有限题库；本阶段使用去重与模板隔离 |
| 原 training-set summed likelihood | 仅训练行为诊断 | 不是 held-out/per-token 泛化证据 |
| 1.5B 旧三组表 .118/.498/.224 | 禁止继续作主结果 | 与本地 four-way 文件冲突，格式协议需要完整追溯 |
| 1.5B four-way .460/.902/.918/.912/.900 | 文件已找到，待协议复验 | `outputs/qwen1_5b_harder_eval500_fourway_accuracy.jsonl`；不直接替换为新有效结果 |
| GSM8K eval100、gold-fallback chosen | pilot | 样本小且存在风格混杂；不能充当标准 GSM8K benchmark |

所有历史文件保留。论文旧表尚未整体替换，阅读时必须结合本清单；本阶段只修正了方法的归一化分母与 reference 符号。

## 运行与审计

```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/prepare_stage1.py  # 数据不存在时运行
.venv/bin/python scripts/run_stage1.py --train-only
.venv/bin/python scripts/evaluate_stage1.py  # GPU 1 独立消费已完成适配器
.venv/bin/python scripts/summarize_stage1.py
```

训练使用 GPU 3；评估使用 GPU 1 的空闲显存，统一 batch=32（`configs/stage1_eval_v2.json`）。只终止了本任务的预检评估，未终止服务器上其他任务。API 使用独立 httpx client 的 `trust_env=False` 绕过环境代理，不改系统代理。服务端实际模型列表与 api.txt 的旧模型名不一致；密钥保持在 gitignore 中。

`outputs/stage1_v1/manifest.json` 记录配置、源码/模型/产物哈希、每个子任务的命令、PID、时间和完成状态。训练完成后才能计入结果；训练启动或 smoke 通过不代表正式实验完成。

正式评估位于 `outputs/stage1_v1/eval512/`，使用独立 `evaluation_manifest.json`，避免训练/评估并行修改同一个状态文件。每个适配器的两组生成和两组 probe 完成后，自动更新 `accuracy.csv`、`paired_statistics.json` 和 `report.md`。原始逐题预测不覆盖历史实验。

## 统计与最终验收

- 每个方法报告三个训练 seed 的均值和样本标准差；base 只评估一次。
- 配对 bootstrap 同时重采样训练 seed 和共享测试题，10,000 次；区间为边际、探索性 95% CI，三个 seed 的不确定性估计仍有限。
- 每个 seed 单独做 exact McNemar；同一个 split 内全部报告的对比统一进行 Holm 校正，不把三次 base 复用当成三份独立数据。
- 机制对比包括 CM-DPO 对 vanilla、prefix-masked、first-error-only、uniform-downweight、normalized、SFT、DPO+NLL；并报告 prefix-masked 对 vanilla、process-positive 对 CM-DPO，以及 vanilla/CM-DPO/SFT 对 base。比较表在运行期间完善、已看过部分结果，不声称事先注册。
- `scripts/validate_stage1_results.py` 要求完整 27 个训练、56 个生成任务（28,000 条预测）和 54 个 held-out probe，逐项验证配置、关键计算源码、模型/数据/产物哈希、优化步数和逐题判分。
- `scripts/plot_stage1_results.py` 只接受完整三 seed 的结果，导出 PNG/PDF；误差棒为 seed 标准差，不是置信区间。
- 27 组训练结束后，剩余评估按 checkpoint 分配给同型号的 GPU 1/3（RTX 3090）；通过 `--runs` 指定互不重叠的任务。第二个 worker 使用 `--manifest-name evaluation_gpu3_manifest.json --skip-summary`，避免状态文件和报告并发覆盖；最终验证合并两份 manifest。只重启了本任务的评估进程，完成产物不变，未完成片段的重试记录保留。
- `scripts/summarize_stage1_masks.py` 仅在 CPU 上读取冻结数据，导出各组实际 token 权重总量，帮助辨别权重位置与整体尺度；不会修改训练或评估协议。
