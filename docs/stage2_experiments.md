# 第二阶段：位置与总权重鉴别实验

配置：`configs/stage2_v1.json`。本轮仅使用本地 Qwen2.5-0.5B-Instruct，无需 API。用户允许将来使用 Qwen 等非 GPT/Claude 模型 API；如有需要须使用流式请求、不自动回退 GPT/Claude。本阶段不执行该类请求。

状态：2026-09-30 全部完成。6 次新训练、15 个复用适配器、44 组生成（22,000 条预测）已通过验收，31 项离线测试通过；中文结论及统计见 [第二阶段结果](stage2_results.md)。冻结主比较没有支持 CM-DPO 的位置优势，两项 OOD 区间均完全低于 0。

## 冻结设计

- 复用第一阶段的 2,000 条训练 pair、样本顺序和参考模型；不改变文本、错误标签、学习率、beta、gamma、epoch、LoRA 或训练 seed。
- 新训练 `mass_matched_uniform`、`prefix_preserving_shuffle`，各 seed 1/2/3，共 6 组。
- 复用 vanilla、prefix-masked、CM-DPO、SFT、uniform .25 的 15 个已验证适配器。复用前核对第一阶段 validation 中的权重、配置和训练产物哈希，不重新训练或覆盖旧结果。
- ID/OOD 各新建 500 题，seed=20260930，排除第一阶段 train/id/ood 的全部 3,000 个 prompt，也排除新 split 内和 split 间重复。模板及错误位置分布仍与第一阶段一致。
- 在同一批新题上重评 base 加 21 个训练适配器，共 44 组生成、22,000 条预测；不把旧测试集分数混入新表。
- 统一 chat、greedy、512 新 token、batch=32；保存逐题正确性、token 长度、EOS/长度上限状态、模板和样本 ID。

## 两个精确对照

设第一阶段 CM-DPO 的某个 rejected 回答有 N 个有效回答 token（包含 EOS），token 权重总量为 M。

1. **总量匹配整体降权**：全部 N 个 token 使用同一权重 M/N。是逐样本精确匹配，不使用全数据平均系数，也不沿用旧 uniform=.25。
2. **保留前缀的随机权重位置**：正确前缀保持 0；末回答 token 与 EOS 权重固定；对其余非零 token 的权重做确定性随机置换。置换由 mask_seed=20260930 与 sample_id 的 SHA-256 决定，三个训练 seed 使用同一份数据。权重集合、总量、零权重位置保持不变；可置换权重非退化时拒绝偶然不变的置换。

浮点验收在 collator 输出的实际 shifted mask 上进行，含 EOS，rtol=atol=1e-6。2,000 条 uniform mask 的最大 float32 总量误差为 3.8147e-6；shuffle 为 0，且 2,000 条 shuffle 都发生实际变化。

总量匹配不等于梯度范数匹配，也不等于对每个位置的作用相同。本轮只控制权重总量及对应的位置安排，不声称排除所有优化动力学因素。

两个新对照也不等于完全不使用 oracle：总量匹配的 M 仍由 CM mask 计算，shuffle 仍使用正确前缀边界。因此本轮检验的是这些条件给定后的权重位置收益，不能把对照叫作“完全无定位信息的方法”。

## 实现核验

- policy collator 与 reference cache 使用同一套权重解析规则：存在 `rejected_token_weights` 时优先使用，否则按原 step weights 展开。
- 修复前 reference cache 忽略自定义 token mask；第一阶段未使用该自定义字段，不受影响。原 step-weight 路径的缓存/实时 loss 与梯度一致性测试继续保留。
- 新测试覆盖自定义 mask 的缓存/实时 loss 与梯度、sum/normalized 两种聚合、置换可复现、边界 token、权重总量、数据排重。
- 使用有因果上下文的微型 GRU LM，eval 模式下保留 autograd：同时删除 chosen/rejected 共同前缀的直接分数项，不改变 vanilla DPO 的 margin、loss 和参数梯度；后续 token 的学习梯度仍非零。这不是关于整个模型梯度为零的断言，也不是有独立 dropout 时逐次 forward 的抵消断言。
- 运行前 29 项离线测试通过。第一阶段源码快照、验证 JSON 和结果文件不变；因为共享源码增加了兼容逻辑，若要重验第一阶段原源码哈希，应使用其历史快照，而非将新文件假装成旧源码。

## 事先固定的统计与判定

- 主要比较仅为 OOD 上 `cmdpo - mass_matched_uniform` 和 `cmdpo - prefix_preserving_shuffle`。
- 配对 bootstrap 同时重采样三个训练 seed 与共享的 500 道题；10,000 次，seed=20260930。每个主要比较报告 97.5% 区间，作为两项比较的 Bonferroni 调整；仅三个 seed 时仍是近似推断，不保证有限样本精确覆盖率。
- 对应 ID 比较为次要分析，报告 95% 区间。另保存主要比较逐 seed 的 exact McNemar p 值及六项检验的 Holm 校正值，不将其冒充总体跨 seed 的单一检验。
- 两个主要区间下界都大于 0：位置收益获初步支持；仅一个下界大于 0：仅一项比较获支持；两个都不是：没有充分的优势证据。区间跨 0 不表示等价。
- 同时报告 ID 取舍、SFT/其他基线、分模板准确率和输出长度；不依据单个 OOD 均值宣布完整方法优越性。
- 不做调参、不用本轮测试集选择 gamma/beta/辅助系数；不启动 GSM8K、1.5B 或新前缀数据实验。

## 记录与执行

`outputs/stage2_v1/provenance.json` 冻结配置、数据、关键源码、模型、reference cache 及复用 checkpoint 哈希；`frozen_source.tar.gz` 保存运行前源码，`frozen_analysis.tar.gz` 在生成评估前保存统计代码和配置。输出均与第一阶段分开。

```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/stage2.py prepare  # 仅首次；冻结后拒绝覆盖
.venv/bin/python scripts/stage2.py train --gpu 3
.venv/bin/python scripts/stage2.py evaluate --gpu 1 --runs base vanilla_seed1  # 示例子队列；完整任务必须互不重叠
.venv/bin/python scripts/analyze_stage2.py  # 完整 44 组通过验收后才导出最终表/图/统计
```

训练与评估支持按 manifest 续跑，完成任务先验证文件哈希再跳过。不同评估 worker 必须使用不同 `--manifest-name` 和不重叠的 `--runs`；不终止其他用户/任务的 GPU 进程。
