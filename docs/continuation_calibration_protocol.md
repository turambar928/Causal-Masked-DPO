# 续写校准与独立验证：冻结协议

沿用用户确认的两阶段方案。本轮仅本地 Qwen2.5-0.5B-Instruct；无模型 API、语言模型训练、论文修改、自然错误采集。原实验数据和代码不修改。配置与本文件在采样前归档。

## 数据与固定预算

同一六种算术模板，新建开发题 48 道（每模板 8 道）、验证题 96 道（每模板 16 道），各含两个位置。排除 stage1/stage2 的已有 train/ID/OOD 数据及 repair pilot/controls 题目，按完整 prompt 去重。题目与位置一起划分，不根据结果补题。新题仍是模板内泛化，不是 OOD。

seed=20261003，数据阶段与生成批次使用独立 SHA256 派生种子。先生成并冻结两阶段输入，但开发分析不读取验证输出。错误仅修改目标步骤结果 ±1；追加内容仅复述目标正确步骤，不追加后续解答。若目标中间值恰等于最终答案，不据此称为过程泄漏或提前结束。

开发：正确历史后分别不追加（base）、追加旧 Review、状态复述（state）、状态复述＋继续指令（continue）。精确文本见配置。每场景每条件 8 次，一个采样批次，共 3,072 条、96 个 batch32 批。

验证：base、正确历史＋选中提示（clean）、错误历史＋完全相同提示与正确步骤（wrong）。每场景每条件两独立采样批次各 8 次，共 9,216 条、288 个 batch32 批。按 bank、场景、条件、sample 固定排序后每 32 条分批；末尾无不足 32 的批次。总预算最多 12,288 条，不追加。

bf16、SDPA、temperature=.7、top_p=.95、top_k=0、repetition_penalty=1.1，最多 512 新 token；tokenizer EOS，assistant 续写前缀不追加 EOS。批次使用独立种子；条件之间不是逐条相同随机数配对。统计配对单位为题目。

## 开发选择与禁止偷看

只在 state/continue 中选择，要求准确率至少 40%、比 base 下降不超过 5 个百分点、候选截断率不超过 5%；base 截断率超过 5% 时整体不合格。Review 只用于诊断，不是候选。

合格候选按准确率降序、平均完整输入 token 长度升序、配置 candidate_order 排序。开发结果、验证记录和选中精确文本保存为不可替换的 selection.json，绑定相应文件哈希。开发失败则冻结失败记录并禁止验证，不修改提示重试。验证运行先核对选择及开发产物哈希。

## 验证与门槛

两个主指标按模板分层、整题聚类 bootstrap 10,000 次，同题两个位置保留，分别取 97.5% 双侧区间。固定投入门槛：

- 协议：clean 至少 40%；clean−base 区间下界严格高于 −5 个百分点；所有验证条件截断率不超过 5%。
- 历史：clean−wrong 至少 3 个百分点、主区间下界大于零、两个独立采样批次的点差值均大于零；精确 token 等长场景至少 90%，其探索性 95% 区间下界大于零。等长子集 bootstrap 保留整题聚类；空子集重采样不能制造通过。
- 协议未通过则不作历史机制支持；区间跨零不等于证明没有效应。上述门槛不是原创性、功效保证或训练收益标准。

报告每条件准确率、平均生成长度、截断、显式答案标记，以及解析答案等于目标正确中间值的代理率；最后一个指标仅在中间值不等于真实最终答案的样本上计算。沿用旧最终答案解析，只判新续写、不拼接已给前缀；代理指标不代表完整过程正确率。

若验证执行，按 seed20261003 和行标识的 SHA256 顺序，每模板×位置×clean/wrong 取两条，共 48 条输出作定性检查。抽样不看正确性，记录遗漏、提前结束、算术错误、无法判断，可以重叠。代理/单次检查不能当作系统人工核验，定性样本不估计总体发生率。

## 运行与恢复

```bash
OMP_NUM_THREADS=4 .venv/bin/python -m pytest -q
.venv/bin/python scripts/continuation_calibration.py prepare
.venv/bin/python -u scripts/continuation_calibration.py rollout --stage development --gpu 3 --max-jobs 1
.venv/bin/python -u scripts/continuation_calibration.py rollout --stage development --gpu 3
.venv/bin/python scripts/continuation_calibration.py analyze --stage development
# 仅在 selection.json 为 eligible 时执行以下两条
.venv/bin/python -u scripts/continuation_calibration.py rollout --stage validation --gpu 3
.venv/bin/python scripts/continuation_calibration.py analyze --stage validation
```

单实验锁、原子批次写入、已完成批次校验后跳过，失败批次按原种子整批恢复。GPU 显存不足则停止，不改变参数，不终止其他任务。prepare 拒绝覆盖已有目录。冻结配置、相关源码/测试、数据、模型、排除题目和历史结果哈希及源码归档。

验收要求输入/输出 token 对应、哈希一致、完整无重复采样矩阵、逐条重判、数据可重建；完整性通过与科学门槛通过分开报告。两门槛通过也不自动进入自然错误采集或训练。本轮不复验旧 R−S 的自适应增量价值。
