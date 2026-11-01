# Mamba4KT 在 Algebra2005 统一 question-window 协议下的评测

日期：2026-11-01

## 结论

本次运行可以报告为：

> 使用第三方 UniKT 仓库中的 Mamba4KT 实现，在本地 Algebra2005 fold 0 / seed 42 和 pyKT question-window 评测协议下得到的一次单折单种子评测。

本次运行不能报告为：

> 作者官方 Mamba4KT 源码的完整端到端复现。

原因有三点：

1. `https://github.com/szhhwh/UniKT` 是第三方实现，不是 Mamba4KT 论文作者发布的官方代码。
2. 本地代码在 commit `0493bfbba31373a3058f287cd6625ed10d48895e` 之上增加了 `prediction_head` 配置，并允许通过环境变量限制 dataloader worker 数。
3. 本次只训练和评估了 fold 0、seed 42。验证集完成 12 组超参数比较，测试集只评估验证集冠军，没有做五折或多随机种子复现。

因此，论文中建议使用以下英文表述：

> We evaluate the third-party UniKT implementation of Mamba4KT under our unified Algebra2005 question-window protocol. This is not an author-released end-to-end reproduction.

## 冠军结果

模型配置：`d_model=256`、`n_blocks=5`、`learning_rate=0.002`、`batch_size=64`、`separate_qa=true`、`prediction_head=unikt`。

| Split / metric | Value |
| --- | ---: |
| Validation AUC | 0.9156417516899358 |
| Validation ACC | 0.8619446976602856 |
| Best epoch, zero-based | 2 |
| Epochs completed | 13 |
| Early-stopping patience | 10 |
| Test `windowauclate_mean` | 0.8140126466044022 |
| Test `windowacclate_mean` | 0.8063670692440175 |
| Test `windowaucconcepts` | 0.8148889786066025 |
| Test `windowaccconcepts` | 0.7776126465847418 |
| Selected targets | 135,246 |
| Question-window groups | 93,481 |

主指标是 pyKT 的统一 question-window `windowauclate_mean`。测试集没有参与超参数选择。冠军 checkpoint 的 SHA256 为 `27d2034162db5e3106ef8d18c7fe1c0e579689e75e178d2448004860ceecd5bb`。

独立地从 `question_window_arrays.npz` 重新聚合，AUC 和 ACC 与 pyKT 报告值完全一致。按 question group 做 bootstrap 的补充区间约为 `[0.81091061, 0.81715710]`。该区间只描述当前测试集分组重采样不确定性，不包含训练随机性或跨折方差。

## 与 0.7978 的关系

不能把 `0.8140 > 0.7978` 写成 Mamba4KT 优于 ASIKT 的证据。

- 当前 Mamba4KT 结果位于 Algebra2005。
- 本地已有 ASIKT 官方源码复现结果 `0.7815264028` 位于 ASSIST2009。
- 项目文档中的 `0.7978` 是 ASSIST2009 的要求或阈值，不是当前 ASIKT 在统一协议上的复现分数。

跨数据集、跨测试文件或跨指标的数值不能直接比较。要形成论文中的同协议胜负结论，必须在完全相同的 Algebra2005 question-window 测试文件上重新训练和评估 ASIKT，或者在同一 ASSIST2009 协议上重新训练和评估 Mamba4KT。

## 实现与训练协议

- 实现来源：`https://github.com/szhhwh/UniKT/tree/main/model/Mamba4KT`
- Git commit：`0493bfbba31373a3058f287cd6625ed10d48895e`
- 模型：Mamba blocks + FFN + Rasch embedding logic，`n_blocks=5`、`d_state=16`、`d_conv=4`、`expand=2`、`dropout=0.1`
- 优化器：Adam，`weight_decay=0`
- 损失：BCELoss
- 训练总参数量：3,063,611
- 验证集超参数网格：`d_model in {64,128,256}` x `learning_rate in {0.0001,0.001,0.002,0.003}`
- 冠军选择规则：验证集 AUC 最大
- 评估协议：pyKT question-window late-mean
- 评估链：causal next-item scores -> selected question-window rows -> `question_window_metrics(..., late_mean)`

本地对第三方源码的改动：

- `Mamba4KT_model.py`：增加 `prediction_head` 开关。`unikt` 保留第三方实现把 `q_next` 拼入预测头的行为；`paper` 只使用 FFN 输出。
- `Mamba4KT_trainer.py`：向模型传递 `prediction_head`。
- `Mamba4KT_data.py`：dataloader worker 数可由 `KT_DATALOADER_WORKERS` 控制，本次设为 1。

这些改动说明本次是受控的第三方实现适配，不是未经修改的上游仓库直接运行。

## 数据来源与计数

压缩包位置：

`E:\StudyWork\sota_kt_lab\pykt-toolkit\data\algebra2005\source\algebra_2005_2006.zip`

文件大小为 22,438,892 bytes，SHA256 为 `118ba5063aa3b428b4072d17533d158f25bde8072645a70058451e04449b2ce9`。来源只能写为从 USTC 公开镜像取得，不能写成直接由 CMU DataShop 下载：

`https://base.ustc.edu.cn/data/KDD_Cup_2010/algebra_2005_2006.zip`

计数链条：

| Stage | Count |
| --- | ---: |
| 原始 interactions | 809,694 |
| 删除缺失 KC 后 interactions | 607,025 |
| pyKT 多概念展开后 train+test concept events | 884,102 |
| `maxlen=200` 并按 `min_seq_len=3` 保留后 train+test | 884,098 |
| 保留 train events | 748,692 |
| 保留 test events | 135,406 |
| Test question-window rows | 125,363 |
| Test selected targets | 135,246 |
| Test question groups | 93,481 |

切窗时丢弃长度不足 3 的尾部窗口共 4 条：训练用户 41 和 124 各 1 条，测试用户 326 2 条。用户提供的 884,102 与多概念展开后的 event 数一致。用户表中的 9,926 项在当前转换器中未复现；当前转换器在删除缺失 KC 后报告 0 个无概念题目，因此 9,926 的原始定义需要在论文中另行核实，不能直接写成同一统计口径。

词典规模：574 users、173,113 questions、112 concepts。测试用户 114 个。fold 0 的训练/验证用户 460 个，其中训练 3,110 个序列、验证 870 个序列。

对齐审计从测试用户 0、57、113 抽样，窗口与 pyKT generator 完全一致。适配器 parquet 为 748,692 rows，与 manifest 一致。

## 资源使用

硬件为 RTX 3060 Laptop，整卡 6,144 MiB。运行时设置：

- `KT_DATALOADER_WORKERS=1`
- 线程数 2
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

网格阶段最高整卡占用约 5,654 MiB；冠军训练最高约 5,160 MiB。资源文件记录冠军 WSL 峰值内存约 5,180 MiB / 7,868 MiB。没有 OOM。评估阶段 Torch 峰值 allocated 为 123,603,968 bytes，reserved 为 140,509,184 bytes。

本机 `D:\Anaconda\envs\machine_learning` 可用，Torch/CUDA 正常，但缺少 Mamba4KT 所需的 `mamba_ssm` 和 `causal_conv1d` CUDA 扩展。本次训练复用了 WSL 中已有的 `/home/lpq/.venvs/unikt-mamba`，没有新建一个可工作的 Python 环境。`work/mamba4kt_algebra2005_20261101/wheels` 仍保留约 3.4 GiB wheel 文件，不能声称已经清理。

## 论文使用建议

可以把本次 AL05 结果放入补充实验或第三方实现基线表，但每一行必须同时写明：

- dataset = Algebra2005
- protocol = unified pyKT question-window late-mean
- implementation = third-party UniKT Mamba4KT
- split = fold 0
- seed = 42
- validation-selected-only test evaluation

主表如果要求与 ASIKT 或作者报告值直接比较，本次结果还不足。至少需要补做同一数据集、同一测试文件上的 ASIKT 对照，以及 Mamba4KT 的五折或多随机种子重复。当前单折单种子结果不能支持总体优越性或稳健性结论。

## 证据文件

- `evaluation_champion_d256_lr0.002/result.json`
- `evaluation_champion_d256_lr0.002/question_window_arrays.npz`
- `evaluation_champion_d256_lr0.002/question_window_predictions.tsv`
- `grid_al05/results.json`
- `pykt_data/al05_preprocess_manifest.json`
- `pykt_data/al05_question_window_alignment_audit.json`
- `unikt_data/al05_unikt_adapter_manifest.json`

上述文件以及源码、checkpoint、测试窗口文件的 SHA256 摘要见 `SHA256SUMS.json`。
