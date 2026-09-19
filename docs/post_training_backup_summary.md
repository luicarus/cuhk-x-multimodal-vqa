# 后训练参考总结：原始多模态数据与历史实验

日期：2026-09-09  
用途：供后续 agent 在制定或实现后训练方案前阅读。本文是背景和证据摘要，不替代当前的实施计划。

## 1. 资料范围与重要边界

本总结根据以下备份读取：

- 原始数据：`<private-backup>\data\raw\`
- 统一数据清单：`<private-backup>\data\interim\clip_manifest.csv`
- 历史项目代码、报告和实验产物：`<private-backup>\cuhk-x-repo-20260906.bundle`

历史项目使用的是 Qwen2.5-VL-3B-Instruct，当前重构项目的固定基座是 Qwen2.5-VL-7B-Instruct。旧模型的分数只能作为方向性证据，不能当作当前 7B 模型的准确率预估。

旧项目的 QLoRA 实验没有完成训练。历史工程门禁在模型构建后发现显存超过冻结上限，于是主动停止；forward、backward、optimizer step、adapter 保存和重载均为 0。因此，旧项目没有“QLoRA 效果不好”的泛化结论，也没有可复用的训练 adapter。相关历史报告路径为 Git bundle 中的 `reports/20_experiments/post_training/pt1_gpu_v1_stop.md`。

当前项目已补齐五折 IR8 训练缓存：4087 QA、1333 个有 QA 的 clip、10664 张 IR JPEG。当前实施计划见 [后训练计划](post_training.md)，首轮仍是 IR + SFT + QLoRA。

## 2. 原始数据中的模态覆盖

原始训练集共有 3912 个 clip、4087 道 QA。下表按有训练 QA 的 clip 统计关联 QA 覆盖；缺少某一模态的 clip 不应被静默删除。

| 模态 | 训练 QA 覆盖 | 测试 QA 覆盖 | 数据形态 | 当前启发 |
|---|---:|---:|---|---|
| IR | 4087 / 4087 | 682 / 682 | 红外视频 | 当前主输入，已有统一 IR8 缓存 |
| Depth | 4087 / 4087 | 682 / 682 | 深度视频 | 覆盖完整，最适合先做 IR 的互补模态实验 |
| Depth_Color | 4037 / 4087 | 660 / 682 | 彩色化深度视频 | 不是普通 RGB；暂不当作独立高价值模态 |
| Thermal | 3521 / 4087 | 594 / 682 | 热成像视频，HAU 为主 | 覆盖不全，接入前必须设计缺失模态策略 |
| Skeleton | 4026 / 4087 | 667 / 682 | 每帧 17 个关键点及置信度 JSON | 适合动作和时序辅助，但需要数值序列编码器或投影 |
| IMU | 4001 / 4087 | 650 / 682 | 多设备时序 CSV，约 10 Hz | 有肢体运动信号，需要对齐、排序和传感器编码 |
| Radar | 4002 / 4087 | 646 / 682 | mmWave 点云检测 CSV | 有运动信息，但数据稀疏、接入成本最高 |

原始非视觉说明见 `<private-backup>\data\raw\NonVisual\README.md`。IMU 行可能有中英文列名和设备内小范围乱序；Radar 可能只有表头；Skeleton 是姿态预测，不含可直接输入模型的可视化图片。

## 3. 历史视觉实验的可用结论

### 3.1 IR 与 Depth 是互补信息，不适合直接固定切换

在旧项目的同一批 117 道 QA 上，公平模态比较为：

| 输入 | 正确数 | 准确率 |
|---|---:|---:|
| IR | 42 / 117 | 35.90% |
| Depth | 40 / 117 | 34.19% |
| Depth_Color | 33 / 117 | 28.21% |

逐题对齐显示，IR 错而 Depth 对有 11 题，IR 对而 Depth 错有 13 题。Depth 不是 IR 的稳定替代品，但包含不同证据，可能适合联合输入或训练时的辅助监督。

旧项目曾固定将 `HARn/object_interaction` 路由到 Depth。在 fold2/fold3 的 1612 道确认 QA 上，IR4 从 635 对变为 636 对，平均收益约 +0.07 pp，未达到当时的晋级门槛。因此不要直接复制旧的固定类别路由规则。

### 3.2 增加 IR 帧数的收益不稳定且成本上升

旧 3B 模型的 IR2 与 IR4 在 fold2/fold3 确认中为：

- fold2：315/862 对两者相同；
- fold3：IR2 为 313/750，IR4 为 320/750；
- 两折非加权平均仅提升约 +0.47 pp，低于当时 +0.5 pp 门槛；
- IR4 累计推理成本约为 IR2 的 1.31 倍。

fold0/fold1 盲评进一步显示 IR4 相对 IR2 的两折平均只提升 +0.18 pp，成本约 2.73 倍，主门禁失败。因此当前后训练首轮应固定项目计划中的帧协议，不预设增加帧数会带来收益。

### 3.3 选项顺序敏感是明确的训练诊断方向

旧模型将选项文本换位后再映射回原语义，只有 75/120 题保持相同预测；`sequence` 题只有 1/20 题保持语义一致。整体准确率从 36.67% 变为 34.17%，但样本较小，不能把该差异当作稳定总体效应。

这说明 SFT 后应检查模型是否仍依赖 A–D 槽位。若需要做选项置换增强，必须同步变换答案：multi 映射后按 A–D 重新规范化，sequence 保留动作顺序。该增强仍属于 SFT + QLoRA，不应与模态、帧数和 prompt 同时改变。

### 3.4 复杂推理策略的性价比很低

旧项目的 candidate scoring 与直接生成都是 44/120，语义一致率 95%，但 candidate scoring 的样本推理成本约 13.57 倍。因此优先训练模型本身，再考虑昂贵的候选打分或多视图推理。

## 4. 对后训练路线的具体启发

### 首轮：保持 IR + SFT + QLoRA

当前首轮继续使用原计划：固定 Qwen2.5-VL-7B revision、IR8 缓存中的第 2/4/6/8 张图、280×280 视觉处理、`cuhkx_mcq_v2` prompt、答案规范化和受约束解码。训练折使用 fold0/1/2，共 2593 QA；fold3 用于开发选择，fold4 中排除 pilot 的 624 QA 用于固定确认，pilot 120 QA 只作历史诊断。

不要因拥有其他模态就改写首轮实验。首轮的价值是建立当前 7B 基座在完整训练数据上的可靠 SFT 对照。

### 第二阶段候选：IR + Depth 联合 SFT

Depth 与 IR 都覆盖全部训练 QA，且旧实验有逐题互补证据，所以它是最值得优先验证的第二阶段方向。建议：

1. 保持同一 QA、subject-grouped folds、答案和评估规则；
2. 为 IR-only、Depth-only、IR+Depth 建立明确的单变量对照；
3. 固定每个模态的帧数和图像尺寸，避免仅因图像数量增加而产生不公平收益；
4. 记录每模态缺失率、跨模态时间对应关系和实际输入 token 长度；
5. 先在 dev，再在固定确认集评估，不直接运行 test。

联合输入可以采用多图 SFT，但需要确认 Qwen processor 的多图占位符、模板和监督 mask。若最终仍要求只用 IR，可把联合模型作为教师，研究跨模态知识蒸馏；只有当教师在独立验证上确实优于 IR-only 时，蒸馏才有意义。

### 更后阶段：Skeleton / IMU / Thermal / Radar

Skeleton 和 IMU 与动作时序关系最直接，但当前图像 VLM 管线不能直接读取 CSV/JSON。需要单独的序列编码器、时间对齐和投影模块，或者预先构造经过验证的时序表示。Thermal 覆盖不全，Radar 事件稀疏，均不应在没有缺失模态策略和独立对照前加入首轮训练。

## 5. 实验与评估纪律

- 继续使用 `subject_grouped_v1`，同一 subject 和同一 clip 的全部 QA 必须在同一 fold。
- 一个实验只改变一个主要变量；模态、帧数、prompt、选项顺序增强和 LoRA 配置不要同时变化。
- 任何新模态都要记录输入覆盖、缺失处理、缓存协议和数据指纹；不能因缺模态静默缩小数据集。
- 训练 loss 只作诊断，主指标是 question-level exact-match；同时记录 category、source、subject 和 paired transitions。
- 先得到同期基座在 dev/确认集的预测，再比较 adapter；不能拿旧 3B 分数或 fold4 pilot 的选择性结果替代当前基线。
- fold4 曾用于历史实验选择，不能称为从未看过的盲测集；确认集一旦反复调参就失去独立确认资格。
- 正式 test 682 QA 只在候选已经选定且确认集有收益后使用，不使用测试答案或伪标签训练。

## 6. 建议后续 agent 的执行顺序

1. 先阅读本文、[后训练计划](post_training.md)、[数据来源说明](data_provenance.md)和 `data/asset_manifest.json`。
2. 归档当前 7B baseline 的真实运行配置、revision、依赖版本和 dev/确认预测。
3. 实现并 CPU 验证 IR-only 的多模态 dataset、collator、assistant-only loss mask、LoRA 元数据和恢复逻辑。
4. 在云端单卡进行真实前向、反向、optimizer step、adapter 保存/重载和恢复训练的短跑；不要把模型构建成功当成训练成功。
5. 完成首轮 IR + SFT + QLoRA 后，再决定是否开展 IR+Depth 联合 SFT。
6. 任何后续模态、选项置换增强或蒸馏实验都需先写清输入、变量、门槛和停止规则。

## 7. 相关文件索引

当前项目：

- `docs/post_training.md`：SFT + QLoRA 实施计划。
- `docs/data_provenance.md`：五折缓存来源、数量、哈希和恢复记录。
- `data/fold_0_3_restore_report.json`、`data/fold_4_restore_report.json`：缓存恢复验收。
- `data/references/folds/subject_grouped_v1/`：冻结 subject、clip、QA 划分。

备份中的历史项目：

- `data/raw/README.md`：视觉模态和数据布局。
- `data/raw/NonVisual/README.md`：IMU、Radar、Skeleton 格式和覆盖。
- `data/interim/clip_manifest.csv`：逐 clip 的模态存在性。
- `reports/20_experiments/modality/visual_cv_modality_comparison_fold4_pilot.md`：IR/Depth/Depth_Color 对比。
- `reports/20_experiments/confirmation/visual_cv_ir2_vs_ir4_fold23_confirmation.md`：IR2/IR4 两折确认。
- `reports/20_experiments/confirmation/fold01_blind_reveal_v1.md`：IR2/IR4 盲评结果。
- `reports/20_experiments/option_order/visual_cv_option_order_diagnostic_fold4_pilot.md`：选项顺序诊断。
- `reports/20_experiments/post_training/pt1_gpu_v1_stop.md`：历史 QLoRA 工程门禁停止记录。

历史报告均位于 `<private-backup>\cuhk-x-repo-20260906.bundle` 对应的仓库版本中；如果后续 agent 无法直接读取 bundle，应先将其以只读方式克隆到临时目录，不要把旧项目报告混入当前项目的实验结果目录。
