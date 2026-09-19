# IR4 + 7B 的既有输入资产

当前数据来自工作区同级 `repo/` bundle；数据报告和 EDA 图表已迁至 `reports/data/`。本轮不需要原视频，不重新 EDA 或抽帧。统一 CLI 已实现，日常运行不依赖迁移来源。

## 可用输入

| 用途 | 无答案 QA | 缓存根目录 | 目标 QA / 缓存 clip / JPEG |
|---|---|---|---|
| test | `data/qa/test.csv` | `data/frames/test/` | 682 / 208 / 1664 |
| pilot | `data/qa/pilot.csv` | `data/frames/pilot/` | 120 / 100 / 800 |
| fold 0 | `data/qa/fold_0.csv` | `data/frames/fold_0/` | 873 / 281 / 2248 |
| fold 1 | `data/qa/fold_1.csv` | `data/frames/fold_1/` | 858 / 281 / 2248 |
| fold 2 | `data/qa/fold_2.csv` | `data/frames/fold_2/` | 862 / 279 / 2232 |
| fold 3 | `data/qa/fold_3.csv` | `data/frames/fold_3/` | 750 / 245 / 1960 |
| fold 4 | `data/qa/fold_4.csv` | `data/frames/fold_4/` | 744 / 247 / 1976 |

各缓存根目录下均保留 `uniform_time_v1/ir/520837f5b798f45a/frame_index.csv` 及其内部相对路径。五折合计 4,087 QA / 1,333 clips / 10,664 JPEG，pilot 是 fold 4 的既有子集，不能再次计入五折总量。

QA 推理字段严格为 `qa_id,source,path,category,question,A,B,C,D`。test 原文件的空 `prediction` 列和 pilot 的 `answer` 列均不进入派生推理输入；行顺序不变。`path` 是历史关联信息，不代表运行时应打开该视频或向模型展示该路径。

pilot 缓存索引总共关联 355 个 QA，仅指定 CSV 的 120 个 QA 进入新基线评测。多出的 235 个关联不删除，不得自动加入评测。

## 参考资料

- `data/qa/sample_submission.csv`：原始提交模板。
- `data/references/pilot_answers.csv`：原 pilot CSV，仅用于评测。
- `data/references/training_qa.csv`：原训练 QA，仅作参考。
- `data/references/test_qa_original.csv`：未改动的测试 QA。
- `data/references/folds/subject_grouped_v1/`：既有划分，不重新划分。
- `data/references/frames_config_original.yaml`：原抽帧配置快照；其中原始数据路径仅说明来源。
- [已有数据报告](../reports/data/README.md)：数据分析与历史抽帧证据。

## 帧协议与验证范围

保留全部 8 张 448×448 RGB letterbox JPEG，不重新编码。新基线拟使用零起始索引 `[1,3,5,7]`，即第 2/4/6/8 张；对应目标归一化时间为 `0.1875,0.4375,0.6875,0.9375`。实际时间取自 metadata。

`data/asset_manifest.json` 对每个迁移文件记录相对目标路径、字节数、SHA256、来源路径及来源 SHA256。来源路径相对于主仓库的父目录；未来删除来源后，此字段仅作溯源。资产清单不包括自身或本次生成的说明文件，避免自引用。

资产迁移时已完整解码图片、检查 QA 关联及 metadata/索引时间字段，并核对源/目标哈希。日常通过 `cuhkx check` 在终端复核，不再保留一次性验收 JSON。没有访问原视频，因此不声称重新验证了原视频 SHA256 或抽帧是否忠实反映原片。

## 独立核验

一次性迁移脚本已退出工作树。使用 `cuhkx check --dataset test` 与 `cuhkx check --dataset pilot` 读取当前资产清单并核验缓存，不再读取迁移来源。来源字段仅作溯源，不要求所记录的历史路径仍然存在。

2026-09-10 已独立核对新增五折的派生 QA、答案与原 training_qa/qa_folds 一致，并核验全部 14,818 个登记资产哈希及五折图片解码。`cuhkx training-check` 检查训练/开发/确认分组，当前分别为 2,593 / 750 / 624 QA，缺失均为 0。本地补齐报告和旧清单备份由用户保留，不参与训练或打包。

`data/` 资产继续由 `.gitignore` 排除；Git 状态不会列出图片和 QA 的复制，交付时需要使用后续数据打包流程。

## Fold0–3 补齐记录（2026-09-09）

重构时已迁入完整 `training_qa.csv` 和五折冻结映射，但视觉训练缓存仅迁入 fold4 pilot。此次从 `<private-data-root>` 恢复缺失四折的完整 IR8 缓存。

| Fold | 原 `visual_cv/subject_grouped_v1/` 下的 scope | QA | Clip | JPEG |
|---|---|---:|---:|---:|
| 0 | `fold_01_blind_v1/fold_0` | 873 | 281 | 2248 |
| 1 | `fold_01_blind_v1/fold_1` | 858 | 281 | 2248 |
| 2 | `fold_23_confirmation_v1/fold_2` | 862 | 279 | 2232 |
| 3 | `fold_23_confirmation_v1/fold_3` | 750 | 245 | 1960 |
| 合计 | | 3343 | 1086 | 8688 |

对于 `k = 0,1,2,3`，本项目中的路径为：

- 无答案推理输入：`data/qa/fold_k.csv`，保持原始行顺序。
- 完整带答案 QA：`data/references/fold_k_answers.csv`，与来源文件字节一致，可供训练数据构建或评测使用。
- IR8 缓存根目录：`data/frames/fold_k/`。
- 帧索引：`data/frames/fold_k/uniform_time_v1/ir/520837f5b798f45a/frame_index.csv`。索引中的帧和 metadata 路径相对于该折缓存根目录。
- 原划分准备元数据：`data/references/visual_cv/`。

按当前 IR 基线补齐，未迁移 Depth 等其他模态或原视频。原始图像、索引、metadata 均直接复制，不重新抽帧或编码。四折 QA ID 已逐折与冻结 `qa_folds.csv` 对齐，且与对应缓存 QA 集合完全一致；原训练 QA 与划分文件也已核对哈希。

新增文件的源/目标 SHA256 已登记在 `data/asset_manifest.json`，其中 `supplemental_training_folds.bindings` 提供四折数据绑定，`counts` 提供规模。复制前的清单备份为 `data/asset_manifest.before_fold_0_3.json`。验收结果保存于 `data/fold_0_3_restore_report.json`，包括全量图片解码、metadata/时间字段检查及原 test/pilot 的回归核验。

通用推理 CLI 的 `--dataset` 仍只支持 test/pilot；训练管线通过独立的 `training-check` 检查五折数据：

```powershell
cuhkx training-check
```

冻结 subject 分组保持不变；训练/验证用途取决于训练计划选定的 held-out fold，不能将待评测 fold 同时用于训练。随后也补齐了完整 fold4，见下节。

换电脑时须单独传输整个 `data/`。Git 不包含这些资产；`scripts/package_cloud.py` 仍只打包 test/pilot 推理数据，完整训练发布应使用 `scripts/package_training.py`。

## Fold4 补齐记录（2026-09-09）

完整 fold4 包含 **744 QA、247 个有 QA 的 clip、1976 张 IR JPEG**。补齐后五折合计 **4087 个不重复 QA、1333 个有 QA 的 clip、10664 张 IR JPEG**，覆盖冻结划分中的全部训练 QA。无 QA 视频及其他模态不在本次 IR QA 数据补齐范围。

- 完整无答案输入：`data/qa/fold_4.csv`。
- 完整带答案 QA：`data/references/fold_4_answers.csv`，直接复制原 `interim/visual_cv/subject_grouped_v1/fold_4_full_v1/fold_4/qa.csv`。
- 完整帧缓存根目录：`data/frames/fold_4/`。
- 完整索引：`data/frames/fold_4/uniform_time_v1/ir/520837f5b798f45a/frame_index.csv`。

原 pilot 有 120 QA，原 remainder 有 624 QA；其缓存分别包含 100、224 个 clip。由于一个 clip 可以对应多个 QA，两套缓存存在 77 个重叠 clip，不能直接把索引拼接。此次逐一核对重叠索引行、metadata 和 JPEG 均完全一致后，按 `sample_id` 去重合并为 247 个 clip，并验证关联 QA 恰好覆盖完整 fold4 的 744 个 ID。合并索引保存了两个来源索引的哈希，原帧及 metadata 保持字节一致。

原 `data/qa/pilot.csv` 和 `data/frames/pilot/` 保留，pilot 是完整 fold4 的子集，不能与完整 fold4 相加计算样本数。两份历史运行摘要保存在 `data/references/visual_cv/<scope>/ir_run_summary_original.json`，不会充当完整 fold4 的新抽帧摘要。

资产清单的 `supplemental_training_folds.bindings` 现已包含 fold0–4，`cuhkx training-check` 会检查训练、开发验证和确认数据。新增验收报告为 `data/fold_4_restore_report.json`；清单补齐前备份为 `data/asset_manifest.before_fold_4.json`。已核验五折 QA 无重复且总计 4087，并完成完整 fold4 图像解码、索引/metadata 校验及原 pilot/test 回归核验。
