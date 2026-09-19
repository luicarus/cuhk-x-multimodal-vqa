# 已有输入资产

本目录除说明文件外由 Git 忽略，包含：

```text
qa/                 test.csv、pilot.csv、sample_submission.csv
references/         pilot 答案、训练 QA、已有划分及抽帧配置快照
qa/fold_0.csv ... fold_4.csv  五折完整无答案 QA
references/fold_0_answers.csv ... fold_4_answers.csv  五折完整带答案 QA
frames/test/        208 个 clip，每个 8 张 IR 图
frames/pilot/       100 个 clip，每个 8 张 IR 图
frames/fold_0/ ... fold_4/  五折完整 IR8 缓存，共 1333 clip、10664 JPEG
asset_manifest.json 逐文件来源、大小和 SHA256
```

test 目标为 682 QA，pilot 目标为 120 QA。pilot 索引的额外 QA 关联不自动参与评测。

`cuhkx check --dataset test` 或 `--dataset pilot` 可独立核验当前数据，不访问来源目录或原视频。没有重新执行 EDA/抽帧。详细说明见 [数据来源](../docs/data_provenance.md)。

2026-09-09 已从重构前的原数据目录补齐 fold0–4，共 4087 QA。完整路径、训练标签与验证说明见 [数据来源及补齐记录](../docs/data_provenance.md)。完整 fold4 有 744 QA，原 120 QA pilot 保留为其子集，不是额外训练数据。

这些数据被 Git 忽略，换电脑需要另外复制整个 `data/`；当前 `scripts/package_cloud.py` 的推理包只包含 test/pilot，不会带上新增五折。
