# CUHK-X 原始数据盘点

审计日期：2026-08-19

## 结论

当前数据为**完整**状态：HAU/HARn 训练视频、测试视频和非视觉数据均已解压，所有 CSV
引用路径、非视觉 manifest unit 以及 submission ID 均通过严格审计。

## 本地布局

```text
data/raw/
├─ Training/
│  ├─ training_qa.csv
│  ├─ HAU/
│  └─ HARn/
├─ Testing/
│  ├─ test_qa.csv
│  ├─ sample_submission.csv
│  └─ large_model_track_test/
├─ NonVisual/
│  ├─ manifest_nonvisual.csv
│  ├─ Training/
│  └─ Testing/
├─ _metadata/
│  └─ __MACOSX/
└─ README.md
```

`_metadata/` 仅保留解压产生的两个 macOS 元数据文件，不参与训练或推理。

## 审计结果

| 数据项 | 结果 |
|---|---:|
| 训练题目 | 4,087 |
| HARn 训练题目 / 唯一引用 clip | 562 / 524，全部存在 |
| HAU 训练题目 / 唯一引用 clip | 3,525 / 809，全部存在 |
| 训练视频文件 | 12,374 |
| 测试题目 | 682 |
| 测试唯一引用文件 | 208，全部存在 |
| 测试视频文件 | 754 |
| sample submission | 682 行，ID 及顺序与测试集一致 |
| 非视觉 manifest unit | 3,932，全部存在 |
| 非视觉文件 | 265,318 |

训练视频按模态计数：Depth 3,908、Depth_Color 3,741、IR 3,912、Thermal 813。
测试视频按模态计数：Depth 208、Depth_Color 198、IR 208、Thermal 140。
模态数量不一致是数据本身的可用性差异，加载器不能假定每个 clip 拥有全部模态。

## 复检命令

以下严格审计当前返回码为 0：

```powershell
conda activate CUHK-X
python scripts/audit_raw_data.py --strict
```

日常查看报告可省略 `--strict`；JSON 输出使用 `--json`。后续若移动、重新解压或补充数据，
应再次运行严格审计。
