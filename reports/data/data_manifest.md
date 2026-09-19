# CUHK-X 统一数据 Manifest

生成日期：2026-08-19  
Schema 版本：1

## 定义

`clip_manifest.csv` 是项目生成的 **clip 级数据资产清单**，每行对应一个物理视频 clip。
它不会复制或修改原始数据，而是统一关联以下信息：

- 训练/测试 split 与 HAU/HARn 来源；
- clip、用户、trial 和 HARn 动作；
- 关联的 QA 数量、ID 和题目类别；
- Depth、Depth_Color、IR、Thermal 视频可用性及路径；
- IMU、Radar、Skeleton 可用性及非视觉 unit 路径。

它与官方的 `data/raw/NonVisual/manifest_nonvisual.csv` 不同：官方文件只描述非视觉模态；
本项目 manifest 覆盖全部视觉 clip，并把 QA 和非视觉信息合并成一个稳定入口。

## 生成物

- `data/interim/clip_manifest.csv`：统一 clip 清单；
- `data/interim/clip_manifest.meta.json`：生成时间、输入文件哈希、输出哈希、字段和统计摘要。

两者属于可重建的私有数据制品，已由 `.gitignore` 排除。

## 数据规模

| Split | Clip | QA | 有 QA 的 clip | HAU | HARn |
|---|---:|---:|---:|---:|---:|
| Train | 3,912 | 4,087 | 1,333 | 814 | 3,098 |
| Test | 208 | 682 | 208 | 144 | 64 |
| **合计** | **4,120** | **4,769** | **1,541** | **958** | **3,162** |

训练集中有 2,579 个 clip 没有直接关联 QA，但仍保留在 manifest 中，可用于动作识别预训练、
自监督学习和传感器对齐。

### 模态覆盖

| Split | Depth | Depth_Color | IR | Thermal | IMU | Radar | Skeleton |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 3,908 | 3,741 | 3,912 | 813 | 3,705 | 3,716 | 3,734 |
| Test | 208 | 198 | 208 | 140 | 190 | 189 | 195 |

## 字段分组

| 分组 | 字段 |
|---|---|
| 身份 | `manifest_version`, `clip_key`, `split`, `source`, `clip_path` |
| 分组与标签 | `subject_id`, `trial_id`, `action_id`, `action_name` |
| QA 关联 | `qa_count`, `qa_ids`, `question_categories`, `has_labeled_qa` |
| 视觉模态 | `visual_modalities`, 四个 `*_path`, `total_video_bytes` |
| 非视觉模态 | `nonvisual_modalities`, `nonvisual_unit_path`, 三个 `has_*` |

`clip_path` 相对于对应的 `data/raw/Training/` 或 `data/raw/Testing/`；各模态路径相对于
`data/raw/`。路径统一使用 `/`，因此 manifest 可在 Windows 和 Linux 之间迁移。

## 完整性与复现

本次生成结果：

```text
rows   = 4120
bytes  = 1723549
sha256 = c79866deefac21a7876a19e0f4c57236039b8452dacffca6ea5243b8bbcc4004
```

生成器在写出前验证：QA 引用均能映射到物理 clip、clip 至少有一个 MP4、官方非视觉
unit 均能映射到视觉 clip、声明的传感器目录真实存在、clip key 不重复。CSV 使用固定排序和
字段顺序，写入采用临时文件替换，避免留下半成品。

重新生成：

```powershell
conda activate CUHK-X
python scripts/build_manifest.py
```

原始数据变动后，应同时运行：

```powershell
python scripts/audit_raw_data.py --strict
python scripts/build_manifest.py
```
