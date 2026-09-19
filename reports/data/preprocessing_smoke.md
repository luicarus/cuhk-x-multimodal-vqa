# CUHK-X 统一抽帧与单模态预处理

> 协议：`uniform_time_v1`  
> 配置哈希：`520837f5b798f45a`  
> 生成时间：2026-08-20T12:49:48+00:00  
> 状态：**PASS**

## 1. 预处理契约

每个视频按归一化时间轴等分区间的中心点采样，并在该模态自身时间轴上选择最近帧。这样 10 FPS 与 25 FPS 视频共享相同的相对时间位置，但不会错误复用原始 frame index。当前运行只读取一个显式指定的视觉模态，不会回退到其他模态。

- 模态：`IR`
- 帧数：`8`
- 采样：`uniform_time_centers`
- 图像：RGB `448 × 448`，等比例缩放并黑边填充
- 编码：JPEG quality `90`
- 路径隔离：输出目录只使用 `clip_key` 的稳定哈希，不把 HARn 动作名称作为模型输入或缓存目录名
- 缓存门禁：配置哈希、QA 关联、解码/图像库版本、源文件大小、mtime、SHA256、帧数量和文件存在性必须全部匹配

## 2. Smoke run 结果

| 指标 | 数量 |
|---|---:|
| 选中 clips | 8 |
| 成功 | 8 |
| 新生成 | 8 |
| 缓存命中 | 0 |
| 缺失目标模态 | 0 |
| 解码失败 | 0 |
| 输出帧 | 64 |

| Split | Source | Selected | OK | Missing | Failed |
|---|---|---:|---:|---:|---:|
| test | HARn | 2 | 2 | 0 | 0 |
| test | HAU | 2 | 2 | 0 | 0 |
| train | HARn | 2 | 2 | 0 | 0 |
| train | HAU | 2 | 2 | 0 | 0 |

## 3. 时间轴审计

- 归一化采样点：`0.0625, 0.1875, 0.3125, 0.4375, 0.5625, 0.6875, 0.8125, 0.9375`
- 源 FPS 范围：`10.000`–`10.000`
- 源时长范围：`0.300`–`55.500` 秒
- 最大目标/实际帧时间误差：`0.081250` 秒
- 重复选帧总数：`5`

## 4. 产物与复现

- 索引：`<project-root>/artifacts/preprocessing_smoke/uniform_time_v1/ir/520837f5b798f45a/frame_index.csv`
- 运行摘要：`<project-root>/artifacts/preprocessing_smoke/uniform_time_v1/ir/520837f5b798f45a/run_summary.json`
- 索引 SHA256：`b462a553cfca51a5c708fd264f59d5c5fad0cfa7456430b7d820f4afd437a71b`
- 解码器：`pyav==18.1.0`
- Pillow：`12.3.0`

```powershell
conda activate CUHK-X
python scripts/preprocess_single_modality.py --modality IR --num-frames 8 --image-size 448 --limit 8 --output-root artifacts/preprocessing_smoke --report reports/preprocessing_smoke.md --json
```

生产缓存去掉 `--limit` 并使用默认 `data/processed/frames`。未提供该模态的 clip 会以 `missing` 状态进入索引和统计，不会伪造帧或静默切换模态；`--strict-missing` 可将其升级为失败。
