# 既有数据记录与本次资产迁移

此目录保留既有 EDA、数据审计及抽帧证据。本次只复制和校验已有文件，没有重新运行 EDA、扫描原始视频或抽帧，也没有运行 7B 推理。

- [数据审计](data_inventory.md)、[数据清单](data_manifest.md)
- [EDA 报告](eda_data_quality.md)及其 `figures/` 下的 4 张原图
- [既有数据划分](fold_design.md)
- [历史抽帧 smoke 记录](preprocessing_smoke.md)

以上 5 份历史报告和 4 张图均按原字节复制，原报告中的日期、绝对路径、统计和复现命令属于当时环境。尤其抽帧 smoke 报告中的 8 clips / 64 frames 不是本次测试和 pilot 缓存的规模。历史原视频路径无需在当前机器存在。

当前缓存规模和逐文件来源、大小、SHA256 位于项目的 `data/asset_manifest.json`，可以用 `cuhkx check` 在终端复核。旧位置的重复 Markdown 报告与旧入口已清理，此目录是数据证据的保留位置。

本步复核图片可读性、448×448 RGB JPEG 格式、8 帧完整性、索引与 metadata 关联、时间信息及复制前后哈希。不重新确认视频内容与原始解码结果的一致性。
