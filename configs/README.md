# 运行配置

- `baseline.yaml`：唯一 IR4 + 7B 协议、模型 revision、生成参数和云端资源预算。
- `qwen35_4b.yaml`：独立 Qwen3.5-4B test 对照；只复用 IR4 输入协议，不替换 baseline。
- `datasets.yaml`：test/pilot 无答案 QA 与 8 帧缓存的绑定，路径相对于数据根。
- `submission.yaml`：模板、评测参考文件与提交格式。

不再维护其他模型 profile。原抽帧配置仅作为历史来源快照保存在 `data/references/frames_config_original.yaml`。
