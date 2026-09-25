# Qwen3.5-4B baseline vLLM 双卡推理实验

选择 Qwen3.5-4B 作为最终模型后，我在 2×T4 配置中补充了 vLLM 推理实验，观察请求批处理、TP/DP 方式对吞吐、延迟和显存的影响。serving 对照使用 `Qwen/Qwen3.5-4B` base model，**不含 QLoRA adapter**；QLoRA adapter serving 尚未进行同口径测试。它是竞赛方案的推理工程延伸，不是独立 serving benchmark。

## 测试口径

- 工作负载：682 条 test QA；沿用 IR8 缓存，每条请求读取第 2/4/6/8 帧，输入尺寸 280×280。
- 模型：`Qwen/Qwen3.5-4B`，revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`。
- 生成：greedy decode、受约束答案空间、最多 8 tokens。
- vLLM 环境：vLLM 0.19.1、PyTorch 2.10.0+cu128、Transformers 5.17.0、2×Tesla T4。
- 同时保留 Transformers baseline 作为跨引擎参考；它使用不同的软件栈，不作为 TP/DP 的控制组。

## 配置与结果

| Run | 执行方式 | Runner 总批次 | 每 replica `max_num_seqs` | 推理吞吐 | 逐请求指标（E2E / TTFT） | 批次摊销执行时间 | 加载 / 端到端总耗时 |
|---|---|---:|---:|---:|---|---:|---:|
| `qwen35_4b_test` | Transformers | 1 | — | 未单独记录 | 未记录 | — | 7.58 / 768.62 s |
| `qwen35_4b_test_vllm_v1` | vLLM TP=2，逐条提交 | 1 | 1 | 1.1672 req/s | mean 856.786 ms；P50 898.535；P95 1010.179；P99 1080.654 | — | 82.81 / 675.00 s |
| `qwen35_4b_test_vllm_v2` | vLLM TP=2 | 32 | 32 | 1.4303 req/s | 未记录 | 699.156 ms / 请求 | 83.24 / 567.83 s |
| `qwen35_4b_test_vllm_v3/` | vLLM DP=2、TP=1，并行加载 | 32 | 16 | 1.5171 req/s | TTFT P50 11297 ms；P95 17138 ms；P99 21669 ms（682/682） | 659.173 ms / 请求 | 97.47 / 554.71 s |

最佳运行的 682 条请求分成 22 批（21 批各 32 条，末批 10 条），平均批次大小 31.0，平均批次耗时 20.434 s；批次总耗时 449.556 s，输出 token rate 为 3.1742 tokens/s。批次摊销时间是批次时长除以请求数，不是单请求延迟。

最佳单次逐请求 TTFT 覆盖率为 682/682（100%），均值 11539 ms，P50 11297 ms、P95 17138 ms、P99 21669 ms，最大 27415 ms。三次各自的 P50 范围为 11.30–12.12 s，P95 为 17.14–17.72 s，P99 为 20.25–26.23 s。TTFT 来自 vLLM 的 `first_token_latency`，从引擎接收请求开始并包含调度排队；不是整条请求的端到端时延。每条 QA 样本保存在对应 `run_summary.json` 的 `latency.ttft_ms.samples`。

## 并发口径

当前 DP2-B32 的 Runner 每次最多提交 32 条，后端将其均分到两个 replica，每个 replica 收到 16 条；每副本 `max_num_seqs=16` 是 vLLM scheduler 上限。两个子进程在等待 readiness 之前都已启动，因此模型加载重叠。日志记录的副本加载时间为 94.82 / 91.89 s，后端总加载时间为 104.26 s，而非二者相加。
