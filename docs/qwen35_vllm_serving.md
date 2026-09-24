# Qwen3.5-4B baseline vLLM 双卡推理实验

选择 Qwen3.5-4B 作为最终模型后，我在 2×T4 配置中补充了 vLLM 推理实验，观察请求批处理、TP/DP 方式对吞吐、延迟和显存的影响。serving 对照使用 `Qwen/Qwen3.5-4B` base model，**不含 QLoRA adapter**；QLoRA adapter serving 尚未进行同口径测试。它是竞赛方案的推理工程延伸，不是独立 serving benchmark。

## 测试口径

- 工作负载：682 条 test QA；沿用 IR8 缓存，每条请求读取第 2/4/6/8 帧，输入尺寸 280×280。
- 模型：`Qwen/Qwen3.5-4B`，revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`。
- 生成：greedy decode、受约束答案空间、最多 8 tokens。
- vLLM 环境：vLLM 0.19.1、PyTorch 2.10.0+cu128、Transformers 5.17.0、2×Tesla T4。
- 同时保留 Transformers baseline 作为跨引擎参考；它使用不同的软件栈，不作为 TP/DP 的控制组。

## 配置与结果

| Run | 执行方式 | Runner 总批次 | 每 replica `max_num_seqs` | 推理吞吐 | 单请求延迟统计 | 批次摊销执行时间 | 加载 / 端到端总耗时 |
|---|---|---:|---:|---:|---|---:|---:|
| `qwen35_4b_test` | Transformers | 1 | — | 未单独记录 | 未记录 | — | 7.58 / 768.62 s |
| `qwen35_4b_test_vllm_v1` | vLLM TP=2，逐条提交 | 1 | 1 | 1.1672 req/s | mean 856.786 ms；P50 898.535；P95 1010.179；P99 1080.654 | — | 82.81 / 675.00 s |
| `qwen35_4b_test_vllm_v2` | vLLM TP=2 | 32 | 32 | 1.4303 req/s | 未记录 | 699.156 ms / 请求 | 83.24 / 567.83 s |
| `qwen35_4b_test_vllm_v3`（当前） | vLLM DP=2、TP=1，并行加载 | 32 | 16 | **1.5001 req/s** | 未形成逐请求分布 | **666.601 ms / 请求** | **97.15 / 559.76 s** |

当前 DP2-B32 的 682 条请求分成 22 批（21 批各 32 条，末批 10 条），平均批次大小 31.0，平均批次耗时 20.665 s；批次总耗时 454.622 s，输出 token rate 为 3.1389 tokens/s。批次摊销时间是批次时长除以请求数，不是单请求延迟。

相较 TP2-B32，当前 DP2-B32 的请求吞吐高 4.9%，批次阶段耗时低 4.7%，端到端时间短 8.1 s；DP2 模型加载多 13.9 s，抵消了部分推理收益。相较此前记录的 DP2-B16，吞吐从 1.5490 降至 1.5001 req/s（低 3.2%），摊销时间从 645.577 增至 666.601 ms；并行加载则将模型加载从 182.53 降至 97.15 s，端到端总耗时从 630.41 降至 559.76 s。批次和启动实现都变化了，不能把吞吐差异单独归因于 batch。

此前 DP2-B16 的原始输出已被新结果替换，上述数值是先前记录的历史指标。当前结果在 Kaggle test 标签不可见的条件下只证明运行与提交格式通过，不能据此判断 accuracy。

## 从现有日志可额外读取的指标

- **稳态吞吐与预热**：TP2 逐条请求 run 跳过前 3 条 warmup 后，679 条稳态请求的吞吐为 1.1925 req/s，P50/P95/P99 总请求耗时为 897.861 / 1010.179 / 1077.883 ms。全量最大值为 12.9999 s，对应 `test_0001`；日志没有把这条长尾归因到具体阶段，因此不直接称为 TTFT。
- **阶段耗时拆分**：TP2 逐条请求的均值为 generation 836.320 ms、图像阶段 7.952 ms、其他 overhead 12.514 ms；记录的时间占比分别为 97.61%、0.93%、1.46%。这组 workload 的时间主要花在模型生成阶段。
- **启动占比与残差**：模型加载占端到端总耗时，TP2 逐条、TP2-B32、当前 DP2-B32 分别约为 12.27%、14.66%、17.36%。端到端耗时减去加载和已记录推理阶段后，仍有约 7.86 / 7.76 / 7.99 s 未进一步拆分；这是残差，不能归因到单一组件。
- **完成率**：4 个 run 均为 PASS，682/682 条记录有效，failed、invalid、pending、prompt leakage 均为 0，每条 checkpoint 只尝试 1 次；4 份提交文件的结构校验也通过。它说明运行和格式完整，不代表 test accuracy。
- **TTFT 覆盖**：此前 TP2 逐条 run 的 `ttft_ms` 为 `count=0, reported_by_engine=false`；当前 DP2-B32 结果也由旧版包生成，`latency.requests=0` 且没有逐请求 TTFT 样本。新打包的 Notebook 会用 `_ttft` run-id 重新运行，并从 vLLM 的每请求 `first_token_latency` 写入覆盖率与分位数；批次端到端延迟仍单独统计。
- **worker telemetry 缺口**：TP2-B32 的 worker telemetry 因 RPC 返回 function 对象而序列化失败，`workers` 为空；当前 DP2-B32 的 worker telemetry 可读到两个 replica，但 allocator `peak_allocated` / `peak_reserved` 为 0。这些日志没有 Peak HBM，不能把运行结束读数标成峰值。

## 显存观测

下表列出设备在运行前和运行结束时报告的空闲显存，单位 MiB。它用于观察系统余量变化，**不是 Peak HBM，也不是模型权重或 KV cache 的独立占用量**。

| Run | 运行前空闲 GPU0 / GPU1 | 运行后空闲 GPU0 / GPU1 |
|---|---:|---:|
| TP2 逐条请求 | 3900.8 / 3900.8 | 3744.8 / 3744.8 |
| TP2-B32 | 3710.8 / 3710.8 | 1316.8 / 1316.8 |
| DP2-B16（历史） | 4332.8 / 4332.8 | 3210.8 / 3104.8 |
| **DP2-B32（当前）** | **4332.8 / 4332.8** | **2268.8 / 2268.8** |

当前 DP2-B32 结束时每卡还报告约 2.22 GiB 空闲显存。worker allocator 的 `peak_allocated_mib` / `peak_reserved_mib` 仍为 0，因此这份旧运行没有可用峰值读数。新 ZIP 会在预测期间每 100 ms 用 NVML 采样每卡全设备 used/free 高水位，保存在 `gpu_memory.peaks.device_polling`；这是设备级采样峰值，包含其他进程占用，采样间隔之间的瞬时尖峰可能漏掉。不同 Kaggle session 的后台占用也会影响结果。

## 并发口径

当前 DP2-B32 的 Runner 每次最多提交 32 条，后端将其均分到两个 replica，每个 replica 收到 16 条；每副本 `max_num_seqs=16` 是 vLLM scheduler 上限。两个子进程在等待 readiness 之前都已启动，因此模型加载重叠。日志记录的副本加载时间为 88.86 / 85.27 s，后端总加载时间为 97.15 s，而非二者相加。

旧 DP2-B16 run 的 Runner 总批次为 16，因此两个 replica 各收到 8 条；它的 `max_num_seqs=16` 是每副本上限，不代表实际每副本都收到 16 条。

## 下一步测量

如果继续完善这组推理实验，优先补齐：

1. 用更新后的 test ZIP 重跑 DP2-B32，确认逐请求 TTFT 样本覆盖率、分位数和 NVML 采样峰值；当前保存的 V3 输出来自旧包，需新 run-id。
2. 采集 GPU utilization 与功耗。批处理下仍不能从整批耗时推导请求级 E2E latency；如需该指标，应另测流式请求路径。
3. 采集 input/prompt tokens per second；当前输出最多 8 tokens，output token rate 的解释力有限。
4. 用同一协议测量 Qwen3.5-4B QLoRA adapter 的 serving 性能，与 base model 区分报告。

## 复现范围与限制

- 当前 Git 版本和 test Notebook 使用 DP=2、每个 replica `max_num_seqs=16`、Runner 全局 `runner_batch_size=32`；两个 replica 并行启动加载。
- 最新 `outputs/qwen35_4b_test/qwen35_4b_test_vllm_v3/run_summary.json` 为 PASS，682/682 条有效，failed、invalid、pending、prompt leakage 均为 0；这不代表 test accuracy。
- 旧 batch run 没有逐请求延迟样本（`latency.requests=0`），不能从它报告 TTFT 或请求级延迟分布。更新后的包只增加引擎报告的 TTFT 样本，不会伪造请求级端到端延迟。
- 显存字段记录运行前与运行结束时的 used/free 容量，不是峰值；当前 worker peak 字段为 0。不同 Kaggle session 的后台占用可能不同。本次没有采集 GPU 利用率、功耗、prompt token throughput 或实际账单成本。
- vLLM 批处理 run 的明细由本地 `outputs/` 中的 `run_summary.json` 保存。`outputs/` 被 Git 忽略，不包含在公开仓库中。

汇总本地已保存的 run：

```powershell
python scripts/bench_report.py
python scripts/bench_report.py --baseline qwen35_4b_test
```

该脚本只汇总 `outputs/` 中已有的结果，不启动模型。要重新测量吞吐或显存，需要 Kaggle 2×T4 GPU session。
