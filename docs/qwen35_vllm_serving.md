# Qwen3.5-4B baseline vLLM 双卡推理实验

选择 Qwen3.5-4B 作为最终模型后，我在 2×T4 配置中补充了 vLLM 推理实验，观察请求批处理、TP/DP 方式对吞吐、延迟和显存的影响。serving 对照使用 `Qwen/Qwen3.5-4B` base model，**不含 QLoRA adapter**；QLoRA adapter serving 尚未进行同口径测试。它是竞赛方案的推理工程延伸，不是独立 serving benchmark。

## 测试口径

- 工作负载：682 条 test QA；沿用 IR8 缓存，每条请求读取第 2/4/6/8 帧，输入尺寸 280×280。
- 模型：`Qwen/Qwen3.5-4B`，revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`。
- 生成：greedy decode、受约束答案空间、最多 8 tokens。
- vLLM 环境：vLLM 0.19.1、PyTorch 2.10.0+cu128、Transformers 5.17.0、2×Tesla T4。
- 同时保留 Transformers baseline 作为跨引擎参考；它使用不同的软件栈，不作为 TP/DP 的控制组。

## 配置与结果

| Run | 执行方式 | 推理吞吐 | 单请求延迟统计 | 批次摊销执行时间 | 加载 / 端到端总耗时 |
|---|---|---:|---|---:|---:|
| `qwen35_4b_test` | Transformers，batch 1 | 未单独记录 | 未记录 | — | 7.58 / 768.62 s |
| `qwen35_4b_test_vllm_v1` | vLLM TP=2，逐条提交 | 1.1672 req/s | mean 856.786 ms；P50 898.535；P95 1010.179；P99 1080.654 | — | 82.81 / 675.00 s |
| `qwen35_4b_test_vllm_v2` | vLLM TP=2，`max_num_seqs=32` | 1.4303 req/s | 未记录 | 699.156 ms / 请求 | 83.24 / 567.83 s |
| `qwen35_4b_test_vllm_v3` | vLLM DP=2、TP=1，`max_num_seqs=16` | 1.5490 req/s | 未记录 | 645.577 ms / 请求 | 182.53 / 630.41 s |

TP2-B32 由 22 个 batch 处理完 682 条请求，平均 batch 大小 31.0、平均 batch 时间 21.674 s；DP2-B16 使用 43 个 batch，平均大小 15.86、平均 batch 时间 10.239 s。批次摊销执行时间是批次时长除以请求数，不是单请求 P50/P95，因此与 TP2 的请求延迟分栏记录。

V2 和 V3 还记录了 output token rate（2.9927 / 3.2411 tokens/s），但输出上限为 8 tokens，实际答案通常只有 1–4 tokens，因此不把 decode token rate 当成主要 workload 指标；input/prompt tokens per second 没有记录。

与 TP2-B32 相比，DP2-B16 的推理阶段吞吐提高 8.3%，摊销耗时降低 7.7%，推理总时间减少约 36.5 s。但它的模型加载多花约 99.3 s，因此 682 条请求的一次性端到端时间反而多 62.6 s。按双卡全程占用估算，含启动的 GPU 卡秒约为 TP2-B32 的 1665、DP2-B16 的 1849 / 千请求。DP 结束时每卡多保留约 1.75–1.85 GiB 空闲显存。

DP2-B16 与 TP2-B32 有 675/682 条预测一致，7 条输出不同；两者的 682 行提交文件均通过格式校验。test 标签不在本地，预测一致率和格式通过率不能替代 accuracy。准确率应以 Kaggle 评估为准。

## 从现有日志可额外读取的指标

- **稳态吞吐与预热**：TP2 逐条请求 run 跳过前 3 条 warmup 后，679 条稳态请求的吞吐为 1.1925 req/s，P50/P95/P99 总请求耗时为 897.861 / 1010.179 / 1077.883 ms。全量最大值为 12.9999 s，对应 `test_0001`；日志没有把这条长尾归因到具体阶段，因此不直接称为 TTFT。
- **阶段耗时拆分**：TP2 逐条请求的均值为 generation 836.320 ms、图像阶段 7.952 ms、其他 overhead 12.514 ms；记录的时间占比分别为 97.61%、0.93%、1.46%。这组 workload 的时间主要花在模型生成阶段。
- **启动占比与残差**：模型加载占端到端总耗时，TP2 逐条、TP2-B32、DP2-B16 分别约为 12.27%、14.66%、28.95%。端到端耗时减去加载和已记录推理阶段后，仍有约 7.86 / 7.76 / 7.60 s 未进一步拆分；这是残差，不能归因到单一组件。
- **完成率**：4 个 run 均为 PASS，682/682 条记录有效，failed、invalid、pending、prompt leakage 均为 0，每条 checkpoint 只尝试 1 次；4 份提交文件的结构校验也通过。它说明运行和格式完整，不代表 test accuracy。
- **TTFT 覆盖**：TP2 逐条 run 的 `ttft_ms` 为 `count=0, reported_by_engine=false`；两个 batch run 没有 TTFT 字段。因此现有日志不能给出 TTFT 分位数，也没有 batch P95。
- **worker telemetry 缺口**：TP2-B32 的 worker telemetry 因 RPC 返回 function 对象而序列化失败，`workers` 为空；DP2-B16 的 RPC 返回两个 replica 的运行结束 used/free 读数，但 allocator `peak_allocated` / `peak_reserved` 为 0。这些日志没有 Peak HBM，不能把运行结束读数标成峰值。

## 显存观测

下表列出设备在运行前和运行结束时报告的空闲显存，单位 MiB。它用于观察系统余量变化，**不是 Peak HBM，也不是模型权重或 KV cache 的独立占用量**。

| Run | 运行前空闲 GPU0 / GPU1 | 运行后空闲 GPU0 / GPU1 |
|---|---:|---:|
| TP2 逐条请求 | 3900.8 / 3900.8 | 3744.8 / 3744.8 |
| TP2-B32 | 3710.8 / 3710.8 | 1316.8 / 1316.8 |
| DP2-B16 | 4332.8 / 4332.8 | 3210.8 / 3104.8 |

当前采集没有可靠的 worker-level Peak HBM、GPU 利用率或功耗。不同 Kaggle session 的初始显存也不完全相同，因此不能把结束时的空闲量当作配置本身的峰值显存。

## 并发口径

V3 的运行合同记录 `data_parallel_size=2`、`tensor_parallel_size=1`、`max_num_seqs=16`，Runner 实际每批提交最多 16 条。现有运行结果没有单独记录每个 replica 的并发上限或 DP 的全局有效并发，因此这里只报告配置值和实际 batch 大小，不把 DP2-B16 换算成 global concurrency 32。

## 下一步测量

如果继续完善这组推理实验，优先补齐：

1. 同一 workload 下的 1×T4 baseline，用它计算增加第二张 GPU 的 scaling efficiency。
2. worker 级 Peak HBM/GPU，以及批处理下的 P50/P95/P99 E2E latency 和 TTFT。
3. GPU utilization 与 input/prompt tokens per second；当前输出最多 8 tokens，output token rate 的解释力有限。
4. 用同一协议测量 Qwen3.5-4B QLoRA adapter 的 serving 性能，与 base model 区分报告。

## 复现范围与限制

- 当前 Git 版本包含 TP=2、`max_num_seqs=32` 的 vLLM 路径，可通过 `notebooks/qwen35-4b-vllm.ipynb` 运行。
- V3 的 DP=2 / TP=1 / `max_num_seqs=16` 来自另一工作区的 Kaggle 实测；本地 `outputs/qwen35_4b_test/qwen35_4b_test_vllm_v3/resolved_config.json` 记录了运行配置，但对应实现尚未同步进当前仓库。因此 V3 是运行结果记录，不能声称当前 checkout 可直接复现。
- 显存字段记录运行前与运行结束时的 used/free 容量，不是峰值；不同 Kaggle session 的后台占用可能不同。本次没有采集 GPU 利用率、功耗或实际账单成本。
- vLLM 批处理 run 的准确 batch 级明细由本地 `outputs/` 中的 `run_summary.json` 保存。`outputs/` 被 Git 忽略，不包含在公开仓库中。

汇总本地已保存的 run：

```powershell
python scripts/bench_report.py
python scripts/bench_report.py --baseline qwen35_4b_test
```

该脚本只汇总 `outputs/` 中已有的结果，不启动模型。要重新测量吞吐或显存，需要 Kaggle 2×T4 GPU session。
