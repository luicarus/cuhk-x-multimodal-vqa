# Qwen3.5-4B vLLM 双卡推理实验

选择 Qwen3.5-4B 作为最终模型后，我在相同的 2×T4 环境中补充了 vLLM 推理实验，观察请求批处理、TP/DP 方式对吞吐、延迟和显存的影响。它是竞赛方案的推理工程延伸，不是独立 serving benchmark。

## 测试口径

- 工作负载：682 条 test QA；沿用 IR8 缓存，每条请求读取第 2/4/6/8 帧，输入尺寸 280×280。
- 模型：`Qwen/Qwen3.5-4B`，revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`。
- 生成：greedy decode、受约束答案空间、最多 8 tokens。
- vLLM 环境：vLLM 0.19.1、PyTorch 2.10.0+cu128、Transformers 5.17.0、2×Tesla T4。
- 同时保留 Transformers baseline 作为跨引擎参考；它使用不同的软件栈，不作为 TP/DP 的控制组。

## 配置与结果

| Run | 执行方式 | 加载时间 | 推理阶段吞吐 | 延迟 | 端到端耗时 | 运行后空闲显存 |
|---|---|---:|---:|---:|---:|---:|
| `qwen35_4b_test` | Transformers，batch 1 | 7.58 s | 未单独记录 | 未记录 | 768.62 s（0.887 req/s） | 未记录 |
| `qwen35_4b_test_vllm_v1` | vLLM TP=2，逐条请求 | 82.81 s | 1.1672 req/s | 均值 856.786 ms；P50 898.535；P95 1010.179；P99 1080.654 | 675.00 s（1.010 req/s） | GPU0/1：3.66 / 3.66 GiB |
| `qwen35_4b_test_vllm_v2` | vLLM TP=2，`max_num_seqs=32` | 83.24 s | 1.4303 req/s；2.9927 output tokens/s | 摊销 699.156 ms / 请求 | 567.83 s（1.201 req/s） | GPU0/1：1.29 / 1.29 GiB |
| `qwen35_4b_test_vllm_v3` | vLLM DP=2、TP=1，`max_num_seqs=16` | 182.53 s | 1.5490 req/s；3.2411 output tokens/s | 摊销 645.577 ms / 请求 | 630.41 s（1.082 req/s） | GPU0/1：3.14 / 3.03 GiB |

V2 由 22 个 batch 处理完 682 条请求，平均 batch 大小 31.0、平均 batch 时间 21.674 s；V3 使用 43 个 batch，平均 batch 大小 15.86、平均 batch 时间 10.239 s。这里的摊销耗时是批次执行时间除以该批请求数，不能当作单请求 P50/P95。两种批处理模式都没有保存可比较的 P95。

与 TP2-B32 相比，DP2-B16 的推理阶段吞吐提高 8.3%，摊销耗时降低 7.7%，推理总时间减少约 36.5 s。但它的模型加载多花约 99.3 s，因此 682 条请求的一次性端到端时间反而多 62.6 s。按双卡全程占用估算，含启动的 GPU 卡秒约为 TP2-B32 的 1665、DP2-B16 的 1849 / 千请求。DP 结束时每卡多保留约 1.75–1.85 GiB 空闲显存。

DP2-B16 与 TP2-B32 有 675/682 条预测一致，7 条输出不同；两者的 682 行提交文件均通过格式校验。test 标签不在本地，预测一致率和格式通过率不能替代 accuracy。准确率应以 Kaggle 评估为准。

## 复现范围与限制

- 当前 Git 版本包含 TP=2、`max_num_seqs=32` 的 vLLM 路径，可通过 `notebooks/qwen35-4b-vllm.ipynb` 运行。
- V3 的 DP=2 / TP=1 / `max_num_seqs=16` 来自另一工作区的 Kaggle 实测；本地 `outputs/qwen35_4b_test/qwen35_4b_test_vllm_v3/resolved_config.json` 记录了运行配置，但对应实现尚未同步进当前仓库。因此 V3 是运行结果记录，不能声称当前 checkout 可直接复现。
- 显存数值是运行前后的设备快照，不是峰值；不同 Kaggle session 的后台占用可能不同。本次没有采集 GPU 利用率、功耗或实际账单成本。
- vLLM 批处理 run 的准确 batch 级明细由本地 `outputs/` 中的 `run_summary.json` 保存。`outputs/` 被 Git 忽略，不包含在公开仓库中。

汇总本地已保存的 run：

```powershell
python scripts/bench_report.py
python scripts/bench_report.py --baseline qwen35_4b_test
```

该脚本只汇总 `outputs/` 中已有的结果，不启动模型。要重新测量吞吐或显存，需要 Kaggle 2×T4 GPU session。
