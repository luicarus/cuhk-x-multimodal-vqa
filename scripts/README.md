# 辅助脚本

- `test.py`：运行 CPU 测试；关闭 bytecode/pytest 缓存，将临时文件放到系统临时目录并自动清除。
- `package_cloud.py`：白名单打包，原样包含用户维护的 `notebooks/cuhk-x-base7b.ipynb`；解包校验在临时目录进行。默认不保存额外验收报告。
- `package_training.py`：只生成包含完整五折缓存的独立后训练 ZIP；数据不齐时拒绝打包。
- `build_training_notebook.py`：只创建完整数据版后训练 Notebook，拒绝覆盖已存在文件，不操作 baseline Notebook。
- `package_qwen35.py`：生成独立 Qwen3.5-4B test 包 `artifacts/cloud/qwen35_4b.zip`，test 推理走 vLLM 0.19.1 双卡；默认 DP=2、Runner batch=32、每 replica `max_num_seqs=16`，包清单声明 `inference_engine: vllm_0.19.1_dual_gpu`；复用 test/pilot IR4 缓存，不包含训练栈和模型权重。
- `build_qwen35_notebook.py`：创建独立 Qwen3.5-4B vLLM test Notebook（`notebooks/qwen35-4b-vllm.ipynb`），拒绝覆盖已存在文件。
- `bench_report.py`：汇总各次运行的**基础设施指标**（吞吐、延迟分位、阶段拆分、逐条一致性），用于对比在不同引擎/并行策略下的表现；只读 `outputs/`，不修改任何运行结果。

业务操作统一使用 CLI，原 baseline Notebook 不提供生成器或覆盖入口。

- `package_qwen35_training.py`: builds the independent complete-data Qwen3.5-4B QLoRA package as `artifacts/cloud_training/qwen35_4b_qlora.zip`, declaring `inference_engine: vllm_0.19.1_dual_gpu`; adapter checks use DP=2 with the notebook's Runner batch default.
- `build_qwen35_training_notebook.py`: builds its reproducible vLLM dual-GPU cloud Notebook, `notebooks/qwen35-4b-qlora-vllm.ipynb`.

## 推理性能对比

项目关注的是服务效率而非答案准确率（准确率以 Kaggle 提交为准），因此对比口径是吞吐与延迟：

```powershell
python scripts/bench_report.py                                    # 所有已完成运行
python scripts/bench_report.py --baseline qwen35_4b_test          # 附逐条输出一致性
python scripts/bench_report.py --json reports/bench.json          # 另存机器可读结果
```

报告包含五类指标：

| 指标 | 说明 |
|---|---|
| 端到端吞吐 | `s/req`、`req/s`，**排除引擎加载时间**（那是可摊销的一次性成本） |
| 延迟分布 | p50/p90/p95/p99/max，分「全量」与「稳态」两个视图 |
| 阶段拆分 | `image_ms`（解码 4 张 JPEG）、`generate_ms`（模型调用）、`overhead_ms`（含 checkpoint 重写） |
| TTFT | 首 token 延迟分位 + token/s；**仅 vLLM 上报**，Transformers 是单次阻塞调用没有时间戳 |
| 显存 | 逐卡 total/used/free/allocated/reserved + 峰值，另有运行前后增长量 |
| 输出一致性 | 与基线逐条比对的相同率，用于确认换引擎没有改变模型行为 |

**关于 TTFT 的口径**：只对**真正上报了首 token 时间戳的请求**求平均，缺失的请求被排除而不是记为 0（记为 0 会得出虚高的漂亮数字），同时输出 `coverage` 说明覆盖率。在本 workload 下（`max_new_tokens=8`，答案 1~4 个 token）prefill 几乎就是全部，**TTFT 预期接近总延迟**——报告如实呈现，不假装是聊天场景的 profile。

> vLLM 的 `LLM` 会在调用方未显式传入时**强制 `disable_log_stats=True`**，此时 output processor 把 `RequestStateStats` 置为 `None`，**`first_token_latency` 根本不会产生**。因此本后端显式传 `disable_log_stats=False`；`VLLM_NO_USAGE_STATS` 只关闭远端上报，与引擎统计无关。

**关于显存**：同时记录驱动视角（`mem_get_info`，决定还能不能开大 batch）与分配器视角（`allocated` / `reserved`，两者差距大意味着碎片而非真缺显存，修法不同）。**逐卡记录**是因为 TP 下两卡不一定均衡，只看总量会掩盖不均衡。

> vLLM 的引擎与每个 TP rank 跑在**独立进程**里，从父进程读 `torch.cuda.memory_allocated()` **恒为 0**——只有驱动视角能跨进程。因此 worker 侧显存通过 `collective_rpc` **在 worker 进程内**采样，KV cache 大小（`available_kv_cache_memory_bytes`、`num_gpu_blocks × block_size`）也从 worker 取，因为引擎在自身 profiling 阶段算出后并不暴露给 `LLM` 对象。

引擎侧指标（TTFT、token 数）通过 `backend.last_metrics` 传递；后端不上报时视为「无数据」而非 0。显存采样失败会被吞掉并返回空字典——**profiling 永远不能成为预测失败的原因**。

**稳态视图跳过前 3 条请求**（`profiling.WARMUP_REQUESTS`）：首次调用包含 kernel 选择与视觉塔首次分配，混进去会低估真实吞吐。全量视图保留这段冷启动尾巴，因为那才是用户实际感受到的。

时序数据写在 `run_summary.json` 的 `latency` 字段，**不参与签名校验**：`_verify_finished` 只比对 signature、target_ids、输出哈希与 counts，因此增删指标不会让已完成的运行失效。

全量恢复（`--resume` 且无待处理请求）的运行不会执行任何请求，报告会标注 `resumed; nothing executed` 而不是给出虚构吞吐。
