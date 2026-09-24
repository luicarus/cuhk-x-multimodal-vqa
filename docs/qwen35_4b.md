# Qwen3.5-4B 独立 test 对照

该 lane 只改变模型，保留 baseline 的 IR4 输入、8 帧缓存、`[1,3,5,7]` 选帧、280×280 处理尺寸、`cuhkx_mcq_v2` prompt、答案规范化、受约束解码和测试模板。

Qwen3.5-4B 官方模型卡使用 `AutoModelForMultimodalLM` 与 `AutoProcessor.apply_chat_template`，模型配置类型为 `qwen3_5`；官方说明要求较新的 Transformers。[模型卡](https://huggingface.co/Qwen/Qwen3.5-4B) · [Transformers 文档](https://huggingface.co/docs/transformers/model_doc/qwen3_5) 本项目为它单独锁定 Transformers 5.17.0，不改变 7B baseline 的 4.57.6 环境。

## 文件

- Notebook：`notebooks/qwen35-4b-vllm.ipynb`
- ZIP：`artifacts/cloud/qwen35_4b.zip`
- 配置：`configs/qwen35_4b.yaml`
- 后端（vLLM 生成）：`src/cuhkx/inference/qwen35_vllm.py`
- 后端（Transformers 参考）：`src/cuhkx/inference/qwen35.py`
- 权重来源校验：`src/cuhkx/inference/qwen35_weights.py`
- 依赖：`requirements/qwen35.lock.txt`

新包不包含模型权重，权重目录需要由 Notebook 下载或作为带 `cuhkx_qwen35_weights.json` 的私有 Input 挂载。模型 revision 首次由 Notebook 解析并写入运行副本；同一运行的 checkpoint 和结果只接受该 revision。

## vLLM 双卡推理

本 lane 的 test 推理默认使用 **vLLM 0.19.1、DP=2**：每个 T4 启动一份完整模型，`max_num_seqs=16` 限制每个 replica，Runner 总批次为 32。DP replicas 并行启动和处理各自的请求块。包清单声明 `inference_engine: vllm_0.19.1_dual_gpu`，Notebook 在解包前会拒绝不匹配的包。TP=2 仍可作为替代配置。这个包只做推理，不含训练栈；QLoRA 后训练见 `docs/qwen35_training.md`。

- 参考后端用有状态的 `prefix_allowed_tokens_fn` 约束解码，vLLM 没有该 hook，改用 `StructuredOutputsParams(choice=[...])` 并锁到相同的字面前缀，保证两条引擎的答案空间一致。
- 引擎构建前会用参考 processor 校验 chat 渲染（`enable_thinking=False`），因为答案边界依赖该精确编码。
- 两张 T4 没有 NVLink。默认 DP 将请求分配到两份独立 replica，不需要跨卡 all-reduce；切换到 TP=2 时，模型层跨 PCIe 通信并使用 `enforce_eager=True`、`disable_custom_all_reduce=True`、`NCCL_P2P_DISABLE=1`。
- 运行合同记录 `engine` / `engine_options`，同一 run-id 不会混用两种引擎的结果。

## 云端运行

将 ZIP 作为私有 Kaggle 输入，或直接使用解压后的内容。打包命令会输出 `manifest_sha256`；必须把该值从可信的本地终端复制到 Notebook 的 `EXPECTED_MANIFEST_SHA256`，然后再运行解包 Cell。Notebook 支持 ZIP 和带 `qwen35_bundle_manifest.json` 的目录：

```text
qwen35_repo/
  configs/qwen35_4b.yaml
  data/frames/test/...
  data/qa/test.csv
  qwen35_bundle_manifest.json
```

打开 Notebook 后按顺序执行。它使用自己的 `/kaggle/working/qwen35_runtime_<包哈希>/`，不会读写 baseline 的 `/kaggle/working/repo`、`ir4_runtime` 或 `outputs/ir4_7b_*`。

默认先运行 test 前 16 QA smoke，再以 Runner batch 32 运行完整 682 QA；smoke 的实际批次受 16 条限额约束：

```bash
cuhkx predict --profile qwen35 --backend vllm --tensor-parallel-size 1 --data-parallel-size 2 --max-num-seqs 16 --runner-batch-size 32 --dataset test --limit 16 --run-id qwen35_4b_smoke_dp2_b32_ttft --weights-dir <weights> --resume
cuhkx predict --profile qwen35 --backend vllm --tensor-parallel-size 1 --data-parallel-size 2 --max-num-seqs 16 --runner-batch-size 32 --dataset test --run-id qwen35_4b_test_dp2_b32_ttft --weights-dir <weights> --resume
cuhkx verify-run --profile qwen35 --run-id qwen35_4b_test_dp2_b32_ttft
cuhkx submit --profile qwen35 --run-id qwen35_4b_test_dp2_b32_ttft
```

Each new batched run records vLLM's per-request `first_token_latency` under `run_summary.json` as `latency.ttft_ms`, including sample count, coverage, mean, P50/P90/P95/P99, and QA-aligned samples. TTFT is measured by vLLM from request arrival and includes scheduler queue time; batch end-to-end latency remains a separate metric. The `_ttft` run-id suffix forces fresh measurements instead of resuming an earlier B32 result without TTFT samples.

The runner also polls NVML every 100 ms during prediction. `gpu_memory.peaks.device_polling` reports the per-device sampled peak used memory and minimum free memory across all processes, including the vLLM replicas. This is a device-wide sampled high-water mark, not a per-process allocator peak; brief spikes between samples may be missed. If NVML is unavailable, the run continues and records the monitor as unavailable.

测试集没有公开答案时，`submit` 只能检查 CSV 完整性，不能计算本地 accuracy。最终对比以相同比赛评估口径/公开榜分数为准。若需要本地 sanity，可把 `--dataset pilot` 作为额外运行，但它不替代 test。

## 重要兼容点

- 新模型不是 Qwen2.5-VL 类；不能调用 `Qwen2_5_VLForConditionalGeneration` 或 4.57.6 运行环境。
- Qwen3.5 使用 Transformers 5.17 的原生实现；权重目录不接受 `.py`/`auto_map`，模型和 processor 均固定 `trust_remote_code=False`。
- Qwen3.5 的 tokenizer EOS 与模型 generation EOS 可能不同；新后端以模型 generation config 的 EOS 集合约束解码，并把同一集合传给 `generate`。
- Qwen3.5 默认可能进入 thinking 模式；新后端通过 `enable_thinking=False` 固定为直接选项输出，避免 8 token 上限截断解释。
- 保留 `use_fast=False` 以继续使用 slow image processor；Transformers 5.17.0 会打印弃用警告，但该警告不影响运行。不能把 `backend="pil"` 传给整个 `AutoProcessor`，否则会被错误传入 Qwen3VL video processor。
- 新 lane 默认使用 FP16 全量权重，不使用 7B baseline 的 NF4 loader。4B 权重大小和 T4 显存需要以 smoke 实测为准，不自动切换量化或输入帧数。

## 复现记录

应下载 `qwen35_4b_test/` 下的 `resolved_config.json`、`input_check.json`、`resume_state.json`、`checkpoint.jsonl`、`predictions.csv`、`audit.jsonl` 和 `run_summary.json`，以及 `submission.csv`、`submission_validation.json`（若导出成功）。其中 `run_summary.json` 的 backend、revision 和环境版本是本次对比的身份记录。

请保留 `qwen35_4b_test` 原始 run-id，不覆盖 7B baseline。完整 test 运行成功且 CSV 校验通过后，才用最终平台分数与 7B baseline 比较；0.49/0.45 等用户提供的远端数字应在对应 Notebook 输出或结果文件中注明来源，不能写入另一模型的 run summary。

## 当前云端结果

根据用户提供的 Kaggle 提交页结果（2026-09-11），当前三条实验线为：

| 实验线 | Kaggle 分数 | 备注 |
|---|---:|---|
| Qwen3.5-4B IR4 | **0.42105** | 当前未后训练模型中领先 |
| Qwen2.5-VL-7B IR4 baseline | 0.36549 | 原始云端 baseline |
| Qwen2.5-VL-7B QLoRA | 0.43859 | 已后训练结果 |

来源：[Kaggle 提交记录](https://www.kaggle.com/competitions/cuhk-x-competition-large-model-track/submissions)。这些是外部平台分数，仅作实验记录；本地 `run_summary.json` 不写入平台分数。
