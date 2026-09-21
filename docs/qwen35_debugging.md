# Qwen3.5 Kaggle debugging 记录

适用入口：`notebooks/qwen35-4b-qlora-vllm.ipynb`。本文只记录本项目实际遇到并已修复的问题。

## 快速定位

`CalledProcessError` 或 Notebook 的 `RuntimeError: <command> exited with code 2` 只是外层包装。应从 `last output` 的最后一个 `Traceback` 开始看，并先确认报错属于安装、加载、训练、adapter 重载还是门禁阶段。

## 已修复问题

| 现象 | 原因 | 当前修复 |
|---|---|---|
| `markupsafe==3.0.3` 哈希不匹配 | 锁文件只记录了另一平台的 wheel 哈希 | 训练锁按 Python 3.11 Linux wheel 生成，并保留 Kaggle manylinux 哈希 |
| 找不到 `torch==2.7.1+cu126` | pip 只查询 PyPI，CUDA 本地版本位于 PyTorch 索引 | 锁文件和安装命令同时声明 PyPI 与 `https://download.pytorch.org/whl/cu126` |
| PEFT 与模型/Transformers 不兼容 | 初版使用 `peft==0.17.1` | 固定为 `peft==0.18.0`，并与 Transformers 5.17.0 一起锁定 |
| `Qwen3VLVideoProcessor requires Torchvision` | 训练依赖没有继承 Qwen3.5 视觉运行依赖 | `train_qwen35.in` 引用 `qwen35.in`，固定 `torchvision==0.22.1+cu126` |
| `TrainingArguments` 拒绝 warmup 参数 | Transformers 5 的参数名与旧版本不同 | 运行时检查 `TrainingArguments` 签名，选择 `warmup_ratio` 或 `warmup_steps` |
| `missing/non-finite LoRA gradients` | 自定义检查在 AMP `GradScaler` unscale 之前检查缩放梯度，把可恢复溢出当成失败 | 分开检查“梯度缺失”和“缩放溢出”；有 scaler 时让 AMP 决定跳步和降倍率 |
| 四步均为 `grad_norm: nan`，最终 adapter 仍为零 | 默认 FP16 scale `65536` 过高，四个 optimizer step 全被跳过 | Qwen3.5 使用 `init_scale=1.0`、`growth_interval=16`；结束时仍严格验证 LoRA B 非零且指纹变化 |
| `verify-run` 查找 `configs/training.yaml`，随后 confirm gate 失败 | Qwen3.5 Notebook 漏传训练配置，CLI 又采用 7B 默认路径 | 三处 `verify-run` 显式传入 `training_qwen35.yaml`；CLI 默认路径也改为按 profile 选择 |
| vLLM 与固定环境冲突 | vLLM 0.21.0 要求 `transformers>=5.5.3` 并固定 `torch==2.11.0` | 本 lane 升到 `torch==2.11.0+cu126` / `torchvision==0.26.0+cu126` 并重新锁定；7B lane 保持 2.7.1 + Transformers 4.57.6，不安装 vLLM |
| vLLM 对答案空间的约束与 Transformers 不一致 | vLLM 没有 `prefix_allowed_tokens_fn` 钩子 | 用 `StructuredOutputsParams(choice=[...])` 约束同一语言，并把每条答案固定为字面前缀；引擎启动前用参考 processor 校验 chat 渲染 |
| 双 T4 上 vLLM 张量并行卡死或崩溃 | 两块 T4 无 NVLink，PCIe 上的 CUDA graph 捕获与 peer-to-peer 探测不稳定 | `enforce_eager=True`、`disable_custom_all_reduce=True`、`NCCL_P2P_DISABLE=1`，并用 `gpu_memory_utilization=0.80` 给 KV cache 留边界 |

7B adapter 重载还遇到过 PEFT 将完整 target path 压缩成 `q_proj`/`v_proj` 后 provenance 校验失败。当前校验接受这种等价序列化，同时仍拒绝扩大的 target 范围。

## 不是致命错误的提示

下面这些日志本身不表示运行失败：

- `use_fast` 已弃用：processor 兼容提示。
- tokenizer 的 PAD/BOS/EOS 与 config 对齐：Transformers 自动同步 token ID。
- gradient checkpointing 将 `use_cache=False`：训练所需行为。
- `causal_conv1d`、`flash-linear-attention` 未安装而使用 PyTorch fallback：速度较慢，不能单独证明结果错误。

## confirm 门禁

`RUN_CONFIRMATION=True` 只开启 confirm 评估。最后一个 Cell 还要求 adapter 的 confirm accuracy 严格高于 baseline：

```python
CONFIRMED = accuracy("qwen35_pt_adapter_confirm") > accuracy("qwen35_pt_base_confirm")
```

正确顺序是：执行配置 Cell；重新执行“完整 SFT、dev 选择和 confirm 门禁”Cell；确认输出 `confirm improved: True`；再执行 test Cell。已有训练和 dev 输出由 `--resume` 复用。

旧包的当前 Kaggle Session 若遇到配置路径错误，可在三条 `verify-run` 中补入：

```python
"--training-config", str(TRAINING_CONFIG),
```

不要在保留现有训练结果的 Session 中途更换 ZIP。新 ZIP 的清单哈希会创建新的 `RUNTIME`，原运行结果仍在旧目录。

## vLLM 双卡相关提示

- vLLM 只做推理。Notebook 中的 `cloud("train", ...)` 不带 `--backend vllm`；训练仍由 Transformers 完成。
- 首次加载 vLLM 需要编译 kernel，耗时明显长于后续调用，属正常现象。
- `VLLM_USE_FLASHINFER_SAMPLER=0`：关闭 FlashInfer sampler，避免在 T4 上额外的 JIT 依赖。
- 若 workspace 只读，`TRITON_CACHE_DIR` / `TORCHINDUCTOR_CACHE_DIR` 已指向 `/tmp`。

## 修复后的最小验证

每次改动后只需按影响范围验证：

1. 运行 `tests/test_qwen35_training.py`。
2. 重新生成 Notebook，并确认所有代码 Cell 可编译。
3. 重新打包 ZIP，检查 14,835 个文件及 test、pilot、五折训练缓存。
4. GPU 相关修复在 Kaggle 先跑四步 smoke 与 vLLM TP=2 短跑，再开始完整 SFT。

成功训练至少应满足：训练 loss 有限、出现真实 optimizer 更新、LoRA B 非零、adapter 可重载。当前已记录的 dev 结果为 baseline `0.4466667`、adapter `0.5413333`，来源为用户完成的 Kaggle 运行。
