# 环境锁

- `bootstrap.lock.txt`：固定 Kaggle Linux Python 3.11 环境使用的 `uv` wheel，并记录其 SHA-256；仅允许用 `--require-hashes --only-binary=:all:` 安装。
- `cpu.lock.txt`：CPU check/evaluate/submit 的依赖及构建工具；不包含 torch 或 bitsandbytes。
- `cloud.lock.txt`：Linux x86_64、Python 3.11、CUDA 12.6 的推理依赖及传递依赖，逐项固定版本和分发包哈希。
- `.in` 文件记录直接依赖。lock 由 uv 0.12.6 根据包索引元数据解析生成；本轮没有在本机安装 CUDA 依赖。
- `train.lock.txt`：在原 cloud lock 约束下解析的独立训练环境，只新增 PEFT 0.17.1；不修改已有推理锁。仅供云端训练环境安装，CPU 测试不需要它。
- `qwen35.lock.txt`：Qwen3.5-4B lane 的独立推理环境，固定 Transformers 5.17.0 + vLLM 0.19.1 + torch 2.10.0；不修改 7B baseline 的 cloud lock。

CPU 使用：`python -m pip install --require-hashes -r requirements/cpu.lock.txt`。
云端在独立 Python 3.11 环境使用：`python -m pip install --require-hashes --only-binary=:all: -r requirements/cloud.lock.txt`。
随后运行 `python -m pip install --no-deps --no-build-isolation -e .`。

解析命令：

```text
uv pip compile requirements/cpu.in --python-version 3.11 --default-index https://pypi.org/simple --generate-hashes --output-file requirements/cpu.lock.txt
uv pip compile requirements/cloud.in --python-version 3.11 --python-platform x86_64-unknown-linux-gnu --default-index https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match --generate-hashes --emit-index-url --output-file requirements/cloud.lock.txt
```

版本组合采用 [PyTorch 官方版本配对](https://docs.pytorch.org/get-started/previous-versions/)中的 torch 2.7.1 / torchvision 0.22.1 CUDA 12.6 和已发布的 [Transformers 4.57.6](https://pypi.org/project/transformers/4.57.6/)。其余包及传递依赖由解析器确定，详见 lock。

`qwen-vl-utils` 带入 av 是它的声明依赖；本项目只给它传递图像，未重新运行视频抽帧。云端锁包含 CUDA 库，仅供 Linux 云端使用。

依赖解析成功不等于 GPU 已验收。真实驱动兼容性、bitsandbytes 加载、显存和推理速度仍需在云端验证。Notebook 使用独立环境，避免与平台预装 torchaudio 等包混装，并保存实际包列表与环境检查结果。

- `train_qwen35.lock.txt`: independent Qwen3.5-4B QLoRA environment (Transformers 5.17.0 + vLLM 0.19.1 + PEFT 0.18.0 + bitsandbytes 0.49.2).

## vLLM 双卡说明

vLLM 固定为 **0.19.1**，因为它是最后一个**预编译二进制**链接 `libcudart.so.12` 的版本。

**关键：不能靠 metadata 判断。** `requires_dist` 里的 `nvidia-cutlass-dsl` 写法与 `.so` 链接哪个 `libcudart` 无关；CUDA 大版本只体现在编译产物里。从 PyPI wheel 中提取 `vllm/_C.abi3.so` 的 `DT_NEEDED`：

| vLLM | `vllm/_C.abi3.so` 链接 | CUDA 12.8 可用 |
|---|---|---|
| 0.19.1 | `libcudart.so.12` | ✅ |
| 0.20.2 | `libcudart.so.13` | ❌ |
| 0.21.0 | `libcudart.so.13` | ❌ |

**真正的分界是 0.20.0，不是 0.22.1。** 链接 `libcudart.so.13` 的 wheel 在 CUDA 12 主机上 import 即失败：

```text
ImportError: libcudart.so.13: cannot open shared object file
```

Kaggle T4 是 SM 7.5 + CUDA 12.8 驱动，宿主提供 `/usr/local/cuda-12.8/lib64/libcudart.so.12`，只有 CUDA 12 这条线可用。

vLLM 0.19.1 固定 `torch==2.10.0`；本 lane 用 cu128 构建以匹配宿主工具链（`torch.version.cuda == 12.8`）。它的 transformers 要求是 `>=4.56`，5.17.0 满足。0.19.1 的 `LLM.chat` 支持 `chat_template_kwargs` / `lora_request` / `mm_processor_kwargs`，`StructuredOutputsParams.choice` 与 `LLM()` 的全部构造参数（`tensor_parallel_size`、`enforce_eager`、`disable_custom_all_reduce`、`enable_prefix_caching`、`limit_mm_per_prompt`、`enable_lora` 等）均已核对存在。

> 教训：判断 CUDA 兼容性必须看 `.so` 的实际链接目标，不能只看依赖声明。`test_qwen35_training.py` 已加入 cu12/cu13 互斥门禁与 vLLM 版本门禁。

- **7B lane 不得安装 vLLM**：`cloud.lock.txt` / `train.lock.txt` 保持 torch 2.7.1 + Transformers 4.57.6，`package_cloud.py` 与 `package_training.py` 的白名单也不包含 `qwen35_vllm.py`。
- `vllm` 在 PyPI 上只有 `cp38-abi3` wheel，可被 CPython 3.11 直接安装；`torch==2.10.0+cu128` 位于 PyTorch cu128 索引。
- 结构化输出依赖 `xgrammar` / `outlines-core`，已包含在 qwen35 lock 中；这是 vLLM 实现答案空间约束的机制。
