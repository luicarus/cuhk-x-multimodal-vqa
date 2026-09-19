# 环境锁

- `bootstrap.lock.txt`：固定 Kaggle Linux Python 3.11 环境使用的 `uv` wheel，并记录其 SHA-256；仅允许用 `--require-hashes --only-binary=:all:` 安装。
- `cpu.lock.txt`：CPU check/evaluate/submit 的依赖及构建工具；不包含 torch 或 bitsandbytes。
- `cloud.lock.txt`：Linux x86_64、Python 3.11、CUDA 12.6 的推理依赖及传递依赖，逐项固定版本和分发包哈希。
- `.in` 文件记录直接依赖。lock 由 uv 0.12.6 根据包索引元数据解析生成；本轮没有在本机安装 CUDA 依赖。
- `train.lock.txt`：在原 cloud lock 约束下解析的独立训练环境，只新增 PEFT 0.17.1；不修改已有推理锁。仅供云端训练环境安装，CPU 测试不需要它。
- `qwen35.lock.txt`：Qwen3.5-4B test lane 的独立推理环境，固定 Transformers 5.17.0；不修改 7B baseline 的 cloud lock。

CPU 使用：`python -m pip install --require-hashes -r requirements/cpu.lock.txt`。
云端在独立 Python 3.11 环境使用：`python -m pip install --require-hashes --only-binary=:all: -r requirements/cloud.lock.txt`。
随后运行 `python -m pip install --no-deps --no-build-isolation -e .`。

解析命令：

```text
uv pip compile requirements/cpu.in --python-version 3.11 --default-index https://pypi.org/simple --generate-hashes --output-file requirements/cpu.lock.txt
uv pip compile requirements/cloud.in --python-version 3.11 --python-platform x86_64-unknown-linux-gnu --default-index https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu126 --index-strategy unsafe-best-match --generate-hashes --emit-index-url --output-file requirements/cloud.lock.txt
```

版本组合采用 [PyTorch 官方版本配对](https://docs.pytorch.org/get-started/previous-versions/)中的 torch 2.7.1 / torchvision 0.22.1 CUDA 12.6 和已发布的 [Transformers 4.57.6](https://pypi.org/project/transformers/4.57.6/)。其余包及传递依赖由解析器确定，详见 lock。

`qwen-vl-utils` 带入 av 是它的声明依赖；本项目只给它传递图像，未重新运行视频抽帧。云端锁包含 CUDA 库，仅供 Linux 云端使用。

依赖解析成功不等于 GPU 已验收。真实驱动兼容性、bitsandbytes 加载、显存和推理速度仍需在云端验证。Notebook 使用独立环境，避免与平台预装 torchaudio 等包混装，并保存实际包列表与环境检查结果。

- `train_qwen35.lock.txt`: independent Qwen3.5-4B QLoRA environment (Transformers 5.17.0 + PEFT).
