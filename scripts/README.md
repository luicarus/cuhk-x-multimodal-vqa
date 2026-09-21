# 辅助脚本

- `test.py`：运行 CPU 测试；关闭 bytecode/pytest 缓存，将临时文件放到系统临时目录并自动清除。
- `package_cloud.py`：白名单打包，原样包含用户维护的 `notebooks/cuhk-x-base7b.ipynb`；解包校验在临时目录进行。默认不保存额外验收报告。
- `package_training.py`：只生成包含完整五折缓存的独立后训练 ZIP；数据不齐时拒绝打包。
- `build_training_notebook.py`：只创建完整数据版后训练 Notebook，拒绝覆盖已存在文件，不操作 baseline Notebook。
- `package_qwen35.py`：生成独立 Qwen3.5-4B test 包，复用 test/pilot IR4 缓存，不包含模型权重。
- `build_qwen35_notebook.py`：创建独立 Qwen3.5-4B test Notebook，拒绝覆盖已存在文件。

业务操作统一使用 CLI，原 baseline Notebook 不提供生成器或覆盖入口。

- `package_qwen35_training.py`: builds the independent complete-data Qwen3.5-4B QLoRA package as `artifacts/cloud_training/qwen35_4b.zip`, declaring `inference_engine: vllm_0.24.0_tensor_parallel`.
- `build_qwen35_training_notebook.py`: builds its reproducible vLLM dual-GPU cloud Notebook, `notebooks/qwen35-4b-qlora-vllm.ipynb`.
