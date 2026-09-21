# 云端运行

以 `notebooks/cuhk-x-base7b.ipynb` 中已验证的 baseline 流程为准。打包脚本只生成独立 ZIP，不执行或自动重写 Notebook。

## 输入与运行

1. 在可信的本地环境运行 `python scripts/package_cloud.py --output <new-package.zip>`，保存命令输出的 `manifest_sha256`。
2. 将 ZIP 挂载为 Kaggle Dataset 输入；也可以挂载已解压的包目录。解压后的包根目录应同时包含 `bundle_manifest.json` 和 `repo/`。数据包含 pilot 答案，应保持私有。
3. 在 Notebook 的解包 Cell 中，把本地命令输出的摘要填入 `EXPECTED_MANIFEST_SHA256`。若输入中有多个匹配包，再设置 `BUNDLE_INPUT` 指向指定 ZIP 或解压目录；不要从包内清单复制期望摘要。

Notebook 在复制包内容前验证外部 Manifest SHA-256、归档路径、文件数量与大小限制，以及每个文件的 SHA-256。验证通过后，代码复制到 `/kaggle/working/ir4_source_<hash>/repo`，并在 `/kaggle/working/ir4_runtime` 中创建运行环境。模型 revision 会在云端解析并固定；权重优先从带来源收据的 Kaggle Input 读取，否则下载到工作目录之外的缓存。
## 脚本接口

Notebook 调用同一套 CLI：

```bash
cuhkx check --dataset test
cuhkx fetch-weights --weights-dir <weights>
cuhkx predict --dataset pilot --limit 16 --run-id ir4_7b_smoke --weights-dir <weights> --resume
cuhkx predict --dataset pilot --run-id ir4_7b_pilot --weights-dir <weights> --resume
cuhkx evaluate --run-id ir4_7b_pilot
cuhkx verify-run --run-id ir4_7b_pilot
cuhkx predict --dataset test --run-id ir4_7b_test --weights-dir <weights> --resume
cuhkx submit --run-id ir4_7b_test
```

支持 `--project-root`、`--data-root`。模型只在云端执行，本地 check/evaluate/verify-run/submit 不加载 GPU 模型。

运行固定 IR 缓存 8 帧、输入第 2/4/6/8 张、视觉处理 280×280、Qwen2.5-VL-7B NF4、受约束解码。缺帧、错误配置或输入变更会失败，不自动降为其他模型/帧数。pilot 仅评分，不设置旧基线比较门槛。

## 打包

```powershell
python scripts/package_cloud.py --output <new-package.zip>
```

仅收集新链路白名单及必需缓存；Notebook 按原字节入包，包含用户保留的执行记录。原有旧 ZIP 不会自动更新或覆盖，上传前应重新打包。打包默认只保留指定 ZIP，校验结果打印到终端；确有需要时才用 `--verification-report <path>` 保存报告。

打包过程的解包和 CPU 检查发生在系统临时目录中。它不执行 Notebook 的下载、环境安装或模型 cells。

## 结果与测试

真实预测所需的 checkpoint、audit、metrics 和 submission 位于云端 `repo/outputs/<run-id>/`，用于恢复和结果核验，应保留。`submit` 只导出文件，不上传。

本地测试使用 `python scripts/test.py -q`，临时结果和 pytest 数据放在系统临时目录，完成后清除，不写入项目 reports/docs/outputs。只在终端汇报测试结果，不生成阶段验收文档。

Qwen3.5-4B 对照使用独立的 `notebooks/qwen35-4b-test.ipynb` 和 `artifacts/cloud/qwen35_4b.zip`，test 推理走 vLLM 0.24.0 双卡张量并行；它不改变本 Notebook、7B 权重或 baseline 运行目录。Qwen3.5 需要单独的 `requirements/qwen35.lock.txt`，不能把 7B 的 4.57.6 环境直接换模型。

用户提供的 Notebook 已保存成功的 cloud smoke、pilot 评测和测试提交导出输出；本地没有对应完整权重或运行目录，不把这些输出替代为本机 GPU 验证。
