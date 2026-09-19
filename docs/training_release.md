# 独立后训练包：完整数据版 v3

使用 `notebooks/cuhk-x-qlora-full-v3.ipynb`，配套 ZIP 为 `artifacts/cloud_training/cuhkx-ir4-qlora-full-v3.zip`。

The ZIP includes the PEFT 0.17.x compact `target_modules` compatibility fix and complete pinned-revision weight hashes. The packaging script prints the current ZIP SHA256; the archive is about 260 MiB.

本包内置五折训练缓存（4,087 QA、1,333 clips、10,664 张 IR JPEG）、原训练 QA/分折，以及 test/pilot 缓存。模型 revision、权重来源清单、代码、配置和依赖锁均固定。无需原 baseline ZIP、原视频或额外挂载五折缓存。

## 使用

1. 运行本地打包命令并保存其输出的 `manifest_sha256`。将 ZIP 挂载为私有输入，或挂载解压后的目录；把可信哈希填入 Notebook 的 `EXPECTED_MANIFEST_SHA256`，并可设置 `BUNDLE_INPUT` 为 ZIP 路径或包含 `training_bundle_manifest.json` 的目录。
2. 打开 `cuhk-x-qlora-full-v3.ipynb`，配置云端 GPU、网络和实验名。完整包没有 `TRAIN_CACHE_INPUT` 参数。
3. 依次执行。Notebook 先验证包、安装 CPU 依赖并复核完整数据指纹，通过后才安装 GPU 依赖和准备权重。
4. 先短跑并重载 adapter，再进行 dev 基座对照与正式训练。确认和测试默认关闭，选定候选后分别开启 `RUN_CONFIRMATION` 与 `RUN_TEST`。

权重可用 `WEIGHTS_INPUT` 指向已经验证的完整目录；也可从固定 revision 下载到 `/tmp/cuhkx_qlora_weights_<revision>/`。包中只有权重来源清单，没有数 GB 的权重本体。

## 数据与隔离

```text
training_repo/data/frames/
  fold_0/   873 QA / 281 clips / 2248 JPEG
  fold_1/   858 QA / 281 clips / 2248 JPEG
  fold_2/   862 QA / 279 clips / 2232 JPEG
  fold_3/   750 QA / 245 clips / 1960 JPEG
  fold_4/   744 QA / 247 clips / 1976 JPEG
  pilot/    原有 120 QA 的验证缓存
  test/     原有 682 QA 的测试缓存
```

每折内部继续使用 `uniform_time_v1/ir/520837f5b798f45a/`，不移动或重编码原图片。训练使用 fold 0–2 的 2,593 QA，dev 使用 fold 3 的 750 QA，confirm 使用 fold 4 去除 pilot 后的 624 QA。

新包的 `package_id` 为 `cuhkx-ir4-qlora-full-v3`，缓存模式为 `embedded_complete`。v3 修复了短跑结束时可能回载零初始化 warmup checkpoint，以及完成恢复时误判为“权重未更新”的问题。当前后训练发布只保留 v3；原 baseline 不受影响。

工作目录为 `/kaggle/working/cuhkx_qlora_<包清单哈希>/<EXPERIMENT>/`，使用自己的 Python 3.11 环境、训练产物和候选预测。不会写入原 `/kaggle/working/repo` 的 baseline 结果。

## 打包与复现

```powershell
python scripts/package_training.py
```

默认使用新的 v3 文件名。同名文件已存在时拒绝覆盖，可传 `--output <新路径>`。生成新的完整数据版 Notebook 使用 `python scripts/build_training_notebook.py --complete`，同样拒绝覆盖已存在文件。

完整包保存每个文件的 SHA256、数据指纹及训练分组覆盖。解包后必须得到相同数据指纹才允许继续。打包和测试都在系统临时目录验证，结束后清除，仅保留请求的交付文件，不生成阶段报告。

现有原始 `training_qa.csv` 和 `qa_folds.csv` 已描述所有训练题；包中不需要补齐报告、旧资产清单备份或每折重复派生 CSV。用户自己的恢复记录仍保留在本地。

## 验证边界

本地可验证数据、哈希、解包和 CPU 接口，不能替代 GPU 实训。模型仍需要实际 CUDA 资源和权重，NF4 反向传播与显存条件应先用短跑确认。保存真实 adapter、必要 checkpoint、预测/评测、环境信息和 session.json；固定输入不承诺跨驱动/硬件逐位一致。
