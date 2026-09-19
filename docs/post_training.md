# IR4 + 7B 后训练计划

日期：2026-09-10。状态：五折缓存已补齐并通过完整检查，训练脚本已适配实际目录；尚未运行 GPU 训练。

本文件是唯一后训练实施计划，不生成阶段验收报告。已有推理 Notebook 保持不变，本机只执行 CPU 测试。

完整数据版入口为 `notebooks/cuhk-x-qlora-full-v3.ipynb`，配套 `cuhkx-ir4-qlora-full-v3.zip` 内置五折缓存，无需另挂训练数据。挂载方式、固定 revision 与独立运行目录见 [训练包说明](training_release.md)。原 baseline Notebook 保持不变。

## 现在可用的脚本

`configs/training.yaml` 独立维护数据路径和训练参数，默认缓存位置如下（可按实际位置编辑 caches 列表）：

```text
data/frames/fold_0/uniform_time_v1/ir/520837f5b798f45a/frame_index.csv
data/frames/fold_1/uniform_time_v1/ir/520837f5b798f45a/frame_index.csv
data/frames/fold_2/uniform_time_v1/ir/520837f5b798f45a/frame_index.csv
data/frames/fold_3/uniform_time_v1/ir/520837f5b798f45a/frame_index.csv
data/frames/fold_4/uniform_time_v1/ir/520837f5b798f45a/frame_index.csv
```

每套缓存保留原内部路径、metadata 和 8 张图。不要同时配置互相重叠的完整缓存与 pilot 子缓存，重复 clip/QA 会被拒绝。

当前可运行：

```powershell
cuhkx training-check
python scripts/test.py -q
```

数据不齐时 training-check 返回 INCOMPLETE 和非零退出码，只打印缺失清单，不写文件。train 在缓存检查通过前不读取模型、不创建训练产物。正式训练至少需要 train 和 dev 完整；确认集可稍后补齐，但更换数据会改变运行指纹，不得在训练中途修改缓存。

独立完整包打包：`python scripts/package_training.py`。该命令要求三组缓存完整，打包和解包时重新核验，仍不抽帧。默认输出新的完整数据版文件名，存在同名文件时拒绝覆盖。

云端另建训练环境，安装 `requirements/train.lock.txt`（完整固定版本和哈希；在 cloud lock 约束下只增加 PEFT），不要在已跑通推理环境里升级依赖。GPU 内存与真实库兼容性仍需短跑验证。安装后执行：

```bash
# 先做最多 20 个 optimizer steps 的功能检查；使用独立 run-id。
cuhkx train --weights-dir <verified-base-weights> --run-id qlora_smoke --smoke-steps 4 --gpu 0

# 正式训练，已有完整 checkpoint 时才恢复 optimizer/scheduler/scaler/RNG。
cuhkx train --weights-dir <verified-base-weights> --run-id qlora_v1 --gpu 0 --resume

# 在同一个 dev 集合比较基座和 adapter。
cuhkx evaluate-training --split dev --weights-dir <verified-base-weights> --run-id base_dev --resume
cuhkx evaluate-training --split dev --weights-dir <verified-base-weights> --adapter-dir artifacts/training/qlora_v1/adapter --run-id qlora_dev --resume

# 确定候选后，先后对同一 confirm 集合运行基座和 adapter。
cuhkx evaluate-training --split confirm --weights-dir <verified-base-weights> --run-id base_confirm --resume
cuhkx evaluate-training --split confirm --weights-dir <verified-base-weights> --adapter-dir artifacts/training/qlora_v1/adapter --run-id qlora_confirm --resume

# 新候选测试预测；原 baseline 不传 --adapter-dir。
cuhkx predict --dataset test --weights-dir <verified-base-weights> --adapter-dir artifacts/training/qlora_v1/adapter --run-id qlora_test --resume
cuhkx submit --run-id qlora_test
```

命令支持 --project-root/--data-root，训练相关命令还支持 --training-config。verify-run 也可核验 dev/confirm 结果（自定义训练配置时同样传 --training-config）。

训练仅单进程单卡；--gpu 控制可见 GPU，必须在新进程启动。每个 epoch 做生成式 dev accuracy 评测并保存可恢复 checkpoint，保留 best/latest；最终 adapter 在 `artifacts/training/<run-id>/adapter/`。smoke 仅使用训练/开发各前 16 个样本，不作为性能对照；其 adapter 带 smoke 标记，无法导出正式提交。

训练中断时只恢复带完整哈希清单的 checkpoint，缺失 optimizer/scaler/RNG 等状态的半成品会跳过。完整恢复要求保持软件、参数、数据和运行目录；若更换云端目录，先保持原 checkpoint 路径布局，不能把仅有 adapter 的文件夹当训练恢复点。保存、重新加载后的小样本预测应在云端实际检查；CPU 测试不代表 NF4 反向传播已通过。

## 1. 首轮目标与边界

以本次真实运行的 `ir4_7b_v1` 为固定对照，先进行监督微调（SFT），采用 NF4 基座上的 LoRA，即 QLoRA。只学习类别合法的选项答案，不生成推理解释或合成思维链。

固定基座 revision、IR 8 帧缓存中的 `[1,3,5,7]`、280×280 视觉处理、`cuhkx_mcq_v2` prompt、答案规范化和受约束解码。首轮不同时改变数据模态、选帧、prompt 或评测规则。

训练产出独立 adapter，不覆盖基座权重、baseline 目录或已跑通的 Notebook。暂不引入全参数训练、DPO/GRPO、蒸馏、伪标签和测试集自训练。

## 2. 当前数据条件：五折 IR8 QA 缓存已齐备

2026-09-09 已从重构前的数据目录导入 fold 0–4 的既有 IR8 缓存，未重新 EDA 或抽帧。当前实际覆盖如下，clip 数仅统计有训练 QA 的视频片段：

| Fold | QA | Clips | IR JPEG |
|---|---:|---:|---:|
| 0 | 873 | 281 | 2,248 |
| 1 | 858 | 281 | 2,248 |
| 2 | 862 | 279 | 2,232 |
| 3 | 750 | 245 | 1,960 |
| 4 | 744 | 247 | 1,976 |
| 合计 | 4,087 | 1,333 | 10,664 |

五折覆盖全部 18 位训练 subject，4,087 个 QA ID 无重复，缺少 IR8 缓存的训练 QA 为 0。该总量是五折可用数据规模；本轮实际训练仍只使用第 3 节规定的 fold 0、1、2，共 2,593 QA。

数据入口（`k = 0,1,2,3,4`）：

- 完整带答案 QA：`data/references/fold_k_answers.csv`，用于构建训练/评测视图。
- 无答案推理输入：`data/qa/fold_k.csv`。
- 缓存根目录：`data/frames/fold_k/`；索引位于其下的 `uniform_time_v1/ir/520837f5b798f45a/frame_index.csv`，索引内部路径相对于该折缓存根目录。
- 冻结划分：`data/references/folds/subject_grouped_v1/qa_folds.csv`。
- 资产哈希与五折绑定：`data/asset_manifest.json` 的 `files` 和 `supplemental_training_folds.bindings`。

图像与 metadata 均保留原字节。完整 fold4 索引由原 pilot/remainder 索引合并，77 个重叠 clip 的索引行与文件逐一核对一致后去重。各折 QA 已与冻结划分及缓存关联核对，并通过 SHA256、全量 JPEG 解码和 metadata/时间字段校验；原 test/pilot 校验也通过。来源和复核方式见 [数据来源及补齐记录](data_provenance.md)。

原 pilot 的 120 QA 保留为完整 fold4 的子集，不额外计入总量，也不用于训练。训练读取器会核对已登记资产的哈希，拒绝注册后发生的修改；未登记的外挂缓存仍须通过结构、图片和关联校验，并纳入运行数据指纹。

本地数据文件被 Git 忽略；迁移到云端时应使用 `scripts/package_training.py` 生成独立完整训练包。任何传输后的缺帧或协议不一致应明确报错，不静默缩减训练集或重新抽帧。

## 3. 数据划分

沿用 `subject_grouped_v1`，不随机按 QA 划分，不重新生成 folds。同一 subject 和同一 clip 的所有 QA 必须在同一组。

| 用途 | 原有 fold | 已核实可用 QA 数 |
|---|---|---:|
| 训练 | 0、1、2 | 2,593 |
| 开发验证：选 epoch/配置 | 3 | 750 |
| 固定确认：选定候选后再看 | 4 去除既有 pilot 120 QA | 624 |
| 已有 pilot | fold 4 中既有 120 QA | 仅保留原基线诊断，不用于训练 |

数量已与冻结分折及恢复缓存核对；训练实现时仍需生成对应数据视图，并在运行环境复核覆盖与分组隔离。确认集按完整 fold4 的 QA ID 排除既有 pilot 的 120 个 QA ID，保留 624 QA；pilot 与确认集可能共享 clip，二者不能当作相互独立的验证组。fold 4 属于相对训练集独立的受试者组，但历史实验已使用过这些数据，不能称为从未看过的盲测集。确认集一旦被用于反复调参，就失去本轮独立确认用途。

当前 baseline 的 pilot 120 QA 分数不能直接与新验证集 750 QA 分数比较。需用未微调的同一基座在 dev/确认集合上补跑一次推理，之后按同一批 QA 做逐题对照；这不涉及重新抽帧。

正式测试 682 QA 只用于候选选定后的预测，不使用测试答案或伪标签训练，不靠反复提交 Public LB 选超参数。

## 4. 训练样本与 loss

一个样本是一道 QA，包含四张现有 IR 图片、与 baseline 完全相同的 user prompt，以及规范化的 assistant 答案。

- 单选类目标为单个有效字母；multi 为按字母排序的非空子集；sequence 为完整排列，保持顺序语义。
- question/options 只进入 user 内容；answer 只进入 assistant 目标，不混入路径、动作目录或身份信息。
- loss 仅作用于 assistant 答案和正确的结束 token。user、图片占位符、视觉 token 和 padding 均设为 ignore index。
- mask 必须按完整多模态模板和实际 processor 输出构造并验证，不能用字符串长度或不加验证的独立 token 数切分。若 pad 与 EOS 共用 token ID，不能把真实 EOS 一起屏蔽。
- 检查解码后的监督 token 恰好对应答案及结束符，每个样本至少有一个监督 token。训练用 teacher forcing，不把推理约束函数施加到 loss 上。
- 不采用默认长度截断破坏图像 token；先确认序列长度分布，超长输入明确报错。首轮不做 packing、选项随机置换或重复采样加权。

首轮采用标准答案 token 平均交叉熵，记录该口径，不同时试验题型加权。按实际存在的可用选项规范化标签。

## 5. 模型与资源方案

冻结基座、视觉编码器、视觉合并/投影模块和 lm_head，只在语言解码器的 attention q_proj/v_proj 加 LoRA。实际模块全名必须从锁定 revision 的模型中核实；不能用不限定范围的 all-linear 把视觉模块一起训练。

建议首轮起点（实验超参数，不是已验证最优值）：

| 参数 | 起点 |
|---|---|
| 基座 | 与 baseline 相同 revision，NF4 double quant |
| 计算精度 | 现有 T4 环境使用 FP16，不假设支持 BF16 |
| LoRA | rank=8、alpha=16、dropout=0.05 |
| micro batch / 累积 | 1 / 16（单卡有效 batch 16） |
| optimizer / LR | AdamW / 5e-5 |
| warmup / clipping | 0.03 / 1.0 |
| epoch | 先 1，最多 2；由 dev 选 checkpoint |
| 内存选项 | gradient checkpointing，训练 use_cache=false |
| seed | 20260909 |

先在现有云端单张 T4 上做真实前向、反向和 optimizer step 的内存试验，不承诺 16 GB 一定足够。两张 T4 的显存不会自动合并成一张 32 GB 卡；首轮不增加分布式训练复杂度。

现有推理器的 `device_map=auto` 和 CPU/disk offload 不能直接作为训练加载方案。训练使用专用模型初始化，显式单卡放置、量化训练准备和梯度检查。若仍 OOM，先停止记录条件，再调整训练方案或资源；不能悄悄降帧/缩图，导致对照不再公平。

训练依赖单独放在 `requirements/train.in`/`train.lock.txt`，尽量继承已跑通的 torch/transformers/bitsandbytes 版本，只增加兼容的 PEFT 等必要依赖；解析和云端短跑后再固定。不修改推理环境锁，不强制升级已跑通 Notebook 的依赖。

## 6. 分阶段实施与通过条件

| 阶段 | 工作 | 通过条件 |
|---|---|---|
| T0 基线归档 | 已完成；保留真实 config、环境、权重来源和运行记录 | baseline 运行核验通过，不根据 Notebook 显示文字伪造完整结果 |
| T1 输入准备 | 已完成；导入缓存、绑定原 folds，并生成固定训练/开发验证/确认数据视图 | 训练、dev、fold4 之间 subject/clip 无交叉；QA 唯一、图片完整，规模分别为 2,593 / 750 / 624（确认）及 120（pilot） |
| T2 CPU 实现 | 已完成；数据集、collator、mask、adapter 元数据、恢复逻辑与 CLI 已有测试 | 不下载大权重、不加载 GPU；测试全部在系统临时目录并清除 |
| T3 云端功能验证 | 从训练折取小批样本，若干 optimizer step，保存/重新加载 adapter | loss/梯度有限，LoRA 确实更新，冻结模块不更新，四帧和监督 mask 正确 |
| T4 正式首轮 | 训练折运行 1–2 epoch，dev 用生成式 exact match 选 checkpoint | 同期基座 dev 预测齐备；invalid/failed=0；记录逐题差异与分层表现 |
| T5 固定确认 | 配置选定后，在确认集合比较 baseline 与 adapter | 同 QA、帧、revision、prompt 和解码；无输入泄漏或结果混用 |
| T6 测试候选 | 确认有收益才对 test 682 QA 生成新候选 | 新 run-id、新 submission，严格校验，原 baseline 不覆盖 |

T3 必须覆盖保存/重载、训练中断恢复、重新开始后的步数和样本顺序；不能只看到 loss 下降就认为训练链路正确。可检查一个小训练子集是否能明显拟合，但这只作为功能测试，不作为泛化结果。

首轮不做大规模超参扫描。若需要第二轮，只调整一个主要变量（优先学习率），并在确认集评估前选定；不得看过确认结果后继续针对它调参。第一轮不再用全部 folds 重训，以免混淆调参与确认边界。

## 7. 判断是否优于 baseline

主指标为相同 QA 集合上的 question-level exact-match accuracy，补充每类、每来源、每 subject 的准确率和配对胜负数。训练 loss 仅用于诊断。

至少要求无无效输出/运行失败，dev 改善且固定确认集合上准确率也高于基座。检查收益是否只集中于少数题型或单个 subject，以及其它组是否明显退化；不承诺训练必然提高成绩。小幅差异要如实报告，有限 subject 数不能支撑夸大的统计结论。

如果未达到上述条件，保留原 baseline 作为主结果，adapter 仅保留为未胜出的实验。若预算允许，可对选定配置做另一随机种子复验，但不能用确认集反复选种子。

## 8. 工程接入与文件边界

实现时最小增量：

```text
configs/training.yaml
requirements/train.in、train.lock.txt
src/cuhkx/training/{dataset,collator,trainer}.py
notebooks/post_train.ipynb                 # 独立训练入口，之后才创建
```

优先使用现有 Transformers Trainer + 专用多模态 collator + PEFT；如需 TRL，先核对它与当前推理版本的兼容性，不为了套模板升级整个工程。

增加训练入口和可选 adapter 推理接口。当前 config 校验只接受固定基线，runner 签名也针对基座；实现时必须显式扩展 adapter 身份、权重哈希、基座 revision、LoRA 配置和训练数据指纹，不能只在加载模型后偷偷附加 adapter。

无 adapter 时保持已有 baseline 签名和验证行为不变，确保现有 Notebook、结果和 verify-run 可继续使用。adapter 运行使用独立 run-id；加载不匹配基座的 adapter 直接失败，不合并覆盖原基座。

真实训练成果仅放在 `artifacts/training/<run-id>/`：最终 adapter、配置/数据身份、精简指标和必要恢复 checkpoint。恢复 checkpoint 应包含 optimizer/scheduler/scaler/RNG 状态；只保存 adapter 不等于可以精确恢复训练。默认保留 best 与 latest，避免大量中间 checkpoint。

正式候选的预测仍放 `outputs/<candidate-run-id>/`。CPU 测试、临时模型和打包试验全部使用系统临时目录，不写 reports/docs/outputs 下的测试产物；本计划之外不新增阶段汇报文件。

## 9. 技术依据与待确认条件

量化基座上的 LoRA 训练和量化训练准备参考 [PEFT 量化指南](https://huggingface.co/docs/peft/developer_guides/quantization)。NF4 的训练用途以及 device_map=auto 的推理限制参考 [Transformers bitsandbytes 指南](https://huggingface.co/docs/transformers/quantization/bitsandbytes)。多模态训练的长度截断风险和 completion/assistant loss 接口参考 [TRL SFT 指南](https://huggingface.co/docs/trl/sft_trainer)；本项目仍需针对实际模板验证 mask。

baseline 结果与全部五折缓存已就绪。正式训练前仍需完成云端单卡反向传播、optimizer step、保存/重载的实际验证，确认内存条件；数据检查通过不等于 GPU 训练通过。
