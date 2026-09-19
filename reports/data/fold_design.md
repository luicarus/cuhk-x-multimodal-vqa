# CUHK-X Subject-Grouped Fold 设计

> Fold 版本：`subject_grouped_v1`  
> 生成时间：2026-08-19T13:51:02+00:00  
> 状态：**PASS**

## 1. 设计原则

训练集的 HAU 与 HARn 共享同一批 18 个用户，并有 799 个重复的 `(subject_id, trial_id)`。因此主验证采用 5-fold subject-grouped split：同一用户在两个来源中的全部 clip，以及同一 clip 的全部 QA，始终位于同一 fold。

分配器只使用训练数据，并联合平衡 QA category、QA source、物理 clip source、HARn action 和模态覆盖。HARn 路径中的动作名仅用于平衡和审计，不会作为模型输入。

## 2. 冻结的用户分配

| Fold | Subjects | Subject 数 |
|---:|---|---:|
| 0 | user1, user18, user19, user21 | 4 |
| 1 | user2, user3, user20, user23 | 4 |
| 2 | user4, user5, user17, user24 | 4 |
| 3 | user6, user8, user9 | 3 |
| 4 | user7, user16, user22 | 3 |

一旦开始记录模型成绩，不得重新随机生成 folds；如必须修改，需要提升 fold 版本并保留旧结果。

## 3. 每折验证集规模

| Fold | Subjects | Clips | HAU clips | HARn clips | QA clips | QA rows | HAU QA | HARn QA | Val actions | Train actions |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 4 | 844 | 171 | 673 | 281 | 873 | 756 | 117 | 44 | 44 |
| 1 | 4 | 838 | 175 | 663 | 281 | 858 | 744 | 114 | 43 | 44 |
| 2 | 4 | 763 | 171 | 592 | 279 | 862 | 744 | 118 | 44 | 44 |
| 3 | 3 | 718 | 147 | 571 | 245 | 750 | 644 | 106 | 43 | 44 |
| 4 | 3 | 749 | 150 | 599 | 247 | 744 | 637 | 107 | 44 | 44 |

## 4. HARn action 覆盖

动作覆盖按其出现过的 subject 数计算理论上限。若某动作只属于少于 5 个 subject，它不可能出现在全部验证折；这种结构性缺失记为 `N/A`，不能当作模型错误或零分。

| Action | Subject coverage | Fold coverage | Theoretical max | Present folds | Missing folds |
|---|---:|---:|---:|---|---|
| Watch TV | 3 | 3 | 3 | 0, 2, 4 | 1, 3 |

每折缺少的验证动作：

- Fold 0：none
- Fold 1：Watch TV
- Fold 2：none
- Fold 3：Watch TV
- Fold 4：none

## 5. QA category 分布

| Fold | single | multi | combination | sequence | object_interaction | emotion |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 264 | 171 | 171 | 72 | 24 | 171 |
| 1 | 259 | 174 | 165 | 57 | 29 | 174 |
| 2 | 258 | 171 | 165 | 66 | 31 | 171 |
| 3 | 226 | 147 | 146 | 57 | 27 | 147 |
| 4 | 231 | 146 | 143 | 56 | 22 | 146 |

## 6. 用户级统计

| Subject | Fold | Clips | HAU | HARn | QA clips | QA rows | HAU QA | HARn QA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| user1 | 0 | 185 | 33 | 152 | 61 | 180 | 150 | 30 |
| user2 | 1 | 219 | 43 | 176 | 73 | 212 | 180 | 32 |
| user3 | 1 | 206 | 36 | 170 | 61 | 193 | 165 | 28 |
| user4 | 2 | 176 | 36 | 140 | 59 | 180 | 156 | 24 |
| user5 | 2 | 145 | 42 | 103 | 54 | 189 | 177 | 12 |
| user6 | 3 | 261 | 51 | 210 | 80 | 249 | 219 | 30 |
| user7 | 4 | 244 | 48 | 196 | 77 | 235 | 201 | 34 |
| user8 | 3 | 213 | 42 | 171 | 77 | 229 | 189 | 40 |
| user9 | 3 | 244 | 54 | 190 | 88 | 272 | 236 | 36 |
| user16 | 4 | 246 | 51 | 195 | 89 | 263 | 222 | 41 |
| user17 | 2 | 223 | 48 | 175 | 77 | 240 | 210 | 30 |
| user18 | 0 | 226 | 42 | 184 | 73 | 222 | 189 | 33 |
| user19 | 0 | 246 | 54 | 192 | 83 | 267 | 237 | 30 |
| user20 | 1 | 210 | 42 | 168 | 71 | 213 | 183 | 30 |
| user21 | 0 | 187 | 42 | 145 | 64 | 204 | 180 | 24 |
| user22 | 4 | 259 | 51 | 208 | 81 | 246 | 214 | 32 |
| user23 | 1 | 203 | 54 | 149 | 76 | 240 | 216 | 24 |
| user24 | 2 | 219 | 45 | 174 | 89 | 253 | 201 | 52 |

## 7. 质量门禁

| Check | Status | Observed | Expected |
|---|---|---|---|
| Every training subject is assigned once | PASS | 18 assignments | 18 subjects |
| Fold IDs are complete | PASS | [0, 1, 2, 3, 4] | [0, 1, 2, 3, 4] |
| Every training clip is assigned once | PASS | 3912 | 3912 |
| Every training QA is assigned once | PASS | 4087 | 4087 |
| A subject never crosses folds | PASS | 18 | 18 |
| All QA for one clip stay in one fold | PASS | 0 cross-fold QA rows | 0 cross-fold QA rows |
| Every fold's training complement retains all HARn actions | PASS | [44, 44, 44, 44, 44] | all 44 actions |
| Validation action coverage reaches its subject-limited maximum | PASS | none | no avoidable missing action-fold pairs |
| Fold subject counts differ by at most one | PASS | [4, 4, 4, 3, 3] | difference <= 1 |
| Every validation fold contains both sources | PASS | [['HARn', 'HAU'], ['HARn', 'HAU'], ['HARn', 'HAU'], ['HARn', 'HAU'], ['HARn', 'HAU']] | HAU and HARn in every fold |
| Every validation fold contains every QA category | PASS | [6, 6, 6, 6, 6] | all 6 categories |

## 8. 使用方式

对于验证 fold `k`：

- 训练：`fold != k`；
- 验证：`fold == k`；
- 最终全量训练：使用全部 18 个 subject，但不得用该结果回填 OOF 分数。

QA 训练脚本优先读取 `qa_folds.csv`；视频预训练或抽帧任务读取 `clip_folds.csv`。绝对不能再次对 QA 行做随机切分。

```powershell
conda activate CUHK-X
python scripts/build_folds.py
```

## 9. 版本与哈希

- 优化 seed：`20260819`
- 随机候选数：`20000`
- Objective：`0.19475144`
- Subject assignment SHA256：`0fab0259b525f0be885578bc324edd6612b54088ef063607348c7be462a0dd4a`
- Clip folds SHA256：`53e135ad7196690041f5ac47e5e333e247de8c9f047f2693961acb66a98c942b`
- QA folds SHA256：`fc12eb65d215e637fc76de2dc2951346faee3f35dd4de6fc5414507c4d0d1205`
