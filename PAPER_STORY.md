# Paper Story — CMCD-LoRA

## Title Direction
Cross-Modal Compensated Drift LoRA for Exemplar-Free HSI+LiDAR Cross-Domain Class-Incremental Learning

---

## Core Narrative

### 1. Observation (动机)
在 HSI+LiDAR 融合的跨域增量学习中，我们发现：
- 输出的光谱表征和空间表征都会漂移
- **空间表征漂移 ≈ 2× 光谱表征漂移**
- LiDAR 空间分支漂移 > HSI 空间分支漂移

→ 需要补充分支级 drift 可视化实验来支撑（spec vs hsi_spa vs lid_spa 的 cosine similarity 衰减曲线）

### 2. Method (方法)
基于漂移不对称观测，提出三个核心设计：

**A. 解耦 Backbone + 不对称 LoRA**
- Mamba 框架下将 HSI 显式拆分为光谱/空间两路，LiDAR 作为独立空间分支
- Warmup 后冻结 backbone，光谱分支作为稳定锚点
- 对 HSI 空间和 LiDAR 空间分支引入 LoRA，不对称低秩配置（LiDAR rank > HSI rank）
- 可塑性集中在漂移大的空间通道

**B. SHINE — Post-hoc Per-domain Whitening（我们提出的）**
- 每个域单独计算各分支的均值/方差
- Z-score 归一化后做 cosine NCM 分类
- 消除 query feature 和 prototype 之间的域偏移
- Related work: DSBN (CVPR'19), Feature Whitening (CVPR'19), Multi-Prototype (CVPR'25)

**C. DCR + Prototype 分类**
- Drift-Compensated Reconstruction 用于旧类 prototype 校正
- Prototype-based 分类替代全连接，效果显著更好

### 3. Results (结果)
**设定**: MTH 顺序（MUUFL → Trento → Houston），3 datasets × 3 tasks = 9 tasks 串行

| 方法 | Exemplar | Avg TAg |
|:---|:---:|:---:|
| Frozen | 0 | 59.2 |
| EWC | 0 | 54.3 |
| LwF | 0 | 61.2 |
| iCaRL | 5/cls | **86.2±2.2** |
| LUCIR | 5/cls | 85.4±3.8 |
| **Ours** | **0** | **83.7±0.9** |

- 无回放达到有回放方法的同等水平
- 方差比 iCaRL 和 LUCIR 都小（稳定性更好）

---

## Contribution Summary
1. 揭示了 HSI+LiDAR 跨域增量场景下"漂移不对称"现象（空间 >> 光谱），并据此设计了解耦-锚定-不对称适配框架
2. 提出 SHINE（per-domain whitening）解决跨域 prototype 偏移
3. 在无回放条件下达到与有回放方法（iCaRL/LUCIR）相当的精度，且稳定性更优

---

## Key Decisions
- **Ours = CMCD-LoRA (default) + SHINE**，报 83.7±0.9%
- SHINE 是我们方法的标配组件，baseline 不加 SHINE
- Mixtrain 变体 (84.0±0.7%) 放 supplementary
- 跨域设定作为实验设定，不强调为 benchmark

---

## Observation Experiment (已确定)

### 目的
证明"跨域场景下空间表征漂移 ≈ 2× 光谱表征漂移"是架构无关的普遍现象。

### 设置
| 项目 | 方案 |
|:---|:---|
| Task 划分 | 3 tasks, dataset-as-task, MTH (MUUFL→Trento→Houston) |
| CIL 策略 | Naive fine-tuning（无 replay、无 KD） |
| Backbone | Coupled CNNs (TGRS'20), HCT (TGRS'23), MAHiDFNet (InfoFusion'22), GAMF (ESWA'24) |
| 度量 | Linear CKA（主）+ Cosine Similarity（辅） |
| 锚点 | Task 0 的 checkpoint |
| Probe set | 三个域各一个测试集，每个 task 后全部测 |
| 种子 | 3 seed |
| 特征维度 | 各分支投影到 d=256 后计算 CKA/cosine |
| 训练/分类器 | 各 backbone 按各自原文设置 |

### 特征提取点
| Backbone | 光谱特征 | HSI 空间特征 | LiDAR 空间特征 |
|:---|:---|:---|:---|
| Coupled CNNs | HSI stream 浅层 band-mixing conv | HSI stream 深层 patch conv | LiDAR stream 输出 |
| HCT | Transformer token 输出 | CNN 特征金字塔输出 | LiDAR 分支 CNN 输出 |
| MAHiDFNet | 原生 spectral branch | 原生 spatial branch | 原生 LiDAR branch |
| GAMF | 光谱节点嵌入 | 空间邻域嵌入 | LiDAR 结构嵌入 |

### 可视化
- 主图：Line plot, x=Task, y=1-CKA (drift), 3条线(spec/hsi_spa/lid_spa), 4个backbone作subplot
- 辅助：Bar chart 各 backbone 的 drift ratio (spatial/spectral)
- Supplementary: t-SNE

---

---

## Baseline Comparison (已确定)

### 原则
- 所有 baseline 按原论文设定（backbone + CIL 策略 + 训练细节）
- 只做 HSI+LiDAR 输入和 9-task 协议所必需的最小适配
- SHINE 只属于 Ours，baseline 不加
- 全部 3-seed，MTH 顺序

### 主表方法（12 个）

| # | 方法 | 类型 | 期刊/年 | Exemplar | 备注 |
|:--|:---|:---|:---|:---:|:---|
| 1 | Naive fine-tune | 下界 | — | 0 | 最基础退化基线 |
| 2 | Frozen + prototype | 下界 | — | 0 | 冻结 backbone |
| 3 | EWC | 经典无回放 | PNAS 2017 | 0 | 正则化基线 |
| 4 | LwF | 经典无回放 | ECCV 2016 | 0 | 蒸馏基线 |
| 5 | ACIL | 现代无回放 | NeurIPS 2022 | 0 | 解析/原型基线 |
| 6 | PASS++ | 现代无回放 | TPAMI 2025 | 0 | 最新无回放 SOTA；若迁移不稳定则用 FCS (CVPR'24) 替补 |
| 7 | iCaRL | 回放 | CVPR 2017 | 20/cls | 经典 replay, NME 推理 |
| 8 | LUCIR | 回放 | CVPR 2019 | 20/cls | 经典 replay, cosine classifier |
| 9 | PODNet | 回放 | ECCV 2020 | 20/cls | 强蒸馏 + replay, 注明 CNN/NME |
| 10 | FOSTER | 回放 | ECCV 2022 | 20/cls | 强 replay, 原文默认 head |
| 11 | Joint training | 上界 | — | — | Oracle upper bound |
| — | **Ours (CMCD-LoRA+SHINE)** | 无回放 | — | **0** | 主方法 |

### Replay 协议
- 统一预算：K = 20 × C_total（32类 → 640 total exemplars）
- 分配：每 task 后 class-balanced shrink, m = K / seen_classes
- 选择策略：Herding
- 推理头：各方法按原文（iCaRL→NME, LUCIR→cosine, PODNet→注明, FOSTER→原文默认）
- 附录补 5/cls 低内存版本作为 memory ablation

### 暂不放主表
- **EASE** (CVPR'24), **RanPAC** (NeurIPS'23)：强依赖 ViT + ImageNet PTM，HSI+LiDAR 无公平的预训练对应物

### 替补
- **FCS** (CVPR'24)：若 PASS++ 迁移不稳定则替换

### 超参协议（三层分离）

**任务层（统一固定）**
| 参数 | 设定 |
|:---|:---|
| Patch size | 先用 Ours + Naive 在验证集 sweep {7,9,11,13,15}，确认趋势一致后锁死；若时间不够直接用 11×11 |
| Normalization | HSI 按 band 标准化，LiDAR 单独标准化 |
| Augmentation | 统一 flip + 90° rotation |
| Class order / split | 所有方法完全一致（MTH，固定 task 划分） |
| Seed | 统一 3 seed |

**方法层（按各自原文）**
| 参数 | 做法 |
|:---|:---|
| Optimizer / scheduler | 按原文 |
| Epoch 数 | 按原文 |
| Learning rate | 按原文 |
| 损失权重 (λ_kd, λ_ewc 等) | 按原文 |
| 分类器类型 | 按原文 |

**各方法参考训练设定**
| 方法 | 设定 |
|:---|:---|
| EWC / LwF / iCaRL / LUCIR | SGD + momentum 0.9, batch=128, lr=0.1, 160 epochs, step decay |
| PODNet | 跟官方实现；无完整 config 则跟 LUCIR 同 recipe |
| FOSTER | SGD, batch=128, lr=0.1, 170 epochs, cosine annealing |
| ACIL | 按原文 analytic 更新 |
| PASS++ | 按原文 |
| Ours | 正常调参，通过验证集选定后固定 |

**最小适配层**
- 只改输入接口（HSI 波段数 + LiDAR 通道），不改训练范式
- 如 ResNet-family 整体不收敛，对该类方法统一降 lr（如 0.1→0.01），不对单个 baseline 单独优化

---

---

## Ablation Study (已确定)

### 主表（6 个配置）

| # | 配置 | 改动 | 验证的 claim |
|:--|:---|:---|:---|
| 1 | Full method | — | 完整方法 |
| 2 | w/o SHINE | 去掉域对齐 | 域对齐的贡献 |
| 3 | w/o DCR | 去掉 prototype 校正，保留 prototype classifier | Prototype 校正的贡献 |
| 4 | Symmetric LoRA | HSI rank = LiDAR rank = 总预算/2（如各 rank=6） | 不对称适配的必要性 |
| 5 | Shared spatial adapter | HSI 空间和 LiDAR 空间共享同一个 LoRA | 解耦设计的必要性 |
| 6 | Spectral LoRA | 只在光谱分支加 LoRA，空间分支冻结 | 反向验证：应适配空间而非光谱 |

阅读顺序：域对齐 → 校正 → 适配策略 → 结构设计 → 核心假设验证

### Supplementary（2 个配置）

| 配置 | 改动 | 验证 |
|:---|:---|:---|
| w/o LoRA | 冻结空间分支，不加任何 LoRA | LoRA 适配本身的贡献 |
| FC head | 用 FC 分类器替换 prototype | Prototype vs FC |

---

---

## Experiment Logging Protocol (已确定)

### 目录结构
```
runs/
  {method_name}/
    {seed}/
      config.yaml                    # 完整配置 + git commit + 环境信息
      metrics_summary.json           # 最终汇总指标
      task_metrics.csv               # 每 task × 每 domain 的 OA/AA/Kappa/per-class
      predictions_task{t}_test.parquet  # 每样本: x,y,gt,pred,confidence,task_id,domain_id
      logits_task{t}_test.npz        # 完整 logits
      features_task{t}_test.npz      # 分类前一层特征（用于 t-SNE/漂移分析）
      confusion_task{t}.csv          # 混淆矩阵
      train_log.csv                  # 每 epoch: loss, val_acc, lr
      ckpt_task{t}.pt                # 每 task 后 checkpoint
```

### 必须记录
- 完整配置（方法名/seed/数据集/task划分/class order/patch size/epoch/lr/memory budget）
- 代码版本（git commit/分支/运行时间/CUDA/PyTorch 版本）
- 每 task 核心指标：OA/AA/Kappa、准确率矩阵、forgetting、BWT
- 每样本预测：x, y, gt_label, pred_label, confidence, task_id, domain_id, split
- 每 task checkpoint

### 强烈建议
- 完整 logits（用于 calibration / bias 分析）
- Feature embedding（用于 t-SNE / 漂移分析 / 类间距离）
- Confusion matrix（每 task + 最终）
- Per-class precision/recall/F1
- 训练曲线（每 epoch）
- Replay 方法的 exemplar index/class counts

### Cross-domain 特有
- task_metrics.csv 包含 task_id × eval_domain 矩阵
- 每个 task 后对所有已见域测试（不只是当前域）
- 保留每个 task 后的 test prediction（用于 classification map 演化图）

### Drift observation 额外
- 4 个第三方 backbone 每 task 后各分支 feature embedding 原始矩阵
- 用于计算 CKA / cosine drift

---

## TODO
- [ ] Patch size sweep（Ours + Naive 验证集）
- [ ] 实现实验 logging 框架
- [ ] 实现 4 个第三方 backbone 的 drift observation 实验
- [ ] 按原文实现 12 个 baseline 的 HSI+LiDAR 适配
- [ ] 全部 3-seed MTH 跑完
- [ ] Ablation 6 配置 + 2 supplementary
