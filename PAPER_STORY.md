# Paper Story — CMCD-LoRA

## Title Direction
Cross-Modal Compensated Drift LoRA for Exemplar-Free HSI+LiDAR Cross-Domain Class-Incremental Learning

---

## Core Narrative

### 1. Observation (动机)

**两层发现：**

**第一层 — 漂移不对称现象：** 在独立双分支架构（spectral/spatial 无共享参数）中，跨域增量学习会导致 spatial drift 显著大于 spectral drift：

| 架构类型 | Backbone | Drift Ratio | 分支独立性 |
|:---|:---|:---|:---|
| 独立双分支 | HCT (TGRS'23) | **5.66±0.99×** | spectral/spatial 完全独立 |
| 独立三分支 | S2CM (Ours) | **4.35±2.41×** | Mamba spectral + VSSBlock spatial，无共享 |
| 原生三分支 | MAHiDFNet (InfoFusion'22) | **1.57±0.49×** | 1D spectral + 2D spatial，独立 CNN |
| 部分共享 | Coupled CNNs (TGRS'20) | 1.36±0.13× | conv2/conv3 跨分支共享（51%参数） |
| 高度耦合 | S2ENet (TGRS'22) | 1.01±0.02× | spec 和 hsi_spa 共享全部卷积参数 |
| 高度耦合 | FusAtNet (CVPR-W'20) | 0.60±0.12× | 共享 HFE，spectral attention 参数反而更重 |

**第二层 — 架构依赖性：** 不对称性与三个架构因素正相关：
1. **分支独立性**（最关键）：共享参数越少 → 不对称越强
2. **参数容量不对称**：spectral 分支参数远少于 spatial → 分母小 → ratio 放大
3. **特征提取点位置**：在耦合层之前提取 → 分支差异被抹平

**关键 insight**:
- 漂移不对称不是普遍规律，而是独立双分支架构的**结构先验**
- 这种先验是可以**被利用的设计信号**：光谱天然稳定 → 锚点；空间需要适配 → LoRA
- 与其均匀抑制所有漂移（EWC/LwF），不如**针对独立分支范式设计非均匀适配策略**
- 这同时解释了为什么我们选择独立三分支 backbone（S2CM）：最大化可利用的不对称性

→ 实验支撑：6 backbone × 3 seed 的 CKA 漂移曲线（主文 Fig. 1 + 架构因素分析）

### 2. Method (方法)
基于"利用独立分支架构的漂移不对称先验"，提出三个设计：

**A. 独立三分支 Backbone + 不对称 LoRA（利用结构先验 → 非均匀适配）**
- 采用独立三分支 Mamba 架构（S2CM），**刻意最大化分支独立性**以放大可利用的漂移不对称
- Warmup 后冻结 backbone，**光谱分支作为稳定锚点**（Observation 证实：独立 spectral branch drift 极小，ratio 4.35×）
- 对 HSI 空间和 LiDAR 空间分支引入 LoRA，**不对称低秩配置**（LiDAR rank > HSI rank，容量分配匹配漂移程度）
- 可塑性集中在漂移大的空间通道，而非均匀分配（EWC/LwF 的均匀正则化忽略了这一结构先验）

**B. SHINE — Post-hoc Per-domain Whitening（我们提出的）**
- 每个域单独计算各分支的均值/方差
- Z-score 归一化后做 cosine NCM 分类
- 消除 query feature 和 prototype 之间的域偏移
- Related work: DSBN (CVPR'19), Feature Whitening (CVPR'19), Multi-Prototype (CVPR'25)

**C. DCR + Prototype 分类**
- Drift-Compensated Reconstruction 用于旧类 prototype 校正
- Prototype-based 分类替代全连接，效果显著更好

### 3. Results (结果)
**设定**: MTH 顺序（MUUFL → Trento → Houston），3 datasets × 3 tasks/dataset = 9 tasks 串行

> 以下为旧 pilot 数据（5/cls），正式主表待 20/cls 全部重跑后更新。

| 方法 | Exemplar | Avg TAg |
|:---|:---:|:---:|
| **Ours** | **0** | **83.7±0.9** |
| iCaRL (pilot, 5/cls) | 5/cls | 86.2±2.2 |
| LUCIR (pilot, 5/cls) | 5/cls | 85.4±3.8 |

- 在无回放条件下与回放方法有竞争力（competitive），且 3-seed std 更小（稳定性更优）
- "稳定性"证据：主表报 mean±std，补充表报 3-seed Forgetting std
- 正式主表将使用 20/cls replay budget，预计 replay 方法数值会更高

---

## Contribution Summary
1. 揭示了 HSI+LiDAR 跨域增量场景下**architecture-dependent 的漂移不对称现象**：独立双分支架构中 spatial drift 显著大于 spectral drift（HCT 5.7×, S2CM 4.4×），而耦合架构中不对称性消失。提出关键 insight：这种不对称是独立分支架构的**结构先验**，可被利用为非均匀适配的设计信号。据此设计**独立三分支 backbone + 不对称 LoRA** 框架
2. 提出 SHINE（per-domain whitening）解决跨域 prototype 偏移
3. 在无回放条件下与回放方法有竞争力（competitive），且 3-seed Avg TAg std 更小（稳定性证据：主表 mean±std + 补充表 Forgetting std）

---

## Key Decisions
- **Ours = CMCD-LoRA (default) + SHINE**，报 83.7±0.9%
- SHINE 是我们方法的标配组件，baseline 不加 SHINE
- Mixtrain 变体 (84.0±0.7%) 放 supplementary，
- 跨域设定作为实验设定，不强调为 benchmark

---

## Observation Experiment (已确定)

### 目的
证明"跨域场景下空间表征漂移 ≈ 2× 光谱表征漂移"是架构无关的普遍现象。

> 注意：此实验使用 3 tasks（dataset-as-task）而非主实验的 9 tasks。
> 目的是隔离纯粹的域漂移，排除类增量漂移的干扰。
> Supplementary 可补 9-task 版本以验证一致性。

### 设置
| 项目 | 方案 |
|:---|:---|
| Task 划分 | 3 tasks, dataset-as-task, MTH (MUUFL→Trento→Houston)（简化诊断实验，非主设定） |
| CIL 策略 | Naive fine-tuning（无 replay、无 KD） |
| Backbone | Coupled CNNs (TGRS'20), HCT (TGRS'23), MAHiDFNet (InfoFusion'22), FusAtNet (CVPR-W'20), S2ENet (TGRS'22), S2CM (Ours backbone) + 更多 |
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
| HCT | Conv3D 后 GAP（光谱维度处理） | Fusion 后 HSI CLS token | Fusion 后 LiDAR CLS token |
| MAHiDFNet | 中心像素 1D conv（纯光谱） | HSI 2D CNN + PAM | LiDAR 2D CNN + PAM |
| FusAtNet | 光谱 self-attention 输出 | LiDAR-guided 空间 cross-attention | LiDAR CNN 分支 |
| S2ENet | SEEM 模块前光谱特征 | SAEM 模块前空间特征 | LiDAR 分支特征 |
| S2CM (Ours) | Mamba SSM 光谱分支（纯光谱，无空间） | VSSBlocks HSI 空间分支 | VSSBlocks LiDAR 空间分支 |

### 可视化
- 主图：Line plot, x=Task, y=1-CKA (drift), 3条线(spec/hsi_spa/lid_spa), 4个backbone作subplot
- 辅助：Bar chart 各 backbone 的 drift ratio (spatial/spectral)
- Supplementary: t-SNE

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
| 1 | Naive fine-tune | 下界 | — | 0 | 使用 S2CM backbone，顺序微调无任何抗遗忘 |
| 2 | Frozen + prototype | 下界 | — | 0 | 使用 S2CM backbone，Task 0 后冻结，cosine NCM |
| 3 | EWC | 经典无回放 | PNAS 2017 | 0 | 正则化基线 |
| 4 | LwF | 经典无回放 | ECCV 2016 | 0 | 蒸馏基线 |
| 5 | ACIL | 现代无回放 | NeurIPS 2022 | 0 | 解析/原型基线 |
| 6 | PASS++ | 现代无回放 | TPAMI 2025 | 0 | 最新无回放 SOTA；若迁移不稳定则用 FCS (CVPR'24) 替补 |
| 7 | iCaRL | 回放 | CVPR 2017 | 20/cls | 经典 replay, NME 推理 |
| 8 | LUCIR | 回放 | CVPR 2019 | 20/cls | 经典 replay, cosine classifier |
| 9 | PODNet | 回放 | ECCV 2020 | 20/cls | 强蒸馏 + replay, 注明 CNN/NME |
| 10 | FOSTER | 回放 | ECCV 2022 | 20/cls | 强 replay, 原文默认 head |
| 11 | Joint training | 上界 | — | all | S2CM backbone，所有 9 task 数据合并一次性训练，同 Ours 的 lr/epoch，使用 FC head，关闭 LoRA/SHINE/DCR（纯上界，无 CIL 组件） |
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

所有 ablation 在 **9-task MTH、3-seed** 上跑，与主实验一致。

### 主表（6 个配置）

| # | 配置 | 改动 | 验证的 claim |
|:--|:---|:---|:---|
| 1 | Full method | — | 完整方法 |
| 2 | w/o SHINE | 去掉域对齐 | 域对齐的贡献 |
| 3 | w/o DCR | 去掉 prototype 校正，保留 prototype classifier | Prototype 校正的贡献 |
| 4 | Symmetric LoRA | HSI rank = LiDAR rank = 总预算/2（如各 rank=6） | 不对称适配的必要性 |
| 5 | Shared spatial adapter | HSI 空间和 LiDAR 空间共享同一个 LoRA，rank = HSI rank（如 rank=4） | 解耦设计的必要性 |
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

---

## Protocol Freeze (锁定协议)

以下为最终锁定的实验协议，不再修改：

| 协议项 | 锁定值 |
|:---|:---|
| 主实验 task 设定 | 9 tasks (3 datasets × 3 tasks/dataset), MTH |
| Observation task 设定 | 3 tasks (dataset-as-task), MTH（诊断实验） |
| Replay budget | 20 exemplars/class (K = 20 × 32 = 640) |
| Patch size | 验证集 sweep 后锁死；fallback = 11×11 |
| 主指标 | **Avg TAg**（主表主列）|
| 补充指标 | OA, AA, Kappa, Forgetting, BWT（补充表/附录） |
| Joint training backbone | S2CM（与 Ours 对齐） |
| Naive / Frozen backbone | S2CM（与 Ours 对齐） |
| 所有其他 baseline backbone | 按各自原文 |
| Seed | 3 seed (0, 1, 2) |

---

## Claim-to-Evidence (贡献-证据对应)

| Contribution | 主要证据 | 位置 |
|:---|:---|:---|
| C1: architecture-dependent 漂移不对称 | Observation：6 backbone × 3 seed CKA 曲线 + 架构因素分析（独立性/参数容量/耦合度） | 主文 Fig. 1 |
| C1: 利用结构先验的非均匀适配 | Ablation: Symmetric LoRA / Shared adapter / Spectral LoRA（验证非均匀优于均匀） | 主文 Ablation Table |
| C2: SHINE 域对齐 | Ablation: w/o SHINE | 主文 Ablation Table |
| C3: 无回放竞争力 | 主表: 12 方法对比 | 主文 Main Table |
| 补充: DCR 贡献 | Ablation: w/o DCR | 主文 Ablation Table |
| 补充: Prototype vs FC | Ablation: FC head | Supplementary |
| 补充: LoRA 本身贡献 | Ablation: w/o LoRA | Supplementary |
| 补充: 低内存 replay | 5/cls replay ablation | Supplementary |

---

## Figure Plan (图表计划)

### 主文
| 图/表 | 内容 | 数据来源 |
|:---|:---|:---|
| Fig. 1 | Drift observation: 6 backbone 的 CKA 漂移曲线 (spec/hsi_spa/lid_spa) + drift ratio vs 分支独立性的关系 | drift observation 实验 |
| Fig. 2 | 方法框架图 | 手绘/AI 绘图 |
| Fig. 3 | Classification maps: GT + Ours + 2-3 代表性 baseline，每 domain 一列 | predictions parquet |
| Fig. 4 | Task-wise accuracy 演化曲线 (Ours vs baselines) | task_metrics.csv |
| Table 1 | 主表: 12 方法 × (Avg TAg, per-dataset, Forgetting) | metrics_summary.json |
| Table 2 | Ablation: 6 配置 | ablation runs |

### Supplementary
| 图/表 | 内容 |
|:---|:---|
| Fig. S1 | t-SNE per domain/task |
| Fig. S2 | 9-task 漂移曲线（验证 observation 在 9-task 下一致性） |
| Fig. S3 | Drift ratio bar chart (spatial/spectral per backbone) |
| Table S1 | 完整 per-class accuracy |
| Table S2 | Supplementary ablation (w/o LoRA, FC head) |
| Table S3 | 5/cls replay memory ablation |
| Table S4 | Confusion matrices |

---

## TODO (按执行顺序)

1. [ ] **Patch size sweep**（Ours + Naive 验证集，{7,9,11,13,15}）→ 锁死 patch size
2. [ ] **实现实验 logging 框架**（统一目录结构 + 记录格式）
3. [ ] **实现 4 个第三方 backbone** 的 drift observation 实验
4. [ ] **按原文实现 11 个 baseline** 的 HSI+LiDAR 适配（Naive/Frozen 用 S2CM 不需额外实现）
5. [ ] **全部 3-seed MTH 跑完**（baseline + ours + ablation）
6. [ ] **Ablation** 6 主表 + 2 supplementary
7. [ ] **画图**：classification maps, drift curves, accuracy curves
