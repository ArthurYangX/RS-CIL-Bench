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

## TODO
- [ ] 补充分支级 drift 可视化实验（spec / hsi_spa / lid_spa cosine similarity 随 task 衰减）
- [ ] Ablation 实验设计
- [ ] Baseline 列表最终确认
