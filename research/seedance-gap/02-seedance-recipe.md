# 02 — Seedance 技术配方拆解(1.0 / 1.5 pro)

> What Seedance's own technical reports disclose: data pipeline, architecture, SFT + three-reward-model RLHF, and the distillation stack behind its ~10x speedup.

[← 返回专题目录](README.md)

一手来源:Seedance 1.0(arXiv [2506.09113](https://arxiv.org/abs/2506.09113),2025-06)、Seedance 1.5 pro(arXiv [2512.13507](https://arxiv.org/abs/2512.13507),2025-12)。标注 ✅ 的条目经 3/3 对抗验证。

---

## 1. 数据管线 ✅

- **多源策展 + 精准密集 caption**:caption 模型基于 Tarsier2,覆盖动态特征(动作、运镜)与静态属性;论文原话:"Video captions largely affect the instruction-following capabilities"——**prompt 遵循优势的第一根因**。
- 管线阶段:切片(shot-aware 时间切分,clip ≤12s)→ 质量/美学过滤 → 去重 → 密集 caption;多源分布再平衡。
- shot-aware 切分同时是多镜头能力的数据基础(每个 shot 独立 caption、按时序组织进训练)。

## 2. 架构(Seedance 1.0)✅

- **时空解耦注意力**:空间层做帧内注意力聚合,时间层只做跨帧注意力——效率取向的设计。
- **MMDiT 式多模态注意力只在空间层**:文本 token 仅在空间层参与跨模态交互;时间层纯视觉自注意力。
- **3D MM-RoPE**:支持视觉/文本 token 交错序列;可扩展到多 shot 训练(shot 按动作时序组织 + per-shot caption)→ **原生多镜头叙事**(镜头正反打、cut-in/cut-away、match cut,主体跨 shot 一致)。
- T2V 与 I2V 联合训练(单模型双任务)。
- 验证 agent 提醒:多镜头能力 = 架构(MM-RoPE)+ 多镜头数据**共同作用**,不是纯架构特性;且 1.5 pro 据报改为不同(统一全注意力式)设计,1.0 的架构描述不能直接套到 1.5。

## 3. 后训练 ✅(与开源差距的核心)

### SFT
- 数百个策展类目的人工核验视频-文本对;分子集训练后做 **model merging**。

### RLHF:三个专用 reward model
| RM | 目标 | 形态 |
|---|---|---|
| Foundational RM | 图文对齐、结构稳定性 | VLM-based |
| Motion RM | 抑制伪影,增强运动幅度与生动性 | 视频输入 |
| Aesthetic RM | 美学 | 图像空间,关键帧输入(承自 Seedream) |

- **偏好标注**:按单一维度选最好/最坏,且"最好者在其他维度不劣于最坏者"——防止维度间此消彼长(与 VisionReward 的 MPO 思想同源,见 [04](04-posttraining-methods.md))。
- **优化方式:直接 reward 最大化**——预测 x0(干净视频),对多 RM 复合 reward 直接反传;论文称对比 DPO/PPO/GRPO 更优;diffusion 与 RM **多轮迭代**。
- RLHF 同时作用于 base model 和 refiner/蒸馏模型(报告 Fig.7 双阶段 reward 曲线稳定上升)。
- 报告 Fig.6:Pre-training → CT → SFT → RLHF 逐阶段质量提升可视化。
- ⚠️ 注意:报告**没有消融隔离** RLHF 对"vs 开源差距"的贡献占比——"后训练是主差距"是获多方证据支持的假设(参见 [06](06-roadmap-feasibility.md) 根因分解),不是测量结论。

### 推理侧 prompt 改写(常被忽略)
- 两阶段(SFT+RL)prompt 改写系统,把用户短 prompt 改写成密集 caption 风格再进模型——训练分布与推理分布对齐。**复现 Seedance 观感时这层成本极低、收益直接。**

## 4. 蒸馏与推理优化 ✅

端到端 **~10x** 加速(5s 1080p 41.4s @ 单张 NVIDIA L20;自报,无独立复现):

| 组件 | 加速 | 说明 |
|---|---|---|
| TSCD(Trajectory Segmented Consistency Distillation,承自 [Hyper-SD](https://arxiv.org/abs/2404.13686)) | ~4x(DiT) | 分段一致性蒸馏 |
| 瘦 VAE decoder | ~2x | 报告称无质量损失 |
| 其余 | 叠加 | RayFlow score distillation、对抗步数蒸馏(APT 系)、混合精度量化、kernel fusion(~15%)、稀疏注意力、混合并行 |

## 5. Seedance 1.5 pro 增量(arXiv 2512.13507)

- **原生音视频联合生成**:双分支 MMDiT(视频/音频)+ 跨模态联合模块;多任务预训练覆盖 T2VA / I2VA / T2V / I2V;多语言/方言口型同步。
  - ❌ **已证伪的流行说法**:"音视频联合是 Seedance 独有 vs 开源"——LTX-2(2026-01 开源权重)同为原生双流音视频 DiT(14B 视频 + 5B 音频)。该差异化仅对 Wan 2.2 / HunyuanVideo 成立。
- 后训练延续 SFT + 多维 RM RLHF ✅;RLHF 引用 **Flow-GRPO、RewardDance、DanceGRPO、UniFL**(即:1.5 代已转向/融合 GRPO 系方法);RLHF 基础设施优化提速近 3x。
- 声称 >10x 推理加速框架(🟡 未完成对抗验证)。
- 评测用内部 SeedVideoBench-1.5(多维);产品页不公布 vs Veo/Kling/Sora/Wan 的直接对比数字。

## 6. 对复现者的要点清单

1. 后训练是配方核心:多维 RM(≥3 个专用维度)+ 直接 reward 最大化或 GRPO 系,多轮迭代。
2. 偏好标注按维度分开、加"不劣于"约束。
3. caption 质量 ≥ 数据量;补 prompt 改写系统。
4. 蒸馏走 TSCD/DMD + 瘦 decoder,RLHF 要同时打在蒸馏后的模型上。
5. 多镜头 = MM-RoPE 类位置编码 + shot 切分数据,可后训练补齐(见 [05](05-distillation-multishot.md))。

---

*下一篇:[03 — 开源模型现状](03-open-source-landscape.md)*
