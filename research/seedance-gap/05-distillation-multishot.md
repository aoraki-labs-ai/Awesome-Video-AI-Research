# 05 — 蒸馏少步生成与多镜头/长视频

> Closing the two "capability" gaps that don't need a new base model: few-step distillation for Seedance-class inference speed, and multi-shot narrative via context-expansion fine-tuning.

[← 返回专题目录](README.md)

---

## A. 蒸馏与少步生成(推理速度差距 —— 最便宜可解)

参照系:Seedance 官方 ~10x = TSCD 4x + 瘦 VAE decoder 2x + 系统优化(见 [02](02-seedance-recipe.md) §4)。

### Self Forcing(arXiv [2506.08009](https://arxiv.org/abs/2506.08009))⭐ 首选

- **思想**:训练时就做自回归 rollout(KV cache),让每帧条件在**模型自己生成的**前文上——消除曝光偏差(train-test gap)。
- **成本(关键数字)**:Wan2.1-T2V-1.3B 上 DMD 目标 **64×H100 约 1.5 小时收敛(~4 H100-天)**;SiD/GAN 目标 2–3 小时。
- **效果**:单 H100 **17.0 FPS、0.69s 延迟**(4 步去噪,chunk-wise),vs 教师类 0.78 FPS/103s;VBench 84.31 **≥ 教师 Wan2.1 的 84.26**(CausVid 同初始化只有 81.20)。
- 代码开源:[github.com/guandeh17/Self-Forcing](https://github.com/guandeh17/Self-Forcing)。直接适配 Wan 2.x checkpoint。

### Seaweed-APT(ByteDance,ICML 2025,arXiv [2501.08316](https://arxiv.org/abs/2501.08316))

- **对抗后训练(APT)**:diffusion 预训练后**对真实数据**做对抗训练(非蒸馏教师输出)→ **一步**生成 2s 1280x720@24fps 实时。
- 稳定性:架构改造 + approximated R1 正则。
- 作者论点:现有步数蒸馏仍有明显质量损失,APT 是替代路径——与 TSCD/DMD 做对照实验的对象。

### 其他
- **CausVid**(arXiv [2412.07772](https://arxiv.org/abs/2412.07772)):双向教师 → 因果学生,自回归实时向的先行工作(Self-Forcing 的对照基线)。
- **Causal Forcing++**(arXiv 2605.15141):可扩展少步自回归蒸馏,交互式实时生成方向。
- **TSCD / Hyper-SD**(arXiv [2404.13686](https://arxiv.org/abs/2404.13686)):Seedance 官方采用的分段一致性蒸馏。

### 实施建议
1. 先在 Wan2.2 TI2V-5B / Wan2.1-1.3B 上复现 Self-Forcing(个位数 H100-天,快速验证流水线);
2. 上 A14B(注意 MoE 双专家蒸馏是开放问题:高/低噪专家可能需分别或联合蒸馏——值得记录的研究点);
3. 与 APT 一步生成对照;蒸馏后模型**再过一轮 RLHF**(Seedance 的做法:RLHF 同时打在 base 和加速版上)。

## B. 多镜头 / 长视频(叙事能力差距 —— 微调量级可解)

参照系:Seedance 原生多镜头 = MM-RoPE + shot 切分数据(见 [02](02-seedance-recipe.md) §2)。

### LCT — Long Context Tuning(arXiv [2503.10589](https://arxiv.org/abs/2503.10589),ICCV 2025)⭐ 路径最通用

- **把预训练单镜头 DiT 的 context 扩到场景级**:全注意力从单 shot 扩展到场景内所有 shot + **interleaved 3D 位置编码** + **异步噪声策略**;**不新增参数**。
- 支持联合生成与自回归逐 shot 生成;LCT 后可再微调成 **context-causal 注意力 + KV cache**,降低长视频推理成本。
- 涌现能力:组合生成、交互式 shot 扩展。

### HoloCine(arXiv [2510.20822](https://arxiv.org/abs/2510.20822),CVPR 2026)⭐ 直接可用

- **基于 Wan 2.2**,用 **40 万策展多镜头样本**微调 → 整体生成电影式多镜头叙事(分钟级、跨 shot 主体一致)。
- 意义:开源界已证明"多镜头 = 数据策展 + 微调",不需要重预训练;可直接在其权重上迭代,或按其数据配方自建。

### 现状对照
- Wan 2.2 / 开源 LTX 原生均单镜头(LTX-2.x 的 "multi-shot storytelling" 靠 Extend 工作流拼接;Wan 2.6 的原生多镜头未开源)✅。
- 数据是关键瓶颈:多镜头训练集需要 shot 切分 + per-shot caption + 跨 shot 主体一致性标注——策展管线成本高于训练本身。

### 实施建议
1. 从 HoloCine 权重/数据配方出发(同为 Wan 2.2 底座,与主线兼容);
2. 需要自有数据分布时按 LCT 配方自训:先小规模(数万场景)验证 interleaved 3D RoPE + 异步噪声;
3. 评测加多镜头专项维度(转场类型、主体一致性、叙事连贯),见 [01](01-benchmark-gap.md) §4。

---

*下一篇:[06 — 根因分解、可行性与路线图](06-roadmap-feasibility.md)*
