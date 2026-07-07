# 03 — 开源模型现状:Wan / LTX / HunyuanVideo / ContentV

> Architecture, data, and (crucially, absent) post-training disclosures of the main open-source video bases, as the counterpart for gap decomposition.

[← 返回专题目录](README.md)

---

## 1. Wan 2.1(arXiv [2503.20314](https://arxiv.org/abs/2503.20314),2025-03)

- **架构**:稠密 14B DiT + flow matching + 全时空注意力;umT5 文本条件(cross-attention)。无 MoE(MoE 是 2.2 才引入)。
- **预训练数据**:数十亿图像/视频,合计 **O(1) 万亿 token**;四步清洗管线(基础维度含 OCR/美学/NSFW/水印/模糊过滤 → 视觉质量聚类+专家打分 → 六级运动质量分类),淘汰约 50% 候选数据。
- **后训练:只有继续监督训练**——数百万人工/专家策展的高质量图像视频(480px/720px),同架构同优化器。**报告没有 RLHF、reward model、偏好优化的任何描述。**

## 2. Wan 2.2([repo](https://github.com/Wan-Video/Wan2.2),2025-07-28,Apache 2.0)

- **MoE DiT**:双专家按去噪阶段分工——高噪专家管布局/结构、低噪专家管细节,按 SNR 阈值切换;27B 总参 / **每步 14B 激活**(推理成本≈稠密 14B)。
- 三个变体:T2V-A14B、I2V-A14B、TI2V-5B(混合)。
- **数据较 2.1**:图像 +65.6%、视频 +83.2%;美学数据带灯光/构图/对比度/色调等细粒度标签(cinematic 取向)。
- **VAE**:Wan2.2-VAE 4x16x16(T×H×W);TI2V-5B 再加 4x32x32 patchify,总压缩 64x → 消费级单卡 <9 分钟生成 5s 720p@24fps。A14B 无优化需 ≥80GB VRAM。
- **repo/报告同样未披露任何 RLHF/RM 后训练。** 自报 Wan-Bench 2.0"多数维度超商业模型"(无第三方复现)。
- 第三方 4 案例横评(302.AI,2025-08):Wan 2.2 在电影感/动态控制/语义遵循上仅"尚可",4 轮全负于 Kling 2.1 / Hailuo-02(各赢 2 轮);评者认为其最大优势是开源本身。
- **→ 差距根因证据链:Wan 系万亿 token 级预训练不弱,缺的是 Seedance 式后训练。**

## 3. LTX-Video(arXiv [2501.00103](https://arxiv.org/abs/2501.00103))与 LTX-2(arXiv [2601.03233](https://arxiv.org/html/2601.03233v1))

### LTX-Video:速度优先的设计点
- **极致压缩 Video-VAE:1:192**(每 token 32x32x8 像素;patchify 移入 VAE)vs 常规 4x8x8——速度优势与细节上限都来自这里。
- H100 上 2 秒生成 5s 768x512@24fps(快于实时)。
- decoder 兼做最后一步去噪(直接在像素空间出干净结果),省掉独立超分/refiner 级联。

### LTX-2:开源阵营的音视频联合模型
- **19B 非对称双流 DiT**:14B 视频流 + 5B 音频流,稠密(无 MoE);文本编码 Gemma3-12B(多层特征提取);modality-aware CFG。
- 单次前向出同步音视频(语音/foley/环境音/口型);20s 连续生成;1080p(multi-tile 至 4K)。
- **效率:720p/121 帧下 1.22 s/step vs Wan 2.2-14B 22.30 s/step(~18x/步)**——来自深压缩 latent 的架构红利,而非蒸馏。
- **未披露**数据规模与任何 SFT/RLHF/DPO 后训练(策展了音频丰富子集 + 联合视听 caption 系统)。
- 人评:超开源 Ovi,接近 Veo 3 / Sora 2(AA 2025-11 快照 I2V #3 / T2V #4);报告未直接对比 Seedance。
- 微调门槛(官方 LTX Trainer):LoRA ~80GB VRAM(INT8 低配 32GB);13B 全参微调 4–8x H100 + FSDP。

## 4. HunyuanVideo 1.5(arXiv [2511.18870](https://arxiv.org/abs/2511.18870),2025-11)——开源后训练配方的最佳公开模板

- 8.3B DiT;3D causal VAE(16x 空间压缩)。
- **数据规模**:>1000 万小时原始视频 → 多级过滤(基础/视觉质量/美学)→ **~8 亿高质量片段 + 50 亿图像**。
- **后训练全配方**(可直接复用的蓝本):
  1. 每任务 ~100 万高质量片段继续训练;
  2. SFT;
  3. **离线 DPO**:O(10K) 均衡 prompt 集,每 prompt 生成 N 个候选组 pair,人工 GSB 标注(语义保真/运动质量/美学);
  4. **在线 MixGRPO**:VLM 微调的四维 reward(文本对齐/图像对齐/视觉质量/运动动态)。
- **结果**:720p T2V GSB vs Seedance Pro **+11.02%(赢)**;I2V **−5.77%(输)**;指令遵循 61.57 vs Veo3 73.77。
- **→ 三个信号**:后训练做满,T2V 可打平/反超上一代 Seedance;I2V 更吃基座;指令遵循差距要靠 caption/数据侧(见 [02](02-seedance-recipe.md) §1)。

## 5. ContentV(arXiv [2506.05343](https://arxiv.org/abs/2506.05343),ByteDance)——中等算力的公开锚点

- 8B,从图像模型(SD3.5L)最大化复用起训;多阶段 flow-matching。
- **算力:256×64GB NPU × 4 周(~7000 加速器-天)**;数据 ~O(100M) 图像 + O(10M) 视频片段(比前沿实验室低一个量级)。
- **免人工标注 RLHF**:MPS(CLIP-based)作视觉质量 reward,可微采样让 x1 可导、reward 直接反传。
- VBench 85.14(> HYV-13B 83.24、Sora 84.28;< Wan2.1-14B 86.22)。
- **→ 说明**:数千加速器-天量级可以训出 VBench 85 档的模型,但要摸到 Wan2.1-14B 之上仍需更大预训练——支持"预训练层差距无法在中小算力弥补,应站在最强开源底座上"的路线判断。

## 6. 横向对比表

| | Wan 2.2 A14B | LTX-2 | HunyuanVideo 1.5 | Seedance 1.5 pro |
|---|---|---|---|---|
| 参数 | 27B MoE(14B 激活) | 19B(14B 视频+5B 音频) | 8.3B | 未披露 |
| License | Apache 2.0 | 开源权重(条款需复核) | 开源 | 闭源 API |
| 预训练数据 | 万亿 token 级 | 未披露 | 800M 片段+5B 图像 | 未披露 |
| RLHF/RM 披露 | **无** | **无** | **DPO+MixGRPO(全公开)** | 多维 RM RLHF |
| 原生多镜头 | 无 | 无(Extend 工作流) | 无 | 有 |
| 原生音频 | 无 | **有** | 无 | 有(1.5+) |
| 蒸馏加速发布 | 无官方 | 架构本身快(~18x/步) | 无官方 | ~10x 官方 |

---

*下一篇:[04 — 后训练方法工具箱](04-posttraining-methods.md)*
