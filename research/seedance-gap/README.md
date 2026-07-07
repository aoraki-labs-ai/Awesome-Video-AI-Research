# Seedance vs 开源视频模型(Wan 2.2 / LTX)差距根因分析与逼近路线

> **English abstract** — A root-cause analysis of the quality gap between ByteDance Seedance (1.0 / 1.5 pro) and open-source video generation models (Wan 2.2, LTX-Video/LTX-2, HunyuanVideo). Main finding: the gap is driven primarily by **post-training (SFT + multi-dimensional reward-model RLHF) and data curation**, not architecture. Evidence: Wan's technical reports disclose no RLHF stage; DanceGRPO shows +181% motion quality from online RL on an open model; HunyuanVideo 1.5 with a full post-training recipe already beats Seedance Pro in 720p T2V human GSB evals. Includes a prioritized, budget-estimated roadmap (reward models → DPO/GRPO → SFT curation → few-step distillation → multi-shot tuning) for approaching Seedance-1.5-pro-level quality on open bases within hundreds-to-thousands of H100-days. All claims cite primary technical reports and 2025–2026 papers.

- 日期:2026-07-06(专题化 2026-07-07)
- 方法:多角度检索 → 抓取一手来源 → 逐条对抗验证(3 票制)→ 人工综合
- 验证标注:✅ = 3/3 对抗验证通过;🟡 = 一手来源(技术报告/论文)抓取核实,未完成三票对抗验证;❌ = 已证伪
- 受众:熟悉 Wan/LTX 的 video AIGC 研究者

## 专题目录

本页为综合结论(TL;DR + 全景);各子篇展开底层素材与可执行细节:

| 篇 | 内容 |
|---|---|
| [01 — 差距量化](01-benchmark-gap.md) | AA 盲测 Elo 全表、GSB 人评、VBench 锚点、时间线、评测集设计启示 |
| [02 — Seedance 技术配方拆解](02-seedance-recipe.md) | 数据管线、MM-RoPE/解耦注意力、三 RM RLHF 细节、TSCD 蒸馏栈、1.5 pro 增量、复现要点清单 |
| [03 — 开源模型现状](03-open-source-landscape.md) | Wan 2.1/2.2、LTX-Video/LTX-2、HunyuanVideo 1.5、ContentV 逐个拆解 + 横向对比表 |
| [04 — 后训练方法工具箱](04-posttraining-methods.md) | VideoAlign/VisionReward、VideoDPO/Flow-DPO/Flow-GRPO/DanceGRPO 全参数与实测收益、reward hacking 坑与对策、选型建议 |
| [05 — 蒸馏与多镜头](05-distillation-multishot.md) | Self-Forcing/APT/CausVid/TSCD 成本与效果;LCT/HoloCine 多镜头路线与实施建议 |
| [06 — 根因分解与路线图](06-roadmap-feasibility.md) | 根因表、可行性/预算评估、P0–P3 路线图、底座选型、风险登记 |

---

## TL;DR

1. **差距主要不在架构,而在后训练和数据策展。** Wan 2.1/2.2 技术报告与 repo 均未披露任何 RLHF / reward model 阶段(后训练只有高质量数据继续训练)🟡;Seedance 1.0/1.5 把 SFT + 多维 reward model RLHF 列为质量核心支柱 ✅。这是最大且最可弥补的一块差距。
2. **RLHF 对开源模型的提升已被定量证明:** DanceGRPO 在 HunyuanVideo 上用 VideoAlign reward 做 online RL,视觉质量 +56%(4.51→7.03)、运动质量 +181%(1.37→3.85)🟡。HunyuanVideo 1.5(8.3B,完整 SFT+DPO+MixGRPO 后训练)在 720p T2V 的 GSB 人评中已反超 Seedance Pro(+11.02%),但 I2V 仍落后(−5.77%)🟡。**结论:开源底座 + 认真做后训练,T2V 质量维度可以打平上一代 Seedance。**
3. **推理速度差距最便宜可解。** Seedance 的 ~10x 加速 = TSCD 蒸馏 4x + 瘦 VAE decoder 2x + 系统优化(5s 1080p 41.4s @L20)✅。开源侧 Self-Forcing 在 Wan2.1-1.3B 上蒸馏只需 ~4 H100-天,单卡 17 FPS 且 VBench 不降 🟡。
4. **多镜头叙事可以后训练补齐,不必重预训练。** Seedance 的原生多镜头来自 MM-RoPE + 多镜头数据(per-shot caption)✅;开源已有 LCT(单镜头模型 → 场景级多镜头,不加参数)和 HoloCine(Wan 2.2 + 40 万多镜头样本微调,CVPR 2026)🟡。
5. **难以弥补的是最前沿一代的综合差距。** 当前(2026 年中)Artificial Analysis 榜:Seedance 2.0 Elo 1222 遥遥领先;最强开源权重 LTX-2.3 ≈ 961–976,差 ~250 Elo;而上一代 Seedance 1.5 pro 已跌至 Elo 1000,LTX-2.3 在 I2V 上距它只差 ~50 Elo 🟡。**现实目标:12 个月内用数百–上千 H100-天,把开源底座后训练到 Seedance 1.5 pro 水平;Seedance 2.0 水平受限于预训练代差,不设为目标。**

---

## 1. 差距量化(人类偏好盲测 Elo,Artificial Analysis,2026 年中快照)🟡

| 模型 | T2V Elo | I2V Elo | 开源权重 |
|---|---|---|---|
| Seedance 2.0 720p(2026-03) | 1222(#1) | 1189(#1) | 否 |
| Wan 2.7(2026-04,API) | 1102 | 1094 | 否 |
| Wan 2.6(2025-12,API) | 1028 | 897 | 否 |
| Seedance 1.5 pro(2025-12) | 1000 | 1000 | 否 |
| LTX-2.3 Fast / Pro(2026-03) | 976 / 961 | 946 / 952 | **是** |

- 历史锚点:2025-06 Seedance 1.0 发布时同时登顶 T2V/I2V 双榜,I2V 领先第二名(Veo 3)100+ Elo ✅。
- Wan 2.2、HunyuanVideo、CogVideoX、Mochi 已掉出当前榜单(无直接 Elo 可比)🟡。
- 解读:**开源权重 vs 闭源最前沿 ≈ 250 Elo;开源权重 vs 上一代旗舰 ≈ 50 Elo 以内。** 阿里自己的 Wan 2.6/2.7 靠(未开源的)迭代已越过 Seedance 1.5 pro,说明这条差距不是"开源方法做不到",而是"最新权重不放出来"。

## 2. Seedance 领先维度及其来源(技术报告披露)

### 2.1 后训练:SFT + 多维 reward model RLHF ✅

Seedance 1.0(arXiv [2506.09113](https://arxiv.org/abs/2506.09113)):

- **三个专用 reward model**:Foundational RM(图文对齐/结构稳定,VLM-based)、Motion RM(抑制伪影、增强运动幅度与生动性)、Aesthetic RM(图像空间,关键帧输入,承自 Seedream)✅。
- 偏好标注按"单一维度选最好/最坏,且最好者在其他维度不劣于最坏者"的多维方式采集 ✅。
- 优化方式是**直接 reward 最大化**(预测 x0,对多 RM 复合 reward 直接反传),论文称优于 DPO/PPO/GRPO;diffusion 与 RM 多轮迭代 ✅。
- SFT:数百个策展类目的人工核验视频-文本对 + model merging 🟡。
- Seedance 1.5 pro(arXiv [2512.13507](https://arxiv.org/abs/2512.13507))延续该配方(SFT on 高质量数据 + 多维 RM 的 RLHF)✅,RLHF 引用 Flow-GRPO、RewardDance、DanceGRPO、UniFL,基础设施优化使 RLHF 训练提速近 3x 🟡。

### 2.2 数据管线:多源策展 + 精准密集 caption ✅

- 多源数据策展 + 精准 video captioning(Tarsier2 系 caption 模型,覆盖动作/运镜/静态属性),论文明确"caption 很大程度决定指令遵循能力" ✅。
- 数据管线含 shot-aware 时间切分(clip ≤12s)、质量/美学过滤、去重 🟡。
- 推理侧还有 prompt 改写系统(SFT+RL,把用户 prompt 改写成密集 caption 风格)🟡 —— 复现 Seedance 观感时别忽略这一层。

### 2.3 原生多镜头叙事 ✅

- 架构:时空解耦注意力(空间层做帧内聚合 + MMDiT 式多模态注意力只在空间层;时间层只做跨帧视觉注意力)+ 3D MM-RoPE,支持把多个 shot 按时序组织、每个 shot 带独立 caption 一起训练 ✅。
- 能力 = 架构 + 多镜头训练数据共同作用(验证 agent 特别提醒:数据侧同等关键)✅。
- Wan 2.2 与开源 LTX 均为单镜头原生(Wan 2.6 才加多镜头,但 API-only)✅。

### 2.4 推理加速:多阶段蒸馏 + 系统优化 ✅

- TSCD(Trajectory Segmented Consistency Distillation,承自 Hyper-SD)给 DiT ~4x;瘦 VAE decoder ~2x 无质量损失;叠加 RayFlow score distillation、对抗蒸馏、量化、kernel fusion、稀疏注意力 → 端到端 ~10x:5s 1080p 41.4s @ 单张 L20 ✅(自报,未独立复现)。
- Seedance 1.5 pro 声称 >10x 加速框架 🟡。

### 2.5 音视频联合生成(1.5 pro)——注意:不再是独占优势 ❌→已修正

- Seedance 1.5 pro 是双分支 MMDiT + 跨模态联合模块的原生音视频联合模型(T2VA/I2VA,多语言口型同步)🟡。
- **但"开源模型都是纯视频"的说法被证伪**:LTX-2(2026-01 开源权重)本身就是 19B 非对称双流 DiT(14B 视频 + 5B 音频),单次前向生成同步音视频 ❌(原声明已按验证结论修正)。该差异化仅对 Wan 2.2 / HunyuanVideo 成立。

## 3. 开源侧现状(用于根因对照)

### Wan 2.1 / 2.2 🟡

- Wan 2.1(arXiv [2503.20314](https://arxiv.org/abs/2503.20314)):稠密 14B DiT + flow matching + 全时空注意力 + umT5;预训练数十亿图像/视频、O(1) 万亿 token,四步清洗淘汰约 50% 候选数据。**后训练 = 高质量数据继续监督训练,无 RLHF/RM/偏好优化。**
- Wan 2.2([repo](https://github.com/Wan-Video/Wan2.2),2025-07-28,Apache 2.0):双专家 MoE DiT(高噪专家管布局、低噪专家管细节,按 SNR 切换;27B 总参 / 14B 激活);数据较 2.1 增加图像 +65.6% / 视频 +83.2%,美学数据带灯光/构图/对比度/色调标签;Wan2.2-VAE 4x16x16(TI2V-5B 加 patchify 达 64x);TI2V-5B 消费级单卡 <9 分钟出 5s 720p;A14B 无优化需 ≥80GB VRAM。**repo 同样未披露任何 RLHF。**

### LTX-Video / LTX-2 🟡

- LTX-Video(arXiv [2501.00103](https://arxiv.org/abs/2501.00103)):极致压缩 VAE(1:192,每 token 32x32x8 像素,patchify 移入 VAE),H100 上 2 秒生成 5s 768x512@24fps;decoder 兼做最后一步去噪。速度优先、细节受压缩上限约束的设计点。
- LTX-2(arXiv [2601.03233](https://arxiv.org/html/2601.03233v1)):19B 非对称双流(14B 视频 + 5B 音频)稠密 DiT,Gemma3-12B 文本编码,20s 连续音视频,1080p(multi-tile 至 4K);720p/121 帧下 1.22 s/step vs Wan 2.2-14B 的 22.30 s/step(~18x/步,来自深压缩 latent 而非蒸馏)。**报告未披露 SFT/RLHF/DPO 后训练与数据规模。**
- 微调门槛:LTX-2 LoRA ~80GB VRAM(INT8 低配 32GB);13B 全参微调需 4–8x H100 FSDP 🟡。

### HunyuanVideo 1.5 —— 开源后训练配方的最佳模板 🟡

arXiv [2511.18870](https://arxiv.org/abs/2511.18870)(2025-11,8.3B):

- 数据:>1000 万小时原始视频 → 多级过滤后 ~8 亿高质量片段 + 50 亿图像。
- 后训练:每任务 ~100 万高质量片段继续训练 → SFT → **离线 DPO + 在线 MixGRPO**,reward 为 VLM 微调的四维评分(文本对齐/图像对齐/视觉质量/运动动态),O(10K) prompt 集 + 人工 GSB 标注偏好对。
- 结果:**720p T2V GSB vs Seedance Pro +11.02%(赢);I2V −5.77%(输);指令遵循 61.57 vs Veo3 73.77(明显落后)。**

## 4. 根因分解:差距在哪一层

| 层 | 证据 | 差距大小 | 可弥补性(数百–数千 H100-天) |
|---|---|---|---|
| **后训练(SFT+RLHF)** | Wan 无 RLHF 披露 vs Seedance 三 RM 多轮 RLHF ✅;DanceGRPO 在开源模型上 MQ +181% 🟡;HYV1.5 靠完整后训练 T2V 反超 Seedance Pro 🟡 | **最大** | **高 —— 首要主攻方向** |
| **数据策展/caption** | Seedance 密集 caption 决定 prompt following ✅;开源模型指令遵循普遍落后(HYV1.5 61.57 vs Veo3 73.77)🟡 | 大 | 中-高:SFT/偏好数据层可补;预训练 caption 质量无法追溯重做 |
| **蒸馏/推理优化** | Seedance ~10x ✅ vs Wan A14B 无蒸馏发布版;Self-Forcing ~4 H100-天即可 🟡 | 大但纯工程 | **很高 —— 最便宜** |
| **多镜头叙事** | MM-RoPE + 多镜头数据 ✅;LCT/HoloCine 证明可微调补齐 🟡 | 中 | 高(微调量级) |
| **预训练规模/物理先验** | Wan 已是万亿 token 量级 🟡;Seedance 未披露规模;复杂运动/物理一致性天花板由基座决定 | 不明,针对最前沿一代(Seedance 2.0)推测显著 | **低 —— 无法在此预算内弥补;靠选最强底座缓解** |

关键校准(验证 agent 的一致提醒):Seedance 报告是厂商自述,四大支柱(数据/架构/后训练/蒸馏)**没有消融隔离各自贡献**,"后训练是主要差距"是被多方证据支持的假设而非测量结论 —— 因此路线图第一阶段要用小规模实验先证实/证伪它。

## 5. 方法工具箱(2025–2026 论文,按用途)

### 5.1 Reward model
- **VideoAlign / VideoReward**(KwaiVGI,NeurIPS 2025,arXiv [2501.13918](https://arxiv.org/abs/2501.13918)):182k 三元组偏好(16k prompts × 12 模型 × 108k 视频),VQ/MQ/TA 三维;附 flow-matching 原生算法 Flow-DPO / Flow-RWR / 推理时 Flow-NRG;**reward model 与代码已开源** —— 首选起点。
- **VisionReward**(THUDM,arXiv [2412.21059](https://arxiv.org/abs/2412.21059)):可解释 checklist 式 RM(视频 9 维/64 个二元判断题 + 线性加权),偏好预测超 VideoScore 17.2%;MPO 只取全维度占优的 pair,避免维度间此消彼长。
- HunyuanVideo 1.5 的 VLM 四维 reward 配方(见上)可作自建 RM 蓝本。

### 5.2 偏好优化 / RL
- **DanceGRPO**(ByteDance,arXiv [2505.07818](https://arxiv.org/abs/2505.07818)):GRPO 统一用于 diffusion 与 rectified flow(SDE 化采样);HunyuanVideo + VideoAlign reward:VQ +56%、MQ +181%;稳定技巧:同 prompt 共享初始噪声防 reward hacking、16 采样取 top-8/bottom-8。
- **Flow-GRPO**(arXiv [2505.05470](https://arxiv.org/abs/2505.05470)):ODE→SDE 转换 + 去噪步数缩减,flow 模型 online RL 的基础组件(也被 Seedance 1.5 引用)。
- **VideoDPO**(arXiv [2412.14167](https://arxiv.org/abs/2412.14167)):全自动偏好对(每 prompt 采 4 个、OmniScore 排序取首尾),10k prompts、3000 步、**4 张 A100** 即可;VBench 提升温和(+0.5~1.5pt)—— 适合当低成本基线,不是终点。
- 注意 Seedance 自己用的是直接 reward 最大化(x0 预测 + 复合 reward 反传)而非 DPO/GRPO ✅,ContentV(下)提供了该路线的开源参考。

### 5.3 蒸馏 / 少步生成
- **Self-Forcing**(arXiv [2506.08009](https://arxiv.org/abs/2506.08009)):自回归 rollout + KV cache 训练消除曝光偏差;Wan2.1-1.3B 上 DMD 目标 **64 张 H100 约 1.5 小时收敛(~4 H100-天)**,单 H100 17 FPS / 0.69s 延迟,VBench 84.31 ≥ 教师 84.26。
- **Seaweed-APT**(ByteDance,ICML 2025,arXiv [2501.08316](https://arxiv.org/abs/2501.08316)):对真实数据的对抗后训练,一步生成 2s 1280x720@24fps 实时;approximated R1 正则稳定训练。
- TSCD(Hyper-SD 系)即 Seedance 官方路径;CausVid / Causal Forcing++(arXiv 2605.15141)为自回归实时向。

### 5.4 多镜头 / 长视频
- **LCT**(arXiv [2503.10589](https://arxiv.org/abs/2503.10589),ICCV 2025):把单镜头 DiT 的 context 扩到场景级(跨 shot 全注意力 + interleaved 3D 位置编码 + 异步噪声),**不加参数**;可再微调为 context-causal 注意力 + KV cache 做自回归扩展。
- **HoloCine**(arXiv [2510.20822](https://arxiv.org/abs/2510.20822),CVPR 2026):基于 Wan 2.2 + 40 万策展多镜头样本微调 —— 直接可用/可复现的开源多镜头方案。

### 5.5 有限算力整体参考
- **ContentV**(ByteDance,arXiv [2506.05343](https://arxiv.org/abs/2506.05343)):8B 模型,256×64GB NPU × 4 周(~7000 加速器-天)从图像模型(SD3.5L)训到 VBench 85.14(> HYV-13B 83.24、Sora 84.28,< Wan2.1-14B 86.22);RLHF 不需人工标注(MPS reward + 可微采样反传)。给出了"中等算力能到哪"的公开锚点。

## 6. 可行性评估(数百–数千 H100-天)

**能逼近(至 Seedance 1.5 pro 水平)的维度:**

| 维度 | 手段 | 预算量级(粗估) |
|---|---|---|
| 推理速度 | Self-Forcing/DMD/APT 蒸馏 Wan 2.2 | 1.3B 上 ~4 H100-天;14B 上 50–150 H100-天 |
| 运动质量 | RM + online GRPO(DanceGRPO 配方) | 数百 H100-天(13B 级已验证) |
| 美学 | 美学 RM + SFT 美学数据(Wan 2.2 已有美学标签基础) | 100–300 H100-天 |
| prompt 遵循 | 密集 caption SFT 数据 + prompt 改写系统 + TA reward | 与 SFT 合并;prompt 改写近乎免费 |
| 多镜头叙事 | LCT / 复现 HoloCine | 100–200 H100-天 |
| 偏好对齐基线 | VideoDPO 自动偏好对 | <10 H100-天 |

**难以逼近的维度:**

- 最前沿一代(Seedance 2.0)的综合质量:预训练基座代差(数据规模、物理/复杂运动先验),数千 H100-天无法重训 —— 缓解手段是**站在最强开源底座上**并跟随其版本升级。
- 复杂物理一致性的天花板:后训练能改善偏好但难以注入基座没学到的物理;需在评测中单独跟踪该维度,避免对后训练抱不现实预期。
- I2V 保真(HYV1.5 后训练做满仍 −5.77% vs Seedance Pro)🟡:比 T2V 更依赖基座,优先级放低或依赖底座升级。

**整体预算画像:完整程序(评测 + RM + SFT + RL + 蒸馏 + 多镜头)约 500–1500 H100-天,在预算内。**

## 7. 技术路线建议(按优先级)

- **P0 评测先行(2–4 周,<50 H100-天)**:建 SeedVideoBench 式多维内部评测(≥200 prompts:运动/物理/多主体/镜头语言/指令遵循)+ 盲测胜率工具;跑 Wan 2.2 A14B、LTX-2、HunyuanVideo 1.5、Seedance 1.5 pro(API)基线。**这是验证"后训练假设"的前提。**
- **P0 Reward model(与评测并行)**:直接采用开源 VideoReward(VideoAlign)起步,用内部偏好数据(多维分开标,防 hacking)微调;对照 VisionReward checklist 路线。
- **P1 偏好优化主线(第 1 轮,~1–2 个月)**:先 VideoDPO 式自动偏好对做廉价基线 → 再上 online GRPO(DanceGRPO/Flow-GRPO 配方,同噪声共享 + best-of-N 筛选)于 **Wan 2.2 A14B**(Apache 2.0,最强开源底座;MoE 双专家意味着 RL 时要分别处理高/低噪专家——本身是一个值得记录的研究点)。目标:内部盲测运动质量与 prompt 遵循显著提升。
- **P1 SFT 数据策展流水线**:密集 caption(Tarsier2/Qwen-VL 级)、美学/运动质量过滤、类目均衡;同时上线推理侧 prompt 改写(照 Seedance 2.4 节配方,成本极低、观感收益直接)。
- **P2 蒸馏加速**:Self-Forcing(DMD)先在 TI2V-5B/1.3B 验证,再上 A14B;对照 APT 一步生成。产出物可直接用于低成本 API/推理服务。
- **P2 多镜头**:LCT 复现于 Wan 2.2,或直接基于 HoloCine 权重迭代。
- **P3 音频**:不自研;若产品需要音视频联合,直接采用/微调 LTX-2(其音频分支已开源)。

## 8. 来源清单

一手技术报告:Seedance 1.0(arXiv 2506.09113)✅、Seedance 1.5 pro(arXiv 2512.13507)✅、Wan(arXiv 2503.20314)、Wan 2.2 repo、LTX-Video(arXiv 2501.00103)、LTX-2(arXiv 2601.03233)、HunyuanVideo 1.5(arXiv 2511.18870)、ContentV(arXiv 2506.05343)。
方法论文:VideoAlign(2501.13918)、VisionReward(2412.21059)、DanceGRPO(2505.07818)、Flow-GRPO(2505.05470)、VideoDPO(2412.14167)、Self-Forcing(2506.08009)、Seaweed-APT(2501.08316)、LCT(2503.10589)、HoloCine(2510.20822)。
榜单:Artificial Analysis T2V/I2V arena(2026 年中快照,盲测 Elo)。

已知局限:🟡 条目未完成三票对抗验证(claims 均直接取自一手报告);Seedance 全部数字为厂商自报,无独立复现;Elo 快照随时间变化。
