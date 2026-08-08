# 07 — MiniMax H3(Hailuo 3.0):首个登顶主流评测的开源权重模型

> MiniMax H3 (open-weighted 2026-08-03) is now the strongest open-weights video model by a wide margin (+260 Elo over LTX-2.3), the first to rank #1 in any mainstream eval dimension — but "open" comes with a 768p self-hosting ceiling (Context-IR & 2K-Regenerate stay API-only), an undisclosed training recipe, and a restrictive license (US/EU/UK/KR excluded, $20M revenue gate, no cross-model output training).

[← 返回专题目录](README.md)

- 调研日期:2026-08-08(定向深调研:一手来源并行抓取 + 关键数字双源交叉;未跑三票对抗验证)
- 来源分级:🟢 一手(官方博客/HF 模型卡/LICENSE 原文/AA 榜单直抓)🟡 权威三方 🟠 社区实测/媒体

---

## 1. H3 是什么,开放边界在哪

- 2026-07-31 发布、**08-03 开放权重**。定位不是"视频模型"而是**全模态生成系统**:统一读文本/图像/视频/音频上下文,输出 4–15s、24fps、原生立体声视频;参考输入上限 **9 图 + 3 视频(每段 2–15s,合计 ≤15s)+ 3 音频** 🟢([官方博客](https://www.minimax.io/blog/minimax-h3)、[开源公告](https://www.minimax.io/news/minimax-h3-open-source))。
- **三件套只开了一件** 🟢:
  | 组件 | 作用 | 状态 |
  |---|---|---|
  | **H3-Base** | 核心生成器,两个 checkpoint:FL2VA(文/图驱动、首尾帧)+ Ref2VA(参考驱动),含 processor/tokenizer/text_encoder/transformer/visual_vae/audio_vae,BF16 | **开源**([HF](https://huggingface.co/MiniMaxAI/MiniMax-H3)) |
  | H3-Context-IR | 多模态指令 → 结构化 Context IR(~100K token 推理压缩到 ~4K) | API-only,未承诺开源 |
  | H3-Regenerate-2K | 768p 输出以原上下文回输模型,In-Context 再生成到 2K | API-only,"ready 后开源" |
- **自部署天花板:768p + 立体声**;官方 2K 产品级质量必须走 API(2K 输出 $7.80/min)🟡([AGIDaily](https://agidaily.cc/articles/minimax-h3-open-weights))。

## 2. 架构(已披露部分)🟢

- **H3-Omni-Transformer**:33B **dense 单流** Transformer,50 层、hidden 5376、56 头、3D MM-RoPE(时/高/宽);注意力与 FFN 无模态专属结构,模态参数只在输入输出层与 AdaLN 分支。
- **~13B 参数在 AdaLN 分支**:调制输出可预计算缓存 → 纯推理部署可不加载这 13B(有效推理负载 ~20B)。
- **H3-Encoder = Qwen3-VL-32B 全量预训练权重**,取其第 50 层 hidden states 供给 Omni-Transformer。
- 视觉 VAE:16x 空间 / 4x 时间压缩、24 latent 通道("有效序列长度 4x 增益");音频 VAE:32kHz → 40Hz latent。
- 预训练任务清单(官方博客):T2I、T2V+原生立体声、**多镜头建模**、T2A、跨模态"广义参考与编辑"——多镜头是预训练原生任务(对比 Seedance 同为原生,Wan 2.2/LTX 无)。
- 发布的是 **CFG-distilled** 权重;sparse attention 训练版"初始开源不含"。

### 2.1 架构深挖(2026-08-08 增补)

**参数解剖:33B ≠ 33B 推理负载。** 50 层 × hidden 5376 × 56 头的单流 dense Transformer,其中 ~13B 参数在 AdaLN 分支——条件注入走 **AdaLN 调制而非 cross-attention**,调制输出对每个 timestep 可预计算缓存,推理必载参数 ~20B。这带来三个工程红利:① 显存/量化友好(社区 "pruned INT8 + ConvRot" 版直接对 AdaLN 剪枝,主干压到 ~20GB);② 与 MMDiT 类(Seedance、SD3)"文本 token 进注意力"的路线相反,条件强度弱但序列短、算力省;③ **LoRA/风格微调的天然落点**(AdaLN 分支小步微调即可改风格/域,社区 default-LoRA 工作流已跑通)。

**两大件,不是一个模型。** 完整系统 = 33B 生成器 + **Qwen3-VL-32B 编码器(全量权重,取第 50 层 hidden states)** + 双 VAE,HF 仓库全量 498GB(FL2VA/Ref2VA 双 checkpoint 各含 transformer)。BF16 部署 ~123.6GB/4 卡的主要原因是把 65B+ 参数的两大件一起装。两件可独立量化:社区常见组合 = 主干 INT8-pruned(~20GB)+ 编码器 NVFP4/AWQ(~15GB)+ video VAE fp16(~5GB),合计 ~40GB——**仍超单张 5090/4090 显存,需 VAE offload(CPU-VAE)或顺序加载**,这是自部署实测的第一道坎。

**单流全模态序列 = 音画同步的来源。** 视频 latent(VAE 16x 空间/4x 时间压缩、24 通道)与音频 latent(32kHz→40Hz)**交错在同一 token 序列里**,由 3D MM-RoPE 统一编位置,音画对齐发生在注意力内部——对比 LTX-2 的双流+跨模态桥、Seedance 1.5 的双分支 MMDiT+联合模块,H3 是三者中最"激进统一"的设计。多镜头、参考图/视频/音频(9+3+3)也都是同一序列里的上下文 token,没有专用模块。

**任务分叉在 checkpoint 层。** FL2VA(文/图/首尾帧驱动)与 Ref2VA(参考驱动)是两份 transformer 权重而非一个模型加开关——部署要装两份、微调要选边,这是"统一架构"营销下的实际工程边界。

**系统件才是产品差距。** Context-IR:多模态指令 → 结构化中间表示,~100K token 推理压缩到 ~4K(疑似 Qwen3-VL 系 agentic 预处理,做参考解析/镜头规划);In-Context Regeneration:768p 成片连同原上下文回炉再生成 2K(自级联,非独立超分网络)。两者都不在开源包里——**H3 再次印证本专题的核心论点:榜单成绩 = 基座 + 后训练 + 系统工程,单独拿到权重只得其一。**

## 3. 横评:是不是最好的开源视频模型?

**按人类偏好 Elo:是,且断层领先;但要按维度和"开放完整度"打折扣。**

### AA 盲测榜(2026-08-08 直抓快照)🟢

**T2V(with audio)**:

| 排名 | 模型 | Elo | 开源权重 |
|---|---|---|---|
| #1 | Gemini Omni Flash | 1244 | 否 |
| **#2** | **MiniMax H3** | **1240** | **是** |
| #3 | Dreamina Seedance 2.0 720p | 1224 | 否 |
| #4 | Wan2.7-260612 | 1161 | 否(API) |
| #7/8 | Kling 3.0 Pro / Wan 2.7 | 1111 | 否 |
| #20 | Seedance 1.5 pro | 1000 | 否 |
| #22/23 | LTX-2.3 Fast / Pro | 980 / 962 | 是 |

- **开源权重内部:H3 领先第二名 LTX-2.3 Fast 约 260 Elo**——本专题 [01 篇](01-benchmark-gap.md)"最强开源权重 ≈ 落后前沿 250 Elo"的结论被 H3 一举清零(距全场第一仅 4 Elo)。
- **Video Editing #1(Elo 1130)**:开源权重模型首次在 AA 主流维度登顶 🟡。
- **I2V #3**(~1185–1187),仍落后 Seedance 2.0(1196)与 Gemini(1193)🟡——I2V 更吃基座的规律(见 [06 篇](06-roadmap-feasibility.md))继续成立。

### 但三个折扣

1. **榜上的 H3 是 API 全链路**(Context-IR + 2K):自部署 H3-Base 只有 768p,且没有 Context-IR 的指令结构化——**自部署质量 ≠ 榜单质量**,差多少无公开定量数据(标记为待实测)。
2. **前沿又动了**:Seedance 2.5 已于 2026-07 底发 API(30s 单次生成、原生 4K、50 个多模态参考、区域级重绘)🟠([第三方实测](https://www.atlascloud.ai/blog/ai-updates/seedance-2.5-vs-minimax-h3)),尚未上榜。
3. **第三方 4-shot 实测**(同参考包同 prompt)🟠:角色一致性上 H3 与 Seedance 2.x 相当,结论是"一致性主要是参考包工程问题而非模型代差";H3 实用优势在 1440p 竖屏直出、英文字幕 8 个剪辑点不跳字、原生立体声。

## 4. 部署要求

| 配置 | 显存/硬件 | 来源 |
|---|---|---|
| BF16 全精度(官方推荐) | ~123.6GB,**4 卡**(SGLang `--num-gpus 4 --ulysses-degree 4`;vLLM/diffusers/ComfyUI 均支持) | 🟢 模型卡 |
| ComfyUI 优化包(调制权重剪枝) | 42.5GB 最小组合(-66%) | 🟡 |
| NVFP4(社区量化) | 单张 RTX 5090(31.7GB),~175s/条 | 🟠 [实测](https://ai-muninn.com/en/blog/minimax-h3-nvfp4-rtx5090) |
| INT8 | ~21–24GB | 🟠 |
| INT4 | ~11.3GB,**RTX 3060 12G 可跑**(需 offload) | 🟠 |

- 两个 checkpoint(FL2VA/Ref2VA)要分开部署,任务不同。
- AdaLN 13B 可缓存不加载是显存优化的官方抓手;sparse attention 版未放。
- 生态速度惊人:开源 5 天内已有 GGUF/NVFP4/INT4 量化、ComfyUI 官方 repack(Comfy-Org/MiniMax-H3)、社区 LoRA([Turbo-LoRA](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora) 等)🟡。

## 5. 算法公开程度:能抄什么,抄不到什么

| 层 | 公开? | 说明 |
|---|---|---|
| 架构 | ✅ 大部分 | 单流 dense、AdaLN 集中化、Qwen3-VL-32B 做 encoder、双 VAE 参数——**"复用大 VLM 全量权重当条件编码器"是最值得抄的设计**(对比 Wan 用 umT5、LTX 用 Gemma3-12B) |
| 权重 | ✅(H3-Base) | BF16,可微调;但发布版已是 CFG-distilled |
| 训练配方 | ❌ | 数据规模/构成、SFT、**RLHF/reward model、蒸馏细节全部未披露**;技术报告"承诺中,尚未发布" 🟢 |
| 训练代码 | ❌ | 无官方 finetune/LoRA 脚本(社区自建中) |
| Context-IR | ❌ | 系统提效核心之一;本质是"多模态指令 → 结构化中间表示"的重推理预处理 |
| 2K 再生成 | ❌(承诺开) | In-Context Regeneration,非独立上采样器 |

**License(HF LICENSE 原文,🟢)——比 Wan Apache 2.0 严苛得多:**

- **地域排除:美国/欧盟/英国/韩国**不在授权范围——排除区内"使用、复制、修改、分发、展示 Works **及其 Outputs**"都不许可。
- 年营收 >$20M 的商业产品需向 MiniMax 单独书面授权(无 MAU 门槛,部分媒体报道的"1M MAU"与原文不符)。
- **不得用 H3 或其输出改进任何其他 AI 模型**(H3 自身及其衍生模型除外)——意味着:❌ 用 H3 输出蒸馏/训练 Wan 系模型;❌ 用 H3 输出训练通用 reward model(若该 RM 用于优化其他模型);✅ 微调 H3 自身。
- 商业产品 UI 必须显著标注 "MiniMax H3"。

## 6. 对我们路线的影响与优化思路

### 结论先行

1. **[01 篇]"开源 vs 前沿 ≈ 250 Elo"的差距叙事作废**:开源权重最强已到 1240(距第一 4 Elo)。竞争焦点从"追质量"转移到"**开放完整度**"(自部署 768p vs API 2K)与"**license 可用性**"。
2. **底座选型变成双底座策略**(替代单选 Wan 2.2):
   - **H3-Base = 能力上限底座**:原生多镜头 + 原生音频 + 最强 Elo,直接覆盖我们 P2/P3 两个 backlog(LCT 多镜头、LTX-2 音频)——这两项如在 H3 上可能不再需要自研;
   - **Wan 2.2 A14B = 后训练研究载体**:Apache 2.0 无枷锁、无 RLHF 底子(增益空间大、归因干净)、14B 激活 rollout 便宜。GRPO 在 33B dense 上的 rollout 成本约为 Wan 激活参数的 2.4x,且 H3-Base 已是 CFG-distilled + 大概率过了 RLHF——**在 H3 上做偏好优化的边际增益与训练稳定性均存疑,先小规模探针再定**。
3. **License 红线进入工程决策**:H3 输出不得反哺其他模型 → 我们的 reward model / 蒸馏管线必须做数据谱系隔离(H3 生成物只用于 H3 系实验);服务美国/欧盟用户的产品不能用 H3(含其输出)——商业化路径需法务复核。

### 具体优化思路(按杠杆排序)

1. **复刻 Context-IR(最高杠杆,纯开源可做)**:自部署与 API 的质量差主要在 Context-IR + 2K。Context-IR 本质 = "重推理的多模态 prompt 结构化"(~100K→4K token),与 Seedance 的 prompt 改写系统([02 篇](02-seedance-recipe.md) §3)同类但更重。**用开源 Qwen3-VL-32B(恰好是 H3-Encoder 同源)自建指令→结构化描述层**,成本低、纯推理侧、可 A/B 定量测增益。
2. **2K 缺口的开源替代**:短期用通用视频超分/refiner 兜底;中期尝试复刻 In-Context Regeneration(低分辨率输出 + 原上下文回输,微调 H3-Base 自身做二段生成——属"改进 H3 自身",license 允许)。
3. **域内 LoRA/SFT on H3-Base**:社区已证明 LoRA 可行;用我们的美学/运动策展数据做垂类微调,是 license 允许且增益明确的路径。
4. **部署优化**:AdaLN 缓存 + NVFP4/INT8 量化(单 5090/4090 级可服务 768p);等 sparse attention 版放出再评估长时长成本。蒸馏加速方向(Self-Forcing 系,[05 篇](05-distillation-multishot.md))对 H3 同样适用,但注意发布版已 CFG-distilled,少步蒸馏收益要重新测。
5. **评测先行不变**:M2 基线评测名单加入 H3-Base(自部署 768p)与 H3 API(全链路)两个条目——**量化"开放缺口"本身就是有价值的公开产出**(社区还没有这个数字)。
6. **RLHF 主线保持在 Wan 2.2 上推进**:归因干净、无 license 枷锁;H3 的出现不改变"后训练是差距主因"的方法论结论,反而是又一例证(H3 榜单成绩 = 全链路系统工程,基座权重只是其中一件)。

### 风险与未知(待技术报告/后续实测)

- H3-Base 是否已含 RLHF(官方未说;若含,偏好优化空间受限)。
- 自部署 768p 与 API 全链路的盲测差距无公开数据——我们实测补位。
- 地缘条款的执行口径(Outputs 的传播限制如何落地)与 $20M 门槛对未来的约束。
- Seedance 2.5 上榜后的格局(原生 4K/30s/50 参考,前沿仍在拉开)。

---

## 来源

🟢 一手:[MiniMax 官方博客](https://www.minimax.io/blog/minimax-h3) · [开源公告](https://www.minimax.io/news/minimax-h3-open-source) · [HF 模型卡 MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) · [LICENSE 原文](https://huggingface.co/MiniMaxAI/MiniMax-H3/raw/main/LICENSE) · [AA T2V 榜单](https://artificialanalysis.ai/video/leaderboard/text-to-video)(2026-08-08 快照)
🟡 权威三方:[AGIDaily 开放边界分析](https://agidaily.cc/articles/minimax-h3-open-weights) · [HF 社区博客](https://huggingface.co/blog/ResterChed/minimax-h3-hailuo-3-0) · [Comfy-Org repack](https://huggingface.co/Comfy-Org)
🟠 社区/媒体:[NVFP4 单 5090 实测](https://ai-muninn.com/en/blog/minimax-h3-nvfp4-rtx5090) · [Seedance 2.5 vs H3 4-shot 实测](https://www.atlascloud.ai/blog/ai-updates/seedance-2.5-vs-minimax-h3) · [部署指南](https://www.atlascloud.ai/blog/guides/minimax-h3-open-source-weights) · [Turbo-LoRA](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora)

*已知局限:未跑三票对抗验证;技术报告未发布,训练配方结论可能随其发布更新;Elo 为 2026-08-08 快照。*
