# 06 — 根因分解、可行性评估与技术路线图

> Root-cause decomposition of the gap, what is (and is not) closable within hundreds-to-thousands of H100-days, and a prioritized execution roadmap.

[← 返回专题目录](README.md)

---

## 1. 根因分解

| 层 | 证据 | 差距大小 | 可弥补性 |
|---|---|---|---|
| **后训练(SFT+RLHF)** | Wan 全系无 RLHF 披露([03](03-open-source-landscape.md))vs Seedance 三 RM 多轮 RLHF([02](02-seedance-recipe.md));DanceGRPO 实测 MQ +181%([04](04-posttraining-methods.md));HYV1.5 靠满配后训练 T2V GSB 反超 Seedance Pro | **最大** | **高 —— 主攻** |
| **数据策展/caption** | Seedance 密集 caption 决定 prompt following;开源指令遵循普遍落后(HYV1.5 61.57 vs Veo3 73.77) | 大 | 中-高:SFT/偏好层可补;预训练 caption 无法追溯 |
| **蒸馏/推理优化** | Seedance ~10x vs 开源无官方蒸馏发布;Self-Forcing ~4 H100-天 | 大(纯工程) | **很高 —— 最便宜** |
| **多镜头叙事** | MM-RoPE+数据;LCT/HoloCine 已证明微调可补 | 中 | 高 |
| **预训练规模/物理先验** | Wan 已万亿 token;Seedance 未披露规模;ContentV 显示中等算力上限 ~VBench 85 | 对最前沿一代(Seedance 2.0)推测显著 | **低 —— 选最强底座缓解,不硬刚** |

**方法论警示**:Seedance 报告对四大支柱(数据/架构/后训练/蒸馏)**无消融隔离**,"后训练是主差距"是多方证据支持的**假设**——路线图 P0 的基线评测就是为了先证实/证伪它。

## 2. 可行性:哪些维度能逼近(目标 = Seedance 1.5 pro 水平)

依据 [01](01-benchmark-gap.md):最强开源权重距 Seedance 1.5 pro 仅 ~25–50 Elo,距 Seedance 2.0 ~250 Elo;Wan 2.6/2.7(闭源)已证明该差距可由迭代跨越。

### ✅ 能逼近

| 维度 | 手段 | 预算量级(粗估) |
|---|---|---|
| 推理速度 | Self-Forcing/DMD/APT 蒸馏 | 1.3B ~4 H100-天;14B 50–150 H100-天 |
| 运动质量 | RM + online GRPO(DanceGRPO 配方) | 数百 H100-天(13B 级已验证) |
| 美学 | 美学 RM + 美学 SFT(Wan 2.2 自带美学标签数据基础) | 100–300 H100-天 |
| prompt 遵循 | 密集 caption SFT + prompt 改写系统 + TA reward | 并入 SFT;prompt 改写近乎免费 |
| 多镜头叙事 | LCT / HoloCine 路线 | 100–200 H100-天 |
| 对齐基线 | VideoDPO 自动偏好对 | <10 H100-天 |

### ❌ 难以逼近

- **Seedance 2.0 级综合质量**:预训练基座代差(数据规模、物理/复杂运动先验),数千 H100-天不可能重训 → 站最强开源底座、跟随其升级。
- **物理一致性天花板**:后训练改偏好、难注入基座没学到的物理;评测中单独跟踪,不给后训练抱不现实预期。
- **I2V 保真**:HYV1.5 后训练做满仍 −5.77% vs Seedance Pro——更吃基座,优先级放低。

**整体预算画像:完整程序(评测+RM+SFT+RL+蒸馏+多镜头)≈ 500–1500 H100-天。**

## 3. 路线图(按优先级)

### P0 — 评测与 RM(2–4 周,<50 H100-天)
1. **多维内部评测集**(≥200 prompts:运动/物理/多主体/镜头语言/指令遵循)+ 盲测成对胜率工具;跑 Wan 2.2 A14B / LTX-2 / HYV 1.5 / Seedance 1.5 pro(API)基线 → **验证后训练假设**。
2. **Reward model**:开源 VideoReward 起步 + 内部偏好数据微调(按维度分开标,防 hacking);对照 VisionReward checklist 路线。

### P1 — 偏好优化主线 + SFT(1–2 个月)
3. VideoDPO 自动 pair 廉价基线 → **online GRPO**(DanceGRPO/Flow-GRPO 配方:噪声共享 + best-of-N)于 **Wan 2.2 A14B**。MoE 双专家的 RL 处理方式是待解研究点(分别/联合更新高低噪专家)。
4. **SFT 数据策展流水线**:密集 caption(Tarsier2/Qwen-VL 级)、美学/运动过滤、类目均衡;同步上线 **prompt 改写系统**(低成本高观感收益)。

### P2 — 能力扩展
5. **蒸馏**:Self-Forcing 先 5B/1.3B 验证再上 A14B;对照 APT;蒸馏版再过一轮 RLHF。
6. **多镜头**:HoloCine 权重迭代或 LCT 自训。

### P3 — 按需
7. **音频**:不自研;需要音视频联合直接采用/微调 LTX-2(音频分支已开源)。

### 底座选型
- **主线:Wan 2.2 A14B**(Apache 2.0、开源最强质量、生态最全);
- 配方蓝本:HunyuanVideo 1.5 后训练配方;
- 音频/极致速度:LTX-2(license 商用条款需复核)。
- 触发重估:Wan/LTX/HYV 发布新一代开源权重时。

## 4. 风险登记

| 风险 | 缓解 |
|---|---|
| 后训练假设被 P0 证伪(差距主要在基座) | 及早小规模验证;若证伪则转向"最强底座 + 蒸馏 + 垂类 SFT"的窄目标 |
| RM 质量不足 → reward hacking | 多维 RM、MPO/占优 pair、噪声共享、人评终检([04](04-posttraining-methods.md) §3) |
| 内部评测与真实用户偏好脱节 | 盲测 + 多维胜率;定期与 AA 榜单校准 |
| MoE RL/蒸馏无先例 | 先在 5B 稠密变体打通,再迁移;记录为研究产出 |

---

*[← 返回专题目录](README.md)*
