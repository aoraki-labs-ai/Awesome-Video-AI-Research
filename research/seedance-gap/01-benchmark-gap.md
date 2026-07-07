# 01 — 差距量化:榜单、人评与时间线

> Quantifying the Seedance-vs-open-source gap with blind human-preference Elo (Artificial Analysis), head-to-head GSB evals, and VBench anchors.

[← 返回专题目录](README.md)

---

## 1. Artificial Analysis 盲测 Elo(2026 年中快照)

Artificial Analysis video arena 用盲测成对人类偏好投票计算 Elo,是目前最中立的闭源 vs 开源横评。

### Text-to-Video

| 排名 | 模型 | Elo | 样本数 | 发布 | 价格 | 开源权重 |
|---|---|---|---|---|---|---|
| #1 | Dreamina Seedance 2.0 720p | **1222** (±9) | 9,618 | 2026-03 | $9.07/min | 否 |
| #6 | Wan 2.7 | 1102 (±10) | 3,290 | 2026-04 | $16.90/min | 否(API) |
| #16 | Wan 2.6 | 1028 (±9) | 5,896 | 2025-12 | $9.00/min | 否(API) |
| #17 | Seedance 1.5 pro | 1000 | 5,687 | 2025-12 | $11.86/min | 否 |
| #19 | LTX-2.3 Fast | 976 (±9) | 6,711 | 2026-03 | $2.40/min | **是** |
| #20 | LTX-2.3 Pro | 961 (±9) | 6,814 | 2026-03 | $3.60/min | **是** |
| #21 | LTX-2 Fast | 948 (±9) | 5,728 | 2025-10 | $2.40/min | **是** |

### Image-to-Video

| 排名 | 模型 | Elo | 说明 |
|---|---|---|---|
| #1 | Dreamina Seedance 2.0 720p | **1189** | 8,136 样本 |
| — | Wan 2.7 | 1094 | |
| #18 | Seedance 1.5 pro | 1000 | |
| #19 | LTX-2.3 Pro | 952 | **最强开源权重** |
| — | LTX-2.3 Fast / LTX-2 Fast | 946 / 936 | |
| — | Wan 2.6 | 897 | |

### 读数

- **开源权重 vs 最前沿闭源 ≈ 240–250 Elo**(LTX-2.3 vs Seedance 2.0)。
- **开源权重 vs 上一代闭源旗舰 ≈ 25–50 Elo**(LTX-2.3 vs Seedance 1.5 pro;I2V 上只差 ~48)。
- 阿里 Wan 2.6/2.7(未开源)已越过 Seedance 1.5 pro——差距不是"开源方法做不到",而是"最新权重不放出来"。
- Wan 2.2、HunyuanVideo、CogVideoX、Mochi 已掉出当前榜单(样本不足/下架),无直接 Elo。
- 开源模型价格优势显著:LTX-2.3 Fast $2.40/min vs Seedance 2.0 $9.07/min(3.8x)。

## 2. 历史锚点

| 时间 | 事件 | 证据 |
|---|---|---|
| 2025-06-10 | Seedance 1.0 同时登顶 AA T2V/I2V 双榜;I2V 领先第二名 Veo 3、第三名 Kling 2.0 **100+ Elo** | Seedance 1.0 报告(arXiv 2506.09113),3/3 对抗验证通过;The Decoder 等第三方同期报道佐证 |
| 2025-07-28 | Wan 2.2 开源(Apache 2.0);自报 Wan-Bench 2.0 上"多数维度超越领先商业模型"(自报,无第三方复现) | Wan2.2 repo |
| 2025-11 | LTX-2 位列 AA I2V #3 / T2V #4(当时快照);人评超开源 Ovi,接近 Veo 3 / Sora 2 | LTX-2 技术报告(arXiv 2601.03233) |
| 2025-11 | HunyuanVideo 1.5(开源 8.3B)GSB 人评:**720p T2V vs Seedance Pro +11.02%(赢)**;I2V −5.77%;指令遵循 61.57 vs Veo3 73.77 | HYV 1.5 技术报告(arXiv 2511.18870) |
| 2025-12 | Seedance 1.5 pro / Wan 2.6 发布 | |
| 2026-03 | Seedance 2.0 发布,重新拉开 ~200 Elo | AA 榜单 |

## 3. VBench 锚点(自动指标,补充参考)

| 模型 | VBench 总分 | 备注 |
|---|---|---|
| Wan2.1-14B | 86.22 | 开源最高档 |
| ContentV-8B | 85.14 | ~7000 加速器-天训出(见 [03](03-open-source-landscape.md)) |
| Sora | 84.28 | |
| Self-Forcing(Wan2.1-1.3B 蒸馏) | 84.31 | 4 步生成,≥教师(84.26) |
| HunyuanVideo-13B | 83.24 | |
| CogVideoX-5B | 81.61 | |
| LTX-Video | 80.00 | 速度优先设计点 |

注意:VBench 与人类偏好 Elo 相关性有限(Wan2.1 VBench 高于 Sora 但 arena 表现不及),**内部评测应以多维盲测胜率为主、VBench 为辅**。

## 4. 对评测集设计的启示

1. 必须分维度:运动质量 / 物理一致性 / prompt 遵循(含多主体)/ 美学 / 多镜头叙事 / I2V 保真,单一综合分会掩盖真实差距结构(HYV1.5 T2V 赢 Seedance 但 I2V 输,就是维度分化的实例)。
2. 盲测成对 GSB/Elo 优先于自动指标;每对比至少数百样本控制 ±10 Elo 噪声。
3. 把 Seedance 1.5 pro(API)作为"可达上界"对照、Seedance 2.0 作为"前沿参考"(不设为目标,见 [06](06-roadmap-feasibility.md))。

---

*来源与验证状态见 [README](README.md#8-来源清单);Elo 数字取自 2026 年中 AA 快照(🟡 一手抓取,未跑三票对抗验证),随时间变化。*
