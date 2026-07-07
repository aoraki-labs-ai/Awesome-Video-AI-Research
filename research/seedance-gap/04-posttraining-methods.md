# 04 — 后训练方法工具箱:视频 Reward Model 与偏好优化(2025–2026)

> The open toolbox for closing the post-training gap: video reward models (VideoAlign, VisionReward) and preference optimization (Flow-DPO, VideoDPO, DanceGRPO, Flow-GRPO), with costs and measured gains.

[← 返回专题目录](README.md)

---

## 1. Reward Model

### VideoAlign / VideoReward(KwaiVGI,NeurIPS 2025,arXiv [2501.13918](https://arxiv.org/abs/2501.13918))⭐ 首选起点

- **数据**:16k 高质量 prompt(8 个元类别)× 12 个 T2V 模型(Gen2/Pika/Luma/Gen3/Kling 1.0/1.5 等)→ 108k 视频 → **182k 成对偏好三元组**;专业标注员按 **VQ(视觉质量)/ MQ(运动质量)/ TA(文本对齐)三维分开标**。
- VideoReward 多维 RM 显著超越既有视频 RM。
- 附 flow-matching 原生对齐算法族(统一 KL 正则 reward 最大化视角推导):
  - **Flow-DPO**(训练时,直接偏好优化)
  - **Flow-RWR**(训练时,reward 加权回归)
  - **Flow-NRG**(推理时,对噪声视频直接 reward 引导——零训练成本可先试)
- **代码与 RM 已开源**:[github.com/KwaiVGI/VideoAlign](https://github.com/KwaiVGI/VideoAlign)。182k 标注量 ≈ 自建同级 RM 的数据规模参考。

### VisionReward(THUDM,arXiv [2412.21059](https://arxiv.org/abs/2412.21059))

- **可解释 checklist 式 RM**:视频 9 维 / 20 子维 / **64 个二元判断题** + 线性加权,非黑箱打分。
- 数据:~48k 图像(3M QA 对)+ 33k 视频(2M QA 对)。
- 偏好预测超 VideoScore **17.2%**;用它做优化的 T2V 模型 pairwise 胜率比用 VideoScore 高 **31.6%**。
- **MPO(多目标偏好优化)**:只取"全维度占优"的 pair 训练,避免维度间此消彼长——与 Seedance 的"最好者不劣于最坏者"标注约束思想一致(见 [02](02-seedance-recipe.md) §3)。
- 已在 CogVideoX 上验证。

### 自建 RM 的第三条路:VLM 微调
- HunyuanVideo 1.5 配方:微调 VLM 按四维打分(文本对齐/图像对齐/视觉质量/运动动态),配 O(10K) prompt 的 GSB 人标——工程上最省(复用现成 VLM),见 [03](03-open-source-landscape.md) §4。

## 2. 偏好优化算法(按成本升序)

### VideoDPO(arXiv [2412.14167](https://arxiv.org/abs/2412.14167))— 廉价基线
- **全自动偏好对**:每 prompt 采 N=4 个视频,OmniScore(视觉质量×文本对齐的多维综合)排序,取首/尾做 win/lose——**零人工标注**。
- 成本:10k prompts(VidProm)、3000 步、**4×A100**、lr 6e-6。
- 收益温和:VBench +0.5~1.5pt(VideoCrafter2 80.44→81.93;CogVideoX 79.30→79.80)。
- 定位:先跑通流水线、拿基线,不是终点。

### Flow-DPO / Flow-RWR(VideoAlign 内)— 离线,flow 原生
- 对 rectified-flow/flow-matching 模型(Wan/LTX 均是)理论上更贴合;成本与 DPO 同级。

### Flow-GRPO(arXiv [2505.05470](https://arxiv.org/abs/2505.05470))— online RL 基础组件
- 两个关键技巧使 GRPO 能上 flow 模型:**ODE→SDE 转换**(引入随机性以支持探索/比较)+ **去噪步数缩减**(训练时少步采样降成本)。
- 被 Seedance 1.5 pro 的 RLHF 引用(见 [02](02-seedance-recipe.md) §5)。

### DanceGRPO(ByteDance,arXiv [2505.07818](https://arxiv.org/abs/2505.07818))— 已验证的最大收益
- 首个统一 diffusion + rectified flow 的 GRPO 框架(采样 SDE 化)。
- **实测(HunyuanVideo + VideoAlign reward)**:视觉质量 4.51→7.03(**+56%**),运动质量 1.37→3.85(**+181%**);I2V 用 SkyReels-I2V 验证。
- 稳定性技巧(直接抄):
  - 同 prompt **共享初始噪声**——防 reward hacking;
  - **best-of-N 轨迹筛选**:16 采样取 top-8 / bottom-8 训练。
- 背景:DDPO/DPOK 等早期 RL 在 >100 prompts 规模就不稳、且与现代 ODE 采样器不兼容——GRPO 系是当前视频 RLHF 的可扩展路径。

### 直接 reward 最大化(Seedance 路线)
- 预测 x0 + 复合 reward 直接反传,多轮迭代(Seedance 声称优于 DPO/PPO/GRPO,无第三方复现)。
- 开源参考:ContentV 的可微采样 reward 反传(MPS reward,免人工标注),见 [03](03-open-source-landscape.md) §5。

## 3. Reward hacking:已知坑与对策

| 坑 | 对策 |
|---|---|
| 单一综合分 → 过饱和/慢动作偏置等伪影 | 多维 RM 分开打分 + 复合;标注按维度分开 |
| 维度间此消彼长 | MPO 全维占优 pair(VisionReward);"不劣于"标注约束(Seedance) |
| 同 prompt 采样噪声差异被 RM 利用 | 共享初始噪声(DanceGRPO) |
| RM 过拟合 → 后期 reward 升、人评降 | 保留人评盲测做终检;RM 定期用新偏好数据刷新(Seedance 多轮迭代) |

## 4. 选型建议(结合预算)

| 阶段 | 方案 | 量级 |
|---|---|---|
| 起步 | 开源 VideoReward + Flow-NRG(推理时引导,零训练)| 数 GPU-天 |
| 基线 | VideoDPO 自动 pair + Flow-DPO | <10 H100-天 |
| 主线 | 内部偏好数据微调 RM → DanceGRPO/MixGRPO online RL | 数百 H100-天(13B 级已验证) |
| 进阶 | 直接 reward 最大化(ContentV/Seedance 式)对照实验 | 与主线同级 |

---

*下一篇:[05 — 蒸馏加速与多镜头](05-distillation-multishot.md)*
