# 08 — 评价闭环:后训练迭代中如何高效评价视频质量

> How to evaluate video generation quality efficiently enough to drive a post-training iteration loop: a three-tier eval stack (objective metrics + calibrated VLM judges + human anchor), what existing benchmarks cover, when to use VLM-as-judge vs specialized models, and the calibration discipline (TPR/TNR, per-dimension judges, rubric learning) that keeps the ruler trustworthy.

[← 返回专题目录](README.md)

- 调研日期:2026-08-09
- 视角:闭环工程(loop engineering)——迭代产出 ≈ 每天转数 × 每转信息量;**代码域的 loop 有编译器/测试当免费验证器,视频生成没有,所以"造尺子"本身是迭代基建的一半**

---

## 0. 先回答核心问题:"是不是就用 VLM/大模型来评?"

**分维度混合,不是单选。** 三个证据:

1. **MJ-Bench**(评"裁判"的 benchmark):闭源 VLM 综合反馈最好,但**专用小打分模型在"图文一致"和"画质"维度比通用 VLM 更准且便宜**;VLM 只在需要推理的维度(安全、偏见、物理常识)胜出。→ **能算的别让 VLM 判**。
2. **VLM judge 有系统性弱点**:位置偏差、评分不稳定、幻觉、可被"好看但错"的样本欺骗([Fooling the LVLM Judges](https://arxiv.org/pdf/2505.15249));且**未校准的 judge 分数不可信**(Anthropic evals 指南:"在有人真正读过 transcript 之前,不把 eval 分数当真")。
3. **不同 VLM 的感知盲区不同**:[JudgeFit](https://arxiv.org/abs/2606.22918)(2026-06)证明对每个 VLM judge 学"它自己的评价维度分类法",比全 judge 共用一套全局 rubric 平均好 ~32%——**rubric 要适配 judge,不能一套走天下**。
4. **有几何真值的维度上,VLM judge 的可信度只有客观指标的一半**:[SpatialEdit-Bench](https://arxiv.org/abs/2604.04911)(2026)评"视角变换对不对"时,用 VGGT 反推相机位姿算出的 Viewpoint Error 与真实位姿排序的 Spearman 相关 **0.932**,而 **vision-language judge 只有 0.445**。这是"能算的别让 VLM 判"目前最硬的单一数字。

结论:**客观可算维度用专用检测器/小模型;开放语义与物理维度用"rubric 化 + 校准过"的 VLM judge;人评做锚,校准前两层。** 且**评测 judge 必须与训练 reward model 分离**——用同一个模型既当训练信号又当验收标准,等于自己判自己及格,reward hacking 会被完美掩盖。

## 1. 现有方案盘点(2025–2026)

### 自动 benchmark 套件

| 名称 | 覆盖 | 特点/局限 |
|---|---|---|
| VBench / [VBench-2.0](https://arxiv.org/abs/2503.21755) | 2.0 = 5 大类 18 细维:人体保真、可控性、创意、**物理、常识** | 通才(VLM/LLM)+ 专才(异常检测器)混合,有人工标注对齐;公共榜单已趋饱和,适合做回归子集而非目标 |
| [Video-Bench](https://arxiv.org/abs/2504.04907) | 首个全维度 MLLM 评审的视频生成 benchmark | few-shot scoring + chain-of-query,声称全维度超越既有指标的人类对齐 |
| [VMBench](https://arxiv.org/html/2503.10076v1) | **运动**专项:感知对齐的运动指标分解 | 我们最关心的 motion quality 维度的现成分解参考 |
| [VideoPhy-2](https://arxiv.org/abs/2503.06800) | **物理常识**(动作中心,3940 prompts,5 级 Likert,标注物理规则) | 附 AutoEval 自动评估器;hard 子集上最好模型的 joint(语义+物理同时达标)v1 报告仅 22%(后续版本收录更强模型约 47.7%)—— 物理是真正拉得开分的"最差集"候选 |
| [EvalCrafter](https://github.com/EvalCrafter/EvalCrafter) | 综合 T2V 质量 | 较早,维度覆盖广 |

### VLM/生成式打分模型(可自部署)

| 名称 | 形态 | 关键数字 |
|---|---|---|
| [VideoScore2](https://arxiv.org/abs/2509.22799) | 多维 + chain-of-thought 理由的打分模型(开源) | 视觉质量/文本对齐/物理常识三维,带可读理由 |
| [UnifiedReward(-Think)](https://github.com/CodeGoat24/UnifiedReward)(NeurIPS 2025) | 统一多模态 CoT reward/judge,支持成对排序 | 生成式判决,可当 judge 也可当 RM——**但同一权重别两用**(见 §0) |
| VideoReward([VideoAlign](https://arxiv.org/abs/2501.13918))/ [VisionReward](https://arxiv.org/abs/2412.21059) | scalar RM / checklist RM | 我们的训练 reward 首选(见 [04 篇](04-posttraining-methods.md));checklist 式天然可审计 |
| [VideoRewardBench](https://arxiv.org/html/2509.00484) | **评 judge/RM 的 benchmark**(1563 三元组) | 选型时先用它筛 judge,别只看论文自报 |

### 人评

- 成对盲测 GSB / Elo 是唯一金标(AA arena、HunyuanVideo 1.5 内部 GSB 都是这个形态);贵且慢,只能当**锚**,不能当日常闸门。

## 2. 三层评价栈(我们的方案)

**设计原则:每层回答"这一转值不值得继续",成本逐层放大、频率逐层降低。**

```
L0 夜跑闸门(每 checkpoint,全自动,~分钟级/百样本)
    客观检测器:主体 ID 一致性跟踪、时序闪烁、OCR 文字正确率、镜头切换检测
    + 已校准的开源 RM 分数(VideoReward VQ/MQ/TA 三维)
    + VBench-2.0 物理/常识子集
    产出:全维度回归表 + hardmin(最差集数字);任何维度回退 >阈值 → 挡住
         ↓ 过闸
L1 VLM Judge 成对评审(每实验,~小时级)
    固定 seed 同 prompt,新旧 checkpoint 成对出片
    每维度独立 judge(运动/物理/prompt 遵循/美学/多镜头),rubric 化 + few-shot
    成对比较(win/lose/tie)而非绝对分 —— 成对判别一致性远高于绝对打分
    产出:分维度胜率 + judge 给出的失败理由(供归因)
         ↓ 里程碑
L2 人评锚(每周或每里程碑,~天级)
    内部 200-prompt 分难度集(常规/硬运动/物理/多主体/多镜头)盲测 GSB
    双用途:① 决策(promote/reject);② 校准 L0/L1 —— 用人评标注测每个
    自动 judge 的 TPR/TNR,不达标的 judge 降级为"提示",不作闸门
```

### 关键设计决定(每条都有出处)

1. **成对优先于绝对分**:同 prompt 固定 seed 的 A/B 成对判定,是 VLM judge 和人评一致性最高的形态(AA arena、HYV1.5 GSB、DanceGRPO best-of-N 全是成对逻辑)。绝对分只在 L0 的回归监控里用。
2. **每维度独立 judge,不用一个 judge 打所有维度**(Anthropic evals 指南原话);进一步,按 [JudgeFit](https://arxiv.org/abs/2606.22918) 的结论,rubric 要针对所选 VLM 校准,而非直接照搬论文 rubric。
3. **校准口径用 TPR/TNR,不用 raw agreement**(Hamel Husain):质量闸门类指标天然类别不平衡——一个 raw agreement 96% 的 judge,抓坏样本的召回可能只有 50%。閘门资格 = 校准达标;不达标只记录不拦截。
4. **评测集分难度,目标写最差集数字(hardmin)不写平均**:平均值会掩盖灾难(物理 hard 集上 SOTA 也只有 ~47.7%,这才是能分出模型高下的地方);复合场景注意乘法效应(单条 92% ≈ 8 条全过仅 51%)。
5. **打分标准先立、并定期怀疑打分器本身**:模糊 case 的判法必须在看到数字前写进文档;打分器的 bug 以"成绩变好"的形式出现,是最难自查的一类错——排期里给"重读打分逻辑"留固定时间。
6. **负样本反向校验**:不仅证明"好的能过",必须证明"坏的会被拦"——建负样本集(已知伪影/物理错误/文字乱码),要求闸门全拦且失败原因命中预期。
7. **留出集纪律**:调参用的 prompt 集和汇报成绩的 prompt 集分开;留出集只跑不看着调,两者差值过大即报警(防对评测集过拟合)。
8. **训练 RM 与评测 judge 分离**:训练用 VideoReward/自建 RM,评测 L1 用不同家族的 VLM(如训练用 Qwen 系 RM,评测混入 Gemini/闭源 judge 抽查)——防止 reward hacking 在评测里自证清白。

## 3. 让尺子随迭代变准:rubric 学习

评价标准不是一次写完的(**criteria drift**,[Shankar et al., UIST 2024](https://arxiv.org/abs/2404.12272):给输出打分的过程本身在帮你定义标准)。两个可执行升级:

- **偏好对 → 学 rubric**:[AutoRubric-T2I](https://arxiv.org/abs/2605.17602) 证明从偏好对自动学出**加权自然语言 rubric**,标注量 <0.01%(几百到几千对即可起步),效果达到或超过微调 scalar RM,且产出可读可审计。我们做 L2 人评天然产生偏好对——**同一批数据既校准 judge 又长 rubric**,是复利资产。
- **rubric 版本化**:每条准则记录它从哪个 case 长出来(origin),修正只追加不覆盖——尺子的演进历史本身是团队知识。

注意一个反直觉的坑:**通用审美对齐会窄化表达**([arXiv 2512.11883](https://arxiv.org/html/2512.11883v3))——judge 会把所有输出推向"平均好看"。若业务需要风格差异化,rubric 要按业务线/风格分组学,不学一套通用审美。

## 4. 迭代节奏(把 eval 栈接进后训练 loop)

| 迭代物 | 转速 | 用哪层 | 判据 |
|---|---|---|---|
| prompt 改写/推理参数 | 天级多转 | L0 + L1 抽样 | 分维度胜率 |
| SFT 数据配比 | 天级 | L0 全量 + L1 | hardmin 不回退 + 目标维度胜率 |
| DPO/GRPO checkpoint | 每 run 多 checkpoint | **L0 夜跑逐 checkpoint** + L1 选优 | RM 分数上升 **且** 独立 judge 胜率上升(两线背离 = reward hacking 警报) |
| 蒸馏版本 | 周级 | L1 + L2 | 盲测胜率降幅 <5pt @ ≥5x 加速 |
| promote 决策 | 里程碑 | **L2 人评** | GSB 显著优 + 无维度显著回退 |

人只出现在三处:定标准(rubric/阈值)、读 L1 失败理由样本(归因)、L2 校准。其余无人值守——**判一次的成本决定转数上限,这是所有设计的出发点**。

## 4.5 两条来自生产的负结果:尺子会骗人,前提也会骗人

> 以下两条来自一个电商图像生成团队的生产实践(已脱敏),它们是**同一类失败的两个方向**,建议成对理解——只防其中一半,另一半会照样发生。

### ① 尺子会骗人:不只会虚高,也会**虚严**

已被广泛讨论的是**虚高**——打分器的 bug 把成绩做上去,而且"成绩变好"这个表象让人不会去查。

少被讨论的是**虚严**:**自建质量闸门的定义域错误会以"悄悄变严"的形式静默误杀,而不是以报错的形式暴露。** 实例:一个用于"量输入图"的像素测量器被复用去量生成图,全史统计 **48 次就绪 vs 39 次被否——45% 的候选被误杀丢掉,日志里只有一行 WARNING**。它不是靠工具发现的,是靠人停下来重读判定逻辑发现的。

**反制**:①闸门必须自带**误杀率监控**(重生成率超阈值即视为误杀信号);②**"成绩变差"和"闸门变严"在报表上长得一模一样,必须能分开**——否则你会去优化模型,而问题在尺子。

### ② 前提会骗人:写进架构文档本身是不可逆操作

同一团队记录:一个错误的技术前提("某类输入下模型只能瞎猜")**活了 18 轮**。原因不是没人质疑,而是**它一旦进了文档正文就获得了默认真值地位**,后续每一轮都在它之上做工作。

**反制**:**架构文档里的每条前提必须带证据徽标——`实测` / `文献` / `判断`,且 `判断` 类前提要有到期复核。** 评审时 `判断` 类段落单独看。

> 这两条与本篇 §2 的纪律是同构的:留出集纪律防的是"在自己判自己及格",证据徽标防的是"在自己给自己的前提上盖楼"。**闭环转得越快,尺子与前提的错误就被放大得越厉害——杠杆两头都放大。**

## 4.6 方法论:怎么让"被推翻"成为可交付物

本篇的多条结论来自一次跨团队的对抗验证协作,其中受托方(研究侧)三次推翻了委托方(架构侧)的主张。可复用的三条:

1. **请对方打自己的地基,并且把"部分推翻"当交付而不是失败。** 委托措辞决定结果——委托方原话是"**尽力推翻它,推翻我就改架构**"。如果委托方想要的是确认,受托方会不自觉地去找确认。这一条是前提,不是态度问题。
2. **不背书的主张要显式标注"不采纳为前提"。** 某论文在 intro 里断言"结构控制类方法做不到像素级保真,因为它们施加的是空间约束"——若成立会让两条技术路线直接出局,但该论文**没有做 head-to-head**。正确处理是记录该主张、标注为 `判断`、并写明本设计不采纳为前提。**影响越大的主张,越不能靠一句 intro。**
3. **更正要与原文并存,并标注"保留什么 / 降级什么 / 口径边界"。** 比直接改正文强——它让后来者看得见判断如何演进,而不是只看到一个结论。

⚠️ **一条容易失效的论证:"两条独立路径推到同一结构,所以可信"。** 它成立的前提是**两边起点不共享**。一旦双方进入紧耦合(一方的证据在塑造另一方的框架,另一方的框架又在决定向对方提什么问题),后续的"收敛"就有相当概率是**回声而非独立验证**,而且**这类论证失效时是无声的**。复核问题只有一句:**这次收敛的两条路径,起点是不是有一条来自对方?** 若是,它只算一致性检查,不算独立证据。

## 5. 落地顺序建议

1. **P0**:内部 200-prompt 分难度评测集 + 成对出片工具 + L2 人评一轮(先有金标,才能校准一切)——正好覆盖基线评测(Wan 2.2 / H3-Base / H3 API / Seedance API)。
2. **P0**:用 [VideoRewardBench](https://arxiv.org/html/2509.00484) + 自己的 L2 标注筛 L1 judge(候选:Qwen3-VL 自部署、VideoScore2、UnifiedReward、闭源 VLM 抽查),按 TPR/TNR 出校准报告。
3. **P1**:L0 夜跑管线(客观检测器 + VideoReward 三维 + VBench-2.0 物理子集),挂到每个训练 run。
4. **P1**:负样本集 + 留出集拆分。
5. **P2**:偏好对落库 → AutoRubric 式 rubric 学习,按季度刷新 judge rubric。

---

## 来源

评测方法:[SpatialEdit-Bench](https://arxiv.org/abs/2604.04911)(几何指标 vs VLM judge:0.932 vs 0.445) · [VBench-2.0](https://arxiv.org/abs/2503.21755) · [Video-Bench](https://arxiv.org/abs/2504.04907) · [VMBench](https://arxiv.org/html/2503.10076v1) · [VideoPhy-2](https://arxiv.org/abs/2503.06800) · [VideoScore2](https://arxiv.org/abs/2509.22799) · [UnifiedReward](https://github.com/CodeGoat24/UnifiedReward) · [VideoRewardBench](https://arxiv.org/html/2509.00484) · [JudgeFit / per-VLM taxonomies](https://arxiv.org/abs/2606.22918) · [MJ-Bench](https://mj-bench.github.io/) · [Fooling the LVLM Judges](https://arxiv.org/pdf/2505.15249)
方法论:[Anthropic — Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents) · [Hamel Husain — LLM-as-a-Judge](https://hamel.dev/blog/posts/llm-judge/) · [Who Validates the Validators (UIST 2024)](https://arxiv.org/abs/2404.12272) · [AutoRubric-T2I](https://arxiv.org/abs/2605.17602) · [审美对齐窄化](https://arxiv.org/html/2512.11883v3) · [Awesome-Evaluation-of-Visual-Generation](https://github.com/ziqihuangg/Awesome-Evaluation-of-Visual-Generation)(跟踪入口)

*§4.5 两条负结果由一个电商图像生成团队提供并授权引用(已脱敏);§4.6 方法论源自该次跨团队对抗验证协作。另参考团队内部闭环工程方法论(loop-engineering,含跨领域负结果档案;其六条规则 R1–R6 与五个反模式是本篇 §2 关键设计决定的直接来源)。*
