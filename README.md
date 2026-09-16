# Awesome Video AI Research 🎬

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

> A curated, dependency-ordered reading path for **generative video** — the models
> that make the pixels, the techniques that make them run, and the post-training
> that makes them good.

**What makes this list different:** every entry says *what it contributes*, not what
it is, and carries an explicit `prereq` list. The corpus is therefore a directed graph,
not a pile of links — you can start anywhere and walk backwards to the things you are
missing.

**Scope.** Generative video, inference efficiency, and post-training. Video
understanding and Video-LLMs are deliberately out of scope; see
[Adjacent Areas](#adjacent-areas) for where to go instead.

> [!NOTE]
> `README.md` is **generated** from [`data/papers.yaml`](data/papers.yaml) and
> [`data/taxonomy.yaml`](data/taxonomy.yaml) by [`scripts/gen_readme.py`](scripts/gen_readme.py).
> Edit the data, not this file. Every arXiv id below is checked against the arXiv API
> by `scripts/verify.py` — titles are the official ones.

**81 papers** across 12 sections · every arXiv id verified against the arXiv API · `prereq` edges make this a graph, not a list

---

## Contents

- [Foundations](#foundations)
  - [Diffusion, Flow and Sampling](#diffusion-flow-and-sampling)
- [Core Architectures](#core-architectures)
  - [Diffusion Transformers](#diffusion-transformers)
- [Video Generation](#video-generation)
  - [Models and Technical Reports](#models-and-technical-reports)
  - [Video VAE and Tokenizers](#video-vae-and-tokenizers)
- [Inference Efficiency](#inference-efficiency)
  - [Step Caching](#step-caching)
  - [Sparse and Structured Attention](#sparse-and-structured-attention)
  - [Kernels and Parallelism](#kernels-and-parallelism)
  - [Few-Step Distillation](#few-step-distillation)
  - [Quantization and Memory](#quantization-and-memory)
- [Post-Training](#post-training)
  - [Reward Models and Preference Optimization](#reward-models-and-preference-optimization)
- [Long Video and Evaluation](#long-video-and-evaluation)
  - [Long Video and Multi-Shot](#long-video-and-multi-shot)
  - [Evaluation](#evaluation)
- [Adjacent Areas](#adjacent-areas)
- [Research Notes](#research-notes)
- [Resources](#resources)
- [Labs & Companies](#labs-companies)
- [Contributing](#contributing)

---

<a id="foundations"></a>

## Foundations

The minimum you need before any video paper makes sense. Read in listed order —
each entry's `prereq` in `data/papers.yaml` encodes the dependency.

### Diffusion, Flow and Sampling

*Modern video models train with flow matching, not the DDPM objective. Skipping this section is the single most common reason later papers stop parsing.*

| Paper | Year | What it contributes | Prereq · Difficulty · Time | Code |
| :--- | :---: | :--- | :--- | :---: |
| **[Attention Is All You Need](https://arxiv.org/abs/1706.03762)** | 2017 | The attention block every model below is built out of. | ●●○○○ · 60min | — |
| **[Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)** | 2020 | Decouples sampling steps from training steps, making few-step sampling possible at all. | after `ddpm` · ●●●○○ · 60min | — |
| **[Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)** | 2020 | Makes denoising diffusion trainable with a simple noise-prediction loss. | ●●●○○ · 90min | [repo](https://github.com/hojonathanho/diffusion) |
| **[Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456)** | 2020 | Unifies DDPM and score matching as one SDE, which is where probability-flow ODE samplers come from. | after `ddpm` · ●●●●● · 150min | — |
| **[High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)** | 2021 | Moves diffusion into a compressed latent space; every model in this list inherits the VAE-plus-denoiser split. | after `ddpm` · ●●●○○ · 90min | [repo](https://github.com/CompVis/latent-diffusion) |
| **[Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598)** | 2022 | Trades diversity for prompt adherence with no extra classifier — and doubles inference cost. | after `ddpm` · ●●○○○ · 30min | — |
| **[Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364)** | 2022 | Turns sampler and noise schedule design into an explicit parameter space; the source of most sampler names in practice. | after `score-sde` · ●●●●○ · 150min | [repo](https://github.com/NVlabs/edm) |
| **[Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)** | 2022 | Replaces the diffusion objective with a straight-line velocity field; current video models train this way, not as DDPMs. | after `score-sde` · ●●●●○ · 120min | — |
| **[Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003)** | 2022 | Straightens the generative path so that fewer solver steps lose less quality — the basis of few-step distillation. | after `flow-matching` · ●●●●○ · 120min | — |
| **[Analyzing and Improving the Training Dynamics of Diffusion Models](https://arxiv.org/abs/2312.02696)** | 2023 | Fixes scale drift during training and introduces post-hoc EMA tuning. | after `edm` · ●●●●○ · 120min | [repo](https://github.com/NVlabs/edm2) |

---

<a id="core-architectures"></a>

## Core Architectures

How the denoiser is built. Everything current descends from DiT plus a joint
multimodal attention scheme.

### Diffusion Transformers

*The backbone. adaLN-Zero conditioning and MMDiT-style joint attention are the two ideas that carry into video.*

| Paper | Year | What it contributes | Prereq · Difficulty · Time | Code |
| :--- | :---: | :--- | :--- | :---: |
| **[Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748)** | 2022 | Replaces the U-Net with a transformer and introduces adaLN-Zero conditioning; the backbone of every model below. | after `ldm`, `transformer` · ●●●○○ · 90min | [repo](https://github.com/facebookresearch/DiT) |
| **[All are Worth Words: A ViT Backbone for Diffusion Models](https://arxiv.org/abs/2209.12152)** | 2022 | Concurrent transformer backbone that keeps long skip connections; useful contrast to DiT's design choices. | after `dit` · ●●●○○ · 60min | — |
| **[PixArt-α: Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis](https://arxiv.org/abs/2310.00426)** | 2023 | Shows a DiT can be trained to competitive quality on a small fraction of the usual compute. | after `dit` · ●●●○○ · 60min | — |
| **[Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206)** | 2024 | MMDiT — text and image tokens share one attention with separate weights; the direct ancestor of single-stream video models. | after `dit`, `rectified-flow` · ●●●●○ · 120min | — |

---

<a id="video-generation"></a>

## Video Generation

The models themselves, oldest first. For recent ones the technical report is
usually worth more than the architecture diagram — read them for data pipelines,
training infrastructure and post-training recipes.

### Models and Technical Reports

*Where the field actually is. Open-weight entries are marked in the notes.*

| Paper | Year | What it contributes | Prereq · Difficulty · Time | Code |
| :--- | :---: | :--- | :--- | :---: |
| **[Imagen Video: High Definition Video Generation with Diffusion Models](https://arxiv.org/abs/2210.02303)** | 2022 | Cascaded spatial and temporal super-resolution; the approach latent video models later replaced. | after `vdm` · ●●●○○ · 60min | — |
| **[Make-A-Video: Text-to-Video Generation without Text-Video Data](https://arxiv.org/abs/2209.14792)** | 2022 | Bootstraps video generation from an image model plus unlabeled video, sidestepping paired data scarcity. | after `vdm` · ●●●○○ · 60min | — |
| **[Video Diffusion Models](https://arxiv.org/abs/2204.03458)** | 2022 | First workable video diffusion model; introduces the factorized space-time U-Net. | after `ddpm` · ●●●○○ · 60min | — |
| **[Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets](https://arxiv.org/abs/2311.15127)** | 2023 | Documents the three-stage data curation pipeline in unusual detail; read it for the data recipe, not the architecture. | after `video-ldm` · ●●●○○ · 90min | [repo](https://github.com/Stability-AI/generative-models) |
| **[Align your Latents - High-Resolution Video Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2304.08818)** | 2023 | Inflates a pretrained image LDM with temporal layers — the cheapest known path from image model to video model. | after `ldm`, `vdm` · ●●●○○ · 90min | — |
| **[VideoPoet: A Large Language Model for Zero-Shot Video Generation](https://arxiv.org/abs/2312.14125)** | 2023 | The autoregressive-token alternative to diffusion; worth reading to see what the diffusion consensus gave up. | after `magvit-v2` · ●●●●○ · 90min | — |
| **[Photorealistic Video Generation with Diffusion Models](https://arxiv.org/abs/2312.06662)** | 2023 | W.A.L.T — window attention over a shared image-video latent space. | after `dit`, `video-ldm` · ●●●●○ · 90min | — |
| **[CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer](https://arxiv.org/abs/2408.06072)** | 2024 | Open weights plus a 3D causal VAE and expert adaptive LayerNorm; the first fully reproducible modern stack. | after `dit`, `sd3` · ●●●○○ · 90min | [repo](https://github.com/THUDM/CogVideo) |
| **[HunyuanVideo: A Systematic Framework For Large Video Generative Models](https://arxiv.org/abs/2412.03603)** | 2024 | 13B open weights with training infrastructure, data pipeline and scaling laws written down. | after `cogvideox` · ●●●●○ · 180min | [repo](https://github.com/Tencent/HunyuanVideo) |
| **[Latte: Latent Diffusion Transformer for Video Generation](https://arxiv.org/abs/2401.03048)** | 2024 | Ablates four ways to factorize space-time attention in a video DiT; the clearest study of that trade-off. | after `dit` · ●●●○○ · 60min | [repo](https://github.com/Vchitect/Latte) |
| **[LTX-Video: Realtime Video Latent Diffusion](https://arxiv.org/abs/2501.00103)** | 2024 | Pushes VAE compression to 1:192 and folds the final denoising step into the decoder to hit real time. | after `video-ldm` · ●●●○○ · 90min | [repo](https://github.com/Lightricks/LTX-Video) |
| **[Movie Gen: A Cast of Media Foundation Models](https://arxiv.org/abs/2410.13720)** | 2024 | 30B joint video-and-audio generation with a fully documented post-training recipe — the most complete public report of its kind. | after `sd3`, `flow-matching` · ●●●●○ · 180min | — |
| **[Pyramidal Flow Matching for Efficient Video Generative Modeling](https://arxiv.org/abs/2410.05954)** | 2024 | Runs most denoising steps at low resolution and only the last at full — a training-time answer to inference cost. | after `flow-matching` · ●●●●○ · 90min | — |
| **[Video generation models as world simulators](https://openai.com/index/video-generation-models-as-world-simulators/)** | 2024 | No method detail, but it established spacetime patches and native-resolution training as the default framing. | after `dit` · ●○○○○ · 30min | — |
| **[Step-Video-T2V Technical Report: The Practice, Challenges, and Future of Video Foundation Model](https://arxiv.org/abs/2502.10248)** | 2025 | 30B open model with a deep VAE and an unusually frank limitations section. | after `hunyuanvideo` · ●●●○○ · 120min | — |
| **[Wan: Open and Advanced Large-Scale Video Generative Models](https://arxiv.org/abs/2503.20314)** | 2025 | The open base most post-training work builds on; note the report discloses no RLHF stage. | after `cogvideox` · ●●●○○ · 120min | [repo](https://github.com/Wan-Video/Wan2.1) |

### Video VAE and Tokenizers

*The compression layer decides how many tokens a second of video costs, which sets everything downstream.*

| Paper | Year | What it contributes | Prereq · Difficulty · Time | Code |
| :--- | :---: | :--- | :--- | :---: |
| **[MAGVIT: Masked Generative Video Transformer](https://arxiv.org/abs/2212.05199)** | 2022 | 3D tokenizer plus masked generative training; establishes video tokens as a unit of work. | after `transformer` · ●●●●○ · 90min | — |
| **[Language Model Beats Diffusion - Tokenizer is Key to Visual Generation](https://arxiv.org/abs/2310.05737)** | 2023 | Lookup-free quantization with a huge vocabulary; the argument that tokenizer quality dominates generator choice. | after `magvit` · ●●●●○ · 90min | — |
| **[CV-VAE: A Compatible Video VAE for Latent Generative Video Models](https://arxiv.org/abs/2405.20279)** | 2024 | Builds a video VAE whose latent space stays compatible with existing image models, so pretrained weights survive. | after `ldm` · ●●●○○ · 60min | — |
| **[OmniTokenizer: A Joint Image-Video Tokenizer for Visual Generation](https://arxiv.org/abs/2406.09399)** | 2024 | One tokenizer trained jointly on images and video, with a progressive training schedule. | after `magvit-v2` · ●●●○○ · 60min | — |

---

<a id="inference-efficiency"></a>

## Inference Efficiency

A 3D DiT does full quadratic attention over the whole clip, once per denoising
step, with nothing cached between steps. These four lines attack that cost from
four different directions and compose with each other.

### Step Caching

*Adjacent denoising steps produce nearly identical features. Reuse them and the step count effectively drops, with no retraining.*

| Paper | Year | What it contributes | Prereq · Difficulty · Time | Code |
| :--- | :---: | :--- | :--- | :---: |
| **[DeepCache: Accelerating Diffusion Models for Free](https://arxiv.org/abs/2312.00858)** | 2023 | Reuses high-level U-Net features across adjacent steps at no training cost; the origin of the caching line. | after `ddpm` · ●●○○○ · 45min | [repo](https://github.com/horseee/DeepCache) |
| **[Adaptive Caching for Faster Video Generation with Diffusion Transformers](https://arxiv.org/abs/2411.02397)** | 2024 | Makes the caching schedule content-dependent instead of fixed. | after `teacache` · ●●●○○ · 45min | — |
| **[Δ-DiT: A Training-Free Acceleration Method Tailored for Diffusion Transformers](https://arxiv.org/abs/2406.01125)** | 2024 | Caches residual deltas rather than features, and shows which blocks matter at which stage of denoising. | after `deepcache`, `dit` · ●●●○○ · 45min | — |
| **[Real-Time Video Generation with Pyramid Attention Broadcast](https://arxiv.org/abs/2408.12588)** | 2024 | Broadcasts attention outputs across steps at different rates per attention type; caching adapted to video DiTs. | after `deepcache`, `latte` · ●●●○○ · 60min | [repo](https://github.com/NUS-HPC-AI-Lab/VideoSys) |
| **[Timestep Embedding Tells: It's Time to Cache for Video Diffusion Model](https://arxiv.org/abs/2411.19108)** | 2024 | TeaCache — uses the timestep embedding to decide when a step can be skipped; currently the most deployed caching method. | after `pab` · ●●○○○ · 45min | [repo](https://github.com/ali-vilab/TeaCache) |

### Sparse and Structured Attention

*Attacks the quadratic term directly by exploiting spatio-temporal locality.*

| Paper | Year | What it contributes | Prereq · Difficulty · Time | Code |
| :--- | :---: | :--- | :--- | :---: |
| **[Radial Attention: O(n log n) Sparse Attention with Energy Decay for Long Video Generation](https://arxiv.org/abs/2506.19852)** | 2025 | Exploits how attention energy decays with spatio-temporal distance to get sub-quadratic cost on long clips. | after `sta` · ●●●●○ · 90min | — |
| **[Sparse VideoGen: Accelerating Video Diffusion Transformers with Spatial-Temporal Sparsity](https://arxiv.org/abs/2502.01776)** | 2025 | Identifies that video attention heads split into spatial and temporal types, and sparsifies each differently. | after `latte`, `flashattention2` · ●●●●○ · 90min | — |
| **[Fast Video Generation with Sliding Tile Attention](https://arxiv.org/abs/2502.04507)** | 2025 | Replaces full 3D attention with hardware-aligned sliding tiles, turning locality into an actual kernel speedup. | after `flashattention2` · ●●●●○ · 90min | — |

### Kernels and Parallelism

*Making one GPU faster, then making many GPUs share one sequence.*

| Paper | Year | What it contributes | Prereq · Difficulty · Time | Code |
| :--- | :---: | :--- | :--- | :---: |
| **[FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)** | 2022 | Never materializes the attention matrix; the reason long sequences are feasible at all. | after `transformer` · ●●●●○ · 90min | — |
| **[FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)** | 2023 | Rebalances work across warps and thread blocks for roughly 2x over the original. | after `flashattention` · ●●●●○ · 60min | — |
| **[Ring Attention with Blockwise Transformers for Near-Infinite Context](https://arxiv.org/abs/2310.01889)** | 2023 | Shards the sequence across devices and rotates KV blocks in a ring; the basis of sequence parallelism. | after `flashattention` · ●●●●○ · 90min | — |
| **[DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models](https://arxiv.org/abs/2309.14509)** | 2023 | All-to-all sequence parallelism that shards by head instead of by block; complementary to Ring. | after `transformer` · ●●●●○ · 60min | — |
| **[DistriFusion: Distributed Parallel Inference for High-Resolution Diffusion Models](https://arxiv.org/abs/2402.19481)** | 2024 | Hides communication behind computation by reusing the previous step's activations at patch boundaries. | after `ldm` · ●●●●○ · 60min | — |
| **[FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608)** | 2024 | Uses Hopper asynchrony and FP8; the current ceiling for exact attention. | after `flashattention2` · ●●●●● · 90min | — |
| **[SageAttention: Accurate 8-Bit Attention for Plug-and-play Inference Acceleration](https://arxiv.org/abs/2410.02367)** | 2024 | INT8 attention that is accurate enough to drop in without retraining. | after `flashattention2` · ●●●●○ · 60min | [repo](https://github.com/thu-ml/SageAttention) |
| **[SageAttention2: Efficient Attention with Thorough Outlier Smoothing and Per-thread INT4 Quantization](https://arxiv.org/abs/2411.10958)** | 2024 | Pushes attention quantization to INT4 with per-thread outlier handling. | after `sageattention` · ●●●●● · 60min | — |
| **[xDiT: an Inference Engine for Diffusion Transformers (DiTs) with Massive Parallelism](https://arxiv.org/abs/2411.01738)** | 2024 | Combines sequence, pipeline and CFG parallelism for multi-GPU DiT inference; the practical entry point. | after `ring-attention`, `ulysses` · ●●●○○ · 60min | [repo](https://github.com/xdit-project/xDiT) |

### Few-Step Distillation

*The only lever that changes the step count itself. Everything else makes a step cheaper; this removes steps.*

| Paper | Year | What it contributes | Prereq · Difficulty · Time | Code |
| :--- | :---: | :--- | :--- | :---: |
| **[Progressive Distillation for Fast Sampling of Diffusion Models](https://arxiv.org/abs/2202.00512)** | 2022 | Repeatedly halves the step count by distilling a teacher into a student; the first practical few-step method. | after `ddim` · ●●●○○ · 60min | — |
| **[Adversarial Diffusion Distillation](https://arxiv.org/abs/2311.17042)** | 2023 | SDXL-Turbo — adds an adversarial loss so one-step output stops looking blurry. | after `progressive-distillation` · ●●●○○ · 60min | — |
| **[Consistency Models](https://arxiv.org/abs/2303.01469)** | 2023 | Trains a model whose output is constant along a probability-flow trajectory, enabling one-step sampling. | after `score-sde` · ●●●●○ · 120min | — |
| **[One-step Diffusion with Distribution Matching Distillation](https://arxiv.org/abs/2311.18828)** | 2023 | Matches distributions rather than trajectories, which is why it degrades less than trajectory distillation. | after `consistency-models` · ●●●●○ · 90min | — |
| **[Latent Consistency Models: Synthesizing High-Resolution Images with Few-Step Inference](https://arxiv.org/abs/2310.04378)** | 2023 | Brings consistency distillation to latent space, where it became usable in production. | after `consistency-models`, `ldm` · ●●●○○ · 60min | — |
| **[From Slow Bidirectional to Fast Autoregressive Video Diffusion Models](https://arxiv.org/abs/2412.07772)** | 2024 | CausVid — distills a bidirectional video model into a causal one so frames can stream out. | after `dmd2` · ●●●●○ · 90min | — |
| **[Improved Distribution Matching Distillation for Fast Image Synthesis](https://arxiv.org/abs/2405.14867)** | 2024 | Drops the regression loss and adds a GAN term; the strongest general-purpose few-step recipe. | after `dmd` · ●●●●○ · 90min | [repo](https://github.com/tianweiy/DMD2) |
| **[Diffusion Adversarial Post-Training for One-Step Video Generation](https://arxiv.org/abs/2501.08316)** | 2025 | Adversarial post-training that reaches one-step 1280x720 video generation. | after `add` · ●●●●○ · 90min | — |
| **[Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion](https://arxiv.org/abs/2506.08009)** | 2025 | Trains on self-generated rollouts to kill exposure bias; real-time single-GPU video at modest distillation cost. | after `causvid` · ●●●●○ · 90min | — |

### Quantization and Memory

*Weight precision decides whether a model fits on a consumer card at all.*

| Paper | Year | What it contributes | Prereq · Difficulty · Time | Code |
| :--- | :---: | :--- | :--- | :---: |
| **[GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323)** | 2022 | Second-order weight rounding that made 4-bit weights respectable. | ●●●●○ · 60min | — |
| **[SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/abs/2211.10438)** | 2022 | Migrates activation outliers into the weights so both sides become quantizable. | ●●●○○ · 60min | — |
| **[AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978)** | 2023 | Protects the small fraction of salient weights identified by activation magnitude; the scheme behind most 4-bit text encoders. | after `smoothquant` · ●●●○○ · 60min | — |
| **[Q-Diffusion: Quantizing Diffusion Models](https://arxiv.org/abs/2302.04304)** | 2023 | Shows why per-timestep activation ranges break naive PTQ on diffusion models. | after `ddpm` · ●●●○○ · 60min | — |
| **[SVDQuant: Absorbing Outliers by Low-Rank Components for 4-Bit Diffusion Models](https://arxiv.org/abs/2411.05007)** | 2024 | Routes outliers into a low-rank branch so the main path can be 4-bit; the current best quality-per-byte for diffusion. | after `awq`, `q-diffusion` · ●●●●○ · 90min | [repo](https://github.com/mit-han-lab/nunchaku) |
| **[ViDiT-Q: Efficient and Accurate Quantization of Diffusion Transformers for Image and Video Generation](https://arxiv.org/abs/2406.02540)** | 2024 | Quantization tailored to video DiTs, where error accumulates across both steps and frames. | after `q-diffusion`, `dit` · ●●●●○ · 60min | — |

---

<a id="post-training"></a>

## Post-Training

The main quality lever separating frontier commercial models from open bases.
Open technical reports rarely disclose an RLHF stage; this is where the gap lives.

### Reward Models and Preference Optimization

*Reward modelling, DPO variants, and online RL adapted to flow matching.*

| Paper | Year | What it contributes | Prereq · Difficulty · Time | Code |
| :--- | :---: | :--- | :--- | :---: |
| **[Training Diffusion Models with Reinforcement Learning](https://arxiv.org/abs/2305.13301)** | 2023 | Frames denoising as a multi-step MDP so policy gradient applies directly. | after `ddpm` · ●●●●○ · 90min | — |
| **[Diffusion Model Alignment Using Direct Preference Optimization](https://arxiv.org/abs/2311.12908)** | 2023 | Ports DPO to diffusion; the cheapest preference-alignment method that reliably works. | after `dpo` · ●●●○○ · 60min | — |
| **[Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)** | 2023 | Removes the separate reward model and RL loop from preference training; read before any diffusion-DPO variant. | ●●●○○ · 90min | — |
| **[Video Diffusion Alignment via Reward Gradients](https://arxiv.org/abs/2407.08737)** | 2024 | Backpropagates reward gradients through the sampler; far more sample-efficient than policy gradient when the reward is differentiable. | after `ddpo` · ●●●●○ · 90min | — |
| **[VideoDPO: Omni-Preference Alignment for Video Diffusion Generation](https://arxiv.org/abs/2412.14167)** | 2024 | Builds the preference pairs automatically from the model's own samples, removing the human-labeling bottleneck. | after `diffusion-dpo` · ●●●○○ · 60min | — |
| **[VisionReward: Fine-Grained Multi-Dimensional Human Preference Learning for Image and Video Generation](https://arxiv.org/abs/2412.21059)** | 2024 | An interpretable checklist reward model instead of a scalar score, which makes reward hacking visible. | after `videoreward` · ●●●○○ · 60min | — |
| **[DanceGRPO: Unleashing GRPO on Visual Generation](https://arxiv.org/abs/2505.07818)** | 2025 | The strongest published evidence that online RL moves open video models a long way on motion quality. | after `flow-grpo` · ●●●●○ · 90min | — |
| **[Flow-GRPO: Training Flow Matching Models via Online RL](https://arxiv.org/abs/2505.05470)** | 2025 | Converts the flow ODE to an SDE so online RL has the stochasticity it needs. | after `flow-matching`, `ddpo` · ●●●●○ · 90min | — |
| **[Improving Video Generation with Human Feedback](https://arxiv.org/abs/2501.13918)** | 2025 | VideoAlign — 182K preference triplets and a multi-dimensional reward model separating visual, motion and alignment quality. | after `diffusion-dpo` · ●●●○○ · 90min | [repo](https://github.com/KwaiVGI/VideoAlign) |

---

<a id="long-video-and-evaluation"></a>

## Long Video and Evaluation

Extending past a single short shot, and how to tell whether any of it worked.

### Long Video and Multi-Shot

*Length breaks memory and identity consistency before it breaks quality.*

| Paper | Year | What it contributes | Prereq · Difficulty · Time | Code |
| :--- | :---: | :--- | :--- | :---: |
| **[StreamingT2V: Consistent, Dynamic, and Extendable Long Video Generation from Text](https://arxiv.org/abs/2403.14773)** | 2024 | Short-term and long-term memory modules that keep identity stable across autoregressive extension. | after `video-ldm` · ●●●○○ · 60min | — |
| **[Frame Context Packing and Drift Prevention in Next-Frame-Prediction Video Diffusion Models](https://arxiv.org/abs/2504.12626)** | 2025 | FramePack — compresses context frames by importance so length stops driving memory. | after `causvid` · ●●●○○ · 60min | — |
| **[Long Context Tuning for Video Generation](https://arxiv.org/abs/2503.10589)** | 2025 | Extends a single-shot model to scene-level multi-shot with no added parameters. | after `wan` · ●●●○○ · 60min | — |

### Evaluation

*Automatic benchmarks correlate imperfectly with human preference. Know each one's blind spots before quoting it.*

| Paper | Year | What it contributes | Prereq · Difficulty · Time | Code |
| :--- | :---: | :--- | :--- | :---: |
| **[EvalCrafter: Benchmarking and Evaluating Large Video Generation Models](https://arxiv.org/abs/2310.11440)** | 2023 | 17 objective metrics calibrated against human opinion, with the calibration gap reported honestly. | ●●○○○ · 45min | — |
| **[VBench: Comprehensive Benchmark Suite for Video Generative Models](https://arxiv.org/abs/2311.17982)** | 2023 | Decomposes video quality into 16 measurable dimensions; the default automatic benchmark, with known blind spots. | ●●○○○ · 60min | [repo](https://github.com/Vchitect/VBench) |
| **[T2V-CompBench: A Comprehensive Benchmark for Compositional Text-to-video Generation](https://arxiv.org/abs/2407.14505)** | 2024 | Targets compositionality — attribute binding, spatial relations, numeracy — where general benchmarks saturate. | after `vbench` · ●●○○○ · 45min | — |

---

## Adjacent Areas

*Out of scope here, kept as a pointer so the boundary is explicit.*

**Video understanding / Video-LLMs** — a different problem (reading video rather than
making it) with its own literature and its own curated lists. This repo does not track it.

| Where to go | Covers |
| :--- | :--- |
| [Awesome-LLMs-for-Video-Understanding](https://github.com/yunlong10/Awesome-LLMs-for-Video-Understanding) | Video-LLMs, benchmarks, instruction tuning |
| [Awesome-Multimodal-Large-Language-Models](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models) | Multimodal LLMs and their video evaluations |
| [MMAction2](https://github.com/open-mmlab/mmaction2) | Action recognition, temporal detection (toolbox, not a list) |

The one exception we do track is **visual tokenizers** (see
[Video VAE and Tokenizers](#video-vae-and-tokenizers)) — they sit on both sides of the
boundary, and the compression ratio they achieve is a first-order constraint on
generation cost.


---

## Research Notes

*In-house deep-dive analyses. Sources cited; claims adversarially verified where marked.*

- 📂 [**Seedance vs Open-Source Gap** (topic series)](research/seedance-gap/) (2026-07, updated 2026-08) — *Root-cause analysis of the quality gap between frontier commercial video models and open-source bases. Main finding: the gap is driven by **post-training and data curation**, not architecture.*
  8 parts: [benchmark gap](research/seedance-gap/01-benchmark-gap.md) · [Seedance recipe](research/seedance-gap/02-seedance-recipe.md) · [open-source landscape](research/seedance-gap/03-open-source-landscape.md) · [post-training methods](research/seedance-gap/04-posttraining-methods.md) · [distillation & multi-shot](research/seedance-gap/05-distillation-multishot.md) · [roadmap & feasibility](research/seedance-gap/06-roadmap-feasibility.md) · [MiniMax H3](research/seedance-gap/07-minimax-h3.md) · [evaluation loop](research/seedance-gap/08-evaluation-loop.md)


---

## Resources

### Datasets

| Name | Type | Scale | Description | Link |
| :--- | :--- | :--- | :--- | :--- |
| **WebVid-10M** | Text-Video | 10M clips | Large-scale weakly captioned | [GitHub](https://github.com/m-bain/webvid) |
| **Panda-70M** | Text-Video | 70M clips | High-quality auto-captions | [GitHub](https://github.com/snap-research/Panda-70M) |
| **InternVid** | Text-Video | 234M clips | Largest public video-text corpus | [GitHub](https://github.com/OpenGVLab/InternVideo/tree/main/Data/InternVid) |
| **HD-VILA-100M** | Text-Video | 100M clips | High resolution clips | [GitHub](https://github.com/microsoft/XPretrain/tree/main/hd-vila-100m) |
| **OpenVid-1M** | Text-Video | 1M clips | Open-source, quality filtered | [GitHub](https://github.com/NJU-PCALab/OpenVid-1M) |

### Benchmarks

| Name | Purpose | Link |
| :--- | :--- | :--- |
| **VBench** | 16-dimension automatic evaluation | [GitHub](https://github.com/Vchitect/VBench) |
| **EvalCrafter** | 17 metrics calibrated to human opinion | [GitHub](https://github.com/EvalCrafter/EvalCrafter) |
| **T2V-CompBench** | Compositional generation | [GitHub](https://github.com/KaiyueSun98/T2V-CompBench) |
| **FVD** | Fréchet Video Distance | [Paper](https://arxiv.org/abs/1812.01717) |
| **Artificial Analysis** | Human-preference Elo leaderboard | [Website](https://artificialanalysis.ai/video) |

### Tools & Libraries

| Name | Description | Link |
| :--- | :--- | :--- |
| **Diffusers** | Reference implementations and schedulers | [GitHub](https://github.com/huggingface/diffusers) |
| **k-diffusion** | Where most practical samplers originate | [GitHub](https://github.com/crowsonkb/k-diffusion) |
| **ComfyUI** | Node-based inference graph, de facto deployment target | [GitHub](https://github.com/comfyanonymous/ComfyUI) |
| **FastVideo** | Sparse attention + distillation, production oriented | [GitHub](https://github.com/hao-ai-lab/FastVideo) |
| **xDiT** | Multi-GPU parallel inference for DiTs | [GitHub](https://github.com/xdit-project/xDiT) |
| **Nunchaku** | SVDQuant 4-bit inference engine | [GitHub](https://github.com/mit-han-lab/nunchaku) |
| **Decord** | Efficient video decoding | [GitHub](https://github.com/dmlc/decord) |


---

## Labs & Companies

### Research Labs

| Lab | Affiliation | Notable Works |
| :--- | :--- | :--- |
| **OpenAI** | - | Sora |
| **Google DeepMind** | Google | Lumiere, VideoPoet, Veo, Gemini |
| **Meta AI (FAIR)** | Meta | Emu Video, Make-A-Video, PyTorchVideo |
| **Shanghai AI Lab** | - | Open-Sora, InternVideo, InternVL |
| **Tsinghua KEG** | Tsinghua | CogVideo, GLM |
| **OpenGVLab** | Shanghai AI Lab | VideoChat, InternVL, VideoMAE |

### Companies

| Company | Products | Focus |
| :--- | :--- | :--- |
| **Runway** | Gen-1, Gen-2, Gen-3 Alpha | Professional video creation |
| **Stability AI** | Stable Video Diffusion | Open-source diffusion |
| **Pika Labs** | Pika | Consumer T2V |
| **Kuaishou** | Kling, LivePortrait | T2V, portrait animation |
| **ByteDance** | MagicVideo, PixelDance | T2V research |
| **Alibaba** | I2VGen-XL, EMO, Animate Anyone | I2V, avatar |
| **Tencent** | HunyuanVideo | Open-source T2V |
| **Genmo** | Mochi 1 | Open-source T2V |
| **Lightricks** | LTX 2.0 | 4K audio-video generation |
| **MiniMax** | Hailuo, H3 | Omni-modal video+audio, open weights |

---


---

## Contributing

Contributions are welcome. **Edit [`data/papers.yaml`](data/papers.yaml), not `README.md`** —
the README is generated and your changes to it will be overwritten.

```bash
python scripts/verify.py       # check arXiv ids, titles and the prereq graph
python scripts/gen_readme.py   # regenerate README.md
```

### Guidelines

1. `desc` says **what the paper contributes**, in one line. "A method for X" is not a description.
2. Fill in `prereq`. An entry with no honest prerequisites is rare; one with the wrong ones is worse than none.
3. Give the arXiv id without a version suffix. `verify.py` will confirm the title against the API.
4. Keep the scope: generative video, inference efficiency, post-training.
5. `difficulty` and `minutes` are rough. Err toward honest rather than encouraging.

---

<p align="center">
  <i>Curated for people who read papers to solve a problem.</i>
</p>
