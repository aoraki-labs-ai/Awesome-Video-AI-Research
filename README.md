# Awesome Video AI Research 🎬

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

> A curated list of Video AI resources, architectures, models, and datasets.
> From Generative Video (Sora-level) to Video Understanding (Video-LLMs)

---

## Table of Contents

- [Basics](#basics)
  - [Mathematical Foundations](#mathematical-foundations)
  - [Key Surveys](#key-surveys)
- [Core Architectures](#core-architectures)
  - [Diffusion Transformer (DiT)](#diffusion-transformer-dit)
  - [U-Net Based Diffusion](#u-net-based-diffusion)
  - [Autoregressive Models](#autoregressive-models)
- [Generative Projects](#generative-projects)
  - [Text-to-Video (T2V)](#text-to-video-t2v)
  - [Image-to-Video (I2V)](#image-to-video-i2v)
  - [Video Editing & Inpainting](#video-editing--inpainting)
  - [Human Motion & Avatar](#human-motion--avatar)
- [Video Understanding](#video-understanding)
  - [Video-LLMs](#video-llms)
  - [Foundation Models](#foundation-models)
- [Infrastructure & Engineering](#infrastructure--engineering)
  - [Inference Optimization](#inference-optimization)
  - [Serving Frameworks](#serving-frameworks)
- [Advanced Topics](#advanced-topics)
  - [Post-Training & Alignment (RLHF)](#post-training--alignment-rlhf)
  - [Few-Step Distillation & Real-Time Generation](#few-step-distillation--real-time-generation)
  - [Temporal Consistency](#temporal-consistency)
  - [Long Video & Multi-Shot Generation](#long-video--multi-shot-generation)
  - [4D Generation (3D + Time)](#4d-generation-3d--time)
- [Research Notes](#research-notes)
- [Resources](#resources)
  - [Datasets](#datasets)
  - [Benchmarks](#benchmarks)
  - [Tools & Libraries](#tools--libraries)
- [Labs & Companies](#labs--companies)
- [Contributing](#contributing)

---

## Basics

📚 **Essential reading before diving into video AI.**

- 📖 [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - *The Transformer architecture that underlies most modern video AI.*
- 🎥 [How Sora Works (Technical Report)](https://openai.com/index/video-generation-models-as-world-simulators/) - *OpenAI's technical explanation of Sora.*
- 📝 [Diffusion Models: A Comprehensive Survey](https://arxiv.org/abs/2209.00796) - *Foundational understanding of diffusion.*

### Mathematical Foundations

*Core mathematical concepts powering video generation - probability, flow, and score matching.*

| Topic | Paper | Description |
| :--- | :--- | :--- |
| **DDPM** | [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) | The foundational diffusion paper |
| **Score SDE** | [Score-Based Generative Modeling through SDEs](https://arxiv.org/abs/2011.13456) | Unified framework via stochastic differential equations |
| **Flow Matching** | [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) | Simulation-free training for continuous normalizing flows |
| **Rectified Flow** | [Flow Straight and Fast](https://arxiv.org/abs/2209.03003) | Straightening flows for faster sampling |

### Key Surveys

- 📊 [A Survey on Video Diffusion Models](https://arxiv.org/abs/2310.10647) (2023) - *Comprehensive overview of video diffusion.*
- 📊 [Video Understanding with Large Language Models: A Survey](https://arxiv.org/abs/2312.17432) (2024) - *State-of-the-art in Video-LLMs.*
- 📊 [Sora: A Review on Background, Technology, Limitations, and Opportunities](https://arxiv.org/abs/2402.17177) (2024) - *Deep dive into Sora and its implications.*

---

## Core Architectures

### Diffusion Transformer (DiT)

*The architecture behind Sora. Scaling Transformers to visual generation.*

| Name | Paper | Code | Notes |
| :--- | :--- | :--- | :--- |
| **DiT** | [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748) | [GitHub](https://github.com/facebookresearch/DiT) | The foundational DiT paper |
| **Latte** | [Latent Diffusion Transformer for Video](https://arxiv.org/abs/2401.03048) | [GitHub](https://github.com/Vchitect/Latte) | First open-source video DiT |
| **Open-Sora** | [Democratizing Efficient Video Production](https://arxiv.org/abs/2403.06394) | [GitHub](https://github.com/hpcaitech/Open-Sora) | Open reproduction of Sora |
| **Open-Sora-Plan** | [Open-Sora Plan](https://github.com/PKU-YuanGroup/Open-Sora-Plan) | [GitHub](https://github.com/PKU-YuanGroup/Open-Sora-Plan) | Alternative open-source Sora |
| **LTX 2.0** | [LTX-2 Model](https://ltx.io/model) | [GitHub](https://github.com/Lightricks/LTX-2) | 19B DiT with audio-video sync |

### U-Net Based Diffusion

*The classic architecture evolved from Stable Diffusion to video domain.*

| Name | Paper | Code | Notes |
| :--- | :--- | :--- | :--- |
| **SVD** | [Stable Video Diffusion](https://arxiv.org/abs/2311.15127) | [GitHub](https://github.com/Stability-AI/generative-models) | Industry-standard I2V model |
| **Video LDM** | [Align your Latents](https://arxiv.org/abs/2304.08818) | - | Temporal layers for video |
| **AnimateDiff** | [Animate Your Personalized T2I Models](https://arxiv.org/abs/2307.04725) | [GitHub](https://github.com/guoyww/AnimateDiff) | Motion modules for any SD model |
| **ModelScope** | [Text-to-Video Synthesis](https://arxiv.org/abs/2308.06571) | [GitHub](https://github.com/modelscope/modelscope) | Early open-source T2V |

### Autoregressive Models

*Treating video generation as a next-token prediction task (LLM-style).*

| Name | Paper/Link | Code | Notes |
| :--- | :--- | :--- | :--- |
| **VideoPoet** | [Google Research](https://sites.research.google/videopoet/) | - | Multimodal LLM for zero-shot video |
| **W.A.L.T** | [Walt Video Diffusion](https://walt-video-diffusion.github.io/) | - | Window attention latent transformer |
| **Emu3** | [Next-Token Prediction is All You Need](https://arxiv.org/abs/2409.18869) | [GitHub](https://github.com/baaivision/Emu3) | Unified multimodal generation |
| **MAGVIT-2** | [Language Model Beats Diffusion](https://arxiv.org/abs/2310.05737) | - | Tokenizer for visual generation |

---

## Generative Projects

### Text-to-Video (T2V)

| Name | Organization | Type | Link | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Sora** | OpenAI | Proprietary | [Link](https://openai.com/sora) | SOTA, DiT-based, up to 1 min |
| **Veo 2** | Google | Proprietary | [Link](https://deepmind.google/technologies/veo/) | 4K resolution, 2+ min |
| **Gen-3 Alpha** | Runway | Proprietary | [Link](https://runwayml.com/) | Commercial T2V leader |
| **Pika** | Pika Labs | Proprietary | [Link](https://pika.art/) | Fast iteration, good motion |
| **Kling** | Kuaishou | Proprietary | [Link](https://klingai.com/) | Strong Chinese T2V model |
| **Seedance** | ByteDance | Proprietary | [1.0 Report](https://arxiv.org/abs/2506.09113) / [1.5 pro Report](https://arxiv.org/abs/2512.13507) | #1 on Artificial Analysis T2V/I2V arenas; native multi-shot, AV joint gen (1.5+) |
| **Lumiere** | Google | Research | [Paper](https://lumiere-video.github.io/) | Space-time diffusion |
| **CogVideoX** | Zhipu AI | Open Source | [GitHub](https://github.com/THUDM/CogVideo) | GLM-based, 6B & 5B models |
| **Mochi 1** | Genmo | Open Source | [GitHub](https://github.com/genmoai/mochi) | High quality open T2V |
| **HunyuanVideo** | Tencent | Open Source | [GitHub](https://github.com/Tencent/HunyuanVideo) | 13B DiT model |
| **LTX 2.0** | Lightricks | Open Source | [GitHub](https://github.com/Lightricks/LTX-2) | 19B DiT, 4K+audio, up to 20s |
| **Wan 2.2** | Alibaba | Open Source | [GitHub](https://github.com/Wan-Video/Wan2.2) | MoE DiT (27B/14B active), Apache 2.0; strongest open-weights base |
| **Wan 2.6** | Alibaba | Proprietary (API) | [Link](https://wan.video/) | 1080p, 15s, multi-shot+audio; weights not released |
| **HunyuanVideo 1.5** | Tencent | Open Source | [GitHub](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5) | 8.3B DiT; full open post-training recipe (SFT+DPO+MixGRPO) |

### Image-to-Video (I2V)

| Name | Organization | Code | Notes |
| :--- | :--- | :--- | :--- |
| **SVD / SVD-XT** | Stability AI | [GitHub](https://github.com/Stability-AI/generative-models) | Industry standard |
| **DynamiCrafter** | CUHK | [GitHub](https://github.com/Doubiiu/DynamiCrafter) | Open-domain image animation |
| **I2VGen-XL** | Alibaba | [GitHub](https://github.com/ali-vilab/i2vgen-xl) | High-resolution I2V |
| **Animate Anyone** | Alibaba | [GitHub](https://github.com/HumanAIGC/AnimateAnyone) | Character animation from image |
| **LivePortrait** | Kuaishou | [GitHub](https://github.com/KwaiVGI/LivePortrait) | Portrait animation |

### Video Editing & Inpainting

| Name | Paper | Code | Notes |
| :--- | :--- | :--- | :--- |
| **ControlVideo** | [Training-free Controllable T2V](https://arxiv.org/abs/2305.08877) | [GitHub](https://github.com/YBYBZhang/ControlVideo) | Depth/edge conditioned |
| **Tune-A-Video** | [One-Shot Tuning](https://arxiv.org/abs/2212.11565) | [GitHub](https://github.com/showlab/Tune-A-Video) | Edit with single video |
| **TokenFlow** | [Consistent Diffusion Editing](https://arxiv.org/abs/2307.10373) | [GitHub](https://github.com/omerbt/TokenFlow) | Temporal consistency via tokens |
| **CoDeF** | [Content Deformation Fields](https://arxiv.org/abs/2308.07926) | [GitHub](https://github.com/qiuyu96/CoDeF) | Canonical representation |
| **ProPainter** | [Video Inpainting with Propagation](https://arxiv.org/abs/2309.03897) | [GitHub](https://github.com/sczhou/ProPainter) | SOTA video inpainting |

### Human Motion & Avatar

| Name | Paper | Code | Notes |
| :--- | :--- | :--- | :--- |
| **EMO** | [Emote Portrait Alive](https://arxiv.org/abs/2402.17485) | [GitHub](https://github.com/HumanAIGC/EMO) | Audio-driven portrait |
| **MagicAnimate** | [Temporal Consistent Human Animation](https://arxiv.org/abs/2311.16498) | [GitHub](https://github.com/magic-research/magic-animate) | ControlNet for humans |
| **Champ** | [Controllable & Consistent Human Animation](https://arxiv.org/abs/2403.14781) | [GitHub](https://github.com/fudan-generative-vision/champ) | 3D-aware human motion |
| **MusePose** | [Pose-Driven Image-to-Video](https://github.com/TMElyralab/MusePose) | [GitHub](https://github.com/TMElyralab/MusePose) | Open-source character animation |

---

## Video Understanding

### Video-LLMs

*Bridging visual encoders with Large Language Models for video comprehension.*

| Name | Backbone | Features | Code |
| :--- | :--- | :--- | :--- |
| **Video-LLaMA 2** | LLaMA + BLIP-2 | Audio-visual understanding | [GitHub](https://github.com/DAMO-NLP-SG/VideoLLaMA2) |
| **VideoChat 2** | Mistral/Vicuna | Multi-turn video dialogue | [GitHub](https://github.com/OpenGVLab/Ask-Anything) |
| **LLaVA-Video** | Qwen2-VL | Strong zero-shot, long video | [GitHub](https://github.com/LLaVA-VL/LLaVA-NeXT) |
| **Qwen2-VL** | Qwen2 | Native video understanding | [GitHub](https://github.com/QwenLM/Qwen2-VL) |
| **InternVL 2** | InternLM | Scaling visual encoders | [GitHub](https://github.com/OpenGVLab/InternVL) |
| **Video-LLaVA** | Vicuna | Unified image-video encoder | [GitHub](https://github.com/PKU-YuanGroup/Video-LLaVA) |
| **mPLUG-Owl** | LLaMA | Modularized training | [GitHub](https://github.com/X-PLUG/mPLUG-Owl) |

### Foundation Models

*Self-supervised video representation learning.*

| Name | Paper | Code | Notes |
| :--- | :--- | :--- | :--- |
| **VideoMAE V2** | [Scaling Video Masked Autoencoders](https://arxiv.org/abs/2303.16727) | [GitHub](https://github.com/OpenGVLab/VideoMAEv2) | Billion-scale pretraining |
| **InternVideo 2** | [Scaling Video Foundation Models](https://arxiv.org/abs/2403.15377) | [GitHub](https://github.com/OpenGVLab/InternVideo) | 6B parameter model |
| **UMT** | [Unmasked Teacher](https://arxiv.org/abs/2303.16727) | [GitHub](https://github.com/OpenGVLab/unmasked_teacher) | Progressive distillation |
| **VideoPrism** | [Foundational Visual Encoder](https://arxiv.org/abs/2402.13217) | - | 36M video pretraining |

---

## Infrastructure & Engineering

### Inference Optimization

*Techniques to run video models efficiently on consumer GPUs.*

| Technique | Paper/Link | Code | Notes |
| :--- | :--- | :--- | :--- |
| **LCM** | [Latent Consistency Models](https://arxiv.org/abs/2310.04378) | [GitHub](https://github.com/luosiallen/latent-consistency-model) | 4-step generation |
| **SDXL-Turbo** | [Adversarial Diffusion Distillation](https://arxiv.org/abs/2311.17042) | [HuggingFace](https://huggingface.co/stabilityai/sdxl-turbo) | 1-step generation |
| **PAB** | [Pyramid Attention Broadcast](https://arxiv.org/abs/2408.12588) | [GitHub](https://github.com/NUS-HPC-AI-Lab/VideoSys) | Real-time video diffusion |
| **TensorRT** | [NVIDIA TensorRT](https://github.com/NVIDIA/TensorRT) | [GitHub](https://github.com/NVIDIA/TensorRT) | Production inference |
| **DeepCache** | [Accelerating Diffusion](https://arxiv.org/abs/2312.00858) | [GitHub](https://github.com/horseee/DeepCache) | Cache-based acceleration |

### Serving Frameworks

| Name | Description | Code |
| :--- | :--- | :--- |
| **vLLM** | High-throughput LLM serving, multimodal support | [GitHub](https://github.com/vllm-project/vllm) |
| **SGLang** | Efficient structured generation | [GitHub](https://github.com/sgl-project/sglang) |
| **Xinference** | Distributed model serving | [GitHub](https://github.com/xorbitsai/inference) |
| **ComfyUI** | Node-based video workflow | [GitHub](https://github.com/comfyanonymous/ComfyUI) |
| **Diffusers** | Hugging Face diffusion library | [GitHub](https://github.com/huggingface/diffusers) |

---

## Advanced Topics

### Post-Training & Alignment (RLHF)

*The main quality lever separating frontier commercial models (Seedance, Veo, Kling) from open-source bases — reward models + preference optimization for video diffusion.*

| Name | Paper | Code | Notes |
| :--- | :--- | :--- | :--- |
| **VideoAlign / VideoReward** | [Improving Video Generation with Human Feedback](https://arxiv.org/abs/2501.13918) | [GitHub](https://github.com/KwaiVGI/VideoAlign) | 182K preference triplets; multi-dim RM (VQ/MQ/TA); Flow-DPO / Flow-RWR / Flow-NRG (NeurIPS 2025) |
| **VisionReward** | [Fine-Grained Multi-Dimensional Preference Learning](https://arxiv.org/abs/2412.21059) | [GitHub](https://github.com/THUDM/VisionReward) | Interpretable checklist RM (64 binary questions); MPO dominant-pair training |
| **DanceGRPO** | [Unleashing GRPO on Visual Generation](https://arxiv.org/abs/2505.07818) | [GitHub](https://github.com/XueZeyue/DanceGRPO) | Online GRPO for diffusion & rectified flow; +181% motion quality on HunyuanVideo |
| **Flow-GRPO** | [Training Flow Matching Models via Online RL](https://arxiv.org/abs/2505.05470) | [GitHub](https://github.com/yifan123/flow_grpo) | ODE→SDE conversion + denoising reduction for RL on flow models |
| **VideoDPO** | [Omni-Preference Alignment for Video Diffusion](https://arxiv.org/abs/2412.14167) | [GitHub](https://github.com/CIntellifusion/VideoDPO) | Fully automatic preference pairs (OmniScore); runs on 4×A100 |
| **Seedance 1.0** | [Technical Report](https://arxiv.org/abs/2506.09113) | - | Reference recipe: 3 specialized RMs (foundational/motion/aesthetic), direct reward maximization |
| **HunyuanVideo 1.5** | [Technical Report](https://arxiv.org/abs/2511.18870) | [GitHub](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5) | Open post-training blueprint: SFT → offline DPO → online MixGRPO with VLM reward |
| **ContentV** | [Efficient Training with Limited Compute](https://arxiv.org/abs/2506.05343) | [GitHub](https://github.com/bytedance/ContentV) | Annotation-free RLHF via differentiable reward backprop; 85.14 VBench on modest compute |

### Few-Step Distillation & Real-Time Generation

*Closing the inference-speed gap: from 50-step diffusion to real-time streaming.*

| Name | Paper | Code | Notes |
| :--- | :--- | :--- | :--- |
| **Self Forcing** | [Bridging the Train-Test Gap in Autoregressive Video Diffusion](https://arxiv.org/abs/2506.08009) | [GitHub](https://github.com/guandeh17/Self-Forcing) | AR rollout + KV cache + DMD; 17 FPS on one H100; converges in ~4 H100-days on Wan2.1-1.3B |
| **Seaweed-APT** | [Adversarial Post-Training for One-Step Video Generation](https://arxiv.org/abs/2501.08316) | - | One-step 720p24 real-time generation (ICML 2025) |
| **CausVid** | [From Slow Bidirectional to Fast Autoregressive](https://arxiv.org/abs/2412.07772) | [GitHub](https://github.com/tianweiy/CausVid) | Bidirectional teacher → causal student distillation |
| **Hyper-SD / TSCD** | [Trajectory Segmented Consistency Distillation](https://arxiv.org/abs/2404.13686) | [HuggingFace](https://huggingface.co/ByteDance/Hyper-SD) | The distillation family behind Seedance's ~10x speedup |

### Temporal Consistency

*Ensuring objects don't morph randomly over time - the core challenge of video generation.*

- 📖 [FreeNoise: Noise Rescheduling for Video Consistency](https://arxiv.org/abs/2310.15169)
- 📖 [VideoComposer: Compositional Video Synthesis](https://arxiv.org/abs/2306.02018)
- 📖 [Make-Your-Video: Customized Video Generation](https://arxiv.org/abs/2306.00943)

### Long Video & Multi-Shot Generation

*Moving beyond 4-10 seconds to minutes of coherent, multi-shot narrative video.*

| Name | Paper | Code | Notes |
| :--- | :--- | :--- | :--- |
| **LCT** | [Long Context Tuning for Video Generation](https://arxiv.org/abs/2503.10589) | [Project](https://guoyww.github.io/projects/long-context-video/) | Single-shot DiT → scene-level multi-shot via context expansion, no new params (ICCV 2025) |
| **HoloCine** | [Holistic Generation of Cinematic Multi-Shot Narratives](https://arxiv.org/abs/2510.20822) | [GitHub](https://github.com/yihao-meng/HoloCine) | Wan 2.2 fine-tuned on 400K multi-shot samples (CVPR 2026) |
| **StreamingT2V** | [Streaming Video Synthesis](https://arxiv.org/abs/2403.14773) | [GitHub](https://github.com/Picsart-AI-Research/StreamingT2V) | Infinite length generation |
| **FreeNoise** | [Tuning-free Long Video](https://arxiv.org/abs/2310.15169) | [GitHub](https://github.com/AILab-CVC/FreeNoise) | Extended context window |
| **Gen-L-Video** | [Multitext to Long Video](https://arxiv.org/abs/2305.18264) | [GitHub](https://github.com/G-U-N/Gen-L-Video) | Story-based generation |

### 4D Generation (3D + Time)

*Generating dynamic 3D content - the frontier of generative AI.*

| Name | Paper | Code | Notes |
| :--- | :--- | :--- | :--- |
| **DreamGaussian4D** | [Gaussian Splatting for 4D](https://arxiv.org/abs/2312.17142) | [GitHub](https://github.com/jiawei-ren/dreamgaussian4d) | Dynamic 3D Gaussians |
| **4D-fy** | [Text-to-4D via Hybrid SDS](https://arxiv.org/abs/2311.17984) | [GitHub](https://github.com/sherwinbahmani/4dfy) | 4D from text |
| **Consistent4D** | [Consistent Dynamic 3D](https://arxiv.org/abs/2311.02848) | [GitHub](https://github.com/yanqinJiang/Consistent4D) | Video to 4D |
| **GaussianFlow** | [Splatting for 4D Reconstruction](https://arxiv.org/abs/2312.03431) | [GitHub](https://github.com/Zerg-Overmind/GaussianFlow) | High-quality 4D |

---

## Research Notes

*In-house deep-dive analyses (sources cited, claims adversarially verified where marked).*

- 📂 [**Seedance vs Open-Source Gap** (topic series)](research/seedance-gap/) (2026-07) - *Root-cause analysis of the Seedance-vs-open-source quality gap: post-training (multi-dim reward-model RLHF) and data curation dominate, not architecture. 6-part series: [benchmark gap](research/seedance-gap/01-benchmark-gap.md) · [Seedance recipe](research/seedance-gap/02-seedance-recipe.md) · [open-source landscape](research/seedance-gap/03-open-source-landscape.md) · [post-training methods](research/seedance-gap/04-posttraining-methods.md) · [distillation & multi-shot](research/seedance-gap/05-distillation-multishot.md) · [roadmap & feasibility](research/seedance-gap/06-roadmap-feasibility.md).*

---

## Resources

### Datasets

| Name | Type | Scale | Description | Link |
| :--- | :--- | :--- | :--- | :--- |
| **WebVid-10M** | Text-Video | 10M clips | Large-scale weakly captioned | [GitHub](https://github.com/m-bain/webvid) |
| **Panda-70M** | Text-Video | 70M clips | High-quality auto-captions | [GitHub](https://github.com/snap-research/Panda-70M) |
| **InternVid** | Text-Video | 234M clips | Largest public video-text | [GitHub](https://github.com/OpenGVLab/InternVideo/tree/main/Data/InternVid) |
| **HD-VILA-100M** | Text-Video | 100M clips | High resolution clips | [GitHub](https://github.com/microsoft/XPretrain/tree/main/hd-vila-100m) |
| **Kinetics-700** | Action | 650K clips | Action recognition benchmark | [DeepMind](https://www.deepmind.com/open-source/kinetics) |
| **Ego4D** | Egocentric | 3.6K hrs | First-person video understanding | [Website](https://ego4d-data.org/) |
| **OpenVid-1M** | Text-Video | 1M clips | Open-source high quality | [GitHub](https://github.com/NJU-PCALab/OpenVid-1M) |

### Benchmarks

| Name | Purpose | Link |
| :--- | :--- | :--- |
| **VBench** | Comprehensive video generation evaluation | [GitHub](https://github.com/Vchitect/VBench) |
| **EvalCrafter** | T2V generation quality assessment | [GitHub](https://github.com/EvalCrafter/EvalCrafter) |
| **FVD** | Fréchet Video Distance - generative quality metric | [Paper](https://arxiv.org/abs/1812.01717) |
| **VideoMME** | Video-LLM understanding evaluation | [GitHub](https://github.com/BradyFU/Video-MME) |
| **MVBench** | Multi-task video understanding | [GitHub](https://github.com/OpenGVLab/Ask-Anything/tree/main/video_chat2) |
| **FETV** | Fine-grained text-video evaluation | [GitHub](https://github.com/llyx97/FETV) |

### Tools & Libraries

| Name | Description | Link |
| :--- | :--- | :--- |
| **Diffusers** | 🤗 Hugging Face diffusion library | [GitHub](https://github.com/huggingface/diffusers) |
| **MMAction2** | OpenMMLab video understanding | [GitHub](https://github.com/open-mmlab/mmaction2) |
| **PyTorchVideo** | Meta's video understanding library | [GitHub](https://github.com/facebookresearch/pytorchvideo) |
| **Decord** | Efficient GPU video loading | [GitHub](https://github.com/dmlc/decord) |
| **Video-FocalNets** | Spatiotemporal focal modulation | [GitHub](https://github.com/TalalWasim/Video-FocalNets) |
| **TimeSformer** | Divided space-time attention | [GitHub](https://github.com/facebookresearch/TimeSformer) |

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

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Guidelines

1. Ensure any new resources are properly categorized
2. Include working links to papers, code repositories, and project pages
3. Provide brief descriptions for new entries
4. Keep the formatting consistent with existing entries

---

## Star History

If you find this resource helpful, please consider giving it a ⭐!

---

<p align="center">
  <i>Curated with ❤️ for the Video AI research community</i>
</p>
