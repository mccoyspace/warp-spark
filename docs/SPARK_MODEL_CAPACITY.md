# DGX Spark model and capacity map

Last verified: 2026-08-28
Host: one NVIDIA GB10 with 128 GiB coherent memory and 3.7 TB internal NVMe
Storage at verification: 2.7 TB used, 898 GB available

This is the master operational inventory for the studio Spark. It distinguishes
what the underlying model can theoretically do from what the installed local
runtime actually exposes and from what has been measured on this machine.

## How to read the performance columns

- **Decode** is post-first-token generation unless the row says effective or
  end-to-end.
- **TTFT** includes prompt handling for the stated prompt shape. It is not
  comparable across different prompt lengths.
- **Peak** means a short, warm, favorable local run. **Qualified** means a
  matched or held-out result with a stronger evidence contract.
- `NR` means not recorded with adequate instrumentation. It does not mean the
  model is slow.
- All local GPU inference engines are mutually exclusive. One of vLLM,
  llama.cpp, DS4, or WARP should own the GB10 at a time.

## Master language-model chart

| Status | Model / release era | Parameters | Local representation | Best local engine | Best measured decode | Best measured TTFT | Evidence and practical meaning |
| --- | --- | ---: | ---: | --- | ---: | ---: | --- |
| **LIVE default** | **Qwen3.8-27B** — 2026-08-05 | 27.8B dense | 25 GiB NVFP4 | vLLM 0.24 + native MTP | **11.0 tok/s** | **0.315 s**, short warm prompt | Fresh 2026-08-23 streamed no-thinking run, 256 output tokens. Medium reasoning is the studio quality default and will take longer. |
| Ready | Gemma-4-26B-A4B uncensored — 2026-04 | 26B / ~4B active | 16 GiB NVFP4 + 0.8 GiB DFlash | vLLM + DFlash | **76.1 tok/s** peak | NR | Short warm 300-token peak. Fastest installed single-stream language model; specialist/uncensored rather than the primary quality default. |
| Ready | Ornith-1.0-35B uncensored — 2026-06 | 35B / ~3B active | 23 GiB NVFP4 + 0.9 GiB DFlash | vLLM + DFlash | **69.3 tok/s** peak | NR | Three 400-token warm runs: 67.3–69.3 tok/s. Agentic/coding-oriented uncensored alternative. |
| Ready | Qwen3.6-35B-A3B heretic — 2026-04 | 35B / ~3B active | 37 GiB Q8 + 0.9 GiB vision projector | llama.cpp + MTP | NR | NR | Qualified for coherence, vision, and tool evaluation; local isolated TTFT/decode capture is still missing. |
| Ready | Nemotron-3-Nano-30B-A3B — 2025-12 | 30B / ~3B active | 19 GiB NVFP4 | vLLM | NR | NR | Text/reasoning/tool specialist. Maintained launcher exists; performance has not been normalized into this measurement scheme. |
| Ready | Qwen3-Coder-Next-80B-A3B — 2026-01 | 79.7B / ~3B active | 43 GiB NVFP4 | vLLM/Marlin | NR | NR | Coherent coding smoke passed. Large cold start, roughly 7–10 minutes on this host; local TTFT/decode capture is missing. |
| Ready | **DeepSeek-V4-Flash 0731** — base 2026-04, checkpoint 2026-07-31 | 284B / 13B active | 81 GiB mixed IQ2/Q2/Q8 GGUF | **DS4 Entrpi fork, ordinary CUDA** | **22.0 tok/s decode; 21.65 effective** | **0.213 s**, 24-token HTTP prompt | Best DS4 single request; four concurrent requests reached **46.14 aggregate tok/s**. Speculative DSpark was slower and is disabled. |
| Experimental ready | **GLM-4.7-Flash** — 2026-01 | 31.2B / ~3B active | 12 GiB WARP VQ3R | WARP CUDA | **12.25 tok/s** peak; 11.89 matched | **5.20 s**, 38-token WARP prefill | Compact, sub-second model open, plain chat only. Historical resident-BF16 vLLM A/B reached 26.85 tok/s and 0.173 s warm TTFT, but its 58 GiB source weights were removed and must be redownloaded to restore that path. |
| Experimental WARP | **GLM-5.3-Flash** — 2026-08-26 | 320B / 18B active | 111.5 GiB base WARP VQ3R; 115 GiB logical MTP container using hardlinked base banks (~7.6 GiB added physical) | WARP CUDA | **3.851 tok/s** short matched | **3.54-3.85 s**, 7-token resident-model prompt | Exact task-major VQ group 4 added 7.22% over a 3.592 tok/s control bracket. Proposal depths 1-3 replayed exactly in shadow tests; depth-one target verification reached 3.611 tok/s and remains experimental/off by default. TTFT predates the fused follow-up. Short warm-cache result, not yet a sustained studio-workload figure. |
| Qualified WARP | **Kimi K2 Instruct** — 2025-07 | 1.026T / 31.69B active | 362 GiB WARP VQ3R | WARP CUDA | **3.001 tok/s peak; 2.748 qualified mean** | **28.78 s** for 53 tokens; 116.84 s for 288 | Median resident-soak request rate including prefill was 1.860 tok/s. Text completion only; no qualified WARP chat template. |
| Experimental WARP | **GLM-5.2** — 2026-06-16 | ~753B / ~40B active | 264 GiB WARP VQ3R | WARP CUDA | **1.725 tok/s peak; 1.706 matched mean** | **6.75 s**, 4-token reset/replay prompt | Source-backed qualification at an exact 2K context bound. Full CUDA was 3.06x the short CPU control; tokens, routes, argmax and top-10 remained stable with zero fallbacks. Short warm-cache result, not yet a sustained studio-workload figure. |
| Experimental WARP | **Full GLM-4.7** — 2025-12 | 358B / ~33B active | 127 GiB WARP VQ3R | WARP CUDA/GQA | **3.63 tok/s development; 2.26 held-out** | **13.84 s**, 34-token prompt | Strong experimental extension with exact routes/tokens and bounded logit drift. Short studio smoke was ~2.52 tok/s. Text/raw format; 4K local profile. |
| **Studio WARP default when selected** | **Kimi K3** — public launch 2026-07-16 | 2.8T; 16 of 896 routed experts/token | 982 GiB WARP container | WARP CUDA, practical top-8 | **1.37 tok/s best calibrated; 1.213 held-out selected** | **72.34 s**, held-out studio prompt | Selected full request rate is **0.515 tok/s**. The top-8 profile is a measured practical approximation; top-16 and exact profiles remain available. Local WARP is text-only despite the native model being multimodal. |
| Test vehicle | Kimi-Linear-48B-A3B — 2025-10 | 48B / ~3B active | 18 GiB WARP VQ4P | WARP CPU/NEON | **11.20 tok/s** peak | NR | Useful architecture/kernel test vehicle and fast small text model, but not a promoted studio assistant profile. |

## Capacity and feature matrix

| Model | Local context profile | Media exposed locally | Tools / agent use | Best studio role | Main limitation |
| --- | ---: | --- | --- | --- | --- |
| Qwen3.8-27B | **262K** | **Text, images, video** | Native tool calls; Hermes; medium reasoning default | General studio assistant, visual analysis, research and agent work | Dense model is slower than the small-active MoEs; reasoning adds latency. |
| Gemma-4-26B-A4B | 262K | Vision | Tool and reasoning parsers installed | Very fast uncensored drafting and ideation | Specialist derivative; not the primary trust/quality baseline. |
| Ornith-1.0-35B | 262K | Image/video-capable runner | Tool calls and opt-in thinking | Fast uncensored agentic/coding work | Derivative model; quality and safety need task-specific judgment. |
| Qwen3.6-35B Q8 | 131K | Vision via BF16 projector | Tool-capable through Hermes | Fast uncensored fallback with higher-weight fidelity | llama.cpp performance/TTFT needs a fresh normalized capture. |
| Nemotron Nano | 65K | Text only in runner | Tool and reasoning parsers | Compact reasoning specialist | No current normalized local speed row. |
| Qwen3-Coder-Next | 131K | Text | Native tool calls | Large coding specialist | Slow service startup and no normalized local speed row. |
| DeepSeek-V4-Flash / DS4 | 65K configured | Text | OpenAI/Anthropic/Responses-compatible server; harness integration is basic | Fast long-form text and concurrent background jobs | Runtime/tool semantics are less mature than vLLM/Hermes; one GPU owner at a time. |
| GLM-4.7-Flash / WARP | 4K qualified A/B profile | Text | Plain no-thinking chat; native tools fail closed | Compact fast experimental assistant | Full vLLM weights are no longer resident; WARP prefill remains much slower. |
| GLM-5.3-Flash / WARP | **2K exact bounded profile**; 1M native model | **Text only locally** | Fixed-Max one-shot plain chat; tools, media, reasoning overrides, and reasoning-bearing history fail closed | Bounded hybrid-architecture research and one-shot background synthesis | Depth-one MTP verification is exact but experimental and slower than ordinary fused decode; deeper proposals are shadow-only. Vision/video, sparse DSA beyond 2K, and a think-aware response parser are omitted. Generated reasoning and answer currently share `content` and must not be replayed as history. |
| Kimi K2 / WARP | 4K practical profile; 131K native model | Text | Raw completions and minimalist harness | Frontier-scale text work where K2 quality is worth ~2.75 tok/s | No supported WARP chat formatter; long prompts have substantial TTFT. |
| GLM-5.2 / WARP | **2K exact bounded profile**; 1M native model | Text | Raw experimental path | Current very-large-model research and bounded background synthesis | Sparse DSA beyond 2K and MTP are omitted; no serving/tool profile or sustained quality campaign yet. |
| Full GLM-4.7 / WARP | 4K configured; 203K native model | Text | Raw experimental path | Research into models too large for ordinary resident serving | Experimental branch, no MTP, no snapshots, limited quality campaign. |
| Kimi K3 / WARP | ~4K practical studio profile; 1M native model | **Text only locally** | Minimal K3 harness; persisted jobs, outputs and learned expert hotlists | Highest-capacity slow background synthesis and long-horizon studio brainstorming | Very long TTFT, ~0.52 full-request tok/s, no local vision path. |
| Kimi-Linear / WARP | 4K practical; 1M native model | Text | Basic WARP path | Fast experiments and kernel/reference work | Not promoted or broadly quality-qualified for studio work. |

## Other installed studio inference capacities

| Capacity | Installed stack/assets | Approximate local footprint | Operational state |
| --- | --- | ---: | --- |
| Image generation | ComfyUI 0.27, DreamShaper 8 / SD 1.5 checkpoint | Part of 5.8 GiB model tree | **Live and healthy** on port 8188 |
| Depth-guided image work | SD 1.5 depth ControlNet | ~0.7 GiB | Installed in ComfyUI |
| Foreground extraction | BiRefNet | ~0.4 GiB | Installed in the studio image pipeline |
| Image-to-3D experiments | TripoSplat + VAE assets | ~1.2 GiB | Installed; specialist experimental workflow |
| Image embedding / analysis support | DINOv3 ViT-H | ~1.6 GiB | Installed in the image stack |
| Speech transcription | faster-whisper-base | 142 MiB | Installed cache; not part of the main LLM dispatcher |
| User interface | OpenWebUI | Persistent chats/preferences | Ready; sees whichever local backend is active |
| Agent harness | Hermes plus K3 minimalist job harness | Code/config only | Hermes fronts ready API models; K3 harness handles durable background jobs and outputs |

## Operational recommendations

1. **Default interactive work:** Qwen3.8-27B through vLLM/Hermes. It has the
   broadest locally exposed modality and tool surface.
2. **Fast unconstrained drafting:** Gemma-4 first, Ornith second. Their large
   speed figures are short warm peaks, not quality rankings.
3. **Coding specialist:** Qwen3-Coder-Next when its long cold start is worth
   paying; otherwise Qwen3.8 or Ornith.
4. **Fast long text or concurrent background requests:** DS4/DeepSeek V4
   Flash. It has the strongest measured combination of sub-second TTFT,
   20+ tok/s single-stream decode, and 46 tok/s aggregate concurrency.
5. **Frontier-scale background thinking:** K2 is the practical WARP sweet
   spot. K3 is reserved for work where its greater model capacity justifies
   roughly 72 seconds to first token and about 0.52 full-request tok/s.
6. **Engine research / models too large for resident vLLM:** GLM-5.3-Flash
   provides a source-backed 320B/18B hybrid option at a short-row 3.85 tok/s;
   GLM-5.2 provides a ~750B-class option at 1.71 tok/s. Both are bounded to
   2K context. Full GLM-4.7 and K3 remain useful explicit profiles for faster
   GQA work and maximum-capacity synthesis respectively.

## Measurement backlog

The chart is complete as an inventory, but three ready models lack comparable
local performance instrumentation: Qwen3.6-35B Q8, Nemotron Nano, and
Qwen3-Coder-Next. A small future qualification should use the same warm
streamed short prompt, 256 output tokens, and record TTFT, post-first-token
decode, end-to-end rate, peak memory, and service startup time. This would
fill the gaps without turning the inventory into an engineering campaign.

## Maintained control points on the Spark

- `~/RUNNERS.md` — runner commands and routing rules
- `~/vllm-runners/` — vLLM launchers
- `~/llamacpp-runners/` — llama.cpp launcher
- `~/ds4-runners/` — DS4 fast/reference launchers
- `~/Development/k3-minimal-harness/` — private durable WARP job harness
- `~/waste-gn100-experiment-20260730/models/` — K3, K2 and Kimi-Linear containers
- `~/Development/glm47-*-experiment/model.waste` — retained GLM conversions
- `~/Development/glm52-experiment/model.waste` — retained bounded GLM-5.2 container
- `~/Development/glm53-flash-experiment/model.waste` — retained bounded GLM-5.3-Flash container
- `~/Development/glm53-flash-mtp-experiment/model.waste` — separate opt-in GLM-5.3 MTP research container
