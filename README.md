# WARP — Weight-Aware Runtime and Paging (formerly WASTE)

> **NVIDIA GB10 / DGX Spark / Acer Veriton GN100 lab fork.** `main` mirrors
> upstream; green, combined GB10 work lives on `spark/integration`; measured
> milestones and raw evidence stay immutable. The qualified K3 profile combines
> CUDA KDA, dense projections, and expert VQ gather with ten pinned performance
> CPUs, child-scoped Q0, two direct-I/O readers at depth two, and Q8. **K3,
> held-out studio prompts, 64 tokens: 0.637 tok/s at 121.35 W wall power** (12
> measurements across six newly frozen families; 187.73 J/generated token). The
> optional lookahead-6 arm reached 0.728 tok/s at 132.13 W and 179.74 J/token
> without changing output. A 4.52-hour promotion soak completed 50/50 exact
> trajectories on the development corpus. These results use the Spark's internal
> NVMe, an otherwise exclusive host, and greedy decoding. See the
> [qualified profile](docs/GB10_QUALIFIED_PROFILE.md),
> [GN100 results and reproduction notes](docs/GN100.md), and
> [upstream issue #11](https://github.com/sqliteai/warp/issues/11).
> `tools/spark_cuda_serve.sh` is the qualified single-user GN100 server path.
>
> **Latest quarantined experiment:** strict CUDA VQ4P coverage is complete on
> Kimi-Linear. In a direct matched GB10 run, VQ3R remained slightly faster
> (12.731 versus 12.326 tok/s with Q0), so this is compatibility coverage rather
> than a K3-profile change. See the
> [VQ4P report](https://github.com/mccoyspace/warp-spark/blob/exp/cuda-vq4p-gb10/docs/GPU_VQ4P_GB10.md).

This fork follows the WARP project name. Upstream intentionally retains the
`waste` executable, C API names, environment variables, and `.waste` container
extension for compatibility.

WARP is an embeddable inference engine written in C, with no third-party runtime dependencies. It keeps the model trunk in memory, streams selected experts directly from disk, and uses the remaining RAM as a bounded expert cache.

The project is driven by humans: the ideas, hypotheses, priorities, tests, and decisions are human. The code is written by LLMs. At this scale, that is the only way to iterate on new algorithms and test hypotheses fast enough.

The goal is to run huge frontier models such as Kimi K3 on consumer hardware. Today, the complete 2.78-trillion-parameter Kimi K3 runs on a 64 GB MacBook Pro at about **0.6 tokens per second**.

**Ultimately we want WARP to execute Kimi K3 locally to improve itself** (we are currently using Opus 5 with extra thinking).

WARP is intentionally narrow, and it exists to find out how far local inference can be pushed when model weights live mostly on fast storage instead of RAM.

```text
$ waste run ~/models/k3.waste 'What is the capital of Italy?'
waste: no --budget, using 46.39 GB of 64.00 GB (expert cache 17.56 GB)
The capital of Italy is **Rome**.
[16 tokens, 26.87 s, 0.60 tok/s | experts 9000 hit / 14552 miss = 38%]
```

**This is the full model, not a distilled or pruned version.** Its published weights occupy 1.42 TB; the converted WARP container is 982 GB.

## How it works

Kimi K3 is a mixture-of-experts model. It has 2.78 trillion parameters, but only about 4% of them are active for each token. WARP keeps the shared part of the model in RAM and reads only the selected experts from disk.

The container is arranged so that one expert requires one aligned read. Those reads overlap with computation, while unused RAM becomes a bounded expert cache. A lookahead router predicts the experts needed by the next layer and starts reading them early; the real router still makes the decision, so this changes timing, not the result. Experts use 3-bit residual vector quantization, while the more sensitive shared weights remain at 4 or 8 bits.

K3's linear attention and compressed latent KV cache also matter: at 4K context, the KV cache is about 0.21 GB instead of 11.25 GB. The result is an engine that needs 29.19 GB to open K3 and uses the rest of the available memory to avoid repeated disk reads.

For the full design and measurements, see [docs/ENGINE.md](docs/ENGINE.md) and [docs/EFFICIENCY.md](docs/EFFICIENCY.md). The on-disk layout is documented in [docs/FORMAT.md](docs/FORMAT.md), while [docs/KDA.md](docs/KDA.md) describes Kimi Delta Attention.

## Performance

Measured on a 64 GB MacBook Pro with an M5 Pro and the model container on the
internal SSD:

| Model | Container | Minimum RAM | Decode speed |
|---|---:|---:|---:|
| Kimi K3 2.78T | 982 GB | 29.19 GB | 0.45–0.62 tok/s |
| Kimi-Linear 48B | 19 GB | 1.32 GB | 10.62 tok/s |

For K3, 64 GB is the practical minimum. A 32 GB machine can open the model but will page heavily. The default memory budget on the test machine is 46.39 GB, including a 17.56 GB expert cache.

Most of that requirement is the 27.28 GB resident trunk rather than the cache. Shrinking the expert cache from 17.32 GB to 3.32 GB costs about 10% of throughput; enlarging it past the default costs everything. Measured across four cache sizes in one process:

| expert cache | hit rate | decode |
|---:|---:|---:|
| 3.32 GB | 29.1% | 0.56–0.58 tok/s |
| 17.32 GB | 36.2% | **0.63 tok/s** |
| 23.32 GB | 38.4% | 0.07–0.09 tok/s |
| 29.32 GB | 41.3% | 0.07–0.08 tok/s |

The last two rows are the failure mode worth knowing about: the hit rate keeps climbing and the bytes read keep falling while throughput drops eightfold. The engine is inside its budget and the machine is not, so a cache hit becomes a page fault. Giving the process more memory is not always faster.

Decoding with fewer experts per token is a knob rather than a rebuild:
`num_experts_per_token` in the container manifest. K3 ships at 16. Measured
on this machine, one load with the arms interleaved:

| experts/token | decode | KL from top-16 | working set |
|---:|---:|---:|---:|
| 16 | 0.59 tok/s | — | 17.01 GiB |
| 12 | 0.70 tok/s | 0.007 | 12.76 GiB |
| **8** | **0.89 tok/s** | **0.037** | **8.50 GiB** |
| 4 | 1.06 tok/s | 0.118 | 4.25 GiB |

Top-8 is 1.49x for a divergence twice that of a quantization this project
rejects elsewhere, and it reproduces top-16's greedy continuation on the
prompts tested. Top-4 does not: its next-token distribution still looks
close, and it stops following the prompt within a few tokens — which is why
the gate here is a continuation and not a KL. This is a quality trade and
the default stays 16.

Storage is the main constraint. A cold K3 token reads about 17 GB of experts. The internal SSD sustains 12.78 GB/s; a tested USB enclosure managed 0.94 GB/s. Put the converted container on internal NVMe storage.

All layers are checked against a PyTorch reference. Final logits agree within 3.6e-06, and the vision tower agrees with its oracle within 2.3e-06.

Additional measurements, profiling data, router-lookahead results, and quantization experiments are collected in [docs/TECHNICAL.md](docs/TECHNICAL.md).

## Vision

Kimi K3 is multimodal, and WARP can use one or more images together with text. Pass `--image` once per image:

```bash
./waste run ~/models/k3.waste "Describe this image" --image photo.jpg
./waste run ~/models/k3.waste "Compare these images" \
    --image before.png --image after.png
```

In interactive mode, `/image FILE` attaches an image to the next message. An image is expanded into many prompt positions: an 896×896 image uses 256 positions at the default patch budget. The vision tower takes about 15.7 seconds for 1024 patches on the test machine, but most of the cost comes afterward because every image position passes through the language model like a text position. In the current K3 measurements, that is about 2.8 seconds per image position.

See [docs/K3.md](docs/K3.md) for the vision architecture and measurements, and [examples/README.md](examples/README.md) for CLI, C, and HTTP multimodal examples.

## Other models

K3 is the target and the best-tested model, and Kimi-Linear is the small one to
start with. Since 0.6.8 the converter and the engine also handle the
**DeepSeek-V3 family — V3, R1 and Kimi K2**, which needed two changes rather
than one.

`convert.py` now reads fp8 block-scaled weights, applying the per-tile scales
these checkpoints ship in a companion tensor, and normalises DeepSeek's MoE
tensor and config names to the single spelling the engine reads. And MLA now
applies rotary to its rope dims. The engine had implemented none: the Kimi
models set `mla_use_nope` and pass those dims through unrotated, which is
correct for them and wrong for everything in the V3 family, where in MLA those
dims are the only positional signal there is. A container built before this was
not degraded, it was unordered — it could not tell which turn of a conversation
came first.

No throughput figures here, because nobody on this project has a K2 container.
What has been measured, by [@fab2s](https://github.com/fab2s) who contributed
both changes: a `Kimi-K2-Instruct` conversion — 61 layers, 384 experts top-8,
VQ3R, a 354 GB expert set and a 6.9 GB trunk — opens and reports 1.03 T
parameters total, 31.69 B active per token; and the rotary arithmetic agrees to
0.000023% relative L2 with an oracle whose YaRN helpers are taken verbatim from
the DeepSeek release's own `modeling_deepseek.py`.

Kimi K3 and Kimi-Linear are unaffected: their forward pass is byte-identical to
0.6.7, by construction rather than by a runtime branch.

## What you need

To build and test WARP:

- a C11 compiler and `make`;
- macOS or Linux; Windows builds with MinGW-w64;
- no BLAS, Python, CUDA, or other external dependency for the current CPU
  inference path.

To run Kimi K3:

- **64 GB of RAM recommended**; 29.19 GB is the hard floor at 4K context;
- **about 1 TB of internal NVMe storage** for the converted model;
- another **1.42 TB of temporary storage** if converting the published weights
  yourself. This staging storage may be external and can be freed afterward.

If you only want to try the engine, start with Kimi-Linear. Its container is 19 GB, it needs 1.32 GB of RAM, and it runs at about 10.6 tok/s on the same machine.

Python, PyTorch, and safetensors are needed only for model conversion and validation, never for inference.

## Getting started

Build the engine and run the model-free test suite:

```bash
git clone https://github.com/sqliteai/warp
cd waste
make
make check
```

`make` builds the `waste` CLI and `libwaste.a`. `make check` creates a small synthetic model, so it does not download weights.

### Get Kimi K3, already converted

The fastest route is to download the already converted container over
BitTorrent. It skips the 1.42 TB source download, the 4.7-hour conversion, and
the temporary staging storage entirely — only the 982 GB container lands on
disk. Any BitTorrent client will do; the torrent's own piece hashes verify the
container as it arrives.

```text
magnet:?xt=urn:btih:54db69b0df8baf5e617744dda5d46c90a2d0f632&dn=k3.waste&tr=udp%3A%2F%2Ftracker.opentrackr.org%3A1337%2Fannounce&tr=udp%3A%2F%2Fopen.demonii.com%3A1337%2Fannounce&tr=udp%3A%2F%2Ftracker.torrent.eu.org%3A451%2Fannounce&tr=https%3A%2F%2Ftracker.tamersunion.org%3A443%2Fannounce
```

With [aria2](https://aria2.github.io/), which resumes and needs no GUI:

```bash
aria2c --dir ~/models --seed-time=60 \
  'magnet:?xt=urn:btih:54db69b0df8baf5e617744dda5d46c90a2d0f632&dn=k3.waste&tr=udp%3A%2F%2Ftracker.opentrackr.org%3A1337%2Fannounce&tr=udp%3A%2F%2Fopen.demonii.com%3A1337%2Fannounce&tr=udp%3A%2F%2Ftracker.torrent.eu.org%3A451%2Fannounce&tr=https%3A%2F%2Ftracker.tamersunion.org%3A443%2Fannounce'
```

This writes `~/models/k3.waste`, the directory the commands below expect. Point
`--dir` at internal NVMe storage: a container downloaded onto an external disk
has to be copied before it is usable at full speed. Raise `--seed-time` if you
can afford to share it back.

### Get Kimi K3, from the published weights

Convert it yourself if you would rather not trust a third-party copy, or if you
already hold the original weights. The download and conversion are resumable:

```bash
# Check required download space.
tools/fetch_weights.sh --dest /Volumes/staging/k3 --dry-run

# Download the original weights.
tools/fetch_weights.sh --dest /Volumes/staging/k3

# Convert them. Put the output on the internal SSD.
uv run --with torch --with safetensors python tools/convert.py \
    --src /Volumes/staging/k3 \
    --out ~/models/k3.waste \
    --jobs 3
```

Conversion takes about 4.7 hours with three workers on the test machine. See [docs/K3.md](docs/K3.md) for validation, recovery, and storage details.

### Run it

```bash
./waste plan ~/models/k3.waste
./waste run  ~/models/k3.waste "The capital of France is" -n 32
./waste chat ~/models/k3.waste
```

Do not set `--budget` unless you have a reason to. By default WARP chooses a
safe memory budget, reports it, and refuses to start below the model's floor.
Inside a container it sizes against the cgroup limit rather than the host's RAM.
Use `./waste --help` for the complete command list.

More CLI examples, including evaluation, tokenization, saved sessions, and multimodal prompts, are in [examples/README.md](examples/README.md).

### Serve it

The optional server implements the OpenAI chat-completions API:

```bash
make libwaste.dylib                 # use libwaste.so on Linux
python3 -m serve ~/models/k3.waste --port 8000

curl localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"k3","messages":[{"role":"user","content":"Why is the sky blue?"}]}'
```

It supports streaming, tools, structured output, thinking controls, and images. See [docs/SERVE.md](docs/SERVE.md) for the protocol and [examples/README.md](examples/README.md) for complete requests.

## Library

WARP is also an embeddable C library. The CLI and server both use the public API in [src/waste.h](src/waste.h). The inference path depends only on libc and pthreads. Text generation, memory planning, session persistence, and multimodal C examples are available in [examples/README.md](examples/README.md).

## Why the name

Every token answered by a cloud service is paid for twice: once on the invoice, and once in the electricity of a datacenter running a model that would fit — barely, awkwardly, but genuinely — on hardware already sitting on a desk. WARP means to be the first concrete step toward ending that waste of tokens — which the project's first name said outright, and which the rename did not change.

## Project status

The format and API are not frozen. K3 is the main target and the best-tested model. The CPU path is currently the fastest measured implementation for this workload, but it is not assumed to be the final answer. CUDA, Metal, and other hardware-specific optimizations remain to be explored and may provide significant gains. Current backend results are documented in [docs/BACKENDS.md](docs/BACKENDS.md), while open directions are tracked in [docs/RESEARCH.md](docs/RESEARCH.md). Read [docs/LEARNED.md](docs/LEARNED.md) before proposing an optimization: failed ideas and negative results are kept there deliberately.

Contributors are more than welcome. New experiments, support for additional hardware, and open discussion about how to improve performance are all encouraged—even when an idea produces a negative result.

The software is currently changing very quickly. Before each release, a large QA run is executed; however, instabilities are definitely possible.

Measurements are treated as experimental results rather than marketing numbers. Each result is tied to the hardware, container, configuration, and commit on which it was obtained; unstable measurements are reported as ranges, and results later found to be wrong remain recorded as such. The detailed snapshots are in [docs/TECHNICAL.md](docs/TECHNICAL.md) and the full history, including negative results, is in [docs/LEARNED.md](docs/LEARNED.md).

Validation covers more than successful generation. The model-free suite builds a synthetic container; real-model checks compare individual layers and final logits against PyTorch, verify conversion round trips, test vision against its oracle, and exercise the server prompt renderer segment by segment against K3's reference encoder. The validation criteria and current evidence are documented in [docs/GATES.md](docs/GATES.md), with server-specific differential tests in [docs/SERVE.md](docs/SERVE.md).

Useful references:

- [docs/FORMAT.md](docs/FORMAT.md): container format;
- [docs/BACKENDS.md](docs/BACKENDS.md): CPU, SIMD, and Metal backends;
- [docs/KDA.md](docs/KDA.md): Kimi Delta Attention;
- [docs/GATES.md](docs/GATES.md): correctness and performance gates;
- [docs/RESEARCH.md](docs/RESEARCH.md): current research directions.
- [docs/TECHNICAL.md](docs/TECHNICAL.md): detailed measurements and technical experiments.

## License

WARP is distributed under the permissive Apache 2.0 license, and **the project will always remain open source under a permissive license**. See [LICENSE](LICENSE).

Copyright 2026 SQLite Cloud, Inc.
