# Qualified GB10 CUDA operating profile

This file is the frozen execution header for the GB10 consolidation soak. It
promotes only the CUDA work qualified through Sprint 13; later cache-policy and
speculative-decoding experiments remain on their own branches.

## Engine profile

| Setting | Frozen value |
|---|---|
| Model | Kimi K3 converted WASTE container |
| CUDA KDA | `WASTE_CUDA_KDA=1` |
| CUDA dense | `WASTE_CUDA_DENSE=2` |
| CUDA VQ | `WASTE_CUDA_VQ=2`, group 1 |
| Compute | 10 threads on CPUs `5-9,15-19` |
| PM QoS | Q0, child-scoped to the engine/server lifetime |
| Quantized trunk | Q8 on; SDOT and I8MM off |
| Storage | direct I/O; two readers at depth two |
| Router lookahead | 0 |
| Cache policy | LFRU; no decode aging or inherited-prior transform |
| Expert cache | 59,340 MiB (62,222,499,840 bytes) |
| Hotlist | frozen Sprint 12 development-corpus usage file |

The public engine's budget is total process memory, not expert-cache bytes.
For this K3 container at context 4096, a 59,340-MiB expert cache requires an
explicit CLI budget of **93,024,636,928 bytes**. The server reserves 2 GiB for
exact prefix snapshots inside the same physical pool, so its matched total
budget is **95,172,120,576 bytes**. The older recommended total of
86,583,021,568 bytes produces a smaller cache and is not this profile.

The named server profile is `spark-cuda`. It fixes the arithmetic and storage
environment atomically and rejects conflicting `WASTE_*` selectors. Run it
through `tools/spark_cuda_serve.sh`, which supplies the ten-core affinity,
matched server budget, and child-scoped Q0 holder. The holder's status artifact
is part of every campaign's evidence because child-scoped Q0 is external to the
server process.

Router lookahead 6 remains an optional prompt-family-dependent setting. It is
not part of the promoted profile because its measured gain used a calibrated
development workload. Any lookahead-6 number must be labeled as a separate
arm.

## Build

The GN100 build is:

```text
make -j8 WASTE_NATIVE=1 WASTE_ENABLE_CUDA=1
```

GCC 13 on the Cortex-X925 does not define the dot-product or I8MM feature
macros under `-mcpu=native`. That does not change this profile because both
runtime paths are off, but builds intended to exercise them must use an
explicit architecture such as `-march=armv8.6-a+dotprod+i8mm -mtune=native`
and verify the resulting macros and disassembly.

## Power definitions

Shelly wall power is sampled at 1 Hz and its lifetime energy accumulator is
the energy authority. NVML is sampled at 4 Hz for the GPU rail, temperature,
clock, and utilization context.

- Total J/token is the Shelly energy change over the measured request window,
  converted to joules and divided by generated tokens.
- Marginal J/token is `(mean request W - loaded-idle W) * seconds / tokens`.

Both values are reported together. The loaded-idle value is measured in the
same session as the workload it qualifies.
