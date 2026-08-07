# Changelog

Every release since 0.6.1. Numbers here were measured on the commit they
ship with, the same rule the rest of the documentation follows — where a
change was measured and *not* adopted, that is recorded too, because the
measurement is the useful part.

`docs/LEARNED.md` carries the full reasoning; this file carries what
changed. Each entry names the section to read for the numbers behind it.

## 0.7.0-spark.2 — 2026-08-07

Spark integration release realigned to upstream 0.6.6. It retains the
qualified GN100 profiles, CUDA dense and VQ3R paths, memory-pressure guard,
state snapshots, and process-ownership controls while adopting upstream's
pluggable VQ metadata, expert-parallel option, compute-pool CPU placement,
converter reclaim path, Linux direct-I/O diskbench fix, and server defaults.

The CUDA VQ implementation consumes the generalized manifest geometry but
remains deliberately **VQ3R-complete**: a VQ4P container is rejected before
CUDA execution pending a separate numerical and performance contract. Under
the qualified CUDA profile, external `taskset` placement continues to own the
whole process, including expert readers; upstream `--cpus` remains available
outside that profile and binds only the compute pool.

## 0.7.0-spark.1 — 2026-08-02

Spark integration release based on upstream 0.6.3. This is API 2, not a
drop-in upstream 0.6.3 library: `waste_cfg` and `waste_memplan` have grown to
account for host-owned memory and process ownership. The four functions that
read or write those structures export `_v2` symbols; source clients use the
ordinary names through header macros, while an API-1 binary using those calls
fails symbol resolution instead of crossing the boundary with smaller
structures. Dynamic bindings can verify `waste_api_version()` and the two
exported structure sizes before making any structure-bearing call.

## 0.6.6 — 2026-08-05

The engine decodes exactly as 0.6.5 did and no container format moved. Two
things make it a tag. A host can now say which CPUs the compute pool runs
on, which is the first answer this project has to a machine whose cores are
not interchangeable. And `tools/diskbench` — the tool whose entire job is to
certify the storage a container will be streamed from — was measuring the
page cache on Linux, so it had been answering that question wrong for
everyone who is not on macOS.

**Callers must recompile against this header.** `cpu_list` was added to
`waste_cfg` between `n_threads` and `cache_policy`, so a caller built
against 0.6.5's struct reads `cache_policy` and every field after it from
the wrong offset. `serve/engine.py`'s mirror moved with it; an out-of-tree
ctypes or FFI binding has to move too. The library is 0.6.x and promises no
stable ABI yet, but a silent misread is worth the sentence.

### Added

- **`--cpus LIST`** on the CLI and the server, `waste_cfg.cpu_list` in the
  API, `WASTE_CPUS` in the environment — a Linux-style cpu list (`0-5`,
  `0-2,6-8`) that the compute pool binds to. Linux and Windows; macOS has
  no call that binds a thread to a core, so a list there is refused with
  `WASTE_E_UNSUPPORTED` rather than ignored.

  It exists because on a machine whose cores are not interchangeable,
  placement is worth more than the thread count.
  [Issue #23](https://github.com/sqliteai/waste/issues/23) measured
  Kimi-Linear-48B on a Ryzen 9 9900X — two 6-core CCDs, separate 32 MB L3 —
  and found six threads on one CCD 16-25% faster than the same six split
  across both, at identical `bytes_read` and identical hit counts. Handing
  those six threads all 24 CPUs to migrate between costs a further ~10%.
  Not reproduced here: this repo has no multi-CCD machine, and the numbers
  above are the reporter's. What *is* checked here is the mechanism —
  `tests/test_cpus.c` reads back every participant's affinity mask.

  **No default changed.** The engine still names no CPUs and leaves
  placement to the OS. `docs/LEARNED.md` §47 measured the tempting default
  — cap the pool at the fast cores — as a 25% gain on Kimi-Linear and a 34%
  loss on K3, so it stays a switch. `--threads 0` with a cpu list means one
  thread per CPU listed; an explicit `--threads` still wins. The thread
  that calls into the engine is bound too, on its first parallel region,
  because it is one of the workers; the expert cache's reader threads are
  not, because they are blocked in `pread` rather than competing for a
  core. `docs/ENGINE.md`, "Thread placement", has the rest.

- **The converted K3 container over BitTorrent**, in `README.md` ahead of
  the conversion recipe. The default conversion is deterministic, so the
  982 GB directory is byte-identical for everyone who produces it, and
  nearly all of what the recipe costs — a 1.42 TB source download, 4.7
  hours, and staging storage that has to exist before it can be freed — is
  paid to reproduce a fixed artifact. The torrent's own piece hashes verify
  it as it arrives. Converting from the published weights stays documented,
  for anyone who would rather not trust a third-party copy.

### Fixed

- **`tools/diskbench` measured the page cache on Linux, not the disk**
  ([#22](https://github.com/sqliteai/waste/pull/22), `docs/LEARNED.md`
  §49). It documented itself as reading with the cache bypassed, and did
  neither: `nocache()` had an `#ifdef __APPLE__` body and nothing else in
  it, and `O_DIRECT` appeared nowhere in the file. Against a Samsung 970
  PRO on Gen3 x4 it reported 44.67 GB/s sequential and 65.72 GB/s random
  over a 3.94 GB/s link — 11x and 17x the ceiling. Bypassed: 3.15 and 3.33
  GB/s, saturating at two threads, which is what that drive should do.

  This is LEARNED §14 in the one place §14 did not reach. The engine's own
  bypass was written blind and fixed on 2026-07-28; the tool that exists to
  characterise the engine's I/O kept reading RAM, which means §46's standing
  rule — run `diskbench` and divide before claiming anything is disk-bound —
  returned a fiction on Linux for that whole window. **No published number
  moves**: every `diskbench` figure in `docs/GATES.md`, `docs/EFFICIENCY.md`
  and LEARNED §44/§46 was measured on macOS, where `F_NOCACHE` did work.

  The flag alone is not enough, for the reason `bank_open` already knows:
  `O_DIRECT` is accepted at open and refused at transfer (tmpfs does this),
  so a bare flag turns a refusing filesystem into a table of zeroes with no
  cause given. It follows `bank_open` instead — probe with one aligned
  transfer, fall back to a plain open plus `POSIX_FADV_RANDOM`, and label
  every row, because a bench that quietly measures something else is worse
  than one that says it could not. The write is bypassed too, and that is
  not symmetry for its own sake: `F_NOCACHE` stops new pages being cached
  but does not evict resident ones, so a buffered write leaves the file in
  the UBC and every read row below it reports RAM — 8.07 GB/s sequential
  with the write bypassed against 26.04 GB/s with it buffered, 1 GB file on
  an M5 Pro. Also fixed alongside: a sub-page record rounded to zero and was
  divided by, and a failed sequential read ended the loop and silently
  shortened the row.

  Reported, diagnosed and fixed by fab2s. Verified here on macOS as
  unchanged within noise; the Linux figures are the reporter's, on hardware
  this repo does not have.

- **`diskbench`'s tok/s column answered for K3 whatever was being sized.**
  The derived column carried 12.5 GB/token in its format string — K3's
  figure — so on a 48B model at a measured 1.61 GB/token it was ~8x off,
  and silent about the assumption, which is what made it a trap rather than
  an approximation. It is now the fifth positional argument with no default:
  without it the column is not printed. A tool cannot derive bytes-per-token
  from a scratch file — that number belongs to a container, and `waste
  bench` already reports it. `docs/GATES.md`'s Gate H table keeps its
  "tok/s @12.5 GB/token" header: that was a K3 decision, and the figure is
  stated in the header rather than hidden in a format string.

## 0.6.5 — 2026-08-04

Nothing in the engine changed: a binary built from this tag decodes exactly
as 0.6.4 did, and no container format moved. What changed is on either side
of it — converting K3 on the disk you actually have, and how long a reply
the server gives a client that never asked for a length.

### Added

- **`tools/convert.py --reclaim {off,dry,on}`** — deletes each source shard
  once its last consumer has published, so peak staging is the container
  plus the shards still owed rather than the container plus all of them. On
  K3 that is the difference between 1.42 TB of staging beside a 982 GiB
  container — two disks — and one. It is safe because every tensor has
  exactly one consumer, so a shard whose last consumer has finished is never
  opened again. `--reclaim` also runs the trunk pass first: it consumes
  every non-expert tensor, and while it ran last almost no shard was ever
  spent. The reordering is neutral — `off`, `dry` and `on` produce
  byte-identical containers.

  **Off by default, and not reversible.** A reclaimed shard has to be
  downloaded again, and `verify_container.py` loses the comparison against
  source for good — which is why `pipeline.sh` now converts one probe layer
  and round-trips *that* while the checkpoint is still whole, before
  converting the rest. Stages renumber to six; stage 4 passes `--skip-trunk`
  because stage 2 built it, a K3 trunk being hours to do twice.

  It refuses **before** deleting rather than during: `--experts`, a
  container inside the checkpoint or the reverse, a shard that is neither on
  disk nor already reclaimed, and a bank that is not a whole bank —
  `bank_is_sound` walks the records by their own block counts for 48 bytes
  each, which catches a bank truncated by a kill, a full disk, or a torn
  rename. Releases are recorded in `<src>/.reclaimed`, fsynced ahead of the
  unlink, because a name recorded but not deleted costs nothing while the
  reverse is indistinguishable from an unfinished download.
  [K3.md](docs/K3.md) has the refusals and the ledger discipline.

  Proven on a copy of a real Kimi-Linear checkpoint rather than on stubs:
  `--layers 1,2 --reclaim on` deleted exactly the one shard the dry pass
  named (4.7 GiB, 92 -> 87 GB), left the other 19 unchanged, and wrote a
  container that reads back 256 records with 0 problems. The second run over
  the now-incomplete source refuses — pointing at `--skip-trunk` — and
  deletes nothing while refusing.

### Changed

- **The server's default `--max-tokens` is 4096, was 512.** Clients mostly
  do not send the field; Open-WebUI does not unless you set it in the
  model's advanced parameters, so the server default was every reply's
  length, and a reply that ends at the cap is indistinguishable from a model
  that stopped on its own. `--ctx` never lifted it and could not: the limit
  is clamped to the room left after the prompt, so raising the context only
  ever lowers the cap. **This is a behaviour change, not a fix** — a
  deployment that relied on 512 to bound per-request cost should now pass
  `--max-tokens` explicitly. `Engine.generate`'s own default is untouched:
  the server always passes the value, so the two never meet.

- **[SERVE.md](docs/SERVE.md) documents Open-WebUI** — the base URL, why no
  compatibility mode is needed, `--host 0.0.0.0` for a client in a
  container, the background title and tag requests that queue behind the
  reply on the lock every generation takes, and `reasoning_content`, which a
  client that does not know the field renders as a server that has stopped.

### Fixed

- **A resumed `--reclaim` run believed the download ledger over the disk.**
  `ST.have()` reads `fetch_weights.sh`'s `.download-state`, so a shard that
  the run had itself consumed still read as present and both refusals were
  skipped. Absence is now asked of the filesystem, and `have()` only asked
  whether a shard that is there finished downloading. The pipeline test
  found it; the file-backed test stub could not have, and now mirrors
  `mxfp4.ST`, trap included.

## 0.6.4 — 2026-08-04

A container format addition and the measurements that price it. **Nothing
changes by default and this is not a speedup release**: no shipped container
uses the new format, the new switches are off, and an engine built from this
tag behaves exactly as 0.6.3 did unless it is asked otherwise. The reason to
tag it is that `fmt 8` is now allocated and public, and a format code that
lives only in a working tree is one somebody else reassigns.

### Added

- **`WQ_VQ4P`, fmt 8** — 4 residual VQ stages of 64 entries with 6-bit
  indices packed four into three bytes. Same 3.00 bits/weight as VQ3R, same
  record size, same blocked index layout; what changes is that a 64-entry
  stage table is 64 bytes, which is one NEON `vqtbl4q`, where VQ3R's
  256-entry table is sixteen vector registers on a machine that has
  thirty-two and cannot be held at all. That is why the VQ3R gather is
  scalar. [FORMAT.md](docs/FORMAT.md) specifies the packing;
  [LEARNED.md](docs/LEARNED.md) §41 has the derivation.

  It is a distinct fmt rather than a manifest flag because a VQ4P payload is
  byte-for-byte the size of a VQ3R one, so a reader taking the three bytes
  for three one-byte indices would decode silently and wrongly. The engine
  additionally refuses a record whose fmt byte disagrees with the manifest's
  `index_bits`, which is the only read-path behaviour that changed.

- **`tools/convert.py`: `--entries`, `--index-bits`, `--stages 4|6`.**
  Default output is unchanged VQ3R. The parameters travel in the job tuple
  rather than a module global: the worker pool uses `spawn`, so a global
  would have written every layer at 256 entries without saying so.

- **`WASTE_XPAR`** — one task per routed expert instead of one per row
  range, **off by default**. `WASTE_XPAR_BATCH` bounds how many experts are
  held at once, `WASTE_P6_CHUNK` sizes the VQ4P apply, and
  `-DWASTE_P6_SCALAR` builds the kernel's portable path, which is
  bit-identical to the NEON one rather than merely close — §43 explains why
  an int8 lookup table raises that bar.

- **`tools/lutbw.c`, `tools/lutmt.c`** — the two benches the sections below
  rest on: kernel throughput against a working set from 4 MB to 1 GB, and
  the same kernels driven through the engine's own `waste_parallel_for`.

### Measured

Numbers on this commit, medians of repeated runs, each container on the same
storage as the baseline it is compared against.

- **The kernel is 3.88x** and that survives a gigabyte of index stream, so it
  is not a cache artefact (§46).
- **In the engine it is 1.24x** on Kimi-Linear best-configuration against
  best-configuration, and **1.74x** against what the engine does untouched;
  **1.09x** on K3 (§46, §47).
- **Quality costs +2.7% perplexity** on Kimi-Linear (10.937 -> 11.237), all
  of it from the smaller codebook. Quantizing the runtime lookup table to
  int8 — which is what makes a byte shuffle possible at all — measured free,
  because the scale is per 32 vector positions rather than global (§41).

The gap between 3.88x on a bench and 1.17x in place is the useful part, and
§47 is the answer: 3.88x is a single-thread ratio, this machine is 6
performance cores and 12 efficiency cores with the pool taking all 18, and
the fast kernel is the one an E-core straggler hurts. **On K3 the same
settings invert** — six threads are 34% worse than eighteen there. No
default was changed because every one of them is right on one model and
wrong on the other.

### Not adopted

- **6x16 codebooks** (FAISS FastScan's shape) are a third faster than 4x64
  and cost 18% more reconstruction error against 4x64's 7.5%. Speed was not
  the binding constraint (§41).
- **Expert-parallel MoE as a default.** Worth 1.24x on Kimi-Linear, a
  regression on K3: the batch that gives it parallelism is the batch that
  barriers the read-ahead, and no batch size wins both (§44).
- **Capping the thread pool at the performance cores.** A 25% win on
  Kimi-Linear and a 34% loss on K3 (§47).

## 0.6.3 — 2026-08-03

### Fixed

- **The automatic budget sized against the host's RAM inside a container**
  ([#14](https://github.com/sqliteai/waste/issues/14)). `waste_physical_ram()`
  is `sysconf(_SC_PHYS_PAGES)` on Linux, which reports the host's `MemTotal`
  from inside a cgroup that is allowed a fraction of it, so a `--budget`-less
  open resolved `floor + 3x` against memory the kernel would never hand over:
  K3 in a 32 GiB cgroup on a large host asks for ~80 GB and is killed. Unlike
  the paging cliff of [LEARNED.md](docs/LEARNED.md) §16 this has no gradual
  form and no cache policy softens it. The ceiling is now
  `min(physical, cgroup limit)` — the smallest finite `memory.max` or
  `memory.high` across the cgroup and its ancestors, since the limit is
  hierarchical. This stable capacity feeds the integration's existing
  current-pressure resolver. See §43.

  Current pressure (`MemAvailable`, `memory.current`) remains deliberately
  outside `waste_usable_ram()`. In this Spark integration it is applied by the
  separate `waste_memory_ceiling()` safety snapshot, along with caller-owned
  `host_reserved_bytes`, after stable capacity has been established.

- **`tools/convert.py` spawned `--jobs` × cores threads**
  ([#13](https://github.com/sqliteai/waste/pull/13), contributed by
  @andrewwhitecdw). torch sizes its intra-op pool from `os.cpu_count()`, and
  the native VQ encoder reads `nthreads=0` as "every core" (capped at 64), so
  N worker processes meant N×cpus threads competing for the machine: on a
  224-core box `--jobs 8` spawned ~1792 threads and the codebook phase ran
  ~20x slower than the measured baseline. Each worker is now capped at its
  fair share, `cpu_count // jobs`, for both pools — set in the parent before
  the workers spawn, since that is the only point torch reads it, and with
  `setdefault`, so an explicit `OMP_NUM_THREADS` stays the caller's.

- **The trace simulator modelled a different cache than the engine.**
  `tools/routing_stats.py simulate` kept a frequency count across evictions
  that `ec_claim` resets and sampled 32 victims where `EC_SAMPLE` is 16:
  against the same trace it read **36.6% where the engine measured 30.4%** —
  optimistic, plausible and wrong. Both constants now come from `ecache.c`,
  which brings it within 1.5 points across a 0–30% range, and `tests/run.sh`
  asserts the agreement rather than remembering it. Two smaller things went
  with it: the route dump writes the absolute position of the token each row
  belongs to, so readers stop re-deriving token boundaries from where the
  layer index wraps (a heuristic that is simply wrong on the chunked path,
  where rows group by layer), and `simulate --data` takes a container, whose
  manifest states the record size the engine actually `pread`s. See §37,
  *The simulator was modelling a different cache*.

### Added

- `waste_usable_ram()`: physical RAM, or a smaller cgroup-v2 limit when one
  applies — what a budget of 0 sizes against, and what an embedding host
  should use as stable capacity. `waste plan --json` reports it beside
  `physical_ram_bytes` and the integration's dynamic
  `memory_ceiling_bytes`.

- **`tests/sweep.c`, a one-process measurement harness.** It loads a
  container once and runs the arms back to back, interleaved, resetting the
  session and clearing the expert cache between each — a warm cache would
  hand the second arm the first one's work and measure the order instead of
  the setting. Kimi-Linear, two arms, three repeats: spreads of 2.6% and
  0.5%, where nine paired runs of the same comparison across processes
  spanned 0.79x to 1.79x. The variance was the harness, not the feature. On
  K3 the deterministic columns come out exact and the clock still drifts,
  which is the machine's memory system rather than the process — what the
  harness buys there is that the noise is visible as noise. §38.

- **`docs/TECHNICAL.md` and `examples/`.** The measurement tables move out of
  `README.md` into TECHNICAL.md, and `examples/` carries three compilable
  programs against the public header — `api_plan.c` (budget arithmetic
  without loading), `api_text.c`, `api_vision.c` — with a README that walks
  through them.

### Changed

- **§4's cache floor still reproduces exactly, and has stopped binding.**
  The oldest load-bearing measurement here — below one token's working set
  the hit rate is zero, not low — was re-measured across four cache sizes in
  one process, two repeats, hit rate and bytes read identical to the digit
  across both:

  | budget | expert cache | slots | hit | decode |
  |---|---|---|---|---|
  | 32 GB | 3.32 GB | 287 | 29.1% | 0.56–0.58 tok/s |
  | 46 GB | 17.32 GB | 1498 | 36.2% | **0.63 tok/s** |
  | 52 GB | 23.32 GB | 2018 | 38.4% | 0.07–0.09 tok/s |
  | 58 GB | 29.32 GB | 2537 | 41.3% | 0.07–0.08 tok/s |

  The 29.1% at 287 slots is not a refutation of §4: with the lookahead off
  the same 287 slots give **0.0%**, exactly §4's zero. What breaks it is that
  a speculative record has to survive one attention rather than one token, so
  a cache far too small for a token's working set is ample to hold six
  experts. A 3.32 GB cache is now within 10% of a 17.32 GB one, which means
  the premise the default resolver is built on — that cache is only worth
  buying in whole multiples of a working set — no longer holds. **The
  resolver is unchanged**: that is a decision, not a measurement, and it is
  [GATES.md](docs/GATES.md) Gate 7, open. The cliff is exactly where it was,
  and the last two rows say what it is — throughput falls eightfold while the
  hit rate rises and the bytes read fall, because the engine is inside its
  budget and the machine is not. §39.

- **0.6.2's "total bytes read unchanged" for the router lookahead was the
  harness.** Measured with both arms starting from an identically cleared
  cache, the byte economics depend on the cache size: **6.6% fewer** bytes at
  1498 slots, **8% more** at 287, where speculative records are evicted
  before use often enough to be re-read. It is a prefetch at small caches and
  a scheduling change at large ones, and 0.6.2 measured only the large end.
  The feature and its default are unchanged. §38, §39.

## 0.6.2 — 2026-08-01

### Fixed

- **`waste info` and `waste run` crashed on K3 on every x86 build**
  ([#10](https://github.com/sqliteai/waste/issues/10)). The tensors the
  loader skips — the vision tower, and anything outside `tensor_prefix` —
  kept `group` at 0, and the row-scratch sizing divided by it. The
  architecture decided what that meant: arm64's `sdiv` answers 0 and the
  run continues, x86's `idiv` raises `#DE`. `waste plan` was unaffected
  because it does not load. §37.
- **`WASTE_Q8=0` could not load a 4-bit trunk**
  ([#6](https://github.com/sqliteai/waste/issues/6)) — that is, any
  container a default `tools/convert.py` run produces. The dequantizer
  read one byte per weight, true of Q8G alone, while catching every
  quantized format. It now decodes through `waste_deq_row`, the one place
  that knows all three widths. The same lines also predated `waste_f16`'s
  subnormal fix and flushed group scales below 6.1e-05 to zero.
- **`embed_tokens` stays on disk under `WASTE_Q8=0`**, as it does
  otherwise: 7.93 → 6.52 GiB of peak RSS on Kimi-Linear, identical logits.
  The f32-equivalence check now differs from the default path in the
  storage width alone, which is what it claims to compare.

### Added

- **Router lookahead in the decode path.** At the end of a MoE layer, once
  its reads are consumed and the disk is about to idle through the next
  layer's attention, layer L+1's router runs on layer L's hidden state and
  issues speculative reads for its top 6. Demand hit rate 14–19% → 38–40%
  in the initial paired runs, with a median 1.17x. A later one-load control
  corrected the byte conclusion at the default cache: 204–205 → 191 GB
  (6.6% less), while a much smaller cache amplified bytes by about 8%.
  `WASTE_LOOKAHEAD=0` disables. §34–36, §41.
- **`WASTE_MLOCK`** wires the trunk and the expert cache; `WASTE_MLOCK=cache`
  wires the cache alone. Off by default — Linux's `RLIMIT_MEMLOCK` is
  commonly 8 MB. Wiring the trunk is worth 3x in the transition zone around
  52 GiB and nothing below it; it does not move the knee. §30, §31, §32.

### Changed

- **`tests/run.sh` generates its own PyTorch oracle** from the container
  under test (16.9 s) instead of diffing against a shipped fixture, which
  can only ever be valid for the container that produced it: expert
  codebooks are k-means, and the same seed on a different `--device`
  trains different books — one layer of 26 moves the logits by 1.24
  against a 1e-3 threshold. The fixture remains as the fallback where `uv`
  is absent, with its provenance recorded beside it.
  ([#7](https://github.com/sqliteai/waste/issues/7)), §33.
- **`tools/make_test_container.py` emits what a real conversion does**: a
  `Q4G/Q8G/F32` trunk rather than Q8G throughout, and `--prefix` for a
  container whose tensors are not all under its `tensor_prefix`. Both are
  shapes the suite could not previously reach, and both had a live bug
  behind them.
- Checks that cannot run now say why instead of reporting a refusal as a
  divergence: `WASTE_Q8=0` on K3 wants 211 GB of f32 trunk on a 64 GB
  machine, the oracle prompt is Kimi-Linear's, and a cold hotlist run that
  already missed nothing demonstrates neither outcome
  ([#5](https://github.com/sqliteai/waste/pull/5)).

### Measured and not adopted

- **Cross-layer prefetch from `next_layer_top`** — gated before building
  and refused: 29.0% recall against a 60% break-even. §29, revisited and
  superseded by the lookahead above in §34.
- **The lookahead in the prefill path.** Built, measured, removed: a chunk
  layer's disk is busy continuously, so a prefetch there does not move a
  read into idle time, it moves it in front of another read and pays an
  eviction for it — 7% more bytes. Decode keeps it. §36.

## 0.6.1 and earlier

Not covered here; see `docs/LEARNED.md`, which is dated and append-only,
and the git history.
