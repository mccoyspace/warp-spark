# Sprint 16 pre-inference amendment

This amendment was frozen before the first new target or draft forward pass.
It supplements, and does not rewrite, the preregistration tagged
`gn100-sprint16-preregistered-2026-08-03`. The original tag remains the audit
record of the first design. H2 remains embargoed and unspent.

## K3 MTP audit

K3 cannot supply a built-in draft head. The converted manifest records
`num_nextn_predict_layers: 0`, and the released checkpoint has one language
head only: `language_model.lm_head.weight`.

A credentialed read-only audit of `moonshotai/Kimi-K3` at Hugging Face commit
`9f62e4e9fffbd0a83ddd60e1c209d828994b3569` reconciled every indexed tensor:

- 497,220 total tensor names;
- 494,592 routed-expert packed/scale tensors
  (`92 * 896 * 3 * 2`);
- 2,628 non-expert tensors, exactly matching the converted trunk; and
- zero names matching MTP, next-token, draft, EAGLE, or fusion heads.

The source `config.json` and tensor index SHA-256 values are recorded in the
machine-readable amendment. No weight shard was downloaded and no model was
run for this audit. MTP is therefore excluded rather than treated as a tested
negative candidate.

## Added zero-residency baseline

One prompt-lookup draft class is added before any trace is observed. At each
canonical K3 root it:

1. takes the suffix of the already-known K3 context;
2. searches suffix lengths four through one for an earlier occurrence that
   has at least one already-known following token;
3. chooses the longest match, then the most recent occurrence; and
4. proposes at most eight tokens that followed that occurrence.

If no such occurrence exists it makes no proposal and K3 emits one direct
token. There is no alternate n-gram limit, tie rule, fallback token, or prompt
format. The registered widths remain 2, 4, and 8 and are all derived from one
maximum-width trace.

This adds three cells to the six Kimi-Linear cells. A+B may select among the
nine cells once. H1 remains veto-only and cannot select a runner-up. The
existing 15% modeled and 10% integrated gates, case regression limits, and
single-use H2 rules do not change. For prompt lookup, the resident comparator
is identical to `F` because the draft has no model or cache allocation. A tie
within one percentage point chooses smaller `k`, then prompt lookup, then
`target_ids`, then `kimi_native`.

## Measurement clarifications

Transactional state is no longer priced from a checkpoint-path anecdote. The
instrument reports four distinct quantities:

- durable file save/load, when measured, is checkpoint I/O and is not charged
  to a speculative block;
- canonical in-memory state export/import is the current exact reference;
- contiguous one-thread and ten-thread shadow copies are explicitly labeled
  optimistic lower bounds for a future double-buffer design; and
- pointer-swap commit is not claimed until the engine actually owns two live
  state banks and passes rollback exactness tests.

The cost model uses the measured in-memory mechanism actually implemented by
the candidate. A lower-bound copy result cannot be substituted for export or
restore cost without implementing and validating the corresponding shadow
state design.

The load-only seal measures the inference-free position-zero root, which is
the fixed KDA/conv floor and contains no MLA latent rows. After the A replay
gate passes, one separate target-state row replays the longest canonical
development prompt through its last 32-token block root. It reports the
actual MLA-inclusive byte size, warm export/import, and one-/ten-thread
contiguous-copy floor. The case is chosen by frozen prompt-token count, not by
observed timing or agreement, and the extra pass cannot select a candidate.
The probe also advances one canonical token, restores the root, and replays
that token; logits, ordered routes, and the complete post-step state must be
byte-identical before the root is restored a final time.

Every agreement result reports first-mismatch/rejection position and survival
by case, family, tier, format, and width. Candidate selection uses family-level
curves as well as aggregate arithmetic.

Target verification is decomposed into:

- exact serial `waste_model_step` verification;
- the current `T=k` chunk path with I8MM disabled, as a non-exact performance
  bound; and
- the same chunk path with I8MM enabled, as a diagnostic bound only.

Timing, CUDA launches, expert-union reads, misses, and bytes are reported
separately so batching and expert deduplication are not double-counted. Neither
chunk arm can qualify unless it is later made byte-identical to the serial
reference.

The paired cache-rent measurement in the original preregistration remains
mandatory: `F` is K3 alone at 59,340 MiB, while `R` is ordinary K3 decode with
the fully resident draft and rollback allocation but speculation disabled.
`S/R` measures income and `R/F` measures rent.

The joint values are frozen at 40,502 MiB for K3, 16,926 MiB for Kimi-Linear,
536,870,912 target rollback bytes, and 134,217,728 draft rollback bytes. The
agreement capture is descriptive and cannot select or model a candidate until
both the paired `F/R` rent row and the registered serial-versus-`T=k` verifier
cost decomposition exist.

Process swap is checked over each complete model subprocess. Major faults are
zero-gated only from the start to the end of the named timed inference or
state-copy window; load/library startup faults remain recorded but are not
misreported as timed faults. Capture additionally requires the exact ten-CPU
affinity and a root-held Q0 child-scope parent at zero microseconds.
