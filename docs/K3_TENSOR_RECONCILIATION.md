# Kimi K3 tensor reconciliation

This audit answers one narrow question: does the qualified Kimi K3 WASTE
container account for every tensor name published in the source checkpoint?
It is a metadata and manifest reconciliation, independent of inference timing.

## Inputs

| Input | Identity |
| --- | --- |
| Source repository | `moonshotai/Kimi-K3` |
| Source revision | `9f62e4e9fffbd0a83ddd60e1c209d828994b3569` |
| `model.safetensors.index.json` | SHA-256 `a1c5210650ce71d2d3ae9ec5a101ac4afd3cf4b10091be589853437eb967febd` |
| Qualified WASTE manifest | 2,628 trunk entries and 92 expert-layer banks |

The source index was fetched with authenticated Hugging Face access at the
pinned revision. Only checkpoint metadata was needed for this audit; no second
copy of the tensor shards was made.

## Result

The source index contains **497,220 unique tensor names**. They reconcile as:

| Class | Count | Reconciliation |
| --- | ---: | --- |
| Expert physical tensors | 494,592 | 92 layers × 896 experts × 3 matrices × packed/scale |
| Non-expert tensors | 2,628 | Exact set equality with the WASTE manifest's trunk names |
| Total | **497,220** | No missing or unparsed names |

Every expert name matched
`language_model.model.layers.L.block_sparse_moe.experts.E.w{1,2,3}.weight_{packed,scale}`.
The observed ranges were layers 1–92 and experts 0–895. The converter maps
`w1`, `w3`, and `w2` to the gate, up, and down records respectively. Its
MXFP4 reader transparently combines each logical matrix's `_packed` and
`_scale` tensors, yielding 247,296 logical expert matrices and one converted
expert record per expert in each of 92 layer banks.

The trunk path deliberately excludes `.experts.`, `_packed`, and `_scale`
names after expert-bank conversion. Its resulting 2,628-name set equals the
2,628 source non-expert names in both directions:

- source non-expert names missing from the manifest: **0**;
- manifest trunk names missing from the source: **0**;
- unparsed expert names: **0**.

The reproducibility hashes are:

| Canonical sorted set | SHA-256 |
| --- | --- |
| All 497,220 source names | `6de6475ef6fac789044688d2b1f127d50cfc3adb17983756c2393c495a535ec4` |
| 2,628 WASTE trunk manifest names | `f17e5be7f593ea93f5030f366585f6fdb49e6e93a097140a1f0c1b556a53c5d6` |
| 247,296 logical expert identities | `2805876b903b66cabfbb8ea69eb11e7d3bb0f136e3b9c543ac94b900dd099ea8` |

The machine-readable form is
[`gn100/k3-tensor-reconciliation-summary.json`](gn100/k3-tensor-reconciliation-summary.json).

## Scope limit

This result proves complete name coverage and a consistent structural mapping.
It does **not** by itself prove numerical equivalence of dequantization or
inference. Those are separate contracts covered by converter round-trip tests,
layer/reference checks, and the exact token/logit/route/state gates reported in
the measured releases.
