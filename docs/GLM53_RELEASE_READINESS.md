# GLM-5.3 release readiness

This is a small, fail-closed intake plan for the anticipated public
`zai-org/GLM-5.3-FP8` checkpoint. It does not claim model support. As of
2026-08-23, Z.ai says GLM-5.3 uses the GLM-5.2 base with post-training gains,
but the public weights have not been released.

## What is ready

- Spark has about 1.3 TiB free after the GLM-5.2 source shards were removed.
  The source-backed GLM-5.2 VQ3R container occupies about 264 GiB; retained
  GLM-4.7 conversions occupy about 139 GB.
- A bounded GLM-5.2 execution path is now qualified through 2,048 context
  tokens, including exact release-shape gates and coherent-memory CUDA
  dense/VQ decode. It is a concrete base for comparison, not evidence that a
  future GLM-5.3 checkpoint is compatible without re-running every gate.
- Normal Qwen vLLM and Comfy workloads remain available. They should be
  stopped only for the eventual conversion/qualification window.
- `tools/check_glm53_release.py` can inspect the release without fetching a
  model shard. It pins the candidate commit, checks repository completeness,
  and compares its execution geometry and FP8 recipe against immutable
  GLM-5.2-FP8 revision `ba978f7d347eaf65d22f1a86833408afdb953541`.

## Release-day sequence

1. Run the metadata gate from a clean checkout:

   ```sh
   python3 tools/check_glm53_release.py --json
   ```

   A missing/private repository or any architecture, DSA, MoE, MTP, tokenizer
   vocabulary, or FP8-recipe change returns `BLOCKED`. Do not start a large
   download in that state. Review the difference first.

2. Record the returned 40-character candidate revision. Pin every subsequent
   source operation to it; never convert a moving `main` revision.

3. Confirm the storage transaction before download. The working estimate is
   roughly 704 GiB of FP8 source plus 264 GiB of VQ3R output. Keep at least
   128 GiB additional headroom, so the go/no-go floor is 1.10 TiB free. These
   are estimates until the release index and a conversion plan are measured.

4. Download metadata and safetensor headers first. Validate every tensor name,
   shape, dtype, shard assignment, and LFS size against the pinned index. The
   metadata gate does not replace this source-boundary gate.

5. Implement and test the actual `GlmMoeDsaForCausalLM` contract before model
   conversion:

   - MLA projections and latent KV cache;
   - the DSA indexer, its full/shared layer pattern, and exact top-2048 choice;
   - 256 routed experts, top-8 sigmoid/noaux routing, correction bias, and the
     shared expert;
   - three dense layers followed by 75 sparse layers;
   - explicit omission of the appended MTP layer unless a separate verified
     MTP contract is implemented;
   - the released tokenizer/chat template and all three EOS identifiers.

   At context lengths of 2048 or less, DSA's top-2048 candidate set covers the
   complete available causal history. A short-context dense-MLA mode may be an
   exact initial implementation, but it must hard-fail above its proven bound.

6. Stop competing inference services only after the source/header gates pass.
   Convert to a new container; do not overwrite the GLM-4.7, K2, or K3 assets.

7. Before deleting source shards, perform source-backed sampled verification,
   minimal reset/replay and semantic checks, then a short matched CPU/CUDA
   qualification. Restore Qwen and Comfy regardless of outcome.

## Promotion boundary

The first useful milestone is not unrestricted model support. It is a pinned,
source-backed GLM-5.3 container that is correct at an explicitly bounded
context length and fails closed everywhere else. Reasoning effort (`low`,
`high`, or `max`) belongs in the harness profile and should be measured for
output length and utility separately from engine tok/s.

No upstream post or PR should be made from preparation alone. Publish only
after the released checkpoint passes the source, correctness, and matched
performance gates.
