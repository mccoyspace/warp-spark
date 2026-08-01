/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/* test_state.c — a saved session must resume exactly where it left off.
 *
 *   ./test_state MODEL
 * Runs a prompt, saves, continues; then reloads the save and continues
 * again. The two continuations must be identical.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/waste.h"

typedef struct { int32_t ids[64]; int n; } sink;

static int on_tok(const waste_token_info *i, const char *piece, void *user)
{
    (void)piece;
    sink *s = (sink *)user;
    if (s->n < 64) s->ids[s->n++] = i->token;
    return 0;
}

/* A semantic family checkpoint is earlier than the exact n-1 checkpoint:
 * restore it, evaluate a genuinely multi-token suffix in one call, and
 * require the result to be indistinguishable from a cold full-prompt eval. */
static int test_family_root_replay(waste_ctx *c, const int32_t *prompt,
                                   size_t n)
{
    const size_t root_n = n / 2;
    const size_t suffix_n = n - root_n;
    const float *logits = NULL;
    size_t cold_vocab = 0, replay_vocab = 0;
    size_t cold_n = 0, root_state_n = 0, replay_n = 0, written = 0;
    unsigned char *cold_state = NULL;
    unsigned char *root_state = NULL;
    unsigned char *replay_state = NULL;
    float *cold_logits = NULL;
    int rc = 1;

    if (n < 4 || root_n == 0 || suffix_n < 2) {
        fprintf(stderr, "family replay prompt needs a multi-token suffix\n");
        return 1;
    }

    waste_state_reset(c);
    if (waste_eval(c, prompt, n, &logits, &cold_vocab) != WASTE_OK ||
        !logits || !cold_vocab) {
        fprintf(stderr, "cold family eval\n");
        goto done;
    }
    cold_logits = (float *)malloc(cold_vocab * sizeof(*cold_logits));
    if (!cold_logits) goto done;
    memcpy(cold_logits, logits, cold_vocab * sizeof(*cold_logits));
    if (waste_state_size(c, &cold_n) != WASTE_OK || !cold_n) goto done;
    cold_state = (unsigned char *)malloc(cold_n);
    if (!cold_state) goto done;
    if (waste_state_export(c, cold_state, cold_n, &written) != WASTE_OK ||
        written != cold_n) {
        fprintf(stderr, "cold family state export\n");
        goto done;
    }

    waste_state_reset(c);
    if (waste_eval(c, prompt, root_n, NULL, NULL) != WASTE_OK ||
        waste_state_size(c, &root_state_n) != WASTE_OK || !root_state_n) {
        fprintf(stderr, "family root eval\n");
        goto done;
    }
    root_state = (unsigned char *)malloc(root_state_n);
    if (!root_state) goto done;
    if (waste_state_export(c, root_state, root_state_n, &written) != WASTE_OK ||
        written != root_state_n) {
        fprintf(stderr, "family root export\n");
        goto done;
    }

    waste_state_reset(c);
    if (waste_state_import(c, root_state, root_state_n) != WASTE_OK) {
        fprintf(stderr, "family root import\n");
        goto done;
    }
    if (waste_eval(c, prompt + root_n, suffix_n,
                   &logits, &replay_vocab) != WASTE_OK || !logits) {
        fprintf(stderr, "family suffix eval\n");
        goto done;
    }
    if (replay_vocab != cold_vocab ||
        memcmp(cold_logits, logits, cold_vocab * sizeof(*cold_logits)) != 0) {
        fprintf(stderr, "family suffix logits changed after restore\n");
        goto done;
    }
    if (waste_state_size(c, &replay_n) != WASTE_OK || !replay_n) goto done;
    replay_state = (unsigned char *)malloc(replay_n);
    if (!replay_state) goto done;
    if (waste_state_export(c, replay_state, replay_n, &written) != WASTE_OK ||
        written != replay_n || replay_n != cold_n ||
        memcmp(cold_state, replay_state, cold_n) != 0) {
        fprintf(stderr, "family suffix state changed after restore\n");
        goto done;
    }

    rc = 0;
done:
    free(cold_logits);
    free(replay_state);
    free(root_state);
    free(cold_state);
    return rc;
}

int main(int argc, char **argv)
{
    if (argc < 2) { fprintf(stderr, "usage: %s MODEL\n", argv[0]); return 2; }
    waste_cfg cfg;
    waste_cfg_init(&cfg);
    /* 0 = let the engine size itself: the container's recommendation,
     * capped at what this machine can hold. This used to be a hardcoded
     * 6 GB, which is a hard ceiling the engine then *fills* with expert
     * cache — 6 GB of it, to test session round-trip on a 1 MB synthetic
     * container. Invisible on a 64 GB laptop and an OOM kill in a CI
     * container, which is a stupid way to lose a check that has nothing
     * to do with memory. The budget still has to clear the floor of
     * whatever container is passed, and letting the engine choose is the
     * only value that does that for both the synthetic one and a real
     * model. */
    cfg.ram_budget_bytes = 0;
    waste_ctx *c;
    if (waste_open(argv[1], &cfg, &c) != WASTE_OK) { fprintf(stderr, "open\n"); return 1; }

    /* A container built by tools/make_test_container.py carries no
     * tokenizer — the round-trip this checks does not need one, so fall
     * back to fixed ids small enough to be valid in any vocabulary. */
    int32_t prompt[64];
    size_t n = 0;
    if (waste_tokenize(c, "The capital of France is Paris, and the capital of Italy is",
                       0, prompt, 64, &n) != WASTE_OK) {
        static const int32_t fixed[] = {3, 7, 11, 5, 9, 13, 2, 17, 4, 8};
        n = sizeof fixed / sizeof *fixed;
        memcpy(prompt, fixed, sizeof fixed);
    }

    waste_gen_params p;
    waste_gen_params_init(&p);
    p.max_tokens = 6;

    if (n < 4) { fprintf(stderr, "prompt too short\n"); return 1; }
    const char *path = "/tmp/waste_state_test.bin";

    if (test_family_root_replay(c, prompt, n)) return 1;
    waste_state_reset(c);

    /* Exact-prefix contract: checkpoint after all but the final prompt
     * token, restore, replay that token, and require every float in the
     * resulting logits to have the same bits. */
    const float *lg;
    size_t vocab = 0;
    if (waste_eval(c, prompt, n - 1, NULL, NULL) != WASTE_OK) return 1;
    size_t snap_n = 0, written = 0;
    if (waste_state_size(c, &snap_n) != WASTE_OK || !snap_n) return 1;
    unsigned char *snap = (unsigned char *)malloc(snap_n);
    unsigned char *again = (unsigned char *)malloc(snap_n);
    if (!snap || !again) return 1;
    memset(again, 0xa5, snap_n);
    if (waste_state_export(c, again, snap_n - 1, &written) != WASTE_E_ARG ||
        written != snap_n || again[0] != 0xa5) {
        fprintf(stderr, "short export was not side-effect free\n"); return 1;
    }
    if (waste_state_export(c, snap, snap_n, &written) != WASTE_OK ||
        written != snap_n) return 1;
    if (waste_eval(c, prompt + n - 1, 1, &lg, &vocab) != WASTE_OK) return 1;
    float *want_logits = (float *)malloc(vocab * sizeof(float));
    if (!want_logits) return 1;
    memcpy(want_logits, lg, vocab * sizeof(float));

    waste_state_reset(c);
    if (waste_state_import(c, snap, snap_n) != WASTE_OK) {
        fprintf(stderr, "memory import\n"); return 1;
    }
    if (waste_state_export(c, again, snap_n, &written) != WASTE_OK ||
        written != snap_n || memcmp(snap, again, snap_n) != 0) {
        fprintf(stderr, "state bytes changed on round trip\n"); return 1;
    }
    if (waste_eval(c, prompt + n - 1, 1, &lg, &vocab) != WASTE_OK ||
        memcmp(want_logits, lg, vocab * sizeof(float)) != 0) {
        fprintf(stderr, "logits changed after memory restore\n"); return 1;
    }
    free(want_logits);
    free(again);
    free(snap);

    /* A rejected import is transactional. Snapshot the now-full prompt,
     * offer a truncated copy, and require live bytes to remain identical. */
    if (waste_state_size(c, &snap_n) != WASTE_OK) return 1;
    snap = (unsigned char *)malloc(snap_n);
    again = (unsigned char *)malloc(snap_n);
    if (!snap || !again) return 1;
    if (waste_state_export(c, snap, snap_n, &written) != WASTE_OK) return 1;
    if (waste_state_import(c, snap, snap_n - 1) != WASTE_E_FORMAT) {
        fprintf(stderr, "truncated memory state accepted\n"); return 1;
    }
    if (waste_state_export(c, again, snap_n, &written) != WASTE_OK ||
        memcmp(snap, again, snap_n) != 0) {
        fprintf(stderr, "bad import modified live state\n"); return 1;
    }
    if (waste_state_save(c, path) != WASTE_OK) return 1;
    FILE *checkpoint = fopen(path, "rb");
    if (!checkpoint || fread(again, 1, snap_n, checkpoint) != snap_n ||
        fgetc(checkpoint) != EOF || memcmp(snap, again, snap_n) != 0) {
        fprintf(stderr, "file and memory state formats diverged\n"); return 1;
    }
    fclose(checkpoint);
    remove(path);
    free(again);
    free(snap);

    /* Run the full prompt, save, then continue through the original file
     * API too: disk and memory snapshots share the same representation. */
    sink a = {{0}, 0};
    if (waste_state_save(c, path) != WASTE_OK) { fprintf(stderr, "save\n"); return 1; }
    int32_t nxt[1] = { 0 };
    /* vocab comes from the model, not from a constant: a synthetic
     * container has 256 entries and the hardcoded 163840 read right off
     * the end of the logits buffer. */
    for (size_t v = 1; v < vocab; v++) if (lg[v] > lg[nxt[0]]) nxt[0] = (int32_t)v;
    waste_generate(c, nxt, 1, &p, on_tok, &a);

    /* fresh context, load the save, continue from the same token */
    waste_close(c);
    if (waste_open(argv[1], &cfg, &c) != WASTE_OK) return 1;
    const waste_status st = waste_state_load(c, path);
    if (st != WASTE_OK) { fprintf(stderr, "load: %s\n", waste_strerror(st)); return 1; }
    sink b = {{0}, 0};
    waste_generate(c, nxt, 1, &p, on_tok, &b);
    waste_close(c);

    int same = (a.n == b.n);
    for (int i = 0; i < a.n && same; i++) same = (a.ids[i] == b.ids[i]);
    printf("fresh   :");
    for (int i = 0; i < a.n; i++) printf(" %d", a.ids[i]);
    printf("\nrestored:");
    for (int i = 0; i < b.n; i++) printf(" %d", b.ids[i]);
    printf("\n%s\n", same ? "STATE OK — memory logits and file continuation are bit-exact"
                          : "STATE MISMATCH");
    remove(path);
    return same ? 0 : 1;
}
