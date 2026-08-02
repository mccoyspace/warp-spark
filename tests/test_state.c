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

    /* run the prompt, save, then continue.
     *
     * The path was "/tmp/waste_state_test.bin". A native Windows binary
     * reads that as C:\tmp\..., which usually does not exist, so the save
     * failed and this check reported the session state broken on the one
     * platform where nothing about the state was wrong. MSYS2 rewrites
     * POSIX paths in argv for native programs; it cannot rewrite a string
     * literal. TEMP first on Windows because MSYS2 sets TMPDIR to /tmp
     * there and that is the same trap again. */
    char pathbuf[512];
#ifdef _WIN32
    const char *td = getenv("TEMP");
    if (!td) td = getenv("TMP");
#else
    const char *td = getenv("TMPDIR");
#endif
    if (!td || !*td) td = ".";
    snprintf(pathbuf, sizeof pathbuf, "%s/waste_state_test.bin", td);
    const char *path = pathbuf;
    const float *lg;
    size_t vocab = 0;
    if (n < 2) { fprintf(stderr, "prompt too short\n"); return 1; }

    /* Exact-prefix contract: checkpoint all but the final prompt token,
     * restore, replay that token, and require both the logits and the full
     * resulting recurrent/KV state to have identical bits. Identical post-
     * step state also covers the deterministic routes used to produce it. */
    if (waste_eval(c, prompt, n - 1, NULL, NULL) != WASTE_OK) return 1;
    size_t snap_n = 0, written = 0;
    if (waste_state_size(c, &snap_n) != WASTE_OK || !snap_n) return 1;
    unsigned char sentinel = 0xa5;
    if (waste_state_export(c, &sentinel, 1, &written) != WASTE_E_ARG ||
        written != snap_n || sentinel != 0xa5) {
        fprintf(stderr, "short export was not side-effect free\n"); return 1;
    }
    unsigned char *snap = (unsigned char *)malloc(snap_n);
    if (!snap || waste_state_export(c, snap, snap_n, &written) != WASTE_OK ||
        written != snap_n) return 1;

    if (waste_eval(c, prompt + n - 1, 1, &lg, &vocab) != WASTE_OK) return 1;
    float *want_logits = (float *)malloc(vocab * sizeof(float));
    if (!want_logits) return 1;
    memcpy(want_logits, lg, vocab * sizeof(float));
    size_t post_n = 0;
    if (waste_state_size(c, &post_n) != WASTE_OK || !post_n) return 1;
    unsigned char *post = (unsigned char *)malloc(post_n);
    if (!post || waste_state_export(c, post, post_n, &written) != WASTE_OK ||
        written != post_n) return 1;

    waste_state_reset(c);
    if (waste_state_import(c, snap, snap_n) != WASTE_OK) {
        fprintf(stderr, "memory import\n"); return 1;
    }
    if (waste_eval(c, prompt + n - 1, 1, &lg, &vocab) != WASTE_OK ||
        memcmp(want_logits, lg, vocab * sizeof(float)) != 0) {
        fprintf(stderr, "logits changed after memory restore\n"); return 1;
    }
    free(want_logits);
    unsigned char *grown = (unsigned char *)realloc(snap, post_n);
    if (!grown) return 1;
    snap = grown;
    if (waste_state_size(c, &snap_n) != WASTE_OK || snap_n != post_n ||
        waste_state_export(c, snap, post_n, &written) != WASTE_OK ||
        written != post_n || memcmp(snap, post, post_n) != 0) {
        fprintf(stderr, "state/routes changed after memory restore\n"); return 1;
    }

    /* Rejected imports are transactional, for both truncation and a corrupt
     * header. The file writer must emit the same canonical bytes too. */
    if (waste_state_import(c, post, post_n - 1) != WASTE_E_FORMAT ||
        waste_state_export(c, snap, post_n, &written) != WASTE_OK ||
        memcmp(snap, post, post_n) != 0) {
        fprintf(stderr, "truncated import modified live state\n"); return 1;
    }
    post[0] ^= 1;
    const waste_status corrupt = waste_state_import(c, post, post_n);
    post[0] ^= 1;
    if (corrupt != WASTE_E_FORMAT ||
        waste_state_export(c, snap, post_n, &written) != WASTE_OK ||
        memcmp(snap, post, post_n) != 0) {
        fprintf(stderr, "corrupt import modified live state\n"); return 1;
    }
    if (waste_state_save(c, path) != WASTE_OK) { fprintf(stderr, "save\n"); return 1; }
    FILE *checkpoint = fopen(path, "rb");
    if (!checkpoint || fread(snap, 1, post_n, checkpoint) != post_n ||
        fgetc(checkpoint) != EOF || memcmp(snap, post, post_n) != 0) {
        fprintf(stderr, "file and memory state formats diverged\n"); return 1;
    }
    fclose(checkpoint);
    free(snap);
    free(post);

    /* Continue through the original file API too. */
    sink a = {{0}, 0};
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
    printf("\n%s\n", same ? "STATE OK — memory logits/state and file continuation are bit-exact"
                          : "STATE MISMATCH");
    remove(path);
    return same ? 0 : 1;
}
