/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/*
 * mtp_shadow.c — measurement-only GLM NextN/MTP qualification harness.
 *
 * This does not speculate in the public generation path.  It keeps the
 * target model's greedy stream authoritative, explicitly runs one MTP draft
 * before each target step, and reports whether the draft predicted the same
 * next token.  That is enough to measure the two quantities a verify2
 * scheduler would trade: proposal cost and one-token acceptance.
 *
 *   mtp_shadow CONTAINER ids[,..] n_gen
 *
 * WASTE_CACHE_MB, WASTE_THREADS, WASTE_CPUS and the accelerator environment
 * variables have their ordinary model-loader meanings.  Token IDs are used
 * directly so a qualification run is exactly reproducible without making a
 * tokenizer or chat-template choice part of this binary.
 */

#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(_WIN32)
#include <windows.h>
#endif

#include "../src/model.h"

static double now(void)
{
#if defined(_WIN32)
    LARGE_INTEGER counter, frequency;
    QueryPerformanceCounter(&counter);
    QueryPerformanceFrequency(&frequency);
    return (double)counter.QuadPart / (double)frequency.QuadPart;
#else
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec / 1e9;
#endif
}

static int parse_nonnegative(const char *s, int *out)
{
    char *end = NULL;
    long v;

    if (!s || !*s) return -1;
    errno = 0;
    v = strtol(s, &end, 10);
    if (errno || !end || *end || v < 0 || v > INT_MAX) return -1;
    *out = (int)v;
    return 0;
}

static int parse_ids(const char *s, int **ids_out, int *n_out)
{
    size_t cap = 1;
    const char *p;
    char *copy = NULL, *field;
    int *ids = NULL, n = 0;

    if (!s || !*s) return -1;
    for (p = s; *p; p++) if (*p == ',') cap++;
    if (cap > (size_t)INT_MAX / sizeof *ids) return -1;
    copy = strdup(s);
    ids = (int *)malloc(cap * sizeof *ids);
    if (!copy || !ids) goto fail;

    for (field = strtok(copy, ","); field; field = strtok(NULL, ",")) {
        if (parse_nonnegative(field, &ids[n])) goto fail;
        n++;
    }
    if (n == 0 || n != (int)cap) goto fail; /* also rejects empty fields */
    free(copy);
    *ids_out = ids;
    *n_out = n;
    return 0;

fail:
    free(copy);
    free(ids);
    return -1;
}

static int argmax(const float *logits, int n)
{
    int best = 0;
    for (int i = 1; i < n; i++)
        if (logits[i] > logits[best]) best = i;
    return best;
}

static int report_failure(waste_model *m, const char *what)
{
    int layer = 0, expert = 0;
    const char *why = waste_model_read_error(m, &layer, &expert);
    if (why)
        fprintf(stderr, "%s failed at expert %d of layer %d: %s\n",
                what, expert, layer, why);
    else
        fprintf(stderr, "%s failed (MTP alignment/cache contract)\n", what);
    return 1;
}

int main(int argc, char **argv)
{
    waste_load_opts opts;
    waste_model model;
    int *ids = NULL;
    int n_prompt = 0, n_gen = 0, loaded = 0;

    if (argc != 4) {
        fprintf(stderr, "usage: %s CONTAINER ids[,..] n_gen\n", argv[0]);
        return 2;
    }
    if (parse_ids(argv[2], &ids, &n_prompt) ||
        parse_nonnegative(argv[3], &n_gen) || n_gen == 0 ||
        n_prompt > INT_MAX - n_gen - 1) {
        fprintf(stderr, "invalid token list or n_gen\n");
        free(ids);
        return 2;
    }

    memset(&opts, 0, sizeof opts);
    const char *cache_mb = getenv("WASTE_CACHE_MB");
    int cache = 0;
    if (cache_mb && (parse_nonnegative(cache_mb, &cache) ||
                     (size_t)cache > SIZE_MAX >> 20)) {
        fprintf(stderr, "invalid WASTE_CACHE_MB\n");
        free(ids);
        return 2;
    }
    opts.cache_bytes = (size_t)cache << 20;
    opts.direct_io = 1;

    const double load_start = now();
    if (waste_model_load(&model, argv[1], n_prompt + n_gen + 1, &opts)) {
        fprintf(stderr, "model load failed: %s\n", argv[1]);
        free(ids);
        return 1;
    }
    loaded = 1;
    const double load_seconds = now() - load_start;
    if (!waste_model_mtp_available(&model)) {
        fprintf(stderr, "container has no qualified MTP layer\n");
        goto fail;
    }
    for (int i = 0; i < n_prompt; i++) {
        if (ids[i] < 0 || ids[i] >= model.cfg.vocab) {
            fprintf(stderr, "prompt token %d is outside vocab %d\n",
                    ids[i], model.cfg.vocab);
            goto fail;
        }
    }
    if (waste_model_mtp_set_enabled(&model, 1)) {
        fprintf(stderr, "could not enable fresh MTP state\n");
        goto fail;
    }

    const float *logits = NULL;
    const double prefill_start = now();
    for (int done = 0; done < n_prompt; ) {
        int chunk = n_prompt - done;
        const int chunk_max = waste_model_chunk_max(&model);
        if (chunk > chunk_max) chunk = chunk_max;
        logits = waste_model_prefill(&model, ids + done, chunk, done);
        if (!logits) {
            report_failure(&model, "prefill");
            goto fail;
        }
        done += chunk;
    }
    const double prefill_seconds = now() - prefill_start;
    int current = argmax(logits, model.cfg.vocab);
    int hidden_pos = -1;
    if (!waste_model_mtp_target_hidden(&model, &hidden_pos) ||
        hidden_pos != n_prompt - 1 ||
        waste_model_mtp_cache_pos(&model) != n_prompt - 1) {
        fprintf(stderr,
                "post-prefill MTP alignment failed: hidden=%d cache=%d "
                "expected=%d\n", hidden_pos,
                waste_model_mtp_cache_pos(&model), n_prompt - 1);
        goto fail;
    }

    const uint64_t steps0 = waste_model_mtp_steps(&model);
    const uint64_t shadows0 = waste_model_mtp_shadow_steps(&model);
    const uint64_t matches0 = waste_model_mtp_shadow_matches(&model);
    const double mtp_seconds0 = waste_model_mtp_seconds(&model);
    double target_seconds = 0.0, draft_wall_seconds = 0.0;
    uint64_t explicit_matches = 0;

    printf("mtp_shadow model=%s prompt=%d generate=%d load_s=%.6f "
           "prefill_s=%.6f prefill_tok_s=%.6f\n",
           argv[1], n_prompt, n_gen, load_seconds, prefill_seconds,
           n_prompt / prefill_seconds);
    printf("# step input_id draft_next_id target_next_id accepted "
           "cache_pos target_hidden_pos draft_s target_s\n");
    const double loop_start = now();
    for (int i = 0; i < n_gen; i++) {
        const int target_pos = n_prompt + i - 1;
        int saved_pos = -1;
        if (!waste_model_mtp_target_hidden(&model, &saved_pos) ||
            saved_pos != target_pos ||
            waste_model_mtp_cache_pos(&model) != target_pos) {
            fprintf(stderr,
                    "pre-proposal MTP alignment failed at step %d: "
                    "hidden=%d cache=%d expected=%d\n", i, saved_pos,
                    waste_model_mtp_cache_pos(&model), target_pos);
            goto fail;
        }

        const double draft_start = now();
        const float *draft = waste_model_mtp_propose(
            &model, current, target_pos, NULL);
        const double draft_elapsed = now() - draft_start;
        draft_wall_seconds += draft_elapsed;
        if (!draft) {
            report_failure(&model, "MTP proposal");
            goto fail;
        }
        const int draft_id = argmax(draft, model.cfg.vocab);

        const double target_start = now();
        logits = waste_model_step(&model, current, target_pos + 1, NULL);
        const double target_elapsed = now() - target_start;
        target_seconds += target_elapsed;
        if (!logits) {
            report_failure(&model, "target step");
            goto fail;
        }
        const int target_id = argmax(logits, model.cfg.vocab);
        const int accepted = draft_id == target_id;
        explicit_matches += (uint64_t)accepted;

        if (!waste_model_mtp_target_hidden(&model, &saved_pos) ||
            saved_pos != target_pos + 1 ||
            waste_model_mtp_cache_pos(&model) != target_pos + 1) {
            fprintf(stderr,
                    "post-target MTP alignment failed at step %d: "
                    "hidden=%d cache=%d expected=%d\n", i, saved_pos,
                    waste_model_mtp_cache_pos(&model), target_pos + 1);
            goto fail;
        }
        printf("%d %d %d %d %d %d %d %.6f %.6f\n", i, current,
               draft_id, target_id, accepted,
               waste_model_mtp_cache_pos(&model), saved_pos,
               draft_elapsed, target_elapsed);
        current = target_id;
    }
    waste_ecache_drain(&model.cache);
    const double loop_seconds = now() - loop_start;
    if (!waste_model_mtp_target_hidden(&model, &hidden_pos)) {
        fprintf(stderr, "final target hidden unexpectedly unavailable\n");
        goto fail;
    }

    const uint64_t step_delta = waste_model_mtp_steps(&model) - steps0;
    const uint64_t shadow_delta =
        waste_model_mtp_shadow_steps(&model) - shadows0;
    const uint64_t match_delta =
        waste_model_mtp_shadow_matches(&model) - matches0;
    const double draft_seconds = waste_model_mtp_seconds(&model) - mtp_seconds0;
    if (step_delta != (uint64_t)n_gen || shadow_delta != (uint64_t)n_gen ||
        match_delta != explicit_matches) {
        fprintf(stderr,
                "shadow accounting mismatch: steps=%" PRIu64
                " shadows=%" PRIu64 " matches=%" PRIu64
                " explicit=%" PRIu64 " expected=%d\n",
                step_delta, shadow_delta, match_delta, explicit_matches, n_gen);
        goto fail;
    }

    printf("summary target_tok_s=%.6f target_s=%.6f "
           "draft_s_per_step=%.6f draft_s=%.6f draft_wall_s=%.6f "
           "shadow_tok_s=%.6f shadow_wall_s=%.6f "
           "acceptance=%" PRIu64 "/%d acceptance_pct=%.3f "
           "mtp_steps=%" PRIu64 " shadow_steps=%" PRIu64
           " shadow_matches=%" PRIu64 " cache_pos=%d hidden_pos=%d\n",
           n_gen / target_seconds, target_seconds,
           draft_seconds / n_gen, draft_seconds, draft_wall_seconds,
           n_gen / loop_seconds, loop_seconds,
           explicit_matches, n_gen, 100.0 * explicit_matches / n_gen,
           waste_model_mtp_steps(&model),
           waste_model_mtp_shadow_steps(&model),
           waste_model_mtp_shadow_matches(&model),
           waste_model_mtp_cache_pos(&model), hidden_pos);

    waste_model_free(&model);
    free(ids);
    return 0;

fail:
    if (loaded) waste_model_free(&model);
    free(ids);
    return 1;
}
