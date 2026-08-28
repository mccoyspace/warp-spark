/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/*
 * mtp_shadow.c — measurement-only GLM NextN/MTP qualification harness.
 *
 * This does not speculate in the public generation path.  It keeps the
 * target model's greedy stream authoritative and measures recursive chains
 * of one to three MTP drafts.  A chain stops being comparable at its first
 * mismatch; an all-accepted chain consumes one target bonus row so its cycle
 * boundary is the one a real speculative scheduler would use.
 *
 *   mtp_shadow CONTAINER ids[,..] n_target_steps [depth]
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

enum { MAX_SHADOW_DEPTH = 3 };

typedef struct {
    int start_step, input_id, depth;
    int draft[MAX_SHADOW_DEPTH], target[MAX_SHADOW_DEPTH];
    int compared, prefix, bonus, produced;
    double draft_model_seconds, draft_wall_seconds, target_seconds;
} shadow_cycle;

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

static uint64_t hash_token(uint64_t hash, int token)
{
    const uint32_t value = (uint32_t)token;
    for (unsigned shift = 0; shift < 32; shift += 8) {
        hash ^= (value >> shift) & 0xffu;
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static uint64_t hash_stream(const int *stream, int n)
{
    uint64_t hash = UINT64_C(14695981039346656037);
    for (int i = 0; i < n; i++) hash = hash_token(hash, stream[i]);
    return hash;
}

static const float *prefill(waste_model *m, const int *ids, int n)
{
    const float *logits = NULL;
    for (int done = 0; done < n; ) {
        int chunk = n - done;
        const int chunk_max = waste_model_chunk_max(m);
        if (chunk_max <= 0) return NULL;
        if (chunk > chunk_max) chunk = chunk_max;
        logits = waste_model_prefill(m, ids + done, chunk, done);
        if (!logits) return NULL;
        done += chunk;
    }
    return logits;
}

static int ordinary_stream_check(waste_model *m, const int *prompt,
                                 int n_prompt, const int *stream,
                                 int n_target_steps, uint64_t *hash_out,
                                 double *decode_seconds)
{
    waste_model_reset(m);
    waste_ecache_clear(&m->cache);
    if (waste_model_mtp_set_enabled(m, 0)) return -1;
    const float *logits = prefill(m, prompt, n_prompt);
    if (!logits) return -1;
    int current = argmax(logits, m->cfg.vocab);
    if (current != stream[0]) return 1;
    uint64_t hash = hash_token(UINT64_C(14695981039346656037), current);
    const double started = now();
    for (int i = 0; i < n_target_steps; i++) {
        logits = waste_model_step(m, current, n_prompt + i, NULL);
        if (!logits) return -1;
        current = argmax(logits, m->cfg.vocab);
        if (current != stream[i + 1]) return 1;
        hash = hash_token(hash, current);
    }
    waste_ecache_drain(&m->cache);
    *decode_seconds = now() - started;
    *hash_out = hash;
    return 0;
}

int main(int argc, char **argv)
{
    waste_load_opts opts;
    waste_model model;
    int *ids = NULL, *stream = NULL;
    shadow_cycle *records = NULL;
    int n_prompt = 0, n_gen = 0, depth = 1, loaded = 0;
    int n_records = 0, done = 0;

    if (argc != 4 && argc != 5) {
        fprintf(stderr,
                "usage: %s CONTAINER ids[,..] n_target_steps [depth]\n",
                argv[0]);
        return 2;
    }
    if (parse_ids(argv[2], &ids, &n_prompt) ||
        parse_nonnegative(argv[3], &n_gen) || n_gen == 0 ||
        (argc == 5 && parse_nonnegative(argv[4], &depth)) ||
        depth < 1 || depth > MAX_SHADOW_DEPTH ||
        n_prompt > INT_MAX - n_gen - 1) {
        fprintf(stderr, "invalid token list, target-step count or depth\n");
        free(ids);
        return 2;
    }
    if ((size_t)n_gen + 1 > SIZE_MAX / sizeof *stream ||
        (size_t)n_gen > SIZE_MAX / sizeof *records) {
        fprintf(stderr, "target-step count is too large\n");
        free(ids);
        return 2;
    }
    stream = (int *)malloc(((size_t)n_gen + 1) * sizeof *stream);
    records = (shadow_cycle *)calloc((size_t)n_gen, sizeof *records);
    if (!stream || !records) {
        fprintf(stderr, "shadow record allocation failed\n");
        goto fail;
    }

    memset(&opts, 0, sizeof opts);
    const char *cache_mb = getenv("WASTE_CACHE_MB");
    int cache = 0;
    if (cache_mb && (parse_nonnegative(cache_mb, &cache) ||
                     (size_t)cache > SIZE_MAX >> 20)) {
        fprintf(stderr, "invalid WASTE_CACHE_MB\n");
        goto usage_fail;
    }
    opts.cache_bytes = (size_t)cache << 20;
    opts.direct_io = 1;

    const double load_start = now();
    if (waste_model_load(&model, argv[1], n_prompt + n_gen + 1, &opts)) {
        fprintf(stderr, "model load failed: %s\n", argv[1]);
        goto fail;
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

    const double prefill_start = now();
    const float *logits = prefill(&model, ids, n_prompt);
    const double prefill_seconds = now() - prefill_start;
    if (!logits) {
        report_failure(&model, "prefill");
        goto fail;
    }
    int current = argmax(logits, model.cfg.vocab);
    stream[0] = current;
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
    double draft_model_seconds = 0.0;
    uint64_t reached[MAX_SHADOW_DEPTH] = {0};
    uint64_t matched[MAX_SHADOW_DEPTH] = {0};
    uint64_t complete_prefix_ge[MAX_SHADOW_DEPTH] = {0};
    uint64_t prefix_hist[MAX_SHADOW_DEPTH + 1] = {0};
    uint64_t complete_prefix_cycles = 0, censored_cycles = 0;
    uint64_t econ_cycles = 0, econ_outputs = 0;
    double econ_draft_wall = 0.0, econ_draft_model = 0.0;

    printf("mtp_shadow model=%s prompt=%d target_steps=%d depth=%d "
           "load_s=%.6f "
           "prefill_s=%.6f prefill_tok_s=%.6f\n",
           argv[1], n_prompt, n_gen, depth, load_seconds, prefill_seconds,
           prefill_seconds > 0.0 ? n_prompt / prefill_seconds : 0.0);
    puts("# cycle start_step input depth d1 d2 d3 t1 t2 t3 "
         "compared prefix bonus produced draft_model_s draft_wall_s target_s");
    const double loop_start = now();
    while (done < n_gen) {
        shadow_cycle *rec = &records[n_records];
        for (int row = 0; row < MAX_SHADOW_DEPTH; row++) {
            rec->draft[row] = -1;
            rec->target[row] = -1;
        }
        rec->bonus = -1;
        rec->start_step = done;
        rec->input_id = current;
        rec->depth = n_gen - done < depth ? n_gen - done : depth;

        const int target_pos = n_prompt + done - 1;
        int saved_pos = -1;
        if (!waste_model_mtp_target_hidden(&model, &saved_pos) ||
            saved_pos != target_pos ||
            waste_model_mtp_cache_pos(&model) != target_pos) {
            fprintf(stderr,
                    "pre-chain MTP alignment failed at target step %d: "
                    "hidden=%d cache=%d expected=%d\n", done, saved_pos,
                    waste_model_mtp_cache_pos(&model), target_pos);
            goto fail;
        }

        const double draft_start = now();
        if (waste_model_mtp_propose_greedy_chain(
                &model, current, target_pos, rec->depth, rec->draft,
                &rec->draft_model_seconds)) {
            report_failure(&model, "MTP recursive chain");
            goto fail;
        }
        rec->draft_wall_seconds = now() - draft_start;
        draft_wall_seconds += rec->draft_wall_seconds;
        draft_model_seconds += rec->draft_model_seconds;
        if (waste_model_mtp_cache_pos(&model) != target_pos + 1 ||
            model.mtp_last_pos != target_pos ||
            model.mtp_last_token != rec->input_id ||
            model.mtp_shadow_argmax != rec->draft[0]) {
            fprintf(stderr,
                    "recursive chain did not unwind to its first row at "
                    "target step %d\n", done);
            goto fail;
        }

        int all_accepted = 1;
        for (int row = 0; row < rec->depth; row++) {
            const double target_start = now();
            logits = waste_model_step(&model, current,
                                      n_prompt + done, NULL);
            const double elapsed = now() - target_start;
            rec->target_seconds += elapsed;
            target_seconds += elapsed;
            if (!logits) {
                report_failure(&model, "target comparison step");
                goto fail;
            }
            const int target_id = argmax(logits, model.cfg.vocab);
            rec->target[row] = target_id;
            rec->compared++;
            reached[row]++;
            stream[++done] = target_id;
            current = target_id;
            if (target_id == rec->draft[row]) {
                rec->prefix++;
                matched[row]++;
            } else {
                all_accepted = 0;
            }

            if (!waste_model_mtp_target_hidden(&model, &saved_pos) ||
                saved_pos != n_prompt + done - 1 ||
                waste_model_mtp_cache_pos(&model) != saved_pos) {
                fprintf(stderr,
                        "post-target MTP alignment failed at target step %d: "
                        "hidden=%d cache=%d expected=%d\n", done - 1,
                        saved_pos, waste_model_mtp_cache_pos(&model),
                        n_prompt + done - 1);
                goto fail;
            }
            if (!all_accepted) break;
        }

        /* When the complete registered chain matches, consume its last
         * candidate once more to obtain the target-authoritative bonus. */
        if (all_accepted && rec->depth == depth && done < n_gen) {
            const double target_start = now();
            logits = waste_model_step(&model, current,
                                      n_prompt + done, NULL);
            const double elapsed = now() - target_start;
            rec->target_seconds += elapsed;
            target_seconds += elapsed;
            if (!logits) {
                report_failure(&model, "target bonus step");
                goto fail;
            }
            rec->bonus = argmax(logits, model.cfg.vocab);
            stream[++done] = rec->bonus;
            current = rec->bonus;
            if (!waste_model_mtp_target_hidden(&model, &saved_pos) ||
                saved_pos != n_prompt + done - 1 ||
                waste_model_mtp_cache_pos(&model) != saved_pos) {
                fprintf(stderr,
                        "post-bonus MTP alignment failed at target step %d\n",
                        done - 1);
                goto fail;
            }
        }

        rec->produced = done - rec->start_step;
        /* A shortened final chain is right-censored only when every available
         * row matches: it did not reject at that prefix, it simply ran out of
         * target budget.  A shortened chain that did reject still has a known
         * exact prefix and belongs in the complete-cycle distribution. */
        if (rec->depth == depth || rec->prefix < rec->depth) {
            complete_prefix_cycles++;
            prefix_hist[rec->prefix]++;
            for (int row = 0; row < rec->prefix; row++)
                complete_prefix_ge[row]++;
        } else {
            censored_cycles++;
        }
        const int complete = rec->depth == depth &&
            (rec->prefix < depth || rec->bonus >= 0);
        if (complete) {
            econ_cycles++;
            econ_outputs += (uint64_t)(rec->prefix < depth
                ? rec->prefix + 1 : depth + 1);
            econ_draft_wall += rec->draft_wall_seconds;
            econ_draft_model += rec->draft_model_seconds;
        }
        n_records++;
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
    const double logical_mtp_seconds =
        waste_model_mtp_seconds(&model) - mtp_seconds0;
    if (step_delta != (uint64_t)n_gen || shadow_delta != (uint64_t)n_gen ||
        match_delta > shadow_delta) {
        fprintf(stderr,
                "shadow accounting mismatch: steps=%" PRIu64
                " shadows=%" PRIu64 " matches=%" PRIu64 " expected=%d\n",
                step_delta, shadow_delta, match_delta, n_gen);
        goto fail;
    }

    const int final_cache_pos = waste_model_mtp_cache_pos(&model);
    const int final_hidden_pos = hidden_pos;
    const uint64_t final_steps = waste_model_mtp_steps(&model);
    const uint64_t final_shadow_steps = waste_model_mtp_shadow_steps(&model);
    const uint64_t final_shadow_matches =
        waste_model_mtp_shadow_matches(&model);
    const uint64_t speculative_hash = hash_stream(stream, n_gen + 1);

    uint64_t ordinary_hash = 0;
    double ordinary_decode_seconds = 0.0;
    const int check_rc = ordinary_stream_check(
        &model, ids, n_prompt, stream, n_gen, &ordinary_hash,
        &ordinary_decode_seconds);
    if (check_rc < 0) {
        report_failure(&model, "ordinary target replay");
        goto fail;
    }
    if (check_rc > 0 || ordinary_hash != speculative_hash) {
        fprintf(stderr,
                "recursive shadow target stream differs from ordinary greedy "
                "generation\n");
        goto fail;
    }

    for (int i = 0; i < n_records; i++) {
        const shadow_cycle *rec = &records[i];
        printf("%d %d %d %d %d %d %d %d %d %d %d %d %d %d "
               "%.6f %.6f %.6f\n",
               i, rec->start_step, rec->input_id, rec->depth,
               rec->draft[0], rec->draft[1], rec->draft[2],
               rec->target[0], rec->target[1], rec->target[2],
               rec->compared, rec->prefix, rec->bonus, rec->produced,
               rec->draft_model_seconds, rec->draft_wall_seconds,
               rec->target_seconds);
    }
    printf("tokens=");
    for (int i = 0; i <= n_gen; i++)
        printf("%s%d", i ? "," : "", stream[i]);
    putchar('\n');
    printf("stream_check=pass stream_fnv1a64=%016" PRIx64
           " ordinary_fnv1a64=%016" PRIx64
           " ordinary_decode_s=%.6f ordinary_tok_s=%.6f\n",
           speculative_hash, ordinary_hash, ordinary_decode_seconds,
           ordinary_decode_seconds > 0.0
               ? n_gen / ordinary_decode_seconds : 0.0);

    for (int row = 0; row < depth; row++) {
        printf("position=%d reached=%" PRIu64 " matched=%" PRIu64
               " conditional_pct=%.3f complete_cycle_cumulative_pct=%.3f\n",
               row + 1, reached[row], matched[row],
               reached[row] ? 100.0 * matched[row] / reached[row] : 0.0,
               complete_prefix_cycles
                   ? 100.0 * complete_prefix_ge[row] /
                         complete_prefix_cycles
                   : 0.0);
    }
    printf("prefix_hist complete_cycles=%" PRIu64
           " censored_cycles=%" PRIu64,
           complete_prefix_cycles, censored_cycles);
    for (int i = 0; i <= depth; i++)
        printf(" accepted_%d=%" PRIu64, i, prefix_hist[i]);
    putchar('\n');

    const double base_s_per_token = n_gen
        ? ordinary_decode_seconds / n_gen : 0.0;
    const double expected_outputs = econ_cycles
        ? (double)econ_outputs / econ_cycles : 0.0;
    const double draft_wall_per_cycle = econ_cycles
        ? econ_draft_wall / econ_cycles : 0.0;
    const double draft_model_per_cycle = econ_cycles
        ? econ_draft_model / econ_cycles : 0.0;
    const double verify_break_even =
        expected_outputs * base_s_per_token - draft_wall_per_cycle;
    const double verify_plus5 =
        expected_outputs * base_s_per_token / 1.05 - draft_wall_per_cycle;
    printf("economics full_cycles=%" PRIu64
           " observed_outputs_per_cycle=%.6f base_s_per_token=%.6f "
           "draft_wall_s_per_cycle=%.6f draft_model_s_per_cycle=%.6f "
           "verify_s_max_break_even=%.6f verify_s_max_plus5=%.6f\n",
           econ_cycles, expected_outputs, base_s_per_token,
           draft_wall_per_cycle, draft_model_per_cycle,
           verify_break_even, verify_plus5);
    printf("summary depth=%d target_steps=%d cycles=%d "
           "target_tok_s=%.6f target_s=%.6f "
           "chain_model_s=%.6f chain_model_s_per_cycle=%.6f "
           "chain_wall_s=%.6f chain_wall_s_per_cycle=%.6f "
           "shadow_tok_s=%.6f shadow_wall_s=%.6f "
           "one_step_matches=%" PRIu64 "/%" PRIu64
           " one_step_acceptance_pct=%.3f logical_mtp_s=%.6f "
           "mtp_steps=%" PRIu64 " shadow_steps=%" PRIu64
           " shadow_matches=%" PRIu64 " cache_pos=%d hidden_pos=%d\n",
           depth, n_gen, n_records,
           target_seconds > 0.0 ? n_gen / target_seconds : 0.0,
           target_seconds, draft_model_seconds,
           n_records ? draft_model_seconds / n_records : 0.0,
           draft_wall_seconds,
           n_records ? draft_wall_seconds / n_records : 0.0,
           loop_seconds > 0.0 ? n_gen / loop_seconds : 0.0,
           loop_seconds, match_delta, shadow_delta,
           shadow_delta ? 100.0 * match_delta / shadow_delta : 0.0,
           logical_mtp_seconds, final_steps, final_shadow_steps,
           final_shadow_matches, final_cache_pos, final_hidden_pos);

    waste_model_free(&model);
    free(records);
    free(stream);
    free(ids);
    return 0;

usage_fail:
    free(records);
    free(stream);
    free(ids);
    return 2;
fail:
    if (loaded) waste_model_free(&model);
    free(records);
    free(stream);
    free(ids);
    return 1;
}
