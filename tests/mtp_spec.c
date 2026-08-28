/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/*
 * mtp_spec.c -- greedy depth-1 MTP scheduler and serial-verifier baseline.
 *
 * This is a measurement tool, not the optimized verify2 path.  It uses the
 * rejection-safe serial oracle to make the target model authoritative:
 *
 *   mtp_spec CONTAINER ids[,..] n_gen
 *
 * The prompt logits supply the first (bootstrap) output token.  Each cycle
 * proposes one draft, verifies current+draft with the target, then either
 * commits the draft and emits the target bonus token or rolls the draft back
 * and emits the target correction.  `n_gen` is an exact output-token budget.
 * The reported emitted_tok_s excludes the already-available bootstrap token.
 *
 * Set WASTE_MTP_SPEC_CHECK=1 to reset the model after measurement, generate
 * the same number of tokens through the ordinary greedy target path, and
 * require byte-for-byte token-stream equality.  All ordinary model-loader,
 * CPU-placement, cache and accelerator environment variables still apply.
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

typedef struct {
    int cycle, pos0, current, draft, target1, accepted;
    int bonus, bonus_emitted, emitted;
    double draft_seconds, verify_begin_seconds, finish_seconds;
} spec_record;

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
    if (n == 0 || n != (int)cap) goto fail;
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

/* Hash token ids as four little-endian bytes so evidence made on ARM and x86
 * has the same compact stream identity.  This is an identity checksum, not a
 * security primitive. */
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

/* Append exactly what greedy target generation makes public.  Returning the
 * verdict lets the caller make the matching oracle commit/rollback choice.
 * This helper is deliberately independent of model state so both budget
 * boundaries are covered by the deterministic --self-test. */
static int append_verdict(int draft, int target1, int bonus,
                          int *stream, int cap, int *emitted,
                          int *current, int *bonus_emitted)
{
    if (!stream || !emitted || !current || !bonus_emitted ||
        *emitted < 0 || *emitted >= cap)
        return -1;
    const int accepted = draft == target1;
    stream[(*emitted)++] = target1;
    *bonus_emitted = 0;
    if (accepted && *emitted < cap) {
        stream[(*emitted)++] = bonus;
        *bonus_emitted = 1;
    }
    *current = accepted ? bonus : target1;
    return accepted;
}

static int scheduler_self_test(void)
{
    int stream[5] = {10, -1, -1, -1, -1};
    int emitted = 1, current = 10, bonus_emitted = -1;

    if (append_verdict(20, 21, 22, stream, 5, &emitted, &current,
                       &bonus_emitted) != 0 ||
        emitted != 2 || current != 21 || bonus_emitted != 0 ||
        stream[1] != 21)
        return 1;
    if (append_verdict(30, 30, 31, stream, 5, &emitted, &current,
                       &bonus_emitted) != 1 ||
        emitted != 4 || current != 31 || bonus_emitted != 1 ||
        stream[2] != 30 || stream[3] != 31)
        return 1;
    if (append_verdict(40, 40, 41, stream, 5, &emitted, &current,
                       &bonus_emitted) != 1 ||
        emitted != 5 || current != 41 || bonus_emitted != 0 ||
        stream[4] != 40 ||
        append_verdict(50, 50, 51, stream, 5, &emitted, &current,
                       &bonus_emitted) != -1)
        return 1;
    puts("MTP SPEC SCHEDULER OK");
    return 0;
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
                                 int n_prompt, const int *stream, int n_gen,
                                 uint64_t *hash_out, double *decode_seconds)
{
    waste_model_reset(m);
    if (waste_model_mtp_set_enabled(m, 0)) return -1;
    const float *logits = prefill(m, prompt, n_prompt);
    if (!logits) return -1;
    int current = argmax(logits, m->cfg.vocab);
    if (current != stream[0]) return 1;
    uint64_t hash = hash_token(UINT64_C(14695981039346656037), current);
    const double started = now();
    for (int i = 1; i < n_gen; i++) {
        logits = waste_model_step(m, current, n_prompt + i - 1, NULL);
        if (!logits) return -1;
        current = argmax(logits, m->cfg.vocab);
        if (current != stream[i]) return 1;
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
    spec_record *records = NULL;
    int n_prompt = 0, n_gen = 0, loaded = 0;
    int emitted = 0, current = -1, pos0 = -1, n_records = 0;
    uint64_t accepted = 0, rejected = 0, bonuses = 0;
    double draft_seconds = 0.0, verify_begin_seconds = 0.0;
    double finish_seconds = 0.0;

    if (argc == 2 && !strcmp(argv[1], "--self-test"))
        return scheduler_self_test();
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
    if ((size_t)n_gen > SIZE_MAX / sizeof *stream ||
        (size_t)n_gen > SIZE_MAX / sizeof *records) {
        fprintf(stderr, "generation budget is too large\n");
        free(ids);
        return 2;
    }
    stream = (int *)malloc((size_t)n_gen * sizeof *stream);
    records = (spec_record *)calloc((size_t)n_gen, sizeof *records);
    if (!stream || !records) {
        fprintf(stderr, "generation record allocation failed\n");
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
    int stream_check = 0;
    const char *check = getenv("WASTE_MTP_SPEC_CHECK");
    if (check && (parse_nonnegative(check, &stream_check) ||
                  (stream_check != 0 && stream_check != 1))) {
        fprintf(stderr, "WASTE_MTP_SPEC_CHECK must be 0 or 1\n");
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
    if (waste_model_ctx_max(&model) > 0 &&
        n_prompt > waste_model_ctx_max(&model) - n_gen) {
        fprintf(stderr, "prompt plus generation exceeds model context\n");
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
    current = argmax(logits, model.cfg.vocab);
    stream[0] = current;
    emitted = 1;
    pos0 = n_prompt;

    int hidden_pos = -1;
    if (!waste_model_mtp_target_hidden(&model, &hidden_pos) ||
        hidden_pos != pos0 - 1 ||
        waste_model_mtp_cache_pos(&model) != pos0 - 1) {
        fprintf(stderr,
                "post-prefill MTP alignment failed: hidden=%d cache=%d "
                "expected=%d\n", hidden_pos,
                waste_model_mtp_cache_pos(&model), pos0 - 1);
        goto fail;
    }

    const double decode_start = now();
    while (emitted < n_gen) {
        spec_record *rec = &records[n_records];
        rec->cycle = n_records;
        rec->pos0 = pos0;
        rec->current = current;

        const double draft_start = now();
        const float *draft_logits = waste_model_mtp_propose(
            &model, current, pos0 - 1, NULL);
        rec->draft_seconds = now() - draft_start;
        draft_seconds += rec->draft_seconds;
        if (!draft_logits) {
            report_failure(&model, "MTP proposal");
            goto fail;
        }
        rec->draft = argmax(draft_logits, model.cfg.vocab);

        waste_mtp_verify2_oracle *oracle = NULL;
        const double verify_start = now();
        const int begin_rc = waste_model_mtp_verify2_oracle_begin(
            &model, current, rec->draft, pos0, NULL, NULL, &oracle);
        rec->verify_begin_seconds = now() - verify_start;
        verify_begin_seconds += rec->verify_begin_seconds;
        if (begin_rc || !oracle) {
            report_failure(&model, "serial verify2 begin");
            goto fail;
        }
        const float *logits0 =
            waste_model_mtp_verify2_oracle_logits(oracle, 0);
        const float *logits1 =
            waste_model_mtp_verify2_oracle_logits(oracle, 1);
        if (!logits0 || !logits1) {
            waste_model_mtp_verify2_oracle_finish(oracle, 0);
            fprintf(stderr, "serial verify2 did not retain both logits rows\n");
            goto fail;
        }
        rec->target1 = argmax(logits0, model.cfg.vocab);
        rec->bonus = argmax(logits1, model.cfg.vocab);
        rec->accepted = rec->draft == rec->target1;

        const double finish_start = now();
        const int finish_rc = waste_model_mtp_verify2_oracle_finish(
            oracle, rec->accepted);
        rec->finish_seconds = now() - finish_start;
        finish_seconds += rec->finish_seconds;
        if (finish_rc) {
            report_failure(&model, "serial verify2 finish");
            goto fail;
        }

        const int verdict = append_verdict(
            rec->draft, rec->target1, rec->bonus, stream, n_gen, &emitted,
            &current, &rec->bonus_emitted);
        if (verdict != rec->accepted) {
            fprintf(stderr, "internal speculative scheduler failure\n");
            goto fail;
        }
        rec->emitted = emitted;
        if (rec->accepted) {
            accepted++;
            bonuses += (uint64_t)rec->bonus_emitted;
            pos0 += 2;
        } else {
            rejected++;
            pos0++;
        }
        n_records++;

        if (!waste_model_mtp_target_hidden(&model, &hidden_pos) ||
            hidden_pos != pos0 - 1 ||
            waste_model_mtp_cache_pos(&model) != pos0 - 1) {
            fprintf(stderr,
                    "post-verdict alignment failed at cycle %d: hidden=%d "
                    "cache=%d expected=%d\n", n_records - 1, hidden_pos,
                    waste_model_mtp_cache_pos(&model), pos0 - 1);
            goto fail;
        }
    }
    waste_ecache_drain(&model.cache);
    const double decode_seconds = now() - decode_start;
    const int final_cache_pos = waste_model_mtp_cache_pos(&model);
    const int final_hidden_pos = hidden_pos;

    const uint64_t speculative_hash = hash_stream(stream, n_gen);
    uint64_t target_hash = 0;
    double target_check_seconds = 0.0;
    int check_rc = 0;
    if (stream_check) {
        check_rc = ordinary_stream_check(&model, ids, n_prompt, stream, n_gen,
                                         &target_hash,
                                         &target_check_seconds);
        if (check_rc < 0) {
            report_failure(&model, "ordinary greedy stream check");
            goto fail;
        }
        if (check_rc > 0) {
            fprintf(stderr,
                    "speculative token stream differs from ordinary greedy "
                    "generation\n");
            goto fail;
        }
    }

    printf("mtp_spec model=%s verifier=serial_oracle prompt=%d generate=%d "
           "load_s=%.6f prefill_s=%.6f prefill_tok_s=%.6f\n",
           argv[1], n_prompt, n_gen, load_seconds, prefill_seconds,
           prefill_seconds > 0.0 ? n_prompt / prefill_seconds : 0.0);
    printf("# cycle pos0 current draft target1 accepted bonus "
           "bonus_emitted emitted draft_s verify_begin_s finish_s\n");
    for (int i = 0; i < n_records; i++) {
        const spec_record *rec = &records[i];
        printf("%d %d %d %d %d %d %d %d %d %.6f %.6f %.6f\n",
               rec->cycle, rec->pos0, rec->current, rec->draft,
               rec->target1, rec->accepted, rec->bonus,
               rec->bonus_emitted, rec->emitted, rec->draft_seconds,
               rec->verify_begin_seconds, rec->finish_seconds);
    }
    printf("tokens=");
    for (int i = 0; i < n_gen; i++)
        printf("%s%d", i ? "," : "", stream[i]);
    putchar('\n');
    printf("stream_fnv1a64=%016" PRIx64 "\n", speculative_hash);
    if (stream_check)
        printf("stream_check=pass target_stream_fnv1a64=%016" PRIx64
               " target_replay_decode_s=%.6f "
               "target_replay_tok_s=%.6f\n",
               target_hash, target_check_seconds,
               target_check_seconds > 0.0
                   ? (n_gen - 1) / target_check_seconds : 0.0);

    const int measured = n_gen - 1;
    const double verify_seconds = verify_begin_seconds + finish_seconds;
    printf("summary generated=%d bootstrap=1 measured_emitted=%d cycles=%d "
           "accepted=%" PRIu64 " rejected=%" PRIu64
           " acceptance_pct=%.3f accepted_bonuses=%" PRIu64
           " emitted_tok_s=%.6f decode_wall_s=%.6f "
           "draft_s=%.6f draft_s_per_cycle=%.6f "
           "verify_s=%.6f verify_s_per_cycle=%.6f "
           "verify_begin_s=%.6f finish_s=%.6f "
           "target_positions=%d cache_pos=%d hidden_pos=%d\n",
           n_gen, measured, n_records, accepted, rejected,
           n_records ? 100.0 * accepted / n_records : 0.0, bonuses,
           decode_seconds > 0.0 ? measured / decode_seconds : 0.0,
           decode_seconds, draft_seconds,
           n_records ? draft_seconds / n_records : 0.0,
           verify_seconds, n_records ? verify_seconds / n_records : 0.0,
           verify_begin_seconds, finish_seconds, 2 * n_records,
           final_cache_pos, final_hidden_pos);

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
