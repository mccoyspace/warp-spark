/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/*
 * mtp_spec.c -- greedy depth-1 MTP scheduler.
 *
 * The default fast verifier evaluates the two target rows layer-major while
 * retaining rejection-safe target authority.  Set WASTE_MTP_VERIFY2=serial
 * to use the slower serial oracle as a matched reference, or `gated` to run
 * row zero through the ordinary target first and evaluate row one only after
 * the draft matches and the output budget has room for its bonus:
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

extern double waste_prof[16];
extern uint64_t waste_prof_n[16];

typedef struct {
    int cycle, pos0, current, draft, target1, accepted, committed;
    int bonus, bonus_emitted, emitted;
    double draft_seconds, verify_begin_seconds, finish_seconds;
} spec_record;

typedef enum {
    VERIFY_FAST,
    VERIFY_SERIAL,
    VERIFY_GATED
} verify_kind;

static const char *verify_name(verify_kind verifier)
{
    switch (verifier) {
    case VERIFY_FAST:   return "fast";
    case VERIFY_SERIAL: return "serial_oracle";
    case VERIFY_GATED:  return "gated";
    }
    return "invalid";
}

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

/* Append exactly what greedy target generation makes public.  The caller
 * separately budget-gates the two-row commit before this helper; returning
 * the predictor verdict checks that the public stream used the same choice.
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
        *current = bonus;
    } else {
        /* Keep the resumable invariant: current is the last public token
         * and has not yet been consumed by the committed target state. */
        *current = target1;
    }
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
        emitted != 5 || current != 40 || bonus_emitted != 0 ||
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
    /* The speculative arm began from a freshly opened cache. Replaying on
     * its warmed records would make the correctness control look faster for
     * an order-dependent reason, so rebuild the same post-prefill state. */
    waste_ecache_clear(&m->cache);
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
    uint64_t accepted = 0, committed = 0, rejected = 0, bonuses = 0;
    uint64_t verified_target_positions = 0, gated_row1_calls = 0;
    double draft_seconds = 0.0, verify_begin_seconds = 0.0;
    double finish_seconds = 0.0;
    double spec_prof[16] = {0};
    uint64_t spec_prof_n[16] = {0};

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
    verify_kind verifier = VERIFY_FAST;
    const char *verify_env = getenv("WASTE_MTP_VERIFY2");
    if (verify_env && *verify_env) {
        if (!strcmp(verify_env, "fast"))
            verifier = VERIFY_FAST;
        else if (!strcmp(verify_env, "serial"))
            verifier = VERIFY_SERIAL;
        else if (!strcmp(verify_env, "gated"))
            verifier = VERIFY_GATED;
        else {
            fprintf(stderr,
                    "WASTE_MTP_VERIFY2 must be fast, serial, or gated\n");
            goto usage_fail;
        }
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

    const int profile = getenv("WASTE_PROFILE") != NULL;
    if (profile) {
        memset(waste_prof, 0, sizeof spec_prof);
        memset(waste_prof_n, 0, sizeof spec_prof_n);
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

        waste_mtp_verify2 *verify = NULL;
        waste_mtp_verify2_oracle *oracle = NULL;
        if (verifier == VERIFY_GATED) {
            /* The proposal has already advanced the MTP cache for `current`.
             * The ordinary target consumes and validates that exact row.  A
             * rejection therefore leaves a complete committed row-zero state
             * and needs neither a checkpoint nor a rollback. */
            const double row0_start = now();
            const float *logits0 = waste_model_step(
                &model, current, pos0, NULL);
            rec->verify_begin_seconds = now() - row0_start;
            verify_begin_seconds += rec->verify_begin_seconds;
            if (!logits0) {
                report_failure(&model, "gated target row zero");
                goto fail;
            }
            verified_target_positions++;
            rec->target1 = argmax(logits0, model.cfg.vocab);
            rec->accepted = rec->draft == rec->target1;
            /* Row one is useful only when its accepted draft and bonus can
             * both become public.  At the exact budget edge, row zero is
             * already the resumable committed state. */
            rec->committed = rec->accepted && emitted + 1 < n_gen;
            rec->bonus = -1;
            if (rec->committed) {
                const double row1_start = now();
                const float *logits1 = waste_model_step(
                    &model, rec->draft, pos0 + 1, NULL);
                rec->finish_seconds = now() - row1_start;
                finish_seconds += rec->finish_seconds;
                if (!logits1) {
                    report_failure(&model, "gated target row one");
                    goto fail;
                }
                verified_target_positions++;
                gated_row1_calls++;
                rec->bonus = argmax(logits1, model.cfg.vocab);
            }
        } else {
            const double verify_start = now();
            const int begin_rc = verifier == VERIFY_FAST
                ? waste_model_mtp_verify2_begin(
                      &model, current, rec->draft, pos0, NULL, NULL, &verify)
                : waste_model_mtp_verify2_oracle_begin(
                      &model, current, rec->draft, pos0, NULL, NULL, &oracle);
            rec->verify_begin_seconds = now() - verify_start;
            verify_begin_seconds += rec->verify_begin_seconds;
            if (begin_rc || (verifier == VERIFY_FAST ? !verify : !oracle)) {
                report_failure(&model, verifier == VERIFY_FAST
                    ? "fast verify2 begin" : "serial verify2 begin");
                goto fail;
            }
            const float *logits0 = verifier == VERIFY_FAST
                ? waste_model_mtp_verify2_logits(verify, 0)
                : waste_model_mtp_verify2_oracle_logits(oracle, 0);
            const float *logits1 = verifier == VERIFY_FAST
                ? waste_model_mtp_verify2_logits(verify, 1)
                : waste_model_mtp_verify2_oracle_logits(oracle, 1);
            if (!logits0 || !logits1) {
                if (verifier == VERIFY_FAST)
                    waste_model_mtp_verify2_finish(verify, 0);
                else
                    waste_model_mtp_verify2_oracle_finish(oracle, 0);
                fprintf(stderr,
                        "%s verify2 did not retain both logits rows\n",
                        verifier == VERIFY_FAST ? "fast" : "serial");
                goto fail;
            }
            rec->target1 = argmax(logits0, model.cfg.vocab);
            rec->bonus = argmax(logits1, model.cfg.vocab);
            rec->accepted = rec->draft == rec->target1;
            /* Accepting row one is useful only when its bonus can also become
             * public.  At the exact budget edge, roll back to row zero so a
             * later continuation starts from the last emitted token. */
            rec->committed = rec->accepted && emitted + 1 < n_gen;

            const double finish_start = now();
            const int finish_rc = verifier == VERIFY_FAST
                ? waste_model_mtp_verify2_finish(verify, rec->committed)
                : waste_model_mtp_verify2_oracle_finish(
                      oracle, rec->committed);
            rec->finish_seconds = now() - finish_start;
            finish_seconds += rec->finish_seconds;
            if (finish_rc) {
                report_failure(&model, verifier == VERIFY_FAST
                    ? "fast verify2 finish" : "serial verify2 finish");
                goto fail;
            }
            verified_target_positions += 2;
        }

        const int verdict = append_verdict(
            rec->draft, rec->target1, rec->bonus, stream, n_gen, &emitted,
            &current, &rec->bonus_emitted);
        if (verdict != rec->accepted ||
            rec->bonus_emitted != rec->committed) {
            fprintf(stderr, "internal speculative scheduler failure\n");
            goto fail;
        }
        rec->emitted = emitted;
        if (rec->accepted) accepted++;
        else rejected++;
        if (rec->committed) {
            committed++;
            bonuses += (uint64_t)rec->bonus_emitted;
            pos0 += 2;
        } else {
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
    if (pos0 != n_prompt + n_gen - 1) {
        fprintf(stderr, "speculative position accounting failure\n");
        goto fail;
    }
    waste_ecache_drain(&model.cache);
    const double decode_seconds = now() - decode_start;
    if (profile) {
        memcpy(spec_prof, waste_prof, sizeof spec_prof);
        memcpy(spec_prof_n, waste_prof_n, sizeof spec_prof_n);
    }
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

    printf("mtp_spec model=%s verifier=%s vq2=%d prompt=%d generate=%d "
           "load_s=%.6f prefill_s=%.6f prefill_tok_s=%.6f\n",
           argv[1], verify_name(verifier),
           waste_model_mtp_verify2_get_vq2(&model),
           n_prompt, n_gen, load_seconds, prefill_seconds,
           prefill_seconds > 0.0 ? n_prompt / prefill_seconds : 0.0);
    printf("# cycle pos0 current draft target1 matched committed bonus "
           "bonus_emitted emitted draft_s verify_begin_s finish_s\n");
    for (int i = 0; i < n_records; i++) {
        const spec_record *rec = &records[i];
        printf("%d %d %d %d %d %d %d %d %d %d %.6f %.6f %.6f\n",
               rec->cycle, rec->pos0, rec->current, rec->draft,
               rec->target1, rec->accepted, rec->committed, rec->bonus,
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
    const double scheduler_residual =
        decode_seconds - draft_seconds - verify_seconds;
    printf("summary generated=%d bootstrap=1 measured_emitted=%d cycles=%d "
           "accepted=%" PRIu64 " rejected=%" PRIu64
           " acceptance_pct=%.3f committed=%" PRIu64
           " commit_pct=%.3f accepted_bonuses=%" PRIu64
           " emitted_tok_s=%.6f decode_wall_s=%.6f "
           "draft_call_s=%.6f draft_call_s_per_cycle=%.6f "
           "verify_transaction_s=%.6f verify_transaction_s_per_cycle=%.6f "
           "verify_begin_call_s=%.6f finish_call_s=%.6f "
           "verify_s_per_target_position=%.6f "
           "scheduler_residual_s=%.6f "
           "verified_target_positions=%" PRIu64 " "
           "gated_row1_calls=%" PRIu64 " "
           "gated_row1_skips=%" PRIu64 " cache_pos=%d hidden_pos=%d\n",
           n_gen, measured, n_records, accepted, rejected,
           n_records ? 100.0 * accepted / n_records : 0.0, committed,
           n_records ? 100.0 * committed / n_records : 0.0, bonuses,
           decode_seconds > 0.0 ? measured / decode_seconds : 0.0,
           decode_seconds, draft_seconds,
           n_records ? draft_seconds / n_records : 0.0,
           verify_seconds, n_records ? verify_seconds / n_records : 0.0,
           verify_begin_seconds, finish_seconds,
           verified_target_positions
               ? verify_seconds / (double)verified_target_positions : 0.0,
           scheduler_residual, verified_target_positions, gated_row1_calls,
           verifier == VERIFY_GATED
               ? (uint64_t)n_records - gated_row1_calls : 0,
           final_cache_pos, final_hidden_pos);
    if (profile) {
        puts("profile_schema p0=lut_build p1=kda p2=mla p3=route_moe "
             "p4=expert_read p5=expert_mm p6=head p7=lut_apply "
             "p8=dense_mm p9=kda_recurrent p10=kda_qkv p11=kda_conv "
             "p12=kda_aux p13=kda_gate p14=kda_norm p15=kda_out");
        printf("profile verifier=%s vq2=%d",
               verify_name(verifier),
               waste_model_mtp_verify2_get_vq2(&model));
        for (int pidx = 0; pidx < 16; pidx++)
            printf(" p%d=%.9f n%d=%" PRIu64,
                   pidx, spec_prof[pidx], pidx, spec_prof_n[pidx]);
        putchar('\n');
    }

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
