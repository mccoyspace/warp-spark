/* SPDX-License-Identifier: Apache-2.0
 * Sprint 16 measurement harness. This is not a public speculative decoder.
 * It freezes target trajectories, teacher-forces the draft on them, creates
 * native Kimi prompts without loading a model, and performs the load-only
 * co-residency check registered in docs/GPU_SPECULATIVE_GB10.md.
 */
#define _GNU_SOURCE
#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <time.h>

#include "../src/model.h"
#include "../src/tokenizer.h"

static uint64_t proc_value_kib(const char *path, const char *key);
static void print_process_safety(const char *scope, long timed_major_faults);

static double now(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec / 1e9;
}

static uint64_t hash_bytes(uint64_t h, const void *data, size_t n)
{
    const uint8_t *p = (const uint8_t *)data;
    for (size_t i = 0; i < n; i++) {
        h ^= p[i];
        h *= UINT64_C(1099511628211);
    }
    return h;
}

static uint64_t hash_one(const void *data, size_t n)
{
    return hash_bytes(UINT64_C(14695981039346656037), data, n);
}

static int argmax(const float *x, int n)
{
    int best = 0;
    for (int i = 1; i < n; i++) if (x[i] > x[best]) best = i;
    return best;
}

static int finite_row(const float *x, int n)
{
    for (int i = 0; i < n; i++) if (!isfinite(x[i])) return 0;
    return 1;
}

static int *parse_ids(const char *text, int *count)
{
    *count = 0;
    if (!text || !text[0] || !strcmp(text, "-")) return NULL;
    const size_t text_len = strlen(text);
    if (text[0] == ',' || text[text_len - 1] == ',' || strstr(text, ",,"))
        return NULL;
    char *copy = strdup(text);
    if (!copy) return NULL;
    int cap = 64;
    int *ids = (int *)malloc((size_t)cap * sizeof *ids);
    if (!ids) { free(copy); return NULL; }
    for (char *p = strtok(copy, ","); p; p = strtok(NULL, ",")) {
        char *end = NULL;
        errno = 0;
        const long value = strtol(p, &end, 10);
        if (errno || !end || *end || value < 0 || value > INT32_MAX) {
            free(ids); free(copy); return NULL;
        }
        if (*count == cap) {
            cap *= 2;
            int *grown = (int *)realloc(ids, (size_t)cap * sizeof *ids);
            if (!grown) { free(ids); free(copy); return NULL; }
            ids = grown;
        }
        ids[(*count)++] = (int)value;
    }
    free(copy);
    return ids;
}

static size_t mb_bytes(const char *text)
{
    char *end = NULL;
    errno = 0;
    const unsigned long long mb = strtoull(text, &end, 10);
    if (errno || !end || *end || mb > (unsigned long long)(SIZE_MAX >> 20))
        return 0;
    return (size_t)mb << 20;
}

static size_t exact_bytes(const char *text)
{
    char *end = NULL;
    errno = 0;
    const unsigned long long value = strtoull(text, &end, 10);
    if (errno || !end || *end || !value || value > SIZE_MAX) return 0;
    return (size_t)value;
}

static int positive_int(const char *text, int maximum, int *value)
{
    char *end = NULL;
    errno = 0;
    const long parsed = strtol(text, &end, 10);
    if (errno || !end || *end || parsed <= 0 || parsed > maximum) return -1;
    *value = (int)parsed;
    return 0;
}

static int valid_ids(const int *ids, int n, int vocab, const char *which)
{
    for (int i = 0; i < n; i++) {
        if (ids[i] < 0 || ids[i] >= vocab) {
            fprintf(stderr, "%s token %d is outside vocabulary [0,%d)\n",
                    which, ids[i], vocab);
            return 0;
        }
    }
    return 1;
}

static int load_model(waste_model *m, const char *path, const char *cache_mb,
                      const char *usage)
{
    waste_load_opts opts;
    memset(&opts, 0, sizeof opts);
    opts.cache_bytes = mb_bytes(cache_mb);
    opts.direct_io = 1;
    if (!opts.cache_bytes || waste_model_load(m, path, 4096, &opts)) {
        fprintf(stderr, "could not load %s with cache %s MiB\n", path, cache_mb);
        return -1;
    }
    if (!m->direct_io) {
        fprintf(stderr, "direct I/O fell back for %s\n", path);
        waste_model_free(m);
        return -1;
    }
    if (usage && strcmp(usage, "-") && waste_model_warm_cache(m, usage) < 0) {
        fprintf(stderr, "could not warm %s from %s\n", path, usage);
        waste_model_free(m);
        return -1;
    }
    return 0;
}

static int cache_ready(const waste_ecache *cache)
{
    int ready = 0;
    for (int i = 0; i < cache->n_slots; i++)
        if (cache->slot[i].state == EC_READY) ready++;
    return ready;
}

static int routed_records(const waste_model *m)
{
    return (m->cfg.n_layers - m->cfg.first_dense) * m->cfg.n_experts;
}

static const float *prefill_cpu(waste_model *m, const int *tokens, int n)
{
    const int kda = m->cuda_kda_mode;
    const int dense = m->cuda_dense_scope;
    const int vq = m->cuda_vq_mode;
    m->cuda_kda_mode = 0;
    m->cuda_dense_scope = 0;
    m->cuda_vq_mode = 0;
    const float *logits = NULL;
    for (int done = 0; done < n; ) {
        int take = n - done;
        const int max = waste_model_chunk_max(m);
        if (take > max) take = max;
        logits = take > 1
               ? waste_model_prefill(m, tokens + done, take, done)
               : waste_model_step(m, tokens[done], done, NULL);
        if (!logits) break;
        done += take;
    }
    waste_ecache_drain(&m->cache);
    m->cuda_kda_mode = kda;
    m->cuda_dense_scope = dense;
    m->cuda_vq_mode = vq;
    return logits;
}

static int state_hash(const waste_model *m, int pos, uint64_t *hash_out,
                      size_t *bytes_out)
{
    size_t bytes = 0, written = 0;
    if (!hash_out || waste_model_state_size(m, pos, &bytes)) return -1;
    void *blob = malloc(bytes ? bytes : 1);
    if (!blob) return -1;
    if (waste_model_state_export(m, pos, blob, bytes, &written) || written != bytes) {
        free(blob); return -1;
    }
    *hash_out = hash_one(blob, bytes);
    free(blob);
    if (bytes_out) *bytes_out = bytes;
    return 0;
}

static void print_ints(const int *x, int n)
{
    putchar('[');
    for (int i = 0; i < n; i++) printf("%s%d", i ? "," : "", x[i]);
    putchar(']');
}

static void print_i32s(const int32_t *x, int n)
{
    putchar('[');
    for (int i = 0; i < n; i++)
        printf("%s%" PRId32, i ? "," : "", x[i]);
    putchar(']');
}

static void print_u64_hex(const uint64_t *x, int n)
{
    putchar('[');
    for (int i = 0; i < n; i++)
        printf("%s\"0x%016" PRIx64 "\"", i ? "," : "", x[i]);
    putchar(']');
}

static void print_u64s(const uint64_t *x, int n)
{
    putchar('[');
    for (int i = 0; i < n; i++)
        printf("%s%" PRIu64, i ? "," : "", x[i]);
    putchar(']');
}

static void print_doubles(const double *x, int n)
{
    putchar('[');
    for (int i = 0; i < n; i++) printf("%s%.9f", i ? "," : "", x[i]);
    putchar(']');
}

static void print_int_matrix(const int *x, int rows, int cols)
{
    putchar('[');
    for (int r = 0; r < rows; r++) {
        if (r) putchar(',');
        print_ints(x + (size_t)r * cols, cols);
    }
    putchar(']');
}

static void print_double_matrix(const double *x, int rows, int cols)
{
    putchar('[');
    for (int r = 0; r < rows; r++) {
        if (r) putchar(',');
        print_doubles(x + (size_t)r * cols, cols);
    }
    putchar(']');
}

static void print_u64_hex_matrix(const uint64_t *x, int rows, int cols)
{
    putchar('[');
    for (int r = 0; r < rows; r++) {
        if (r) putchar(',');
        print_u64_hex(x + (size_t)r * cols, cols);
    }
    putchar(']');
}

typedef struct {
    waste_model *draft;
    void *target_rollback;
    size_t target_rollback_bytes;
    size_t draft_rollback_bytes;
    size_t target_prompt_state_bytes;
    size_t draft_prompt_state_bytes;
    int draft_prompt_tokens;
    int draft_cache_slots;
    int draft_routed_records;
    int draft_warm_ready;
    double target_load_seconds;
    double draft_load_seconds;
    double touch_seconds;
    double draft_prefill_seconds;
    double draft_snapshot_seconds;
    double target_snapshot_seconds;
    uint64_t memavailable_before_kib;
    uint64_t memavailable_after_target_load_kib;
    uint64_t memavailable_after_draft_load_kib;
    uint64_t memavailable_after_touch_kib;
    uint64_t memavailable_after_prompt_snapshots_kib;
    uint64_t rss_after_prompt_snapshots_kib;
    struct rusage timed0;
} resident_control;

static int target_capture(waste_model *m, const int *prompt, int n_prompt,
                          int n_gen, resident_control *resident)
{
    const int warm_ready = cache_ready(&m->cache);
    if (!valid_ids(prompt, n_prompt, m->cfg.vocab, "target prompt")) return 1;
    const int target_ctx = waste_model_ctx_max(m);
    if (target_ctx && (n_prompt > target_ctx || n_gen > target_ctx - n_prompt)) {
        fprintf(stderr, "target prompt plus generation exceeds context\n");
        return 1;
    }
    const uint64_t ph0 = m->cache.hits;
    const uint64_t pm0 = m->cache.misses;
    const uint64_t pb0 = m->cache.bytes_read;
    struct rusage timed0, timed1;
    memset(&timed0, 0, sizeof timed0); memset(&timed1, 0, sizeof timed1);
    if (resident) timed0 = resident->timed0;
    else getrusage(RUSAGE_SELF, &timed0);
    const double p0 = now();
    const float *logits = prefill_cpu(m, prompt, n_prompt);
    const double prefill_seconds = now() - p0;
    if (!logits) { fprintf(stderr, "target prompt failed\n"); return 1; }
    if (!finite_row(logits, m->cfg.vocab)) {
        fprintf(stderr, "target prompt produced non-finite logits\n");
        return 1;
    }
    if (resident) {
        size_t written = 0;
        if (waste_model_state_size(m, n_prompt,
                                   &resident->target_prompt_state_bytes) ||
            !resident->target_prompt_state_bytes ||
            resident->target_prompt_state_bytes > resident->target_rollback_bytes) {
            fprintf(stderr, "target prompt state exceeds resident rollback\n");
            return 1;
        }
        const double snapshot_started = now();
        if (waste_model_state_export(m, n_prompt, resident->target_rollback,
                                     resident->target_rollback_bytes, &written) ||
            written != resident->target_prompt_state_bytes) {
            fprintf(stderr, "target prompt snapshot failed\n");
            return 1;
        }
        resident->target_snapshot_seconds = now() - snapshot_started;
        resident->memavailable_after_prompt_snapshots_kib =
            proc_value_kib("/proc/meminfo", "MemAvailable");
        resident->rss_after_prompt_snapshots_kib =
            proc_value_kib("/proc/self/status", "VmRSS");
    }

    int *tokens = (int *)calloc((size_t)n_gen, sizeof *tokens);
    uint64_t *logit_rows = (uint64_t *)calloc((size_t)n_gen + 1, sizeof *logit_rows);
    uint64_t *route_rows = (uint64_t *)calloc((size_t)n_gen, sizeof *route_rows);
    uint64_t *token_prefix = (uint64_t *)calloc((size_t)n_gen, sizeof *token_prefix);
    uint64_t *logit_prefix = (uint64_t *)calloc((size_t)n_gen + 1, sizeof *logit_prefix);
    uint64_t *route_prefix = (uint64_t *)calloc((size_t)n_gen, sizeof *route_prefix);
    int *routed = (int *)malloc((size_t)WASTE_MAX_LAYERS * 64 * sizeof *routed);
    if (!tokens || !logit_rows || !route_rows || !token_prefix ||
        !logit_prefix || !route_prefix || !routed) {
        free(tokens); free(logit_rows); free(route_rows); free(token_prefix);
        free(logit_prefix); free(route_prefix); free(routed); return 1;
    }
    uint64_t token_hash = UINT64_C(14695981039346656037);
    uint64_t logit_hash = UINT64_C(14695981039346656037);
    uint64_t route_hash = UINT64_C(14695981039346656037);
    logit_rows[0] = hash_one(logits, (size_t)m->cfg.vocab * sizeof *logits);
    logit_hash = hash_bytes(logit_hash, logits,
                            (size_t)m->cfg.vocab * sizeof *logits);
    logit_prefix[0] = logit_hash;
    const int route_layers = m->cfg.n_layers - m->cfg.first_dense;
    const size_t route_count = (size_t)route_layers * m->cfg.top_k;
    const uint64_t h0 = m->cache.hits, mi0 = m->cache.misses;
    const uint64_t b0 = m->cache.bytes_read;
    const double started = now();
    for (int i = 0; i < n_gen; i++) {
        tokens[i] = argmax(logits, m->cfg.vocab);
        token_hash = hash_bytes(token_hash, &tokens[i], sizeof tokens[i]);
        token_prefix[i] = token_hash;
        memset(routed, 0xff, (size_t)WASTE_MAX_LAYERS * 64 * sizeof *routed);
        logits = waste_model_step(m, tokens[i], n_prompt + i, routed);
        if (!logits) { fprintf(stderr, "target step %d failed\n", i); goto fail; }
        if (!finite_row(logits, m->cfg.vocab)) {
            fprintf(stderr, "target step %d produced non-finite logits\n", i);
            goto fail;
        }
        waste_ecache_decode_tick(&m->cache);
        logit_rows[i + 1] = hash_one(logits,
                                     (size_t)m->cfg.vocab * sizeof *logits);
        logit_hash = hash_bytes(logit_hash, logits,
                                (size_t)m->cfg.vocab * sizeof *logits);
        logit_prefix[i + 1] = logit_hash;
        const int *rr = routed + (size_t)m->cfg.first_dense * m->cfg.top_k;
        route_rows[i] = hash_one(rr, route_count * sizeof *rr);
        route_hash = hash_bytes(route_hash, rr, route_count * sizeof *rr);
        route_prefix[i] = route_hash;
    }
    waste_ecache_drain(&m->cache);
    const double seconds = now() - started;
    getrusage(RUSAGE_SELF, &timed1);
    const long timed_major_faults = timed1.ru_majflt - timed0.ru_majflt;
    size_t state_bytes = 0;
    uint64_t sh = 0;
    if (state_hash(m, n_prompt + n_gen, &sh, &state_bytes)) {
        fprintf(stderr, "target final state export failed\n");
        goto fail;
    }
    const uint64_t target_fallbacks = waste_model_cuda_kda_fallbacks(m);
    const uint64_t draft_fallbacks = resident
        ? waste_model_cuda_kda_fallbacks(resident->draft) : 0;
    const uint64_t vmswap_kib = proc_value_kib("/proc/self/status", "VmSwap");
    if (resident && (timed_major_faults != 0 || vmswap_kib != 0 ||
                     target_fallbacks != 0 || draft_fallbacks != 0 ||
                     resident->memavailable_after_prompt_snapshots_kib <
                         UINT64_C(24) * 1024 * 1024)) {
        fprintf(stderr, "resident target safety gate failed: faults=%ld "
                "swap=%" PRIu64 " target_fallbacks=%" PRIu64
                " draft_fallbacks=%" PRIu64 " memavailable=%" PRIu64 " KiB\n",
                timed_major_faults, vmswap_kib, target_fallbacks,
                draft_fallbacks,
                resident->memavailable_after_prompt_snapshots_kib);
        goto fail;
    }
    printf("{\n  \"schema\":\"waste.gn100.spec_target.v1\",\n");
    printf("  \"model\":\"target\",\"prompt_tokens\":%d,\"generated\":%d,\n",
           n_prompt, n_gen);
    printf("  \"tokens\":"); print_ints(tokens, n_gen); puts(",");
    printf("  \"logit_row_hashes\":"); print_u64_hex(logit_rows, n_gen + 1); puts(",");
    printf("  \"route_row_hashes\":"); print_u64_hex(route_rows, n_gen); puts(",");
    printf("  \"token_prefix_hashes\":"); print_u64_hex(token_prefix, n_gen); puts(",");
    printf("  \"logit_prefix_hashes\":"); print_u64_hex(logit_prefix, n_gen + 1); puts(",");
    printf("  \"route_prefix_hashes\":"); print_u64_hex(route_prefix, n_gen); puts(",");
    printf("  \"token_hash\":\"0x%016" PRIx64 "\",\n", token_hash);
    printf("  \"logit_hash\":\"0x%016" PRIx64 "\",\n", logit_hash);
    printf("  \"route_hash\":\"0x%016" PRIx64 "\",\n", route_hash);
    printf("  \"state_bytes\":%zu,\"state_hash\":\"0x%016" PRIx64 "\",\n",
           state_bytes, sh);
    printf("  \"prefill_seconds\":%.9f,\"decode_seconds\":%.9f,\"tok_s\":%.9f,\n",
           prefill_seconds, seconds, n_gen / seconds);
    printf("  \"prefill_hits\":%" PRIu64 ",\"prefill_misses\":%" PRIu64
           ",\"prefill_bytes\":%" PRIu64 ",\n",
           m->cache.hits - ph0 - (m->cache.hits - h0),
           m->cache.misses - pm0 - (m->cache.misses - mi0),
           (b0 - pb0));
    printf("  \"decode_hits\":%" PRIu64 ",\"decode_misses\":%" PRIu64
           ",\"decode_bytes\":%" PRIu64 ",\n",
           m->cache.hits - h0, m->cache.misses - mi0,
           m->cache.bytes_read - b0);
    if (resident) {
        printf("  \"resident_control\":{\"enabled\":true,"
               "\"speculation_enabled\":false,"
               "\"draft_kept_loaded_during_target_decode\":true,"
               "\"target_cache_mib\":40502,\"draft_cache_mib\":16926,"
               "\"target_rollback_bytes\":%zu,\"draft_rollback_bytes\":%zu,"
               "\"target_prompt_state_bytes\":%zu,"
               "\"draft_prompt_state_bytes\":%zu,\"draft_prompt_tokens\":%d,"
               "\"target_load_seconds\":%.9f,\"draft_load_seconds\":%.9f,"
               "\"touch_seconds\":%.9f,\"draft_prefill_seconds\":%.9f,"
               "\"draft_snapshot_seconds\":%.9f,"
               "\"target_snapshot_seconds\":%.9f,"
               "\"memavailable_before_kib\":%" PRIu64 ","
               "\"memavailable_after_target_load_kib\":%" PRIu64 ","
               "\"memavailable_after_draft_load_kib\":%" PRIu64 ","
               "\"memavailable_after_touch_kib\":%" PRIu64 ","
               "\"memavailable_after_prompt_snapshots_kib\":%" PRIu64 ","
               "\"rss_after_prompt_snapshots_kib\":%" PRIu64 ","
               "\"draft_cache_slots\":%d,\"draft_routed_records\":%d,"
               "\"draft_warm_ready\":%d,\"draft_fully_resident\":true,"
               "\"draft_direct_io\":%d,\"draft_readers\":%d,"
               "\"draft_depth\":%d,\"draft_cuda_kda\":%d,"
               "\"draft_cuda_dense\":%d,\"draft_cuda_vq\":%d,"
               "\"draft_cuda_fallbacks\":%" PRIu64 "},\n",
               resident->target_rollback_bytes, resident->draft_rollback_bytes,
               resident->target_prompt_state_bytes,
               resident->draft_prompt_state_bytes, resident->draft_prompt_tokens,
               resident->target_load_seconds, resident->draft_load_seconds,
               resident->touch_seconds, resident->draft_prefill_seconds,
               resident->draft_snapshot_seconds,
               resident->target_snapshot_seconds,
               resident->memavailable_before_kib,
               resident->memavailable_after_target_load_kib,
               resident->memavailable_after_draft_load_kib,
               resident->memavailable_after_touch_kib,
               resident->memavailable_after_prompt_snapshots_kib,
               resident->rss_after_prompt_snapshots_kib,
               resident->draft_cache_slots, resident->draft_routed_records,
               resident->draft_warm_ready, resident->draft->direct_io,
               waste_ecache_io_threads(&resident->draft->cache),
               waste_ecache_io_depth(&resident->draft->cache),
               waste_model_cuda_kda_effective(resident->draft),
               waste_model_cuda_dense_effective(resident->draft),
               waste_model_cuda_vq_effective(resident->draft),
               draft_fallbacks);
    }
    print_process_safety(resident
        ? "draft_prefill_plus_target_prefill_decode" : "prefill_plus_decode",
        timed_major_faults);
    printf("  \"io\":{\"direct\":%d,\"readers\":%d,\"depth\":%d},\n",
           m->direct_io, waste_ecache_io_threads(&m->cache),
           waste_ecache_io_depth(&m->cache));
    printf("  \"cache\":{\"slots\":%d,\"routed_records\":%d,"
           "\"warm_ready\":%d},\n",
           m->cache.n_slots, routed_records(m), warm_ready);
    printf("  \"cuda\":{\"kda\":%d,\"dense\":%d,\"vq\":%d,"
           "\"fallbacks\":%" PRIu64 ",\"kda_calls\":%" PRIu64
           ",\"dense_calls\":%" PRIu64 ",\"vq_experts\":%" PRIu64
           ",\"vq_applies\":%" PRIu64 ",\"vq_lut_builds\":%" PRIu64
           ",\"vq_launches\":%" PRIu64 ",\"vq_syncs\":%" PRIu64 "}\n}\n",
           waste_model_cuda_kda_effective(m),
           waste_model_cuda_dense_effective(m),
           waste_model_cuda_vq_effective(m), target_fallbacks,
           waste_model_cuda_kda_calls(m),
           waste_model_cuda_dense_calls(m),
           waste_model_cuda_vq_experts(m),
           waste_model_cuda_vq_applies(m),
           waste_model_cuda_vq_lut_builds(m),
           waste_model_cuda_vq_launches(m),
           waste_model_cuda_vq_syncs(m));
    free(tokens); free(logit_rows); free(route_rows); free(token_prefix);
    free(logit_prefix); free(route_prefix); free(routed); return 0;

fail:
    free(tokens); free(logit_rows); free(route_rows); free(token_prefix);
    free(logit_prefix); free(route_prefix); free(routed); return 1;
}

static int target_mode(int argc, char **argv)
{
    if (argc != 7) return -2;
    int n_prompt = 0, n_gen = 0;
    int *prompt = parse_ids(argv[5], &n_prompt);
    if (!prompt || n_prompt <= 0 || positive_int(argv[6], 4096, &n_gen)) {
        free(prompt); return -2;
    }
    waste_model m;
    memset(&m, 0, sizeof m);
    if (load_model(&m, argv[2], argv[3], argv[4])) { free(prompt); return 1; }
    const int rc = target_capture(&m, prompt, n_prompt, n_gen, NULL);
    free(prompt); waste_model_free(&m); return rc;
}

static int teacher_mode(int argc, char **argv)
{
    enum { BRANCH_WIDTH = 8 };
    if (argc != 7) return -2;
    int n_prompt = 0, n_target = 0;
    int *prompt = parse_ids(argv[5], &n_prompt);
    int *target = parse_ids(argv[6], &n_target);
    if (!prompt || !target || n_prompt <= 0 || n_target <= 0) {
        free(prompt); free(target); return -2;
    }
    waste_model m;
    memset(&m, 0, sizeof m);
    if (load_model(&m, argv[2], argv[3], argv[4])) {
        free(prompt); free(target); return 1;
    }
    const int warm_ready = cache_ready(&m.cache);
    if (!valid_ids(prompt, n_prompt, m.cfg.vocab, "draft prompt") ||
        !valid_ids(target, n_target, m.cfg.vocab, "draft target"))
        goto fail;
    const int draft_ctx = waste_model_ctx_max(&m);
    if (draft_ctx && (n_prompt > draft_ctx ||
        n_target > draft_ctx - n_prompt)) {
        fprintf(stderr, "draft prompt plus branch trace exceeds context\n");
        goto fail;
    }
    struct rusage timed0, timed1;
    memset(&timed0, 0, sizeof timed0); memset(&timed1, 0, sizeof timed1);
    getrusage(RUSAGE_SELF, &timed0);
    const double p0 = now();
    const float *logits = prefill_cpu(&m, prompt, n_prompt);
    const double prefill_seconds = now() - p0;
    if (!logits) { fprintf(stderr, "draft prompt failed\n"); goto fail; }
    if (!finite_row(logits, m.cfg.vocab)) {
        fprintf(stderr, "draft prompt produced non-finite logits\n");
        goto fail;
    }
    int *pred = (int *)calloc((size_t)n_target, sizeof *pred);
    int *match = (int *)calloc((size_t)n_target, sizeof *match);
    int *branch_widths = (int *)calloc((size_t)n_target, sizeof *branch_widths);
    int *match_lengths = (int *)calloc((size_t)n_target, sizeof *match_lengths);
    int *branch_pred = (int *)malloc((size_t)n_target * BRANCH_WIDTH *
                                     sizeof *branch_pred);
    uint64_t *logit_rows = (uint64_t *)calloc((size_t)n_target, sizeof *logit_rows);
    uint64_t *branch_logit_rows = (uint64_t *)calloc(
        (size_t)n_target * BRANCH_WIDTH, sizeof *branch_logit_rows);
    uint64_t *branch_route_rows = (uint64_t *)calloc(
        (size_t)n_target * BRANCH_WIDTH, sizeof *branch_route_rows);
    double *step_seconds = (double *)calloc((size_t)n_target, sizeof *step_seconds);
    double *branch_seconds = (double *)calloc((size_t)n_target * BRANCH_WIDTH,
                                               sizeof *branch_seconds);
    uint64_t *branch_hits = (uint64_t *)calloc((size_t)n_target,
                                                sizeof *branch_hits);
    uint64_t *branch_misses = (uint64_t *)calloc((size_t)n_target,
                                                  sizeof *branch_misses);
    uint64_t *branch_bytes = (uint64_t *)calloc((size_t)n_target,
                                                 sizeof *branch_bytes);
    int *routed = (int *)malloc((size_t)WASTE_MAX_LAYERS * 64 * sizeof *routed);
    void *snapshot = NULL;
    size_t snapshot_cap = 0, snapshot_bytes_total = 0;
    double snapshot_seconds = 0, restore_seconds = 0;
    if (!pred || !match || !branch_widths || !match_lengths || !branch_pred ||
        !logit_rows || !branch_logit_rows || !branch_route_rows ||
        !step_seconds || !branch_seconds || !branch_hits ||
        !branch_misses || !branch_bytes || !routed) {
        free(pred); free(match); free(branch_widths); free(match_lengths);
        free(branch_pred); free(logit_rows); free(step_seconds);
        free(branch_seconds); free(branch_hits); free(branch_misses);
        free(branch_bytes); free(branch_logit_rows); free(branch_route_rows);
        free(routed); goto fail;
    }
    for (int i = 0; i < n_target * BRANCH_WIDTH; i++) branch_pred[i] = -1;
    const uint64_t h0 = m.cache.hits, mi0 = m.cache.misses, b0 = m.cache.bytes_read;
    const size_t route_count = (size_t)(m.cfg.n_layers - m.cfg.first_dense) *
                               m.cfg.top_k;
    int matched = 0;
    for (int i = 0; i < n_target; i++) {
        pred[i] = argmax(logits, m.cfg.vocab);
        match[i] = pred[i] == target[i];
        matched += match[i];
        logit_rows[i] = hash_one(logits, (size_t)m.cfg.vocab * sizeof *logits);

        size_t snapshot_bytes = 0, written = 0;
        if (waste_model_state_size(&m, n_prompt + i, &snapshot_bytes)) {
            fprintf(stderr, "draft snapshot size failed at prefix %d\n", i);
            goto teacher_fail;
        }
        if (snapshot_bytes > snapshot_cap) {
            void *grown = realloc(snapshot, snapshot_bytes);
            if (!grown) {
                fprintf(stderr, "draft snapshot allocation failed at prefix %d\n", i);
                goto teacher_fail;
            }
            snapshot = grown;
            snapshot_cap = snapshot_bytes;
        }
        double s = now();
        if (waste_model_state_export(&m, n_prompt + i, snapshot,
                                     snapshot_cap, &written) ||
            written != snapshot_bytes) {
            fprintf(stderr, "draft snapshot export failed at prefix %d\n", i);
            goto teacher_fail;
        }
        snapshot_seconds += now() - s;
        snapshot_bytes_total += snapshot_bytes;

        const int width = n_target - i < BRANCH_WIDTH
                        ? n_target - i : BRANCH_WIDTH;
        branch_widths[i] = width;
        const float *branch_logits = logits;
        int prefix_matches = 0;
        const uint64_t branch_h0 = m.cache.hits;
        const uint64_t branch_m0 = m.cache.misses;
        const uint64_t branch_b0 = m.cache.bytes_read;
        for (int j = 0; j < width; j++) {
            branch_logit_rows[(size_t)i * BRANCH_WIDTH + j] =
                hash_one(branch_logits,
                         (size_t)m.cfg.vocab * sizeof *branch_logits);
            const int proposal = argmax(branch_logits, m.cfg.vocab);
            branch_pred[(size_t)i * BRANCH_WIDTH + j] = proposal;
            if (prefix_matches == j && proposal == target[i + j])
                prefix_matches++;
            if (j + 1 < width) {
                memset(routed, 0xff,
                       (size_t)WASTE_MAX_LAYERS * 64 * sizeof *routed);
                s = now();
                branch_logits = waste_model_step(&m, proposal,
                                                  n_prompt + i + j, routed);
                branch_seconds[(size_t)i * BRANCH_WIDTH + j] = now() - s;
                if (!branch_logits) {
                    fprintf(stderr, "draft branch step failed at prefix %d/%d\n",
                            i, j);
                    goto teacher_fail;
                }
                if (!finite_row(branch_logits, m.cfg.vocab)) {
                    fprintf(stderr,
                            "draft branch step produced non-finite logits at %d/%d\n",
                            i, j);
                    goto teacher_fail;
                }
                branch_route_rows[(size_t)i * BRANCH_WIDTH + j] = hash_one(
                    routed + (size_t)m.cfg.first_dense * m.cfg.top_k,
                    route_count * sizeof *routed);
            }
        }
        waste_ecache_drain(&m.cache);
        branch_hits[i] = m.cache.hits - branch_h0;
        branch_misses[i] = m.cache.misses - branch_m0;
        branch_bytes[i] = m.cache.bytes_read - branch_b0;
        match_lengths[i] = prefix_matches;
        if (branch_pred[(size_t)i * BRANCH_WIDTH] != pred[i]) {
            fprintf(stderr, "draft branch root mismatch at prefix %d\n", i);
            goto teacher_fail;
        }
        int restored_pos = -1;
        s = now();
        if (waste_model_state_import(&m, snapshot, snapshot_bytes,
                                     &restored_pos) ||
            restored_pos != n_prompt + i) {
            fprintf(stderr, "draft snapshot restore failed at prefix %d\n", i);
            goto teacher_fail;
        }
        restore_seconds += now() - s;

        s = now();
        logits = waste_model_step(&m, target[i], n_prompt + i, NULL);
        step_seconds[i] = now() - s;
        if (!logits) { fprintf(stderr, "draft step %d failed\n", i); goto teacher_fail; }
        if (!finite_row(logits, m.cfg.vocab)) {
            fprintf(stderr, "draft step %d produced non-finite logits\n", i);
            goto teacher_fail;
        }
        waste_ecache_decode_tick(&m.cache);
    }
    waste_ecache_drain(&m.cache);
    getrusage(RUSAGE_SELF, &timed1);
    const long timed_major_faults = timed1.ru_majflt - timed0.ru_majflt;
    size_t state_bytes = 0;
    uint64_t sh = 0;
    if (state_hash(&m, n_prompt + n_target, &sh, &state_bytes)) {
        fprintf(stderr, "draft final state export failed\n");
        goto teacher_fail;
    }
    double total = 0;
    for (int i = 0; i < n_target; i++) total += step_seconds[i];
    printf("{\n  \"schema\":\"waste.gn100.spec_teacher.v1\",\n");
    printf("  \"model\":\"draft\",\"prompt_tokens\":%d,\"target_tokens\":%d,\n",
           n_prompt, n_target);
    printf("  \"targets\":"); print_ints(target, n_target); puts(",");
    printf("  \"predictions\":"); print_ints(pred, n_target); puts(",");
    printf("  \"matches\":"); print_ints(match, n_target); puts(",");
    printf("  \"branch_width\":%d,\"branch_widths\":", BRANCH_WIDTH);
    print_ints(branch_widths, n_target); puts(",");
    printf("  \"branch_predictions\":");
    print_int_matrix(branch_pred, n_target, BRANCH_WIDTH); puts(",");
    printf("  \"branch_logit_row_hashes\":");
    print_u64_hex_matrix(branch_logit_rows, n_target, BRANCH_WIDTH); puts(",");
    printf("  \"branch_route_row_hashes\":");
    print_u64_hex_matrix(branch_route_rows, n_target, BRANCH_WIDTH); puts(",");
    printf("  \"prefix_match_lengths\":");
    print_ints(match_lengths, n_target); puts(",");
    printf("  \"logit_row_hashes\":"); print_u64_hex(logit_rows, n_target); puts(",");
    printf("  \"step_seconds\":"); print_doubles(step_seconds, n_target); puts(",");
    printf("  \"branch_step_seconds\":");
    print_double_matrix(branch_seconds, n_target, BRANCH_WIDTH); puts(",");
    printf("  \"branch_hits\":"); print_u64s(branch_hits, n_target); puts(",");
    printf("  \"branch_misses\":"); print_u64s(branch_misses, n_target); puts(",");
    printf("  \"branch_bytes\":"); print_u64s(branch_bytes, n_target); puts(",");
    printf("  \"matched\":%d,\"marginal_agreement\":%.9f,\n",
           matched, (double)matched / n_target);
    printf("  \"prefill_seconds\":%.9f,\"teacher_seconds\":%.9f,\"tok_s\":%.9f,\n",
           prefill_seconds, total, n_target / total);
    printf("  \"snapshot_count\":%d,\"snapshot_bytes_total\":%zu,"
           "\"snapshot_seconds\":%.9f,\"restore_count\":%d,"
           "\"restore_seconds\":%.9f,\n",
           n_target, snapshot_bytes_total, snapshot_seconds, n_target,
           restore_seconds);
    printf("  \"decode_hits\":%" PRIu64 ",\"decode_misses\":%" PRIu64
           ",\"decode_bytes\":%" PRIu64 ",\n",
           m.cache.hits - h0, m.cache.misses - mi0, m.cache.bytes_read - b0);
    printf("  \"state_bytes\":%zu,\"state_hash\":\"0x%016" PRIx64 "\",\n",
           state_bytes, sh);
    print_process_safety("prefill_plus_teacher_branches_snapshots_restores",
                         timed_major_faults);
    printf("  \"io\":{\"direct\":%d,\"readers\":%d,\"depth\":%d},\n",
           m.direct_io, waste_ecache_io_threads(&m.cache),
           waste_ecache_io_depth(&m.cache));
    printf("  \"cache\":{\"slots\":%d,\"routed_records\":%d,"
           "\"warm_ready\":%d,\"fully_warm_at_start\":%s},\n",
           m.cache.n_slots, routed_records(&m), warm_ready,
           warm_ready == routed_records(&m) ? "true" : "false");
    printf("  \"cuda\":{\"kda\":%d,\"dense\":%d,\"vq\":%d,"
           "\"fallbacks\":%" PRIu64 ",\"kda_calls\":%" PRIu64
           ",\"dense_calls\":%" PRIu64 ",\"vq_experts\":%" PRIu64
           ",\"vq_applies\":%" PRIu64 ",\"vq_lut_builds\":%" PRIu64
           ",\"vq_launches\":%" PRIu64 ",\"vq_syncs\":%" PRIu64 "}\n}\n",
           waste_model_cuda_kda_effective(&m),
           waste_model_cuda_dense_effective(&m),
           waste_model_cuda_vq_effective(&m),
           waste_model_cuda_kda_fallbacks(&m),
           waste_model_cuda_kda_calls(&m),
           waste_model_cuda_dense_calls(&m),
           waste_model_cuda_vq_experts(&m),
           waste_model_cuda_vq_applies(&m),
           waste_model_cuda_vq_lut_builds(&m),
           waste_model_cuda_vq_launches(&m),
           waste_model_cuda_vq_syncs(&m));
    free(snapshot); free(pred); free(match); free(branch_widths);
    free(match_lengths); free(branch_pred); free(logit_rows);
    free(step_seconds); free(branch_seconds); free(branch_hits);
    free(branch_misses); free(branch_bytes); free(branch_logit_rows);
    free(branch_route_rows); free(routed);
    free(prompt); free(target); waste_model_free(&m); return 0;

teacher_fail:
    free(snapshot); free(pred); free(match); free(branch_widths);
    free(match_lengths); free(branch_pred); free(logit_rows);
    free(step_seconds); free(branch_seconds); free(branch_hits);
    free(branch_misses); free(branch_bytes); free(branch_logit_rows);
    free(branch_route_rows); free(routed);
fail:
    free(prompt); free(target); waste_model_free(&m); return 1;
}

static int append_tokens(const waste_tok *tok, int32_t *out, int cap, int *n,
                         const char *text, int special)
{
    const int got = waste_tok_encode(tok, text, out + *n, cap - *n, special);
    if (got < 0) return -1;
    *n += got;
    return 0;
}

static int require_special(const waste_tok *tok, const char *marker)
{
    int32_t id = -1;
    char decoded[128];
    const int n = waste_tok_encode(tok, marker, &id, 1, 1);
    const int bytes = n == 1
                    ? waste_tok_decode1(tok, id, decoded, (int)sizeof decoded)
                    : -1;
    if (n != 1 || bytes != (int)strlen(marker) ||
        memcmp(decoded, marker, strlen(marker))) {
        fprintf(stderr, "tokenizer does not register structural marker %s\n",
                marker);
        return -1;
    }
    return 0;
}

static int prompt_mode(int argc, char **argv)
{
    if (argc != 5) return -2;
    waste_tok *tok = waste_tok_open(argv[2]);
    if (!tok) { fprintf(stderr, "could not open tokenizer %s\n", argv[2]); return 1; }
    static const char *const markers[] = {
        "<|im_system|>", "<|im_user|>", "<|im_assistant|>",
        "<|im_middle|>", "<|im_end|>"
    };
    for (size_t i = 0; i < sizeof markers / sizeof markers[0]; i++) {
        if (require_special(tok, markers[i])) {
            waste_tok_free(tok);
            return 1;
        }
    }
    const int cap = 16384;
    int32_t *ids = (int32_t *)malloc((size_t)cap * sizeof *ids);
    int n = 0;
    if (!ids ||
        append_tokens(tok, ids, cap, &n, "<|im_system|>system<|im_middle|>", 1) ||
        append_tokens(tok, ids, cap, &n, argv[3], 0) ||
        append_tokens(tok, ids, cap, &n, "<|im_end|><|im_user|>user<|im_middle|>", 1) ||
        append_tokens(tok, ids, cap, &n, argv[4], 0) ||
        append_tokens(tok, ids, cap, &n,
                      "<|im_end|><|im_assistant|>assistant<|im_middle|>", 1)) {
        fprintf(stderr, "native prompt exceeded tokenizer buffer\n");
        free(ids); waste_tok_free(tok); return 1;
    }
    printf("{\"schema\":\"waste.gn100.spec_prompt.v1\",\"tokens\":");
    print_i32s(ids, n);
    printf(",\"token_count\":%d}\n", n);
    free(ids); waste_tok_free(tok); return 0;
}

static void touch_cache(waste_ecache *cache)
{
    const size_t page = 4096;
    for (int i = 0; i < cache->n_slots; i++) {
        volatile uint8_t *p = cache->slot[i].data;
        for (size_t off = 0; off < cache->rec_bytes; off += page) p[off] = p[off];
        if (cache->rec_bytes) p[cache->rec_bytes - 1] = p[cache->rec_bytes - 1];
    }
}

static void touch_blob(void *blob, size_t bytes)
{
    volatile uint8_t *p = (volatile uint8_t *)blob;
    for (size_t off = 0; off < bytes; off += 4096) p[off] = 1;
    if (bytes) p[bytes - 1] = 1;
}

/* R, the resident control: ordinary K3 decode at the registered reduced
 * cache while one fully-warm, prefilled Kimi-Linear model and both exact
 * rollback allocations stay live in this process. This measures rent only;
 * the draft performs no work after K3 decode begins. */
static int resident_target_mode(int argc, char **argv)
{
    enum {
        TARGET_CACHE_MIB = 40502,
        DRAFT_CACHE_MIB = 16926,
    };
    const size_t target_rollback_bytes = (size_t)512 << 20;
    const size_t draft_rollback_bytes = (size_t)128 << 20;
    if (argc != 9) return -2;

    int n_target_prompt = 0, n_draft_prompt = 0, n_gen = 0;
    int *target_prompt = parse_ids(argv[6], &n_target_prompt);
    int *draft_prompt = parse_ids(argv[7], &n_draft_prompt);
    if (!target_prompt || !draft_prompt || n_target_prompt <= 0 ||
        n_draft_prompt <= 0 || positive_int(argv[8], 4096, &n_gen)) {
        free(target_prompt); free(draft_prompt); return -2;
    }

    waste_model target, draft;
    memset(&target, 0, sizeof target); memset(&draft, 0, sizeof draft);
    int target_loaded = 0, draft_loaded = 0;
    void *target_rollback = NULL, *draft_rollback = NULL;
    resident_control resident;
    memset(&resident, 0, sizeof resident);
    resident.memavailable_before_kib =
        proc_value_kib("/proc/meminfo", "MemAvailable");

    double started = now();
    if (load_model(&target, argv[2], "40502", argv[3])) goto fail;
    target_loaded = 1;
    resident.target_load_seconds = now() - started;
    resident.memavailable_after_target_load_kib =
        proc_value_kib("/proc/meminfo", "MemAvailable");

    /* Kimi-Linear has no VQ path. Set this only after K3 has captured its
     * registered VQ=2 mode in the already-loaded target object. */
    if (setenv("WASTE_CUDA_VQ", "0", 1)) {
        fprintf(stderr, "could not select the draft CUDA profile\n");
        goto fail;
    }
    started = now();
    if (load_model(&draft, argv[4], "16926", argv[5])) goto fail;
    draft_loaded = 1;
    resident.draft_load_seconds = now() - started;
    resident.memavailable_after_draft_load_kib =
        proc_value_kib("/proc/meminfo", "MemAvailable");
    resident.draft_cache_slots = draft.cache.n_slots;
    resident.draft_routed_records = routed_records(&draft);
    resident.draft_warm_ready = cache_ready(&draft.cache);

    if (resident.draft_cache_slots < resident.draft_routed_records ||
        resident.draft_warm_ready != resident.draft_routed_records) {
        fprintf(stderr, "resident draft is not fully warm (%d/%d records)\n",
                resident.draft_warm_ready, resident.draft_routed_records);
        goto fail;
    }
    if (!valid_ids(draft_prompt, n_draft_prompt, draft.cfg.vocab,
                   "resident draft prompt"))
        goto fail;
    const int draft_ctx = waste_model_ctx_max(&draft);
    if (draft_ctx && n_draft_prompt > draft_ctx) {
        fprintf(stderr, "resident draft prompt exceeds context\n");
        goto fail;
    }
    if (waste_model_get_cuda_kda(&target) != 1 ||
        waste_model_get_cuda_dense(&target) != 2 ||
        waste_model_get_cuda_vq(&target) != 2 ||
        waste_model_get_cuda_kda(&draft) != 1 ||
        waste_model_get_cuda_dense(&draft) != 2 ||
        waste_model_get_cuda_vq(&draft) != 0 ||
        waste_ecache_io_threads(&target.cache) != 2 ||
        waste_ecache_io_depth(&target.cache) != 2 ||
        waste_ecache_io_threads(&draft.cache) != 2 ||
        waste_ecache_io_depth(&draft.cache) != 2) {
        fprintf(stderr, "resident target profile drifted\n");
        goto fail;
    }

    target_rollback = calloc(1, target_rollback_bytes);
    draft_rollback = calloc(1, draft_rollback_bytes);
    if (!target_rollback || !draft_rollback) {
        fprintf(stderr, "resident rollback allocation failed\n");
        goto fail;
    }
    started = now();
    touch_cache(&target.cache); touch_cache(&draft.cache);
    touch_blob(target_rollback, target_rollback_bytes);
    touch_blob(draft_rollback, draft_rollback_bytes);
    resident.touch_seconds = now() - started;
    resident.memavailable_after_touch_kib =
        proc_value_kib("/proc/meminfo", "MemAvailable");

    memset(&resident.timed0, 0, sizeof resident.timed0);
    getrusage(RUSAGE_SELF, &resident.timed0);
    started = now();
    const float *draft_logits = prefill_cpu(&draft, draft_prompt, n_draft_prompt);
    resident.draft_prefill_seconds = now() - started;
    if (!draft_logits || !finite_row(draft_logits, draft.cfg.vocab)) {
        fprintf(stderr, "resident draft prompt failed or produced non-finite logits\n");
        goto fail;
    }
    resident.draft_prompt_tokens = n_draft_prompt;
    if (waste_model_state_size(&draft, n_draft_prompt,
                               &resident.draft_prompt_state_bytes) ||
        !resident.draft_prompt_state_bytes ||
        resident.draft_prompt_state_bytes > draft_rollback_bytes) {
        fprintf(stderr, "draft prompt state exceeds resident rollback\n");
        goto fail;
    }
    size_t written = 0;
    started = now();
    if (waste_model_state_export(&draft, n_draft_prompt, draft_rollback,
                                 draft_rollback_bytes, &written) ||
        written != resident.draft_prompt_state_bytes) {
        fprintf(stderr, "resident draft prompt snapshot failed\n");
        goto fail;
    }
    resident.draft_snapshot_seconds = now() - started;

    resident.draft = &draft;
    resident.target_rollback = target_rollback;
    resident.target_rollback_bytes = target_rollback_bytes;
    resident.draft_rollback_bytes = draft_rollback_bytes;
    const int rc = target_capture(&target, target_prompt, n_target_prompt,
                                  n_gen, &resident);
    free(target_rollback); free(draft_rollback);
    free(target_prompt); free(draft_prompt);
    waste_model_free(&draft); waste_model_free(&target);
    return rc;

fail:
    free(target_rollback); free(draft_rollback);
    free(target_prompt); free(draft_prompt);
    if (draft_loaded) waste_model_free(&draft);
    if (target_loaded) waste_model_free(&target);
    return 1;
}

enum { SHADOW_MAX_THREADS = 10, SHADOW_REPEATS = 7 };

typedef struct shadow_team shadow_team;

typedef struct {
    shadow_team *team;
    const uint8_t *src;
    uint8_t *dst;
    size_t off, bytes;
    unsigned seen;
} shadow_worker;

struct shadow_team {
    pthread_mutex_t mutex;
    pthread_cond_t start;
    pthread_cond_t done;
    pthread_t threads[SHADOW_MAX_THREADS];
    shadow_worker workers[SHADOW_MAX_THREADS];
    int n_threads, created, remaining, stop;
    int mutex_ready, start_ready, done_ready;
    unsigned generation;
};

typedef struct {
    double best_seconds;
    double median_seconds;
    double best_gib_s;
    double median_gib_s;
    uint64_t copied_hash;
} shadow_result;

static volatile uint8_t shadow_touch_sink;

static void touch_blob_read(const void *blob, size_t bytes)
{
    const volatile uint8_t *p = (const volatile uint8_t *)blob;
    uint8_t value = 0;
    for (size_t off = 0; off < bytes; off += 4096) value ^= p[off];
    if (bytes) value ^= p[bytes - 1];
    shadow_touch_sink ^= value;
}

static void *shadow_worker_main(void *arg)
{
    shadow_worker *worker = (shadow_worker *)arg;
    shadow_team *team = worker->team;
    pthread_mutex_lock(&team->mutex);
    for (;;) {
        while (!team->stop && worker->seen == team->generation)
            pthread_cond_wait(&team->start, &team->mutex);
        if (team->stop) break;
        worker->seen = team->generation;
        pthread_mutex_unlock(&team->mutex);
        memcpy(worker->dst + worker->off, worker->src + worker->off,
               worker->bytes);
        pthread_mutex_lock(&team->mutex);
        team->remaining--;
        if (!team->remaining) pthread_cond_signal(&team->done);
    }
    pthread_mutex_unlock(&team->mutex);
    return NULL;
}

static void shadow_team_destroy(shadow_team *team)
{
    if (team->created && team->mutex_ready) {
        pthread_mutex_lock(&team->mutex);
        team->stop = 1;
        team->generation++;
        if (team->start_ready) pthread_cond_broadcast(&team->start);
        pthread_mutex_unlock(&team->mutex);
        for (int i = 0; i < team->created; i++)
            pthread_join(team->threads[i], NULL);
    }
    if (team->done_ready) pthread_cond_destroy(&team->done);
    if (team->start_ready) pthread_cond_destroy(&team->start);
    if (team->mutex_ready) pthread_mutex_destroy(&team->mutex);
    memset(team, 0, sizeof *team);
}

static int shadow_team_init(shadow_team *team, const void *src, void *dst,
                            size_t bytes, int n_threads)
{
    memset(team, 0, sizeof *team);
    if (n_threads < 1 || n_threads > SHADOW_MAX_THREADS) return -1;
    if (pthread_mutex_init(&team->mutex, NULL)) return -1;
    team->mutex_ready = 1;
    if (pthread_cond_init(&team->start, NULL)) {
        shadow_team_destroy(team); return -1;
    }
    team->start_ready = 1;
    if (pthread_cond_init(&team->done, NULL)) {
        shadow_team_destroy(team); return -1;
    }
    team->done_ready = 1;
    team->n_threads = n_threads;
    const size_t base = bytes / (size_t)n_threads;
    const size_t extra = bytes % (size_t)n_threads;
    size_t off = 0;
    for (int i = 0; i < n_threads; i++) {
        shadow_worker *worker = &team->workers[i];
        worker->team = team;
        worker->src = (const uint8_t *)src;
        worker->dst = (uint8_t *)dst;
        worker->off = off;
        worker->bytes = base + ((size_t)i < extra);
        off += worker->bytes;
        if (pthread_create(&team->threads[i], NULL,
                           shadow_worker_main, worker)) {
            shadow_team_destroy(team); return -1;
        }
        team->created++;
    }
    return 0;
}

static int shadow_team_copy(shadow_team *team, double *seconds)
{
    if (pthread_mutex_lock(&team->mutex)) return -1;
    team->remaining = team->n_threads;
    team->generation++;
    const double started = now();
    if (pthread_cond_broadcast(&team->start)) {
        pthread_mutex_unlock(&team->mutex); return -1;
    }
    while (team->remaining)
        if (pthread_cond_wait(&team->done, &team->mutex)) {
            pthread_mutex_unlock(&team->mutex); return -1;
        }
    *seconds = now() - started;
    pthread_mutex_unlock(&team->mutex);
    return 0;
}

static int double_cmp(const void *a, const void *b)
{
    const double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}

static int shadow_copy_floor(const void *src, void *dst, size_t bytes,
                             int n_threads, shadow_result *result)
{
    shadow_team team;
    double samples[SHADOW_REPEATS], warmup = 0;
    touch_blob_read(src, bytes);
    touch_blob(dst, bytes);
    if (shadow_team_init(&team, src, dst, bytes, n_threads)) return -1;
    if (shadow_team_copy(&team, &warmup)) {
        shadow_team_destroy(&team); return -1;
    }
    for (int i = 0; i < SHADOW_REPEATS; i++)
        if (shadow_team_copy(&team, &samples[i])) {
            shadow_team_destroy(&team); return -1;
        }
    shadow_team_destroy(&team);
    result->copied_hash = hash_one(dst, bytes);
    qsort(samples, SHADOW_REPEATS, sizeof samples[0], double_cmp);
    result->best_seconds = samples[0];
    result->median_seconds = samples[SHADOW_REPEATS / 2];
    const double gib = (double)bytes / (double)(UINT64_C(1) << 30);
    result->best_gib_s = result->best_seconds > 0
                       ? gib / result->best_seconds : 0;
    result->median_gib_s = result->median_seconds > 0
                         ? gib / result->median_seconds : 0;
    return 0;
}

/* Price the exact target state at a real speculative block root.  Unlike
 * load_mode's inference-free pos=0 floor, this includes the live MLA latent
 * rows and AttnRes state produced by the canonical prompt and continuation. */
static int state_mode(int argc, char **argv)
{
    if (argc != 7) return -2;
    int n_prompt = 0, n_target = 0;
    int *prompt = parse_ids(argv[5], &n_prompt);
    int *target = parse_ids(argv[6], &n_target);
    if (!prompt || !target || n_prompt <= 0 || n_target <= 0) {
        free(prompt); free(target); return -2;
    }

    waste_model m;
    memset(&m, 0, sizeof m);
    void *blob = NULL, *shadow = NULL, *post_a = NULL, *post_b = NULL;
    float *baseline_logits = NULL;
    int *routed_a = NULL, *routed_b = NULL;
    size_t state_bytes = 0;
    if (load_model(&m, argv[2], argv[3], argv[4])) {
        free(prompt); free(target); return 1;
    }
    const int warm_ready = cache_ready(&m.cache);
    if (!valid_ids(prompt, n_prompt, m.cfg.vocab, "state prompt") ||
        !valid_ids(target, n_target, m.cfg.vocab, "state continuation"))
        goto fail;
    const int ctx = waste_model_ctx_max(&m);
    if (ctx && (n_prompt > ctx || n_target - 1 > ctx - n_prompt)) {
        fprintf(stderr, "state prompt plus canonical root exceeds context\n");
        goto fail;
    }

    struct rusage timed0, timed1;
    memset(&timed0, 0, sizeof timed0); memset(&timed1, 0, sizeof timed1);
    getrusage(RUSAGE_SELF, &timed0);
    double started = now();
    const float *logits = prefill_cpu(&m, prompt, n_prompt);
    const double prefill_seconds = now() - started;
    if (!logits || !finite_row(logits, m.cfg.vocab)) {
        fprintf(stderr, "state prompt failed or produced non-finite logits\n");
        goto fail;
    }
    started = now();
    for (int i = 0; i + 1 < n_target; i++) {
        if (argmax(logits, m.cfg.vocab) != target[i]) {
            fprintf(stderr, "state continuation drifted from greedy target at %d\n", i);
            goto fail;
        }
        logits = waste_model_step(&m, target[i], n_prompt + i, NULL);
        if (!logits || !finite_row(logits, m.cfg.vocab)) {
            fprintf(stderr, "state continuation step %d failed\n", i);
            goto fail;
        }
        waste_ecache_decode_tick(&m.cache);
    }
    waste_ecache_drain(&m.cache);
    const double continuation_seconds = now() - started;
    if (argmax(logits, m.cfg.vocab) != target[n_target - 1]) {
        fprintf(stderr, "state final root drifted from greedy target\n");
        goto fail;
    }
    const int state_pos = n_prompt + n_target - 1;
    if (waste_model_state_size(&m, state_pos, &state_bytes) || !state_bytes) {
        fprintf(stderr, "post-prompt target state size failed\n");
        goto fail;
    }
    blob = mmap(NULL, state_bytes, PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    shadow = mmap(NULL, state_bytes, PROT_READ | PROT_WRITE,
                  MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (blob == MAP_FAILED) blob = NULL;
    if (shadow == MAP_FAILED) shadow = NULL;
    if (!blob || !shadow) {
        fprintf(stderr, "post-prompt target state mapping failed\n");
        goto fail;
    }
    touch_blob(blob, state_bytes);
    touch_blob(shadow, state_bytes);

    size_t written = 0;
    int restored_pos = -1;
    if (waste_model_state_export(&m, state_pos, blob, state_bytes, &written) ||
        written != state_bytes ||
        waste_model_state_import(&m, blob, state_bytes, &restored_pos) ||
        restored_pos != state_pos) {
        fprintf(stderr, "post-prompt target state warm round trip failed\n");
        goto fail;
    }
    started = now();
    written = 0;
    const int export_rc = waste_model_state_export(
        &m, state_pos, blob, state_bytes, &written);
    const double export_seconds = now() - started;
    started = now();
    restored_pos = -1;
    const int import_rc = waste_model_state_import(
        &m, blob, state_bytes, &restored_pos);
    const double import_seconds = now() - started;
    if (export_rc || import_rc || written != state_bytes ||
        restored_pos != state_pos) {
        fprintf(stderr, "timed post-prompt target state round trip failed\n");
        goto fail;
    }

    shadow_result shadow_1, shadow_10;
    memset(&shadow_1, 0, sizeof shadow_1);
    memset(&shadow_10, 0, sizeof shadow_10);
    const uint64_t blob_hash = hash_one(blob, state_bytes);
    if (shadow_copy_floor(blob, shadow, state_bytes, 1, &shadow_1) ||
        shadow_1.copied_hash != blob_hash ||
        shadow_copy_floor(blob, shadow, state_bytes, 10, &shadow_10) ||
        shadow_10.copied_hash != blob_hash) {
        fprintf(stderr, "post-prompt target shadow-copy benchmark failed\n");
        goto fail;
    }
    getrusage(RUSAGE_SELF, &timed1);
    const long timed_major_faults = timed1.ru_majflt - timed0.ru_majflt;
    /* A restore into an unchanged root is not evidence: even a no-op import
     * would pass. Advance once, restore, replay, and compare all three strict
     * contract surfaces before restoring the measured root again. */
    const size_t route_cap = (size_t)WASTE_MAX_LAYERS * 64;
    const size_t route_count =
        (size_t)(m.cfg.n_layers - m.cfg.first_dense) * m.cfg.top_k;
    baseline_logits = (float *)malloc((size_t)m.cfg.vocab * sizeof(float));
    routed_a = (int *)malloc(route_cap * sizeof(int));
    routed_b = (int *)malloc(route_cap * sizeof(int));
    if (!baseline_logits || !routed_a || !routed_b) {
        fprintf(stderr, "post-prompt replay verification allocation failed\n");
        goto fail;
    }
    memset(routed_a, 0xff, route_cap * sizeof(int));
    const int replay_token = target[n_target - 1];
    const float *replay_a = waste_model_step(
        &m, replay_token, state_pos, routed_a);
    if (!replay_a || !finite_row(replay_a, m.cfg.vocab)) {
        fprintf(stderr, "post-prompt baseline replay step failed\n");
        goto fail;
    }
    memcpy(baseline_logits, replay_a, (size_t)m.cfg.vocab * sizeof(float));
    waste_ecache_decode_tick(&m.cache);
    waste_ecache_drain(&m.cache);
    size_t post_bytes = 0, post_written = 0;
    if (waste_model_state_size(&m, state_pos + 1, &post_bytes) || !post_bytes) {
        fprintf(stderr, "post-prompt replay state size failed\n");
        goto fail;
    }
    post_a = mmap(NULL, post_bytes, PROT_READ | PROT_WRITE,
                  MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    post_b = mmap(NULL, post_bytes, PROT_READ | PROT_WRITE,
                  MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (post_a == MAP_FAILED) post_a = NULL;
    if (post_b == MAP_FAILED) post_b = NULL;
    if (!post_a || !post_b ||
        waste_model_state_export(&m, state_pos + 1, post_a, post_bytes,
                                 &post_written) ||
        post_written != post_bytes) {
        fprintf(stderr, "post-prompt baseline replay state export failed\n");
        goto fail;
    }

    restored_pos = -1;
    if (waste_model_state_import(&m, blob, state_bytes, &restored_pos) ||
        restored_pos != state_pos) {
        fprintf(stderr, "post-prompt replay root restore failed\n");
        goto fail;
    }
    memset(routed_b, 0xff, route_cap * sizeof(int));
    const float *replay_b = waste_model_step(
        &m, replay_token, state_pos, routed_b);
    if (!replay_b || !finite_row(replay_b, m.cfg.vocab)) {
        fprintf(stderr, "post-prompt restored replay step failed\n");
        goto fail;
    }
    waste_ecache_decode_tick(&m.cache);
    waste_ecache_drain(&m.cache);
    post_written = 0;
    if (memcmp(baseline_logits, replay_b,
               (size_t)m.cfg.vocab * sizeof(float)) ||
        memcmp(routed_a + (size_t)m.cfg.first_dense * m.cfg.top_k,
               routed_b + (size_t)m.cfg.first_dense * m.cfg.top_k,
               route_count * sizeof(int)) ||
        waste_model_state_export(&m, state_pos + 1, post_b, post_bytes,
                                 &post_written) ||
        post_written != post_bytes || memcmp(post_a, post_b, post_bytes)) {
        fprintf(stderr, "post-prompt rollback replay was not bit-exact\n");
        goto fail;
    }
    restored_pos = -1;
    if (waste_model_state_import(&m, blob, state_bytes, &restored_pos) ||
        restored_pos != state_pos) {
        fprintf(stderr, "post-prompt final root restore failed\n");
        goto fail;
    }
    uint64_t live_hash = 0;
    size_t live_bytes = 0;
    if (state_hash(&m, state_pos, &live_hash, &live_bytes) ||
        live_bytes != state_bytes || live_hash != blob_hash) {
        fprintf(stderr, "post-prompt final root hash mismatch\n");
        goto fail;
    }

    int mla_layers = 0;
    for (int L = 0; L < m.cfg.n_layers; L++)
        if (!m.cfg.kda_layer[L]) mla_layers++;
    const uint64_t mla_latent_bytes = (uint64_t)mla_layers *
        (uint64_t)state_pos * (uint64_t)(m.cfg.kv_lora + m.cfg.qk_rope) *
        sizeof(float);
    printf("{\n  \"schema\":\"waste.gn100.spec_state.v1\",\n");
    printf("  \"model\":\"target\",\"actual_model_state\":true,"
           "\"synthetic_mla_rows\":false,\n");
    printf("  \"prompt_tokens\":%d,\"canonical_continuation_tokens\":%d,"
           "\"continuation_tokens_applied\":%d,\"state_position\":%d,\n",
           n_prompt, n_target, n_target - 1, state_pos);
    printf("  \"state_bytes\":%zu,\"state_hash\":\"0x%016" PRIx64 "\","
           "\"mla_layers\":%d,\"mla_latent_bytes\":%" PRIu64 ",\n",
           state_bytes, blob_hash, mla_layers, mla_latent_bytes);
    printf("  \"prefill_seconds\":%.9f,\"continuation_seconds\":%.9f,\n",
           prefill_seconds, continuation_seconds);
    printf("  \"roundtrip_replay\":{\"token\":%d,"
           "\"logits_byte_identical\":true,"
           "\"ordered_routes_byte_identical\":true,"
           "\"post_state_byte_identical\":true,"
           "\"root_restored_after_check\":true,"
           "\"post_state_bytes\":%zu},\n",
           replay_token, post_bytes);
    printf("  \"in_memory_state_serialization\":{"
           "\"label\":\"warm post-prompt in-memory export/import; "
           "includes actual MLA latent rows and excludes durable file I/O\","
           "\"warmup_roundtrips\":1,\"bytes\":%zu,"
           "\"export_seconds\":%.9f,\"import_seconds\":%.9f},\n",
           state_bytes, export_seconds, import_seconds);
    printf("  \"shadow_copy_floor\":{"
           "\"label\":\"optimistic contiguous-memory copy of actual "
           "post-prompt serialized state; not pointer-swap or durable I/O\","
           "\"optimistic\":true,\"is_pointer_swap\":false,"
           "\"is_durable_file_io\":false,\"warmup_copies\":1,"
           "\"pages_pretouched\":true,"
           "\"thread_creation_timed\":false,\"worker_dispatch_timed\":true,"
           "\"throughput_basis\":\"logical state bytes per one-way copy; "
           "not aggregate read-plus-write bus traffic\","
           "\"repeats\":%d,\"bytes\":%zu,"
           "\"threads_1\":{\"threads\":1,\"best_seconds\":%.9f,"
           "\"median_seconds\":%.9f,\"best_gib_s\":%.6f,"
           "\"median_gib_s\":%.6f},"
           "\"threads_10\":{\"threads\":10,\"best_seconds\":%.9f,"
           "\"median_seconds\":%.9f,\"best_gib_s\":%.6f,"
           "\"median_gib_s\":%.6f}},\n",
           SHADOW_REPEATS, state_bytes,
           shadow_1.best_seconds, shadow_1.median_seconds,
           shadow_1.best_gib_s, shadow_1.median_gib_s,
           shadow_10.best_seconds, shadow_10.median_seconds,
           shadow_10.best_gib_s, shadow_10.median_gib_s);
    printf("  \"io\":{\"direct\":%d,\"readers\":%d,\"depth\":%d},\n",
           m.direct_io, waste_ecache_io_threads(&m.cache),
           waste_ecache_io_depth(&m.cache));
    printf("  \"cache\":{\"slots\":%d,\"routed_records\":%d,"
           "\"warm_ready\":%d},\n",
           m.cache.n_slots, routed_records(&m), warm_ready);
    print_process_safety("prefill_continuation_export_import_shadow_copy",
                         timed_major_faults);
    printf("  \"cuda\":{\"kda\":%d,\"dense\":%d,\"vq\":%d,"
           "\"fallbacks\":%" PRIu64 "}\n}\n",
           waste_model_cuda_kda_effective(&m),
           waste_model_cuda_dense_effective(&m),
           waste_model_cuda_vq_effective(&m),
           waste_model_cuda_kda_fallbacks(&m));

    munmap(post_b, post_bytes); munmap(post_a, post_bytes);
    munmap(shadow, state_bytes); munmap(blob, state_bytes);
    free(baseline_logits); free(routed_a); free(routed_b);
    free(prompt); free(target); waste_model_free(&m); return 0;

fail:
    if (post_b) {
        size_t bytes = 0;
        if (!waste_model_state_size(&m, state_pos + 1, &bytes) && bytes)
            munmap(post_b, bytes);
    }
    if (post_a) {
        size_t bytes = 0;
        if (!waste_model_state_size(&m, state_pos + 1, &bytes) && bytes)
            munmap(post_a, bytes);
    }
    if (shadow) munmap(shadow, state_bytes);
    if (blob) munmap(blob, state_bytes);
    free(baseline_logits); free(routed_a); free(routed_b);
    free(prompt); free(target); waste_model_free(&m); return 1;
}

#if defined(WASTE_ENABLE_DIAGNOSTIC_VERIFY)
/* One representative T=2..4 verifier calibration. This deliberately is not
 * a speculative runner or a candidate qualifier: each invocation measures
 * one fresh-process arm from a byte-hashed canonical root. */
static int verify4_mode(int argc, char **argv)
{
    enum { ARM_SERIAL, ARM_CHUNK0, ARM_CHUNK1 } arm;
    if (argc != 9) return -2;
    if (!strcmp(argv[2], "serial")) arm = ARM_SERIAL;
    else if (!strcmp(argv[2], "chunk0")) arm = ARM_CHUNK0;
    else if (!strcmp(argv[2], "chunk1")) arm = ARM_CHUNK1;
    else return -2;

    int n_prompt = 0, n_root = 0, n_proposal = 0;
    int *prompt = parse_ids(argv[6], &n_prompt);
    int *root = parse_ids(argv[7], &n_root);
    int *proposal = parse_ids(argv[8], &n_proposal);
    if (!prompt || n_prompt <= 0 ||
        (!root && strcmp(argv[7], "-")) ||
        !proposal || n_proposal < 2 || n_proposal > 4) {
        free(prompt); free(root); free(proposal); return -2;
    }

    waste_model m;
    memset(&m, 0, sizeof m);
    void *root_blob = NULL;
    float *rows = NULL;
    int *routed = NULL;
    if (waste_model_set_i8mm_diagnostic(0)) {
        fprintf(stderr, "verify4 could not force diagnostic I8MM off\n");
        goto fail_unloaded;
    }
    if (load_model(&m, argv[3], argv[4], argv[5])) goto fail_unloaded;
    const int warm_ready = cache_ready(&m.cache);
    if (!valid_ids(prompt, n_prompt, m.cfg.vocab, "verify4 prompt") ||
        !valid_ids(root, n_root, m.cfg.vocab, "verify4 root") ||
        !valid_ids(proposal, n_proposal, m.cfg.vocab, "verify4 proposal"))
        goto fail;
    const int state_pos = n_prompt + n_root;
    const int ctx = waste_model_ctx_max(&m);
    if (ctx && (state_pos > ctx || n_proposal > ctx - state_pos)) {
        fprintf(stderr, "verify4 root plus proposal exceeds context\n");
        goto fail;
    }

    double started = now();
    const float *logits = prefill_cpu(&m, prompt, n_prompt);
    const double prompt_seconds = now() - started;
    if (!logits || !finite_row(logits, m.cfg.vocab)) {
        fprintf(stderr, "verify4 prompt failed or produced non-finite logits\n");
        goto fail;
    }
    started = now();
    for (int i = 0; i < n_root; i++) {
        if (argmax(logits, m.cfg.vocab) != root[i]) {
            fprintf(stderr, "verify4 root is not canonical greedy at token %d\n", i);
            goto fail;
        }
        logits = waste_model_step(&m, root[i], n_prompt + i, NULL);
        if (!logits || !finite_row(logits, m.cfg.vocab)) {
            fprintf(stderr, "verify4 canonical root step %d failed\n", i);
            goto fail;
        }
        waste_ecache_decode_tick(&m.cache);
    }
    waste_ecache_drain(&m.cache);
    const double root_advance_seconds = now() - started;
    const uint64_t root_logit_hash =
        hash_one(logits, (size_t)m.cfg.vocab * sizeof(float));
    const int root_argmax = argmax(logits, m.cfg.vocab);

    size_t root_bytes = 0, written = 0;
    if (waste_model_state_size(&m, state_pos, &root_bytes) || !root_bytes ||
        !(root_blob = malloc(root_bytes))) {
        fprintf(stderr, "verify4 root snapshot allocation failed\n");
        goto fail;
    }
    started = now();
    const int snapshot_rc = waste_model_state_export(
        &m, state_pos, root_blob, root_bytes, &written);
    const double snapshot_seconds = now() - started;
    if (snapshot_rc || written != root_bytes) {
        fprintf(stderr, "verify4 root snapshot failed\n");
        goto fail;
    }
    const uint64_t root_hash = hash_one(root_blob, root_bytes);

    waste_model_reset(&m);
    int restored_pos = -1;
    started = now();
    const int restore_rc = waste_model_state_import(
        &m, root_blob, root_bytes, &restored_pos);
    const double restore_seconds = now() - started;
    written = 0;
    if (restore_rc || restored_pos != state_pos ||
        waste_model_state_export(&m, state_pos, root_blob, root_bytes, &written) ||
        written != root_bytes || hash_one(root_blob, root_bytes) != root_hash) {
        fprintf(stderr, "verify4 restored root hash mismatch\n");
        goto fail;
    }

    if (waste_model_set_i8mm_diagnostic(arm == ARM_CHUNK1)) {
        fprintf(stderr, "verify4 chunk1 requested unavailable I8MM\n");
        goto fail;
    }
    const int measured_i8mm = waste_model_get_i8mm_diagnostic();
    uint64_t logit_hashes[4] = {0}, route_hashes[4] = {0};
    int row_argmax[4] = {0};
    if (arm == ARM_SERIAL) {
        routed = (int *)malloc((size_t)WASTE_MAX_LAYERS * 64 * sizeof *routed);
        if (!routed) {
            fprintf(stderr, "verify4 route allocation failed\n");
            goto fail;
        }
    } else {
        if (m.cfg.vocab <= 0 ||
            (size_t)m.cfg.vocab > SIZE_MAX / (size_t)n_proposal /
                                  sizeof *rows) {
            fprintf(stderr, "verify4 row allocation overflow\n");
            goto fail;
        }
        const size_t count = (size_t)n_proposal * (size_t)m.cfg.vocab;
        rows = (float *)malloc(count * sizeof *rows);
        if (!rows) {
            fprintf(stderr, "verify4 row allocation failed\n");
            goto fail;
        }
    }

    const uint64_t h0 = m.cache.hits, mi0 = m.cache.misses;
    const uint64_t b0 = m.cache.bytes_read, er0 = m.expert_reads;
    const uint64_t u0 = waste_model_chunk_expert_union(&m);
    const uint64_t k0 = waste_model_cuda_kda_calls(&m);
    const uint64_t d0 = waste_model_cuda_dense_calls(&m);
    const uint64_t ve0 = waste_model_cuda_vq_experts(&m);
    const uint64_t va0 = waste_model_cuda_vq_applies(&m);
    const uint64_t vl0 = waste_model_cuda_vq_launches(&m);
    const uint64_t vs0 = waste_model_cuda_vq_syncs(&m);
    const uint64_t f0 = waste_model_cuda_kda_fallbacks(&m);
    struct rusage ru0, ru1;
    memset(&ru0, 0, sizeof ru0); memset(&ru1, 0, sizeof ru1);
    getrusage(RUSAGE_SELF, &ru0);
    started = now();
    if (arm == ARM_SERIAL) {
        const size_t route_count =
            (size_t)(m.cfg.n_layers - m.cfg.first_dense) * m.cfg.top_k;
        for (int i = 0; i < n_proposal; i++) {
            memset(routed, 0xff,
                   (size_t)WASTE_MAX_LAYERS * 64 * sizeof *routed);
            logits = waste_model_step(&m, proposal[i], state_pos + i, routed);
            if (!logits || !finite_row(logits, m.cfg.vocab)) {
                fprintf(stderr, "verify4 serial step %d failed\n", i);
                goto fail;
            }
            logit_hashes[i] = hash_one(
                logits, (size_t)m.cfg.vocab * sizeof(float));
            row_argmax[i] = argmax(logits, m.cfg.vocab);
            route_hashes[i] = hash_one(
                routed + (size_t)m.cfg.first_dense * m.cfg.top_k,
                route_count * sizeof(int));
            waste_ecache_decode_tick(&m.cache);
        }
    } else {
        logits = waste_model_prefill_diagnostic_rows(
            &m, proposal, n_proposal, state_pos, rows,
            (size_t)n_proposal * (size_t)m.cfg.vocab);
        if (!logits) {
            fprintf(stderr, "verify4 chunk failed\n");
            goto fail;
        }
        for (int i = 0; i < n_proposal; i++) {
            const float *row = rows + (size_t)i * m.cfg.vocab;
            if (!finite_row(row, m.cfg.vocab)) {
                fprintf(stderr, "verify4 chunk row %d is non-finite\n", i);
                goto fail;
            }
            logit_hashes[i] = hash_one(
                row, (size_t)m.cfg.vocab * sizeof(float));
            row_argmax[i] = argmax(row, m.cfg.vocab);
            waste_ecache_decode_tick(&m.cache);
        }
    }
    waste_ecache_drain(&m.cache);
    const double verifier_seconds = now() - started;
    getrusage(RUSAGE_SELF, &ru1);

    /* The rollback root stayed resident for the complete timed verifier.
     * Release it before the separate final-state export to avoid a second
     * target-sized snapshot inflating peak memory for no measurement value. */
    free(root_blob);
    root_blob = NULL;
    size_t final_bytes = 0;
    uint64_t final_hash = 0;
    if (state_hash(&m, state_pos + n_proposal, &final_hash, &final_bytes)) {
        fprintf(stderr, "verify4 final state export failed\n");
        goto fail;
    }
    printf("{\n  \"schema\":\"waste.gn100.verify4_diagnostic.v1\",\n");
    printf("  \"representative_diagnostic_only\":true,"
           "\"candidate_qualifier\":false,\"arm\":\"%s\",\n", argv[2]);
    printf("  \"prompt_tokens\":%d,\"root_tokens\":%d,"
           "\"proposal_tokens\":%d,\"state_position\":%d,\n",
           n_prompt, n_root, n_proposal, state_pos);
    printf("  \"proposals\":"); print_ints(proposal, n_proposal); puts(",");
    printf("  \"root\":{\"state_bytes\":%zu,"
           "\"state_hash\":\"0x%016" PRIx64 "\","
           "\"logit_hash\":\"0x%016" PRIx64 "\",\"argmax\":%d,"
           "\"first_proposal_matches\":%s,"
           "\"rollback_resident_during_verifier\":true},\n",
           root_bytes, root_hash, root_logit_hash, root_argmax,
           root_argmax == proposal[0] ? "true" : "false");
    printf("  \"timing\":{\"prompt_seconds\":%.9f,"
           "\"root_advance_seconds\":%.9f,\"snapshot_seconds\":%.9f,"
           "\"restore_seconds\":%.9f,\"verifier_seconds\":%.9f,"
           "\"positions_per_second\":%.9f},\n",
           prompt_seconds, root_advance_seconds, snapshot_seconds,
           restore_seconds, verifier_seconds,
           verifier_seconds > 0 ? n_proposal / verifier_seconds : 0);
    printf("  \"logit_row_hashes\":");
    print_u64_hex(logit_hashes, n_proposal); puts(",");
    printf("  \"row_argmax\":"); print_ints(row_argmax, n_proposal); puts(",");
    if (arm == ARM_SERIAL) {
        printf("  \"ordered_routes_available\":true,"
               "\"ordered_route_row_hashes\":");
        print_u64_hex(route_hashes, n_proposal); puts(",");
    } else {
        puts("  \"ordered_routes_available\":false,"
             "\"ordered_route_row_hashes\":null,");
    }
    printf("  \"final_state\":{\"bytes\":%zu,"
           "\"hash\":\"0x%016" PRIx64 "\"},\n", final_bytes, final_hash);
    printf("  \"cache_delta\":{\"hits\":%" PRIu64
           ",\"misses\":%" PRIu64 ",\"bytes\":%" PRIu64
           ",\"physical_expert_reads\":%" PRIu64
           ",\"chunk_expert_union\":%" PRIu64 "},\n",
           m.cache.hits - h0, m.cache.misses - mi0, m.cache.bytes_read - b0,
           m.expert_reads - er0, waste_model_chunk_expert_union(&m) - u0);
    printf("  \"cuda_delta\":{\"kda_calls\":%" PRIu64
           ",\"dense_calls\":%" PRIu64 ",\"vq_experts\":%" PRIu64
           ",\"vq_applies\":%" PRIu64 ",\"vq_launches\":%" PRIu64
           ",\"vq_syncs\":%" PRIu64 ",\"fallbacks\":%" PRIu64 "},\n",
           waste_model_cuda_kda_calls(&m) - k0,
           waste_model_cuda_dense_calls(&m) - d0,
           waste_model_cuda_vq_experts(&m) - ve0,
           waste_model_cuda_vq_applies(&m) - va0,
           waste_model_cuda_vq_launches(&m) - vl0,
           waste_model_cuda_vq_syncs(&m) - vs0,
           waste_model_cuda_kda_fallbacks(&m) - f0);
    print_process_safety("verifier_forward_plus_cache_drain",
                         ru1.ru_majflt - ru0.ru_majflt);
    printf("  \"exact_profile\":{\"root_i8mm\":0,\"measured_i8mm\":%d,"
           "\"prompt_path\":\"cpu_chunk\","
           "\"root_path\":\"canonical_ordinary_t1\","
           "\"verifier_path\":\"%s\","
           "\"cuda_requested\":{\"kda\":%d,\"dense\":%d,\"vq\":%d},"
           "\"cuda_effective\":{\"kda\":%d,\"dense\":%d,\"vq\":%d},"
           "\"vq_group\":%d,\"lookahead\":%d,"
           "\"direct_io\":%d,\"io_threads\":%d,\"io_depth\":%d,"
           "\"cache_slots\":%d,\"warm_ready_at_load\":%d}\n}\n",
           measured_i8mm,
           arm == ARM_SERIAL ? "ordinary_t1" : "cpu_chunk_all_rows",
           waste_model_get_cuda_kda(&m), waste_model_get_cuda_dense(&m),
           waste_model_get_cuda_vq(&m), waste_model_cuda_kda_effective(&m),
           waste_model_cuda_dense_effective(&m),
           waste_model_cuda_vq_effective(&m),
           waste_model_get_cuda_vq_group(&m), waste_model_get_lookahead(),
           m.direct_io, waste_ecache_io_threads(&m.cache),
           waste_ecache_io_depth(&m.cache), m.cache.n_slots, warm_ready);

    free(routed); free(rows); free(root_blob);
    free(prompt); free(root); free(proposal); waste_model_free(&m); return 0;

fail:
    free(routed); free(rows); free(root_blob);
    free(prompt); free(root); free(proposal); waste_model_free(&m); return 1;
fail_unloaded:
    free(prompt); free(root); free(proposal); return 1;
}
#else
static int verify4_mode(int argc, char **argv)
{
    if (argc != 9) return -2;
    (void)argv;
    fprintf(stderr, "verify4 requires WASTE_ENABLE_DIAGNOSTIC_VERIFY\n");
    return 1;
}
#endif

static uint64_t proc_value_kib(const char *path, const char *key)
{
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    char line[256];
    uint64_t value = 0;
    while (fgets(line, sizeof line, f)) {
        if (!strncmp(line, key, strlen(key))) {
            char *p = line + strlen(key);
            while (*p == ':' || *p == ' ' || *p == '\t') p++;
            value = strtoull(p, NULL, 10);
            break;
        }
    }
    fclose(f);
    return value;
}

static void print_process_safety(const char *scope, long timed_major_faults)
{
    printf("  \"process_safety\":{\"vmswap_kib\":%" PRIu64
           ",\"timed_major_faults_delta\":%ld,\"timed_scope\":\"%s\"},\n",
           proc_value_kib("/proc/self/status", "VmSwap"),
           timed_major_faults, scope);
}

static int load_mode(int argc, char **argv)
{
    if (argc != 10) return -2;
    const size_t target_rollback = exact_bytes(argv[8]);
    const size_t draft_rollback = exact_bytes(argv[9]);
    if (!target_rollback || !draft_rollback) return -2;
    waste_model target, draft;
    memset(&target, 0, sizeof target);
    memset(&draft, 0, sizeof draft);
    struct rusage ru0;
    memset(&ru0, 0, sizeof ru0);
    getrusage(RUSAGE_SELF, &ru0);
    const uint64_t avail_before =
        proc_value_kib("/proc/meminfo", "MemAvailable");
    if (load_model(&target, argv[2], argv[3], argv[4])) return 1;
    const uint64_t avail_after_target =
        proc_value_kib("/proc/meminfo", "MemAvailable");
    setenv("WASTE_CUDA_VQ", "0", 1);
    if (load_model(&draft, argv[5], argv[6], argv[7])) {
        waste_model_free(&target); return 1;
    }
    if (draft.cache.n_slots < routed_records(&draft)) {
        fprintf(stderr, "draft cache has %d slots for %d expert records\n",
                draft.cache.n_slots, routed_records(&draft));
        waste_model_free(&draft); waste_model_free(&target); return 1;
    }
    const uint64_t avail_after_draft =
        proc_value_kib("/proc/meminfo", "MemAvailable");
    void *tb = calloc(1, target_rollback);
    void *db = calloc(1, draft_rollback);
    if (!tb || !db) {
        fprintf(stderr, "rollback allocation failed\n");
        free(tb); free(db); waste_model_free(&draft); waste_model_free(&target);
        return 1;
    }
    size_t ts = 0, ds = 0;
    if (waste_model_state_size(&target, 0, &ts) ||
        waste_model_state_size(&draft, 0, &ds) ||
        !ts || !ds || ts > target_rollback || ds > draft_rollback) {
        fprintf(stderr, "rollback capacity is smaller than live root state\n");
        free(tb); free(db); waste_model_free(&draft); waste_model_free(&target);
        return 1;
    }
    /* Make both the safety snapshot and the timings describe resident
     * memory. The full rollback allocations remain touched for the final
     * co-residency gate, including capacity beyond the root-state payload. */
    touch_cache(&target.cache);
    touch_cache(&draft.cache);
    touch_blob(tb, target_rollback);
    touch_blob(db, draft_rollback);

    /* One untimed round trip removes first-touch and lazy-allocation work
     * from the explicitly in-memory serialization timings below. */
    size_t tw = 0, dw = 0;
    int tp = -1, dp = -1;
    if (waste_model_state_export(&target, 0, tb, target_rollback, &tw) ||
        waste_model_state_export(&draft, 0, db, draft_rollback, &dw) ||
        tw != ts || dw != ds ||
        waste_model_state_import(&target, tb, ts, &tp) ||
        waste_model_state_import(&draft, db, ds, &dp) || tp != 0 || dp != 0) {
        fprintf(stderr, "load-only root state round trip failed\n");
        free(tb); free(db); waste_model_free(&draft); waste_model_free(&target);
        return 1;
    }

    double started = now();
    tw = 0;
    const int target_export_rc =
        waste_model_state_export(&target, 0, tb, target_rollback, &tw);
    const double target_export_seconds = now() - started;
    started = now();
    dw = 0;
    const int draft_export_rc =
        waste_model_state_export(&draft, 0, db, draft_rollback, &dw);
    const double draft_export_seconds = now() - started;
    started = now();
    tp = -1;
    const int target_import_rc = waste_model_state_import(&target, tb, ts, &tp);
    const double target_import_seconds = now() - started;
    started = now();
    dp = -1;
    const int draft_import_rc = waste_model_state_import(&draft, db, ds, &dp);
    const double draft_import_seconds = now() - started;
    if (target_export_rc || draft_export_rc || tw != ts || dw != ds ||
        target_import_rc || draft_import_rc || tp != 0 || dp != 0) {
        fprintf(stderr, "timed in-memory state round trip failed\n");
        free(tb); free(db); waste_model_free(&draft); waste_model_free(&target);
        return 1;
    }

    /* This deliberately measures only an optimistic lower bound: copying
     * one already-contiguous, resident target snapshot to another resident
     * contiguous buffer. It is neither a pointer swap nor durable file I/O. */
    void *shadow = mmap(NULL, ts, PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (shadow == MAP_FAILED) shadow = NULL;
    shadow_result shadow_1, shadow_10;
    memset(&shadow_1, 0, sizeof shadow_1);
    memset(&shadow_10, 0, sizeof shadow_10);
    const uint64_t target_state_blob_hash = hash_one(tb, ts);
    if (!shadow || shadow_copy_floor(tb, shadow, ts, 1, &shadow_1) ||
        shadow_1.copied_hash != target_state_blob_hash ||
        shadow_copy_floor(tb, shadow, ts, 10, &shadow_10) ||
        shadow_10.copied_hash != target_state_blob_hash) {
        fprintf(stderr, "target contiguous shadow-copy benchmark failed\n");
        if (shadow) munmap(shadow, ts);
        free(tb); free(db);
        waste_model_free(&draft); waste_model_free(&target);
        return 1;
    }
    /* Do not let this second target-sized buffer inflate the memory gate.
     * The rollback buffers above remain allocated and resident. */
    if (munmap(shadow, ts)) {
        fprintf(stderr, "could not release target shadow-copy mapping\n");
        free(tb); free(db);
        waste_model_free(&draft); waste_model_free(&target);
        return 1;
    }

    struct rusage ru;
    memset(&ru, 0, sizeof ru);
    getrusage(RUSAGE_SELF, &ru);
    const uint64_t avail = proc_value_kib("/proc/meminfo", "MemAvailable");
    const uint64_t swap = proc_value_kib("/proc/self/status", "VmSwap");
    const uint64_t rss = proc_value_kib("/proc/self/status", "VmRSS");
    const uint64_t swap_total = proc_value_kib("/proc/meminfo", "SwapTotal");
    const uint64_t swap_free = proc_value_kib("/proc/meminfo", "SwapFree");
    printf("{\n  \"schema\":\"waste.gn100.spec_load.v1\",\n");
    printf("  \"target\":{\"cache_slots\":%d,\"record_bytes\":%zu,"
           "\"cache_bytes\":%zu,\"state0_bytes\":%zu,"
           "\"routed_records\":%d,\"warm_ready\":%d,"
           "\"direct_io\":%d,\"readers\":%d,\"depth\":%d,"
           "\"cuda_kda_requested\":%d,\"cuda_dense_requested\":%d,"
           "\"cuda_vq_requested\":%d},\n",
           target.cache.n_slots, target.cache.rec_bytes,
           (size_t)target.cache.n_slots * target.cache.rec_bytes, ts,
           routed_records(&target), cache_ready(&target.cache),
           target.direct_io, waste_ecache_io_threads(&target.cache),
           waste_ecache_io_depth(&target.cache),
           waste_model_get_cuda_kda(&target),
           waste_model_get_cuda_dense(&target),
           waste_model_get_cuda_vq(&target));
    printf("  \"draft\":{\"cache_slots\":%d,\"record_bytes\":%zu,"
           "\"cache_bytes\":%zu,\"state0_bytes\":%zu,"
           "\"routed_records\":%d,\"warm_ready\":%d,"
           "\"cache_covers_all_records\":%s,"
           "\"direct_io\":%d,\"readers\":%d,\"depth\":%d,"
           "\"cuda_kda_requested\":%d,\"cuda_dense_requested\":%d,"
           "\"cuda_vq_requested\":%d},\n",
           draft.cache.n_slots, draft.cache.rec_bytes,
           (size_t)draft.cache.n_slots * draft.cache.rec_bytes, ds,
           routed_records(&draft), cache_ready(&draft.cache),
           draft.cache.n_slots >= routed_records(&draft) ? "true" : "false",
           draft.direct_io, waste_ecache_io_threads(&draft.cache),
           waste_ecache_io_depth(&draft.cache),
           waste_model_get_cuda_kda(&draft),
           waste_model_get_cuda_dense(&draft),
           waste_model_get_cuda_vq(&draft));
    printf("  \"target_rollback_bytes\":%zu,\"draft_rollback_bytes\":%zu,\n",
           target_rollback, draft_rollback);
    printf("  \"in_memory_state_serialization\":{"
           "\"label\":\"warm in-memory state export/import; excludes durable file I/O\","
           "\"warmup_roundtrips\":1,\"rollback_pages_pretouched\":true,"
           "\"target\":{\"bytes\":%zu,\"export_seconds\":%.9f,"
           "\"import_seconds\":%.9f},"
           "\"draft\":{\"bytes\":%zu,\"export_seconds\":%.9f,"
           "\"import_seconds\":%.9f}},\n",
           ts, target_export_seconds, target_import_seconds,
           ds, draft_export_seconds, draft_import_seconds);
    printf("  \"target_shadow_copy_floor\":{"
           "\"label\":\"optimistic contiguous-memory shadow-copy floor; "
           "not pointer-swap or durable file timing\","
           "\"optimistic\":true,\"contiguous_buffers\":true,"
           "\"is_pointer_swap\":false,\"is_durable_file_io\":false,"
           "\"pages_pretouched\":true,\"warmup_copies\":1,"
           "\"thread_creation_timed\":false,\"worker_dispatch_timed\":true,"
           "\"throughput_basis\":\"logical state bytes per one-way copy; "
           "not aggregate read-plus-write bus traffic\","
           "\"repeats\":%d,\"bytes\":%zu,"
           "\"temporary_mapping_released_before_memory_snapshot\":true,"
           "\"verification_hash\":\"0x%016" PRIx64 "\","
           "\"threads_1\":{\"threads\":1,\"best_seconds\":%.9f,"
           "\"median_seconds\":%.9f,\"best_gib_s\":%.6f,"
           "\"median_gib_s\":%.6f},"
           "\"threads_10\":{\"threads\":10,\"best_seconds\":%.9f,"
           "\"median_seconds\":%.9f,\"best_gib_s\":%.6f,"
           "\"median_gib_s\":%.6f}},\n",
           SHADOW_REPEATS, ts, target_state_blob_hash,
           shadow_1.best_seconds, shadow_1.median_seconds,
           shadow_1.best_gib_s, shadow_1.median_gib_s,
           shadow_10.best_seconds, shadow_10.median_seconds,
           shadow_10.best_gib_s, shadow_10.median_gib_s);
    printf("  \"rss_kib\":%" PRIu64 ",\"memavailable_before_kib\":%" PRIu64
           ",\"memavailable_after_target_load_kib\":%" PRIu64
           ",\"memavailable_after_draft_load_kib\":%" PRIu64
           ",\"memavailable_after_touch_kib\":%" PRIu64 ",\n",
           rss, avail_before, avail_after_target, avail_after_draft, avail);
    printf("  \"vmswap_kib\":%" PRIu64 ",\"swap_total_kib\":%" PRIu64
           ",\"swap_free_kib\":%" PRIu64
           ",\"major_faults_delta\":%ld\n}\n",
           swap, swap_total, swap_free, ru.ru_majflt - ru0.ru_majflt);
    free(tb); free(db);
    waste_model_free(&draft); waste_model_free(&target); return 0;
}

static void usage(const char *name)
{
    fprintf(stderr,
        "usage:\n"
        "  %s prompt DRAFT_MODEL SYSTEM USER\n"
        "  %s target MODEL CACHE_MB USAGE|- PROMPT_IDS N_GEN\n"
        "  %s resident-target TARGET TARGET_USAGE|- DRAFT DRAFT_USAGE|- "
        "TARGET_PROMPT_IDS DRAFT_PROMPT_IDS N_GEN\n"
        "  %s teacher MODEL CACHE_MB USAGE|- PROMPT_IDS TARGET_IDS\n"
        "  %s state MODEL CACHE_MB USAGE|- PROMPT_IDS TARGET_IDS\n"
        "  %s verify4 serial|chunk0|chunk1 MODEL CACHE_MB USAGE|- "
        "PROMPT_IDS ROOT_IDS|- PROPOSAL_IDS\n"
        "  %s load TARGET TARGET_CACHE_MB USAGE|- DRAFT DRAFT_CACHE_MB "
        "USAGE|- TARGET_ROLLBACK_BYTES DRAFT_ROLLBACK_BYTES\n",
        name, name, name, name, name, name, name);
}

int main(int argc, char **argv)
{
    if (argc < 2) { usage(argv[0]); return 2; }
    int rc = -2;
    if (!strcmp(argv[1], "prompt")) rc = prompt_mode(argc, argv);
    else if (!strcmp(argv[1], "target")) rc = target_mode(argc, argv);
    else if (!strcmp(argv[1], "resident-target"))
        rc = resident_target_mode(argc, argv);
    else if (!strcmp(argv[1], "teacher")) rc = teacher_mode(argc, argv);
    else if (!strcmp(argv[1], "state")) rc = state_mode(argc, argv);
    else if (!strcmp(argv[1], "verify4")) rc = verify4_mode(argc, argv);
    else if (!strcmp(argv[1], "load")) rc = load_mode(argc, argv);
    if (rc == -2) { usage(argv[0]); return 2; }
    return rc;
}
