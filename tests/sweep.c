/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/*
 * sweep.c — one model load, many configurations, one machine state.
 *
 * A K3 measurement costs about 50 seconds of which 48 are model load and
 * prefill, and that is the smaller problem. The larger one is that every
 * heavy run changes the machine the next one lands on, so two arms measured
 * in two processes are measured on two computers: docs/LEARNED.md §32 and
 * §33 are both records of a conclusion that came out wrong for exactly that
 * reason, and §16's "sweep upward, never downward" exists because of it.
 *
 * So: load once, then run the arms back to back, interleaved, with the
 * expert cache cleared and the session reset between each. What is left
 * varying between two adjacent measurements is the setting and roughly
 * nothing else.
 *
 * This is an internal measurement harness, not the public open path: it
 * deliberately takes an expert-cache size rather than resolving a total
 * RAM budget, and it does not take the model ownership lock. Run it on an
 * exclusive host with an explicitly qualified WASTE_CACHE_MB. Set
 * WASTE_USAGE to restore the same learned hotlist before every arm.
 *
 * The one thing it still cannot vary is the context length, which sizes the
 * KV and KDA state at load.
 *
 * Set WASTE_CAPTURE_DIR to an existing directory to retain every full logit
 * row and routed expert id for tools/compare_gpu_runs.py. Capture is opt-in:
 * a 64-token K3 arm writes about 41 MiB of logits.
 *
 *   sweep CONTAINER ids,.. n_gen lookahead=0,6 [repeat]
 *   sweep CONTAINER ids,.. n_gen iodepth=2,4,8 [repeat]
 *   sweep CONTAINER ids,.. n_gen cache=3400,17736,23879 [repeat]
 *   sweep CONTAINER ids,.. n_gen cuda=0,1,2 [repeat]
 *   WASTE_CUDA_KDA=1 sweep CONTAINER ids,.. n_gen cuda_dense=0,1,2,3 [repeat]
 *
 * `cache` is in MB and re-makes the expert cache in place. The trunk is what
 * a load costs, not the cache, so a budget sweep no longer needs a process
 * per budget — which is what made §32 and §33 come out wrong, each process
 * meeting a machine the one before it had changed. The footprint at each arm
 * is what that budget would have made it: the same trunk plus the cache
 * being asked for.
 */
#include <inttypes.h>
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

static uint64_t hash_bytes(uint64_t h, const void *data, size_t n)
{
    const uint8_t *p = (const uint8_t *)data;
    for (size_t i = 0; i < n; i++) {
        h ^= p[i];
        h *= UINT64_C(1099511628211);
    }
    return h;
}

static uint64_t cuda_call_target(const waste_model *m, int n_gen)
{
    uint64_t calls = 0;
    for (int layer = 0; layer < m->cfg.n_layers; layer++)
        if (m->cfg.kda_layer[layer])
            calls += (uint64_t)(7 + (m->cfg.full_rank_gate ? 1 : 2)) *
                     (uint64_t)n_gen;
    return calls;
}

static uint64_t cuda_dense_call_target(const waste_model *m, int scope,
                                       int n_gen)
{
    uint64_t calls = 0;
    if (scope >= 1) {
        for (int layer = 0; layer < m->cfg.n_layers; layer++) {
            const char *prefix = m->cfg.prefix;
            char name[192];
            snprintf(name, sizeof name,
                     "%smodel.layers.%d.block_sparse_moe.gate.weight",
                     prefix, layer);
            if (!waste_find(m, name)) continue;
            calls += 3;                         /* shared gate/up/down */
            if (m->cfg.latent_dim) calls += 2; /* latent down/up */
        }
    }
    if (scope >= 2) {
        for (int layer = 0; layer < m->cfg.n_layers; layer++) {
            if (m->cfg.kda_layer[layer]) continue;
            calls += m->cfg.q_lora ? 2 : 1;    /* q_a/q_b or q */
            calls += 2;                         /* kv_a and o */
            if (m->cfg.mla_output_gate) calls++;
        }
    }
    if (scope >= 3) {
        for (int layer = 0; layer < m->cfg.n_layers; layer++) {
            char name[192];
            snprintf(name, sizeof name,
                     "%smodel.layers.%d.block_sparse_moe.gate.weight",
                     m->cfg.prefix, layer);
            if (!waste_find(m, name)) calls += 3;
        }
    }
    return calls * (uint64_t)n_gen;
}

static int write_capture(const char *dir, const char *key, int value, int rep,
                         const waste_model *m, int prompt_tokens, int n_gen,
                         const int *inputs, const int *routes,
                         const float *logits, double seconds,
                         uint64_t token_hash, uint64_t logit_hash,
                         uint64_t route_hash)
{
    char stem[96], manifest_path[1024], logits_path[1024], logits_name[128];
    snprintf(stem, sizeof stem, "%s-%d-rep%d", key, value, rep);
    snprintf(logits_name, sizeof logits_name, "%s.logits.f32", stem);
    snprintf(manifest_path, sizeof manifest_path, "%s/%s.json", dir, stem);
    snprintf(logits_path, sizeof logits_path, "%s/%s", dir, logits_name);
    FILE *raw = fopen(logits_path, "wb");
    const size_t rows = (size_t)n_gen + 1;
    const size_t count = rows * (size_t)m->cfg.vocab;
    if (!raw) {
        fprintf(stderr, "could not write capture logits %s\n", logits_path);
        return -1;
    }
    const int write_failed = fwrite(logits, sizeof *logits, count, raw) != count;
    const int close_failed = fclose(raw) != 0;
    if (write_failed || close_failed) {
        fprintf(stderr, "could not write capture logits %s\n", logits_path);
        return -1;
    }

    FILE *json = fopen(manifest_path, "w");
    if (!json) {
        fprintf(stderr, "could not write capture manifest %s\n", manifest_path);
        return -1;
    }
    const int dense_arm = !strcmp(key, "cuda_dense");
    const int effective = dense_arm ? waste_model_cuda_dense_effective(m)
                                    : waste_model_cuda_kda_effective(m);
    const uint64_t calls = dense_arm ? waste_model_cuda_dense_calls(m)
                                     : waste_model_cuda_kda_calls(m);
    const uint64_t expected = dense_arm
        ? cuda_dense_call_target(m, value, n_gen)
        : (value ? cuda_call_target(m, n_gen) : UINT64_C(0));
    const uint64_t kda_expected = waste_model_get_cuda_kda(m)
        ? cuda_call_target(m, n_gen) : UINT64_C(0);
    fprintf(json,
            "{\n  \"schema\": \"waste.gpu_capture.v1\",\n"
            "  \"dtype\": \"float32-le\",\n"
            "  \"logits_file\": \"%s\",\n"
            "  \"vocab\": %d,\n  \"top_k\": %d,\n  \"greedy\": true,\n"
            "  \"arm\": {\"key\": \"%s\", \"value\": %d, "
            "\"repeat\": %d, \"effective\": %d, \"fallbacks\": %" PRIu64
            ", \"calls\": %" PRIu64 ", \"expected_calls\": %" PRIu64
            ", \"kda_mode\": %d, \"kda_effective\": %d"
            ", \"kda_calls\": %" PRIu64
            ", \"kda_expected_calls\": %" PRIu64
            ", \"seconds\": %.9f, \"token_hash\": \"0x%016" PRIx64
            "\", \"logit_hash\": \"0x%016" PRIx64
            "\", \"route_hash\": \"0x%016" PRIx64 "\"},\n"
            "  \"steps\": [\n",
            logits_name, m->cfg.vocab, m->cfg.top_k, key, value, rep,
            effective, waste_model_cuda_kda_fallbacks(m), calls, expected,
            waste_model_get_cuda_kda(m), waste_model_cuda_kda_effective(m),
            waste_model_cuda_kda_calls(m), kda_expected, seconds,
            token_hash, logit_hash, route_hash);
    const int route_layers = m->cfg.n_layers - m->cfg.first_dense;
    const size_t route_stride = (size_t)route_layers * m->cfg.top_k;
    for (int step = 0; step <= n_gen; step++) {
        fprintf(json,
                "    {\"index\": %d, \"position\": %d, \"input_token\": ",
                step, prompt_tokens - 1 + step);
        if (step == 0) fputs("null", json);
        else fprintf(json, "%d", inputs[step - 1]);
        fputs(", \"routes\": [", json);
        if (step > 0) {
            const int *row = routes + (size_t)(step - 1) * route_stride;
            for (int layer = m->cfg.first_dense; layer < m->cfg.n_layers; layer++) {
                if (layer > m->cfg.first_dense) fputs(", ", json);
                fprintf(json, "{\"layer\": %d, \"experts\": [", layer);
                const int *ids = row + (size_t)(layer - m->cfg.first_dense) *
                                       m->cfg.top_k;
                for (int k = 0; k < m->cfg.top_k; k++) {
                    if (k) fputs(", ", json);
                    fprintf(json, "%d", ids[k]);
                }
                fputs("]}", json);
            }
        }
        fprintf(json, "]}%s\n", step == n_gen ? "" : ",");
    }
    fputs("  ]\n}\n", json);
    if (fclose(json)) {
        fprintf(stderr, "could not finish capture manifest %s\n", manifest_path);
        return -1;
    }
    printf("capture %s\n", manifest_path);
    return 0;
}

#define MAX_ARMS 16
#define MAX_IDS 512

int main(int argc, char **argv)
{
    if (argc < 5) {
        fprintf(stderr,
                "usage: %s CONTAINER ids,.. n_gen KEY=v1,v2,.. [repeat]\n"
                "  KEY is lookahead, iodepth, cache (MB), cuda, or cuda_dense\n",
                argv[0]);
        return 2;
    }
    int ids[MAX_IDS], n = 0;
    for (char *p = strtok(argv[2], ","); p && n < MAX_IDS; p = strtok(NULL, ","))
        ids[n++] = atoi(p);
    const int n_gen = atoi(argv[3]);
    if (n == 0 || n_gen <= 0) {
        fprintf(stderr, "prompt and n_gen must both be nonzero\n");
        return 2;
    }

    char *eq = strchr(argv[4], '=');
    if (!eq) { fprintf(stderr, "expected KEY=v1,v2,..\n"); return 2; }
    *eq = 0;
    const char *key = argv[4];
    int arm[MAX_ARMS], n_arms = 0;
    for (char *p = strtok(eq + 1, ","); p && n_arms < MAX_ARMS; p = strtok(NULL, ","))
        arm[n_arms++] = atoi(p);
    const int repeat = argc > 5 ? atoi(argv[5]) : 4;
    if (n_arms == 0 || repeat <= 0) {
        fprintf(stderr, "at least one arm and one repeat are required\n");
        return 2;
    }

    const int is_look = !strcmp(key, "lookahead");
    const int is_depth = !strcmp(key, "iodepth");
    const int is_cache = !strcmp(key, "cache");
    const int is_cuda = !strcmp(key, "cuda");
    const int is_dense = !strcmp(key, "cuda_dense");
    if (!is_look && !is_depth && !is_cache && !is_cuda && !is_dense) {
        fprintf(stderr, "unknown key %s\n", key);
        return 2;
    }

    waste_model m;
    waste_load_opts lo;
    memset(&lo, 0, sizeof lo);
    const char *cmb = getenv("WASTE_CACHE_MB");
    const unsigned long long cache_mb = cmb ? strtoull(cmb, NULL, 10) : 0;
    if ((is_look || is_depth || is_cuda || is_dense) && cache_mb == 0) {
        fprintf(stderr, "WASTE_CACHE_MB must be positive for %s sweeps\n", key);
        return 2;
    }
    if (cache_mb > (unsigned long long)(SIZE_MAX >> 20)) {
        fprintf(stderr, "WASTE_CACHE_MB is too large\n");
        return 2;
    }
    lo.cache_bytes = (size_t)cache_mb << 20;
    lo.direct_io = 1;
    double t0 = now();
    if (waste_model_load(&m, argv[1], 4096, &lo)) {
        fprintf(stderr, "load failed\n");
        return 1;
    }
    if (!m.direct_io) {
        fprintf(stderr, "direct I/O fell back on at least one expert bank\n");
        waste_model_free(&m);
        return 1;
    }
    if ((is_look || is_depth || is_cuda || is_dense) &&
        waste_ecache_io_threads(&m.cache) == 0) {
        fprintf(stderr, "%s sweep has no effective reader threads\n", key);
        waste_model_free(&m);
        return 1;
    }
    if (is_dense && !waste_model_get_cuda_kda(&m)) {
        fprintf(stderr,
                "cuda_dense sweep requires WASTE_CUDA_KDA=1 or 2\n");
        waste_model_free(&m);
        return 1;
    }
    const char *usage = getenv("WASTE_USAGE");
    if (usage && !*usage) usage = NULL;
    const char *capture_dir = getenv("WASTE_CAPTURE_DIR");
    if (capture_dir && !*capture_dir) capture_dir = NULL;
    float *capture_logits = NULL;
    int *capture_inputs = NULL, *capture_routes = NULL;
    const int route_layers = m.cfg.n_layers - m.cfg.first_dense;
    size_t route_stride = 0;
    if (capture_dir) {
        if (route_layers <= 0 || m.cfg.top_k <= 0 || m.cfg.vocab <= 0 ||
            (size_t)route_layers > SIZE_MAX / (size_t)m.cfg.top_k) {
            fprintf(stderr, "correctness capture requires a valid MoE layout\n");
            waste_model_free(&m);
            return 1;
        }
        route_stride = (size_t)route_layers * (size_t)m.cfg.top_k;
        const size_t logit_rows = (size_t)n_gen + 1;
        if (logit_rows > SIZE_MAX / (size_t)m.cfg.vocab ||
            (size_t)n_gen > SIZE_MAX / route_stride ||
            (size_t)n_gen * route_stride > SIZE_MAX / sizeof(int)) {
            fprintf(stderr, "correctness capture is too large\n");
            waste_model_free(&m);
            return 1;
        }
        const size_t logit_count = logit_rows * (size_t)m.cfg.vocab;
        if (logit_count > SIZE_MAX / sizeof(float) ||
            (size_t)n_gen > SIZE_MAX / sizeof(int) ||
            !(capture_logits = malloc(logit_count * sizeof(float))) ||
            !(capture_inputs = malloc((size_t)n_gen * sizeof(int))) ||
            !(capture_routes = malloc((size_t)n_gen * route_stride * sizeof(int)))) {
            fprintf(stderr, "could not allocate correctness capture\n");
            free(capture_logits); free(capture_inputs); free(capture_routes);
            waste_model_free(&m);
            return 1;
        }
    }
    const char *io_env = getenv("WASTE_IO_THREADS");
    const int requested_readers = io_env ? atoi(io_env) : 2;
    printf("loaded in %.1fs — cache %d slots, direct I/O %d, readers %d, "
           "depth %d; %d arms x %d repeats, %d prompt + %d generated%s%s\n\n",
           now() - t0, m.cache.n_slots, m.direct_io,
           waste_ecache_io_threads(&m.cache),
           waste_ecache_io_depth(&m.cache), n_arms, repeat, n, n_gen,
           usage ? "; usage " : "", usage ? usage : "");

    /* Interleaved rather than grouped: if the machine drifts over the run —
     * and it does — a grouped sweep charges the drift to the last arm and
     * an interleaved one spreads it across all of them. */
    printf("%10s %4s %7s %3s %3s %3s %4s %7s %6s %10s %11s %9s %9s %8s %14s "
           "%18s %18s %18s %4s %7s\n",
           key, "rep", "slots", "io", "qd", "eff", "fall", "calls", "warm",
           "seconds", "tok/s", "hits", "misses", "hit", "bytes",
           "token_hash", "logit_hash", "route_hash", "deff", "dcalls");
    const uint64_t expected_cuda_calls = is_cuda
        ? cuda_call_target(&m, n_gen) : UINT64_C(0);
    for (int r = 0; r < repeat; r++) {
        for (int a = 0; a < n_arms; a++) {
            const int ai = (r & 1) ? n_arms - 1 - a : a;
            int value = arm[ai];
            if (is_look) {
                waste_model_set_lookahead(value);
                value = waste_model_get_lookahead();
            } else if (is_depth) {
                value = value < 1 ? 1 : value;
                const int max_depth = m.cache.n_slots / 4;
                if (max_depth > 0 && value > max_depth) value = max_depth;
                m.cache.depth = value;
            } else if (is_cuda) {
                const int requested = value;
                if (waste_model_set_cuda_kda(&m, requested) ||
                    waste_model_get_cuda_kda(&m) != requested) {
                    fprintf(stderr, "cuda=%d is unavailable for this build/model\n",
                            requested);
                    waste_model_free(&m);
                    return 1;
                }
                value = requested;
            } else if (is_dense) {
                const int requested = value;
                if (waste_model_set_cuda_dense(&m, requested) ||
                    waste_model_get_cuda_dense(&m) != requested) {
                    fprintf(stderr,
                            "cuda_dense=%d is unavailable for this build/model\n",
                            requested);
                    waste_model_free(&m);
                    return 1;
                }
                value = requested;
            } else if (value < 0 ||
                       waste_model_resize_cache(&m, (size_t)value << 20)) {
                fprintf(stderr, "resize to %d MB failed\n", value);
                waste_model_free(&m);
                return 1;
            }
            if (is_cache && value > 0 && m.cache.n_slots == 0) {
                fprintf(stderr, "positive cache arm produced no usable slot\n");
                waste_model_free(&m);
                return 1;
            }
            if (requested_readers > 0 && m.cache.n_slots > 0 &&
                waste_ecache_io_threads(&m.cache) == 0) {
                fprintf(stderr, "%s=%d lost all requested reader threads\n",
                        key, value);
                waste_model_free(&m);
                return 1;
            }

            waste_model_reset(&m);
            waste_ecache_clear(&m.cache);
            int warmed = 0;
            if (usage) {
                warmed = waste_model_warm_cache(&m, usage);
                if (warmed < 0) {
                    fprintf(stderr, "could not restore usage hotlist %s\n", usage);
                    waste_model_free(&m);
                    return 1;
                }
            }

            /* CUDA is a decode-only experiment. Force even a one-token
             * prompt or a one-token final chunk through CPU so the timed
             * call target and both arms begin from the same prompt state. */
            const int decode_cuda_mode = m.cuda_kda_mode;
            const int decode_dense_scope = m.cuda_dense_scope;
            if (is_cuda || is_dense) {
                m.cuda_kda_mode = 0;
                m.cuda_dense_scope = 0;
            }
            const float *lg = NULL;
            for (int done = 0; done < n; ) {
                int k = n - done;
                const int cmax = waste_model_chunk_max(&m);
                if (k > cmax) k = cmax;
                lg = (k > 1) ? waste_model_prefill(&m, ids + done, k, done)
                             : waste_model_step(&m, ids[done], done, NULL);
                if (!lg) break;
                done += k;
            }
            if (is_cuda || is_dense) {
                m.cuda_kda_mode = decode_cuda_mode;
                m.cuda_dense_scope = decode_dense_scope;
            }
            if (!lg) {
                fprintf(stderr, "prompt failed\n");
                waste_model_free(&m);
                return 1;
            }

            int cur = 0;
            for (int v = 1; v < m.cfg.vocab; v++) if (lg[v] > lg[cur]) cur = v;
            if (capture_logits)
                memcpy(capture_logits, lg,
                       (size_t)m.cfg.vocab * sizeof *capture_logits);
            const uint64_t h0 = m.cache.hits, mi0 = m.cache.misses;
            const uint64_t b0 = m.cache.bytes_read;
            uint64_t token_hash = UINT64_C(14695981039346656037);
            uint64_t logit_hash = UINT64_C(14695981039346656037);
            uint64_t route_hash = UINT64_C(14695981039346656037);
            logit_hash = hash_bytes(logit_hash, lg,
                                    (size_t)m.cfg.vocab * sizeof *lg);
            int routed[WASTE_MAX_LAYERS * 64];
            const double s = now();
            for (int i = 0; i < n_gen; i++) {
                token_hash = hash_bytes(token_hash, &cur, sizeof cur);
                if (capture_inputs) capture_inputs[i] = cur;
                memset(routed, 0xff, sizeof routed);
                lg = waste_model_step(&m, cur, n + i, routed);
                if (!lg) {
                    fprintf(stderr, "step failed\n");
                    waste_model_free(&m);
                    return 1;
                }
                logit_hash = hash_bytes(logit_hash, lg,
                                        (size_t)m.cfg.vocab * sizeof *lg);
                if (capture_logits)
                    memcpy(capture_logits + (size_t)(i + 1) * m.cfg.vocab,
                           lg, (size_t)m.cfg.vocab * sizeof *lg);
                if (capture_routes)
                    memcpy(capture_routes + (size_t)i * route_stride,
                           routed + (size_t)m.cfg.first_dense * m.cfg.top_k,
                           route_stride * sizeof *routed);
                route_hash = hash_bytes(
                    route_hash,
                    routed + (size_t)m.cfg.first_dense * m.cfg.top_k,
                    (size_t)(m.cfg.n_layers - m.cfg.first_dense) *
                    m.cfg.top_k * sizeof *routed);
                cur = 0;
                for (int v = 1; v < m.cfg.vocab; v++) if (lg[v] > lg[cur]) cur = v;
            }
            waste_ecache_drain(&m.cache);
            const double dt = now() - s;
            const uint64_t h = m.cache.hits - h0;
            const uint64_t mi = m.cache.misses - mi0;
            const uint64_t bytes = m.cache.bytes_read - b0;
            const int cuda_effective = waste_model_cuda_kda_effective(&m);
            const uint64_t cuda_fallbacks =
                waste_model_cuda_kda_fallbacks(&m);
            const uint64_t cuda_calls = waste_model_cuda_kda_calls(&m);
            const int dense_effective = waste_model_cuda_dense_effective(&m);
            const uint64_t dense_calls = waste_model_cuda_dense_calls(&m);
            const uint64_t expected_kda_calls = decode_cuda_mode
                ? cuda_call_target(&m, n_gen) : UINT64_C(0);
            const uint64_t expected_dense_calls = cuda_dense_call_target(
                &m, decode_dense_scope, n_gen);
            printf("%10d %4d %7d %3d %3d %3d %4" PRIu64 " %7" PRIu64
                   " %6d %10.6f %11.6f %9" PRIu64
                   " %9" PRIu64 " %7.2f%% %14" PRIu64 " 0x%016" PRIx64
                   " 0x%016" PRIx64 " 0x%016" PRIx64 " %4d %7" PRIu64 "\n",
                   value, r + 1, m.cache.n_slots,
                   waste_ecache_io_threads(&m.cache),
                   waste_ecache_io_depth(&m.cache),
                   cuda_effective, cuda_fallbacks, cuda_calls, warmed,
                   dt, n_gen / dt,
                   h, mi, 100.0 * (double)h / (double)(h + mi ? h + mi : 1),
                   bytes, token_hash, logit_hash, route_hash,
                   dense_effective, dense_calls);
            fflush(stdout);
            if (capture_dir && write_capture(
                    capture_dir, key, value, r + 1, &m, n, n_gen,
                    capture_inputs, capture_routes, capture_logits, dt,
                    token_hash, logit_hash, route_hash)) {
                free(capture_logits); free(capture_inputs); free(capture_routes);
                waste_model_free(&m);
                return 1;
            }
            if (is_cuda &&
                ((value == 0 && (cuda_effective != 0 || cuda_fallbacks != 0 ||
                                 cuda_calls != 0)) ||
                 (value != 0 && (cuda_effective != value ||
                                 cuda_fallbacks != 0 ||
                                 cuda_calls != expected_cuda_calls)))) {
                fprintf(stderr,
                        "cuda=%d acceptance failed: effective=%d, "
                        "fallbacks=%" PRIu64 ", calls=%" PRIu64
                        " (expected=%" PRIu64 ")\n",
                        value, cuda_effective, cuda_fallbacks, cuda_calls,
                        value ? expected_cuda_calls : UINT64_C(0));
                free(capture_logits); free(capture_inputs); free(capture_routes);
                waste_model_free(&m);
                return 1;
            }
            if (is_dense &&
                (cuda_effective != decode_cuda_mode || cuda_fallbacks != 0 ||
                 cuda_calls != expected_kda_calls ||
                 (value == 0 && (dense_effective != 0 || dense_calls != 0)) ||
                 (value != 0 && (dense_effective != value ||
                                 dense_calls != expected_dense_calls)))) {
                fprintf(stderr,
                        "cuda_dense=%d acceptance failed: kda=%d/%d calls=%" PRIu64
                        "/%" PRIu64 ", dense=%d calls=%" PRIu64 "/%" PRIu64
                        ", fallbacks=%" PRIu64 "\n",
                        value, cuda_effective, decode_cuda_mode,
                        cuda_calls, expected_kda_calls,
                        dense_effective, dense_calls, expected_dense_calls,
                        cuda_fallbacks);
                free(capture_logits); free(capture_inputs); free(capture_routes);
                waste_model_free(&m);
                return 1;
            }
        }
    }
    free(capture_logits); free(capture_inputs); free(capture_routes);
    waste_model_free(&m);
    return 0;
}
