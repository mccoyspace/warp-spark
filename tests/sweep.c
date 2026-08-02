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
 *   sweep CONTAINER ids,.. n_gen lookahead=0,6 [repeat]
 *   sweep CONTAINER ids,.. n_gen iodepth=2,4,8 [repeat]
 *   sweep CONTAINER ids,.. n_gen cache=3400,17736,23879 [repeat]
 *   sweep CONTAINER ids,.. n_gen cuda=0,1,2 [repeat]
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

#define MAX_ARMS 16
#define MAX_IDS 512

int main(int argc, char **argv)
{
    if (argc < 5) {
        fprintf(stderr,
                "usage: %s CONTAINER ids,.. n_gen KEY=v1,v2,.. [repeat]\n"
                "  KEY is lookahead, iodepth, cache (MB), or cuda\n", argv[0]);
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
    if (!is_look && !is_depth && !is_cache && !is_cuda) {
        fprintf(stderr, "unknown key %s\n", key);
        return 2;
    }

    waste_model m;
    waste_load_opts lo;
    memset(&lo, 0, sizeof lo);
    const char *cmb = getenv("WASTE_CACHE_MB");
    const unsigned long long cache_mb = cmb ? strtoull(cmb, NULL, 10) : 0;
    if ((is_look || is_depth || is_cuda) && cache_mb == 0) {
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
    if ((is_look || is_depth || is_cuda) &&
        waste_ecache_io_threads(&m.cache) == 0) {
        fprintf(stderr, "%s sweep has no effective reader threads\n", key);
        waste_model_free(&m);
        return 1;
    }
    const char *usage = getenv("WASTE_USAGE");
    if (usage && !*usage) usage = NULL;
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
    printf("%8s %4s %7s %3s %3s %3s %4s %6s %10s %11s %9s %9s %8s %14s "
           "%18s %18s %18s\n",
           key, "rep", "slots", "io", "qd", "eff", "fall", "warm",
           "seconds", "tok/s", "hits", "misses", "hit", "bytes",
           "token_hash", "logit_hash", "route_hash");
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
                waste_model_set_cuda_kda(&m, value);
                value = waste_model_get_cuda_kda(&m);
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
            if (!lg) {
                fprintf(stderr, "prompt failed\n");
                waste_model_free(&m);
                return 1;
            }

            int cur = 0;
            for (int v = 1; v < m.cfg.vocab; v++) if (lg[v] > lg[cur]) cur = v;
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
                memset(routed, 0xff, sizeof routed);
                lg = waste_model_step(&m, cur, n + i, routed);
                if (!lg) {
                    fprintf(stderr, "step failed\n");
                    waste_model_free(&m);
                    return 1;
                }
                logit_hash = hash_bytes(logit_hash, lg,
                                        (size_t)m.cfg.vocab * sizeof *lg);
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
            printf("%8d %4d %7d %3d %3d %3d %4" PRIu64
                   " %6d %10.6f %11.6f %9" PRIu64
                   " %9" PRIu64 " %7.2f%% %14" PRIu64 " 0x%016" PRIx64
                   " 0x%016" PRIx64 " 0x%016" PRIx64 "\n",
                   value, r + 1, m.cache.n_slots,
                   waste_ecache_io_threads(&m.cache),
                   waste_ecache_io_depth(&m.cache),
                   waste_model_cuda_kda_effective(&m),
                   waste_model_cuda_kda_fallbacks(&m),
                   warmed, dt, n_gen / dt,
                   h, mi, 100.0 * (double)h / (double)(h + mi ? h + mi : 1),
                   bytes, token_hash, logit_hash, route_hash);
            fflush(stdout);
        }
    }
    waste_model_free(&m);
    return 0;
}
