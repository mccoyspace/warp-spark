/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/*
 * test_forward.c — run the C forward pass and dump logits for the oracle diff.
 *
 *   cc -O2 -fopenmp -o test_forward tests/test_forward.c src/model.c src/kda.c \
 *      src/kda_neon.c src/backend.c -lm
 *   ./test_forward model.waste 1008,10484,318,15383,387 out.bin
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

#include "../src/model.h"
#include "../src/waste_backend.h"
#include "../src/waste.h"

static double now(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec / 1e9;
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "usage: %s container ids[,..] [out.bin] [n_gen]\n", argv[0]);
        return 2;
    }
    const char *dir = argv[1];
    int ids[512], n = 0;
    for (char *p = strtok(argv[2], ","); p && n < 512; p = strtok(NULL, ","))
        ids[n++] = atoi(p);
    const char *out = argc > 3 ? argv[3] : NULL;
    const int n_gen = argc > 4 ? atoi(argv[4]) : 0;

    waste_model m;
    double t0 = now();
    const char *cmb = getenv("WASTE_CACHE_MB");
    const size_t cache_bytes = (size_t)(cmb ? atoi(cmb) : 0) << 20;
    waste_load_opts lo;
    memset(&lo, 0, sizeof lo);
    lo.cache_bytes = cache_bytes;
    lo.direct_io = 1;
    if (waste_model_load(&m, dir, 4096, &lo)) { fprintf(stderr, "load failed\n"); return 1; }
    printf("%s\n", waste_build_info());
    printf("loaded in %.1fs — %d layers, %d experts, top-%d, vocab %d; "
           "expert cache %d slots (%.0f MB, %.1f%% of the expert set)\n",
           now() - t0, m.cfg.n_layers, m.cfg.n_experts, m.cfg.top_k, m.cfg.vocab,
           m.cache.n_slots, (double)(m.cache.n_slots * m.cache.rec_bytes) / 1048576.0,
           100.0 * m.cache.n_slots / (double)(m.cfg.n_experts * 26));

    const float *lg = NULL;
    const int chunked = getenv("WASTE_CHUNK") && atoi(getenv("WASTE_CHUNK")) != 0;
    t0 = now();
    if (chunked) {
        int done = 0;
        while (done < n) {
            int c = n - done;
            if (c > waste_model_chunk_max(&m)) c = waste_model_chunk_max(&m);
            lg = waste_model_prefill(&m, ids + done, c, done);
            done += c;
        }
    } else {
        for (int i = 0; i < n; i++) lg = waste_model_step(&m, ids[i], i, NULL);
    }
    const double tp = now() - t0;

    /* NULL means an expert record did not survive the read: a short read,
     * a header that is not the one the bank index describes, or a payload
     * that is not what the converter checksummed. Say which record, and
     * stop — there are no logits to look at. */
    if (!lg) {
        int layer = 0, expert = 0;
        const char *why = waste_model_read_error(&m, &layer, &expert);
        fprintf(stderr, "expert %d of layer %d: %s\n", expert, layer,
                why ? why : "read failed");
        waste_model_free(&m);
        return 1;
    }

    int best = 0;
    for (int v = 1; v < m.cfg.vocab; v++) if (lg[v] > lg[best]) best = v;
    printf("prefill %d tok in %.2fs (%.2f tok/s); argmax %d, max %.4f\n",
           n, tp, n / tp, best, lg[best]);

    if (out) {
        FILE *f = fopen(out, "wb");
        fwrite(lg, sizeof(float), (size_t)m.cfg.vocab, f);
        fclose(f);
        printf("wrote %s\n", out);
    }

    int cur = best;
    for (int i = 0; i < n_gen; i++) {
        t0 = now();
        lg = waste_model_step(&m, cur, n + i, NULL);
        if (!lg) {
            int layer = 0, expert = 0;
            const char *why = waste_model_read_error(&m, &layer, &expert);
            fprintf(stderr, "expert %d of layer %d: %s\n", expert, layer,
                    why ? why : "read failed");
            waste_model_free(&m);
            return 1;
        }
        best = 0;
        for (int v = 1; v < m.cfg.vocab; v++) if (lg[v] > lg[best]) best = v;
        printf("  [%3d] %6d  (%.2fs, %llu expert reads)\n", i, best, now() - t0,
               (unsigned long long)m.expert_reads);
        cur = best;
    }

    printf("cuda kda: requested %d, effective %d, fallbacks %llu, calls %llu\n",
           waste_model_get_cuda_kda(&m),
           waste_model_cuda_kda_effective(&m),
           (unsigned long long)waste_model_cuda_kda_fallbacks(&m),
           (unsigned long long)waste_model_cuda_kda_calls(&m));

    extern double waste_prof[16];
    if (getenv("WASTE_PROFILE")) {
        /* indented names are sub-totals of the line above and are excluded
         * from `tot`, so the percentages add to 100 */
        const char *names[16] = {"  LUT build","kda","mla","moe(all)",
                                 "  expert I/O","  expert mm","lm_head",
                                 "  LUT apply","  batched mm",
                                 "  kda recurrence","  kda qkv projections",
                                 "  kda short conv","  kda auxiliaries",
                                 "  kda output gate","  kda output norm",
                                 "  kda output projection"};
        double tot = 0;
        for (int i = 0; i < 16; i++)
            tot += (i == 0 || i == 4 || i == 5 || i == 7 || i >= 8)
                 ? 0 : waste_prof[i];
        printf("\n-- profile (s, %d steps) --\n", n + n_gen);
        for (int i = 0; i < 16; i++)
            if (waste_prof[i] > 0)
                printf("  %-14s %7.2f  %5.1f%%\n", names[i], waste_prof[i],
                       100.0 * waste_prof[i] / tot);
        printf("  %-14s %7.2f\n", "accounted", tot);
    }
    printf("\ncache: %llu hits / %llu misses = %.1f%% hit, %llu evictions, "
           "%.2f GB read\n",
           (unsigned long long)m.cache.hits, (unsigned long long)m.cache.misses,
           100.0 * waste_ecache_hit_rate(&m.cache),
           (unsigned long long)m.cache.evictions,
           (double)m.cache.bytes_read / 1073741824.0);
    waste_model_free(&m);
    return 0;
}
