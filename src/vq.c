/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/*
 * vq.c — residual vector quantization encoder.
 *
 * Assigning 8-dim vectors to a 256-entry codebook is trivially parallel and
 * entirely compute-bound *if* the distance matrix never exists. In torch it
 * does: [n, 256] floats get written and read back for the argmin, which for
 * one expert matrix is ~1.4 GB of traffic per stage and pins the throughput
 * at a fraction of a percent of peak.
 *
 * Here each vector's 256 distances live in registers, the codebook (8 KB)
 * stays in L1, and only the winning byte is written. All `stages` are done
 * in one pass so the vector is loaded once.
 *
 * Called from tools/convert.py through ctypes.
 */

#include <math.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>

#include "threads.h"

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#endif

/* The encoder has no waste_cfg to read, so WASTE_CPUS is the only way in.
 * A list this platform cannot bind, or cannot parse, leaves placement to
 * the OS — the strict refusal belongs at the engine's front door, where a
 * caller asked for it explicitly and can be told. */
static void vq_pool_init(int nthreads)
{
    waste_cpumask cpus;
    const int cr = waste_cpus_resolve(NULL, &cpus);
    waste_pool_init(nthreads, cr == WASTE_CPUS_OK ? &cpus : NULL);
}

typedef struct {
    const float *X;        /* [n][dim]                                     */
    const float *books;    /* [stages][entries][dim]                       */
    uint8_t *out;          /* [n][stages]                                  */
    int dim, entries, stages;
} vq_job;

/* Nearest entry to `v` under squared Euclidean distance, and subtract it. */
static inline int vq_step(const float *v, const float *C, int entries, int dim,
                          float *resid)
{
    int best = 0;
    float bestd = INFINITY;

#if defined(__ARM_NEON) || defined(__aarch64__)
    if (dim == 8) {
        const float32x4_t v0 = vld1q_f32(v), v1 = vld1q_f32(v + 4);
        for (int c = 0; c < entries; c++) {
            const float *e = C + (size_t)c * 8;
            const float32x4_t d0 = vsubq_f32(v0, vld1q_f32(e));
            const float32x4_t d1 = vsubq_f32(v1, vld1q_f32(e + 4));
            const float d = vaddvq_f32(vfmaq_f32(vmulq_f32(d0, d0), d1, d1));
            if (d < bestd) { bestd = d; best = c; }
        }
        const float *e = C + (size_t)best * 8;
        vst1q_f32(resid, vsubq_f32(v0, vld1q_f32(e)));
        vst1q_f32(resid + 4, vsubq_f32(v1, vld1q_f32(e + 4)));
        return best;
    }
#endif
    for (int c = 0; c < entries; c++) {
        const float *e = C + (size_t)c * dim;
        float d = 0;
        for (int j = 0; j < dim; j++) { const float t = v[j] - e[j]; d += t * t; }
        if (d < bestd) { bestd = d; best = c; }
    }
    for (int j = 0; j < dim; j++) resid[j] = v[j] - C[(size_t)best * dim + j];
    return best;
}

static void vq_range(int b, int e, void *p)
{
    const vq_job *j = (const vq_job *)p;
    float cur[64], nxt[64];
    for (int i = b; i < e; i++) {
        const float *v = j->X + (size_t)i * j->dim;
        for (int d = 0; d < j->dim; d++) cur[d] = v[d];
        for (int s = 0; s < j->stages; s++) {
            const float *C = j->books + (size_t)s * j->entries * j->dim;
            j->out[(size_t)i * j->stages + s] =
                (uint8_t)vq_step(cur, C, j->entries, j->dim, nxt);
            for (int d = 0; d < j->dim; d++) cur[d] = nxt[d];
        }
    }
}

/* n vectors of `dim`, `stages` codebooks of `entries` each -> n*stages bytes */
void waste_vq_encode(const float *X, int n, const float *books, int stages,
                     int entries, int dim, uint8_t *out, int nthreads)
{
    vq_pool_init(nthreads);
    vq_job j = { X, books, out, dim, entries, stages };
    waste_parallel_for(n, 4096, vq_range, &j);
}

/* One Lloyd iteration: assign, then recompute centroids. Returns the number
 * of non-empty clusters. Used to fit the codebooks. */
int waste_vq_lloyd(const float *X, int n, float *C, int entries, int dim,
                   int nthreads)
{
    uint8_t *asg = (uint8_t *)malloc((size_t)n);
    if (!asg) return -1;
    vq_pool_init(nthreads);
    vq_job j = { X, C, asg, dim, entries, 1 };
    waste_parallel_for(n, 4096, vq_range, &j);

    double *sum = (double *)calloc((size_t)entries * dim, sizeof(double));
    long *cnt = (long *)calloc((size_t)entries, sizeof(long));
    if (!sum || !cnt) { free(asg); free(sum); free(cnt); return -1; }
    for (int i = 0; i < n; i++) {
        const int a = asg[i];
        cnt[a]++;
        for (int d = 0; d < dim; d++) sum[(size_t)a * dim + d] += X[(size_t)i * dim + d];
    }
    int live = 0;
    for (int c = 0; c < entries; c++) {
        if (cnt[c]) {
            live++;
            for (int d = 0; d < dim; d++)
                C[(size_t)c * dim + d] = (float)(sum[(size_t)c * dim + d] / (double)cnt[c]);
        }
    }
    free(asg); free(sum); free(cnt);
    return live;
}
