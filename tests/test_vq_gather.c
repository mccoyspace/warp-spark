/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/*
 * test_vq_gather.c — exact scalar/SVE expert-VQ comparison and optional
 * same-binary microbenchmark.
 *
 * Default mode is a fast correctness check.  The benchmark is deliberately
 * opt-in so the portable suite does not turn hardware timing into a test:
 *
 *   test_vq_gather --bench compare ITERATIONS THREADS
 *   test_vq_gather --bench cpu     ITERATIONS THREADS
 *   test_vq_gather --bench sve     ITERATIONS THREADS
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../src/simd.h"
#include "../src/threads.h"
#include "../src/waste_backend.h"

#if defined(WASTE_HAVE_SVE)

typedef struct {
    const char *name;
    int rows, nv, stages, entries;
    size_t idx_count, lut_count;
    uint8_t *idx;
    uint16_t *scale;
    float *lut, *cpu_y, *sve_y;
    vq_arg cpu_arg, sve_arg;
} fixture;

static uint64_t rng_state = 0x243f6a8885a308d3ULL;

static uint32_t rng32(void)
{
    rng_state = rng_state * 6364136223846793005ULL +
                1442695040888963407ULL;
    return (uint32_t)(rng_state >> 32);
}

static void *xmalloc(size_t n)
{
    void *p = malloc(n ? n : 1);
    if (!p) {
        fprintf(stderr, "allocation failed for %zu bytes\n", n);
        exit(1);
    }
    return p;
}

static void fixture_init(fixture *f, const char *name, int rows, int nv,
                         int stages, int entries)
{
    memset(f, 0, sizeof *f);
    f->name = name;
    f->rows = rows;
    f->nv = nv;
    f->stages = stages;
    f->entries = entries;

    const size_t blocks = ((size_t)rows + WASTE_VQ_TILE - 1) / WASTE_VQ_TILE;
    f->idx_count = blocks * (size_t)nv * WASTE_VQ_TILE * stages;
    f->lut_count = (size_t)nv * stages * entries;
    f->idx = (uint8_t *)xmalloc(f->idx_count);
    f->scale = (uint16_t *)xmalloc((size_t)rows * sizeof(uint16_t));
    f->lut = (float *)xmalloc(f->lut_count * sizeof(float));
    f->cpu_y = (float *)xmalloc((size_t)rows * sizeof(float));
    f->sve_y = (float *)xmalloc((size_t)rows * sizeof(float));

    /* The odd multiplier covers every byte value before repeating. */
    for (size_t i = 0; i < f->idx_count; i++)
        f->idx[i] = (uint8_t)(73u * (uint32_t)i + (rng32() & 0xffu));

    static const uint16_t scales[] = {
        0x3c00, /*  1.0       */
        0x3800, /*  0.5       */
        0xbc00, /* -1.0       */
        0x3555, /* ~0.333     */
        0x0400, /* min normal */
        0x0001, /* subnormal  */
    };
    for (int i = 0; i < rows; i++)
        f->scale[i] = scales[(unsigned)i % (sizeof scales / sizeof scales[0])];

    /* A broad but finite exponent range makes reassociation visible while
     * keeping even a complete K3-shaped accumulation far from overflow. */
    for (size_t i = 0; i < f->lut_count; i++) {
        const int32_t mant = (int32_t)(rng32() % 2000001u) - 1000000;
        const int shift = (int)(rng32() % 17u) - 8;
        f->lut[i] = (float)mant / 1000000.0f;
        if (shift >= 0) f->lut[i] *= (float)(1u << shift);
        else f->lut[i] /= (float)(1u << -shift);
    }

    memset(f->cpu_y, 0xa5, (size_t)rows * sizeof(float));
    memset(f->sve_y, 0x5a, (size_t)rows * sizeof(float));
    f->cpu_arg = (vq_arg){ f->cpu_y, f->idx, f->scale, f->lut,
                           nv, stages, entries };
    f->sve_arg = (vq_arg){ f->sve_y, f->idx, f->scale, f->lut,
                           nv, stages, entries };
}

static void fixture_free(fixture *f)
{
    free(f->idx);
    free(f->scale);
    free(f->lut);
    free(f->cpu_y);
    free(f->sve_y);
    memset(f, 0, sizeof *f);
}

static int check_case(const char *name, int rows, int nv, int stages,
                      int b, int e)
{
    fixture f;
    fixture_init(&f, name, rows, nv, stages, 256);

    /* Distinct sentinels make writes outside [b,e) independently visible;
     * equality between two equally wrong kernels is not enough. */
    for (int r = 0; r < rows; r++) {
        f.cpu_y[r] = 10000.0f + (float)r;
        f.sve_y[r] = f.cpu_y[r];
    }
    waste_vq_rows_cpu(b, e, &f.cpu_arg);
    waste_vq_rows_sve(b, e, &f.sve_arg);

    int ok = memcmp(f.cpu_y, f.sve_y, (size_t)rows * sizeof(float)) == 0;
    for (int r = 0; ok && r < rows; r++) {
        if ((r < b || r >= e) && f.cpu_y[r] != 10000.0f + (float)r)
            ok = 0;
    }
    if (!ok) {
        for (int r = 0; r < rows; r++) {
            if (memcmp(&f.cpu_y[r], &f.sve_y[r], sizeof(float)) != 0) {
                fprintf(stderr, "%s differs at row %d: cpu=%a sve=%a\n",
                        name, r, f.cpu_y[r], f.sve_y[r]);
                break;
            }
        }
    }
    fixture_free(&f);
    return ok;
}

static int correctness(void)
{
    if (!check_case("k3-gate-up", 3072, 3584 / 8, 3, 0, 3072)) return 0;
    if (!check_case("k3-down", 3584, 3072 / 8, 3, 0, 3584)) return 0;
    if (!check_case("sve-and-block-tails", 193, 17, 3, 0, 193)) return 0;
    if (!check_case("nonzero-range", 321, 13, 3, WASTE_VQ_RANGE, 321)) return 0;
    if (!check_case("vq2-fallback", 193, 17, 2, 0, 193)) return 0;

    /* Exercise the same runtime-selected function pointer and range splitting
     * as vq_apply(), not just direct calls to the two implementations. */
    fixture f;
    fixture_init(&f, "dispatch-and-parallel", 321, 13, 3, 256);
    waste_vq_rows_cpu(0, f.rows, &f.cpu_arg);
    waste_backend_init(WASTE_BE_NO_DEVICE);
    const char *forced = getenv("WASTE_BACKEND");
    const waste_range_fn expected = forced && strcmp(forced, "cpu") == 0
                                  ? waste_vq_rows_cpu : waste_vq_rows_sve;
    if (waste_k.vq_rows != expected) {
        fprintf(stderr, "runtime dispatch selected %s instead of %s VQ gather\n",
                waste_backend_name(), expected == waste_vq_rows_cpu ? "scalar" : "SVE");
        fixture_free(&f);
        return 0;
    }
    waste_pool_init(4);
    waste_parallel_for(f.rows, WASTE_VQ_RANGE, waste_k.vq_rows, &f.sve_arg);
    waste_pool_shutdown();
    const int ok = memcmp(f.cpu_y, f.sve_y,
                          (size_t)f.rows * sizeof(float)) == 0;
    if (!ok) fprintf(stderr, "runtime-dispatched parallel output differs\n");
    fixture_free(&f);
    if (!ok) return 0;
    return 1;
}

static double now(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec / 1e9;
}

static double one_call(waste_range_fn fn, fixture *f, int sve)
{
    const double t0 = now();
    waste_parallel_for(f->rows, WASTE_VQ_RANGE, fn,
                       sve ? (void *)&f->sve_arg : (void *)&f->cpu_arg);
    return now() - t0;
}

static int cmp_double(const void *ap, const void *bp)
{
    const double a = *(const double *)ap, b = *(const double *)bp;
    return (a > b) - (a < b);
}

static double median(const double *values, int n)
{
    double *v = (double *)xmalloc((size_t)n * sizeof(double));
    memcpy(v, values, (size_t)n * sizeof(double));
    qsort(v, (size_t)n, sizeof(double), cmp_double);
    const double m = (n & 1) ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) * 0.5;
    free(v);
    return m;
}

static uint64_t checksum(const float *v, int n)
{
    const uint8_t *p = (const uint8_t *)v;
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < (size_t)n * sizeof(float); i++) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

static int bench_fixture(fixture *f, const char *mode, int iterations)
{
    if (strcmp(mode, "compare") == 0) {
        /* Untimed exactness check and warmup. */
        waste_parallel_for(f->rows, WASTE_VQ_RANGE,
                           waste_vq_rows_cpu, &f->cpu_arg);
        waste_parallel_for(f->rows, WASTE_VQ_RANGE,
                           waste_vq_rows_sve, &f->sve_arg);
        if (memcmp(f->cpu_y, f->sve_y,
                   (size_t)f->rows * sizeof(float)) != 0) {
            fprintf(stderr, "%s warmup outputs differ\n", f->name);
            return 0;
        }
        double *cpu = (double *)xmalloc((size_t)iterations * sizeof(double));
        double *sve = (double *)xmalloc((size_t)iterations * sizeof(double));
        for (int i = 0; i < iterations; i++) {
            if ((i & 1) == 0) {
                cpu[i] = one_call(waste_vq_rows_cpu, f, 0);
                sve[i] = one_call(waste_vq_rows_sve, f, 1);
            } else {
                sve[i] = one_call(waste_vq_rows_sve, f, 1);
                cpu[i] = one_call(waste_vq_rows_cpu, f, 0);
            }
        }
        const double cm = median(cpu, iterations), sm = median(sve, iterations);
        printf("BENCH %s cpu_median=%.9f sve_median=%.9f ratio=%.6f "
               "reduction=%.2f%% checksum=%016llx\n",
               f->name, cm, sm, sm / cm, 100.0 * (1.0 - sm / cm),
               (unsigned long long)checksum(f->sve_y, f->rows));
        free(cpu);
        free(sve);
    } else {
        const int sve = strcmp(mode, "sve") == 0;
        const waste_range_fn fn = sve ? waste_vq_rows_sve : waste_vq_rows_cpu;
        /* Do not execute the opposite kernel in single-kernel modes.  Use
         * enough iterations that fixture setup is negligible to perf stat. */
        (void)one_call(fn, f, sve);
        double total = 0;
        for (int i = 0; i < iterations; i++) total += one_call(fn, f, sve);
        const float *y = sve ? f->sve_y : f->cpu_y;
        printf("BENCH %s mode=%s iterations=%d total=%.9f mean=%.9f "
               "checksum=%016llx\n",
               f->name, mode, iterations, total, total / iterations,
               (unsigned long long)checksum(y, f->rows));
    }
    return 1;
}

static int benchmark(const char *mode, int iterations, int threads)
{
    if (strcmp(mode, "compare") != 0 && strcmp(mode, "cpu") != 0 &&
        strcmp(mode, "sve") != 0) {
        fprintf(stderr, "benchmark mode must be compare, cpu, or sve\n");
        return 0;
    }
    if (iterations < 1 || iterations > 10000 || threads < 1 || threads > 64) {
        fprintf(stderr, "iterations must be 1..10000 and threads 1..64\n");
        return 0;
    }

    waste_pool_init(threads);
    fixture gate, down;
    fixture_init(&gate, "k3-gate-up", 3072, 3584 / 8, 3, 256);
    fixture_init(&down, "k3-down", 3584, 3072 / 8, 3, 256);
    const int ok = bench_fixture(&gate, mode, iterations) &&
                   bench_fixture(&down, mode, iterations);
    fixture_free(&gate);
    fixture_free(&down);
    waste_pool_shutdown();
    return ok;
}

int main(int argc, char **argv)
{
    if (!(waste_cpu_features() & WASTE_CPU_SVE)) {
        printf("SKIP SVE is compiled but unavailable at runtime\n");
        return 0;
    }
    if (argc == 1) {
        if (!correctness()) return 1;
        printf("PASS scalar and SVE VQ gathers are bit-identical\n");
        return 0;
    }
    if (argc != 5 || strcmp(argv[1], "--bench") != 0) {
        fprintf(stderr, "usage: %s [--bench compare|cpu|sve ITERATIONS THREADS]\n",
                argv[0]);
        return 2;
    }
    return benchmark(argv[2], atoi(argv[3]), atoi(argv[4])) ? 0 : 1;
}

#else

int main(void)
{
    printf("SKIP SVE is not built for this target\n");
    return 0;
}

#endif /* WASTE_HAVE_SVE */
