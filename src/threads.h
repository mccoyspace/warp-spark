/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/*
 * threads.h — minimal fork-join pool for row-parallel kernels.
 *
 * No OpenMP: Apple's clang does not ship it, and a build dependency for
 * one parallel-for is not worth it. Workers are created once and parked on
 * a condvar; waste_parallel_for splits [0,n) across them and blocks until
 * all have finished. Splitting is by row, so results are bit-identical to
 * the serial version regardless of thread count.
 *
 * Windows needs nothing here: MinGW implements pthreads over winpthreads,
 * so the pool below compiles and runs unchanged. Only the CPU count was
 * POSIX-only, and that now comes from platform.h.
 *
 * Placement is the OS's unless a caller names a cpuset. It is worth naming
 * on a machine whose cores are not interchangeable: on a two-CCD Ryzen,
 * six threads on one CCD measured 16-25% faster than the same six split
 * across both, at identical bytes read (issue #23), because the split pair
 * reaches the other die's L3 over Infinity Fabric and the barrier waits
 * for whoever is slowest. LEARNED §47 is the same shape with P-cores and
 * E-cores instead of dies. Neither is a default — §47 measured "cap the
 * pool" as a 25% win on one model and a 34% loss on another — so this is
 * a switch, and the engine still chooses nothing on its own.
 */

#ifndef WASTE_THREADS_H
#define WASTE_THREADS_H

#include <pthread.h>
#include <stdlib.h>
#include <string.h>

#include "platform.h"

typedef void (*waste_range_fn)(int begin, int end, void *arg);

typedef struct {
    pthread_t th[64];
    pthread_mutex_t mu;
    pthread_cond_t start, done;
    waste_range_fn fn;
    void *arg;
    int n, nthreads, next_chunk, chunk, active, epoch, stop;
    /* The cpuset every participant binds to, and a generation so a thread
     * knows whether it has already done so. 0 = placement is the OS's. */
    waste_cpumask cpus;
    int aff_gen;
} waste_pool;

static waste_pool g_pool;
/* The pool is shared by every model in this translation unit.  One lock
 * makes first-use initialization deterministic; the other owns a complete
 * submitted job, because the fields in waste_pool are the job descriptor
 * itself and cannot represent two callers at once. */
static pthread_mutex_t g_pool_init_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t g_pool_run_mu = PTHREAD_MUTEX_INITIALIZER;

/* Outside the pool because waste_pool_shutdown memsets the pool and the
 * per-thread "already bound" marks survive it: a second pool that reused
 * generation 1 would be skipped by every thread the first one bound, and
 * silently keep the old cpuset. Counting up costs nothing and cannot. */
static int g_aff_gen;

/* Bind this thread to the pool's cpuset, at most once per thread.
 *
 * The pool's own threads do it when they start. The calling thread does it
 * on its first parallel region rather than at pool init, because it is a
 * worker too (see the tail of waste_parallel_for) and because it is the
 * host's thread, not the engine's: restricting it when it is handed to a
 * kernel is defensible, restricting it because a model was opened on it is
 * not. It is also the only way this works in a server, where the thread
 * that opens the model is rarely the thread that decodes on it.
 *
 * Leaving the caller out was measured as pointless: with six threads it is
 * one of the six, so an unbound caller on the far die is exactly the
 * straggler the option exists to remove. */
static inline void waste__bind_self(void)
{
    static _Thread_local int seen;      /* per thread, per translation unit */
    const int gen = g_pool.aff_gen;
    if (!gen || seen == gen) return;
    seen = gen;
    waste_bind_thread_cpus(&g_pool.cpus);
}

static void *waste__worker(void *p)
{
    waste_pool *P = (waste_pool *)p;
    int seen = 0;
    waste__bind_self();
    for (;;) {
        pthread_mutex_lock(&P->mu);
        while (!P->stop && P->epoch == seen) pthread_cond_wait(&P->start, &P->mu);
        if (P->stop) { pthread_mutex_unlock(&P->mu); return NULL; }
        seen = P->epoch;
        pthread_mutex_unlock(&P->mu);

        for (;;) {
            pthread_mutex_lock(&P->mu);
            const int b = P->next_chunk;
            if (b >= P->n) { pthread_mutex_unlock(&P->mu); break; }
            P->next_chunk = b + P->chunk;
            pthread_mutex_unlock(&P->mu);
            int e = b + P->chunk;
            if (e > P->n) e = P->n;
            P->fn(b, e, P->arg);
        }

        pthread_mutex_lock(&P->mu);
        if (--P->active == 0) pthread_cond_signal(&P->done);
        pthread_mutex_unlock(&P->mu);
    }
}

static inline int waste_hw_threads(void) { return waste_cpu_count(); }

/* How many the pool ended up with, which is not always what was asked for:
 * a cpuset sizes it, the cap trims it, and a pthread_create that failed
 * leaves it short. 0 before the first waste_pool_init. */
static inline int waste_pool_threads(void) { return g_pool.nthreads; }

/* nthreads 0 = one per CPU the pool may use; cpus NULL = wherever the OS
 * puts them. Both are decided by the first caller in the process: the pool
 * is shared, and a second model opened with different wishes gets the
 * first model's pool. */
static inline void waste_pool_init(int nthreads, const waste_cpumask *cpus)
{
    pthread_mutex_lock(&g_pool_init_mu);
    if (g_pool.nthreads) { pthread_mutex_unlock(&g_pool_init_mu); return; }
    if (cpus) {
        g_pool.cpus = *cpus;
        g_pool.aff_gen = ++g_aff_gen;
        /* A cpuset without a thread count means the cpuset: 24 threads
         * over the 6 CPUs a user just asked for is not what they asked
         * for, and it is worse than either half of the choice. */
        if (nthreads <= 0) nthreads = waste_cpumask_count(cpus);
    }
    if (nthreads <= 0) nthreads = waste_hw_threads();
    if (nthreads > 64) nthreads = 64;
    if (nthreads < 1) nthreads = 1;
    pthread_mutex_init(&g_pool.mu, NULL);
    pthread_cond_init(&g_pool.start, NULL);
    pthread_cond_init(&g_pool.done, NULL);
    /* Count only workers that actually exist.  Waiting for a failed
     * pthread_create as though it were alive otherwise hangs forever. */
    g_pool.nthreads = 1;
    for (int i = 0; i < nthreads - 1; i++) {
        if (pthread_create(&g_pool.th[i], NULL, waste__worker, &g_pool)) break;
        g_pool.nthreads++;
    }
    pthread_mutex_unlock(&g_pool_init_mu);
}

/* Runs fn over [0,n) split into chunks; every chunk boundary is a multiple
 * of min_chunk. Blocks until complete. */
static inline void waste_parallel_for(int n, int min_chunk, waste_range_fn fn, void *arg)
{
    /* Before the serial path too: a pool of one still runs the kernel, and
     * on the CPUs it was told to. A thread-local check, so this costs a
     * load per dispatch and a syscall once per thread. */
    waste__bind_self();
    if (g_pool.nthreads <= 1 || n <= min_chunk) { fn(0, n, arg); return; }
    int chunk = (n + g_pool.nthreads - 1) / g_pool.nthreads;
    if (chunk < min_chunk) chunk = min_chunk;
    /* Round up to a whole number of min_chunk units: callers that block
     * their data (the VQ tile) need every range to start on a boundary. */
    chunk = ((chunk + min_chunk - 1) / min_chunk) * min_chunk;

    /* Keep the descriptor stable until every worker has left this job.
     * Distinct waste_ctx instances may be called concurrently even though
     * they reuse this process-wide pool. */
    pthread_mutex_lock(&g_pool_run_mu);
    pthread_mutex_lock(&g_pool.mu);
    g_pool.fn = fn; g_pool.arg = arg; g_pool.n = n; g_pool.chunk = chunk;
    g_pool.next_chunk = 0;
    g_pool.active = g_pool.nthreads - 1;
    g_pool.epoch++;
    pthread_cond_broadcast(&g_pool.start);
    pthread_mutex_unlock(&g_pool.mu);

    /* the calling thread is a worker too */
    for (;;) {
        pthread_mutex_lock(&g_pool.mu);
        const int b = g_pool.next_chunk;
        if (b >= n) { pthread_mutex_unlock(&g_pool.mu); break; }
        g_pool.next_chunk = b + chunk;
        pthread_mutex_unlock(&g_pool.mu);
        int e = b + chunk;
        if (e > n) e = n;
        fn(b, e, arg);
    }

    pthread_mutex_lock(&g_pool.mu);
    while (g_pool.active > 0) pthread_cond_wait(&g_pool.done, &g_pool.mu);
    pthread_mutex_unlock(&g_pool.mu);
    pthread_mutex_unlock(&g_pool_run_mu);
}

static inline void waste_pool_shutdown(void)
{
    if (!g_pool.nthreads) return;
    pthread_mutex_lock(&g_pool.mu);
    g_pool.stop = 1;
    pthread_cond_broadcast(&g_pool.start);
    pthread_mutex_unlock(&g_pool.mu);
    for (int i = 0; i < g_pool.nthreads - 1; i++) pthread_join(g_pool.th[i], NULL);
    memset(&g_pool, 0, sizeof g_pool);
}

#endif /* WASTE_THREADS_H */
