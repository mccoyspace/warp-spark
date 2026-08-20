/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/* The cpu list: parsing it, and the pool actually running on it.
 *
 * Two checks with different reaches, which is why they are two arguments
 * rather than one binary:
 *
 *   parse — the string to the mask. Pure arithmetic, so it runs on every
 *           host the suite does, including the ones that cannot bind a
 *           thread at all. The cases that matter are the malformed ones:
 *           a cpu list is a performance option, and a typo that parses to
 *           a smaller set than the user meant is indistinguishable from
 *           the option not helping.
 *
 *   bind  — the pool's threads, and the calling thread, really are
 *           restricted to what was asked for. Only where the platform has
 *           the call (exits 77 = SKIP otherwise) and only where the mask
 *           can be read back, because a check that cannot observe the
 *           result is not a check.
 *
 * The CPUs used are taken from the ones this process is already allowed,
 * never assumed to be 0 and 1: the suite runs inside containers whose
 * cpuset is somebody else's decision, and binding to a CPU outside it
 * fails in a way that would read as a bug here.
 */

#include "../src/threads.h"

#include <stdio.h>
#include <string.h>

#if defined(__linux__)
#include <errno.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>
#define HAVE_READBACK 1
#else
#define HAVE_READBACK 0
#endif

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "check failed at line %d: %s\n", __LINE__, #x); \
    return 1; \
} } while (0)

/* ---- parse -------------------------------------------------------------- */

static int one(const char *s, int want_count, const char *want_set)
{
    waste_cpumask m;
    const int n = waste_cpuset_parse(s, &m);
    if (n != want_count) {
        fprintf(stderr, "\"%s\": parsed %d, expected %d\n", s, n, want_count);
        return 1;
    }
    if (n < 0) return 0;                       /* rejected, as expected */
    /* want_set lists exactly the CPUs that must be set, "1,2,3"-style. */
    waste_cpumask expect;
    if (waste_cpuset_parse(want_set, &expect) < 0) return 1;
    if (memcmp(&m, &expect, sizeof m) != 0) {
        fprintf(stderr, "\"%s\": wrong CPUs (expected %s)\n", s, want_set);
        return 1;
    }
    return 0;
}

static int check_parse(void)
{
    int bad = 0;

    bad |= one("0", 1, "0");
    bad |= one("3", 1, "3");
    bad |= one("0-5", 6, "0,1,2,3,4,5");
    bad |= one("0-2,6-8", 6, "0,1,2,6,7,8");
    bad |= one("2,2,2", 1, "2");               /* a set, so duplicates fold */
    bad |= one("5-5", 1, "5");
    bad |= one(" 0 - 2 , 4 ", 4, "0,1,2,4");   /* spaces around the punctuation */
    bad |= one("1023", 1, "1023");             /* the last CPU that fits */

    /* Everything below is a typo, and every one of them has a reading that
     * would have "worked" — which is the argument for refusing. */
    bad |= one("5-0", WASTE_CPUS_BAD, NULL);   /* backwards, not empty */
    bad |= one("", WASTE_CPUS_BAD, NULL);
    bad |= one(",", WASTE_CPUS_BAD, NULL);
    bad |= one("0,", WASTE_CPUS_BAD, NULL);
    bad |= one("0,,3", WASTE_CPUS_BAD, NULL);
    bad |= one("-3", WASTE_CPUS_BAD, NULL);
    bad |= one("0-", WASTE_CPUS_BAD, NULL);
    bad |= one("0..3", WASTE_CPUS_BAD, NULL);
    bad |= one("0 3", WASTE_CPUS_BAD, NULL);   /* a missing comma is not a range */
    bad |= one("abc", WASTE_CPUS_BAD, NULL);
    bad |= one("0-5x", WASTE_CPUS_BAD, NULL);
    bad |= one("1024", WASTE_CPUS_BAD, NULL);  /* past WASTE_MAX_CPUS */
    bad |= one("99999999999999999999", WASTE_CPUS_BAD, NULL);   /* no overflow */
    bad |= one(NULL, WASTE_CPUS_BAD, NULL);

    if (bad) return 1;
    printf("parse: %d cases\n", 22);
    return 0;
}

/* ---- bind --------------------------------------------------------------- */

#if HAVE_READBACK

static int read_affinity(waste_cpumask *m)
{
    waste_cpumask_zero(m);
    return syscall(SYS_sched_getaffinity, 0, sizeof m->w, m->w) < 0 ? -1 : 0;
}

/* Every range records the thread that ran it and what that thread was
 * allowed to run on, then waits for the other chunks at a rendezvous.
 *
 * The wait used to be a fixed 2M-iteration spin per element, on the
 * reasoning that the caller would still be burning time when the worker
 * woke and would therefore not come back for the next chunk itself. That is
 * a wall-clock assumption, and it stopped holding under load: 6 failures in
 * 12 full suite runs on a 24-thread box, 1 in 40 standalone with the
 * machine busy, 0 in 8 idle — the suite is its own load, which is why it
 * failed there most (#30). When it lost the race the caller drained every
 * chunk and `distinct` came back 1, which reads as "the pool did not
 * participate" when the pool only did not need to.
 *
 * A rendezvous does not race, because a thread holding a chunk cannot come
 * back for another one while it is waiting inside it. The caller blocking
 * here is precisely what gives a worker the chance to take the next chunk,
 * so the property is established rather than hoped for.
 *
 * Timed rather than a bare wait: a pool that genuinely does not participate
 * has to fail this check, not hang the suite on it. */
#define MAXREC 256
#define RENDEZVOUS_S 10
static struct { pthread_t th; waste_cpumask mask; } rec[MAXREC];
static int nrec;
static pthread_mutex_t rec_mu = PTHREAD_MUTEX_INITIALIZER;

static pthread_mutex_t rv_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t rv_cv = PTHREAD_COND_INITIALIZER;
static int rv_arrived, rv_want, rv_timedout;

static void note_range(int b, int e, void *arg)
{
    (void)arg; (void)b; (void)e;
    waste_cpumask mine;
    const int ok = read_affinity(&mine) == 0;

    pthread_mutex_lock(&rec_mu);
    if (ok && nrec < MAXREC) {
        rec[nrec].th = pthread_self();
        rec[nrec].mask = mine;
        nrec++;
    }
    pthread_mutex_unlock(&rec_mu);

    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    ts.tv_sec += RENDEZVOUS_S;

    pthread_mutex_lock(&rv_mu);
    rv_arrived++;
    pthread_cond_broadcast(&rv_cv);
    while (rv_arrived < rv_want && !rv_timedout) {
        if (pthread_cond_timedwait(&rv_cv, &rv_mu, &ts) == ETIMEDOUT) {
            /* Release the others too. One timeout is the whole answer, and
             * waiting for each of them to reach it in turn would multiply
             * the suite's worst case by the thread count. */
            rv_timedout = 1;
            pthread_cond_broadcast(&rv_cv);
        }
    }
    pthread_mutex_unlock(&rv_mu);
}

static int check_bind(void)
{
    waste_cpumask allowed;
    if (read_affinity(&allowed) != 0) {
        printf("SKIP: cannot read this thread's affinity\n");
        return 77;
    }

    /* The first two CPUs this process may already use — or one, on a
     * machine or container that only has one. */
    waste_cpumask want;
    waste_cpumask_zero(&want);
    int picked = 0;
    for (int i = 0; i < WASTE_MAX_CPUS && picked < 2; i++)
        if (waste_cpumask_test(&allowed, i)) { waste_cpumask_set(&want, i); picked++; }
    if (picked < 1) {
        printf("SKIP: no CPU in this process's affinity mask\n");
        return 77;
    }

    /* Threads 0 means one per CPU named, which is the behaviour being
     * checked as much as the binding is. */
    waste_pool_init(0, &want);
    const int nthreads = waste_pool_threads();
    CHECK(nthreads == picked);

    /* waste_parallel_for hands out ceil(n / nthreads) per chunk, not
     * min_chunk, so n = nthreads * 4 is exactly `nthreads` chunks of 4 —
     * one per thread. That is what makes a barrier of nthreads the precise
     * assertion rather than an approximation of one. */
    nrec = 0;
    rv_arrived = 0;
    rv_timedout = 0;
    rv_want = nthreads;
    waste_parallel_for(nthreads * 4, 1, note_range, NULL);
    if (rv_timedout) {
        fprintf(stderr, "only %d of %d chunks reached the rendezvous in %d s: "
                        "the pool did not take the chunks left for it\n",
                rv_arrived, rv_want, RENDEZVOUS_S);
        return 1;
    }

    /* Every range, on whichever thread ran it. */
    CHECK(nrec > 0);
    for (int i = 0; i < nrec; i++) {
        if (memcmp(&rec[i].mask, &want, sizeof want) == 0) continue;
        fprintf(stderr, "range %d ran on a thread with the wrong mask\n", i);
        return 1;
    }

    /* The calling thread is a worker too, and the one most easily left
     * out: it is not created by the pool, so nothing binds it unless
     * waste_parallel_for does. */
    waste_cpumask self;
    CHECK(read_affinity(&self) == 0);
    if (memcmp(&self, &want, sizeof want) != 0) {
        fprintf(stderr, "the calling thread was not bound\n");
        return 1;
    }

    /* With more than one CPU, more than the caller should have taken a
     * chunk — otherwise the check above proves only that the caller was
     * bound and says nothing about the pool. Guaranteed rather than raced
     * for once the rendezvous above completed: every chunk was held by a
     * thread blocked in it, so distinct == nthreads. Kept as its own check
     * because it is the statement of intent, and it is what someone reading
     * a failure will look at first. */
    if (picked > 1) {
        int distinct = 0;
        for (int i = 0; i < nrec; i++) {
            int seen = 0;
            for (int j = 0; j < i; j++)
                if (pthread_equal(rec[j].th, rec[i].th)) { seen = 1; break; }
            distinct += !seen;
        }
        CHECK(distinct > 1);
        printf("bind: %d threads on %d CPUs, %d ranges\n", distinct, picked, nrec);
    } else {
        printf("bind: 1 CPU available, caller bound\n");
    }
    return 0;
}

#else

static int check_bind(void)
{
    /* WASTE_HAVE_AFFINITY is the engine's own answer to the same question,
     * so a platform that says it can bind but cannot be read back here is
     * worth saying out loud rather than passing quietly. */
    printf("SKIP: no way to read a thread's affinity on this platform "
           "(engine says it can%s bind)\n", WASTE_HAVE_AFFINITY ? "" : "not");
    return 77;
}

#endif

int main(int argc, char **argv)
{
    const char *what = argc > 1 ? argv[1] : "parse";
    if (!strcmp(what, "parse")) return check_parse();
    if (!strcmp(what, "bind")) return check_bind();
    fprintf(stderr, "usage: test_cpus [parse|bind]\n");
    return 2;
}
