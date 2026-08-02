/* SPDX-License-Identifier: Apache-2.0 */
/* test_ecache.c — reset semantics used by the one-process sweep harness. */
#if !defined(_WIN32)
#define _POSIX_C_SOURCE 200809L
#endif

#include "../src/ecache.h"
#include "../src/model.h"

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#if defined(_WIN32)
#include <windows.h>
#endif

#define REC_BYTES 16384

typedef struct {
    pthread_mutex_t mu;
    pthread_cond_t cv;
    int started;
    int release;
} fetch_gate;

static int failed;

#define CHECK(expr, what) do {                                             \
    if (!(expr)) {                                                         \
        fprintf(stderr, "FAIL line %d: %s\n", __LINE__, (what));          \
        failed = 1;                                                        \
    }                                                                      \
} while (0)

static int gated_fetch(void *user, int layer, int expert, uint8_t *dst)
{
    fetch_gate *g = (fetch_gate *)user;
    pthread_mutex_lock(&g->mu);
    g->started = 1;
    pthread_cond_broadcast(&g->cv);
    while (!g->release) pthread_cond_wait(&g->cv, &g->mu);
    pthread_mutex_unlock(&g->mu);
    memset(dst, (layer + expert) & 0xff, REC_BYTES);
    return 0;
}

static void delay_50ms(void)
{
#if defined(_WIN32)
    Sleep(50);
#else
    const struct timespec t = {0, 50 * 1000 * 1000};
    nanosleep(&t, NULL);
#endif
}

static void *release_fetch(void *arg)
{
    fetch_gate *g = (fetch_gate *)arg;
    delay_50ms();
    pthread_mutex_lock(&g->mu);
    g->release = 1;
    pthread_cond_broadcast(&g->cv);
    pthread_mutex_unlock(&g->mu);
    return NULL;
}

int main(void)
{
    waste_model_set_lookahead(-1);
    CHECK(waste_model_get_lookahead() == 0, "negative lookahead clamps to zero");
    waste_model_set_lookahead(WASTE_PF_MAX + 1);
    CHECK(waste_model_get_lookahead() == WASTE_PF_MAX,
          "lookahead cannot exceed the fixed prediction buffer");

    waste_ecache c;
    CHECK(waste_ecache_init(&c, 4 * REC_BYTES, REC_BYTES, 0) == 0,
          "cache initialization");
    if (failed) return 1;

    fetch_gate g;
    memset(&g, 0, sizeof g);
    pthread_mutex_init(&g.mu, NULL);
    pthread_cond_init(&g.cv, NULL);
    const int io_rc = waste_ecache_io_start(&c, gated_fetch, &g, 1, 1);
    CHECK(io_rc == 0, "reader initialization");
    if (io_rc) {
        waste_ecache_free(&c);
        pthread_cond_destroy(&g.cv);
        pthread_mutex_destroy(&g.mu);
        return 1;
    }

    const int id = 3;
    waste_ecache_prefetch(&c, 1, &id, 1);
    pthread_mutex_lock(&g.mu);
    while (!g.started) pthread_cond_wait(&g.cv, &g.mu);
    pthread_mutex_unlock(&g.mu);

    /* clear() must wait for this read. Without the drain, it marks the slot
     * empty now and the worker changes it back to READY after returning. */
    c.rng = 1234;
    c.pf_gen = 99;
    c.purged = 7;
    pthread_t release;
    const int thread_rc = pthread_create(&release, NULL, release_fetch, &g);
    CHECK(thread_rc == 0, "release-thread creation");
    if (thread_rc) {
        pthread_mutex_lock(&g.mu);
        g.release = 1;
        pthread_cond_broadcast(&g.cv);
        pthread_mutex_unlock(&g.mu);
    }
    waste_ecache_clear(&c);
    if (!thread_rc) pthread_join(release, NULL);
    waste_ecache_io_stop(&c);

    for (int i = 0; i < c.n_slots; i++) {
        CHECK(c.slot[i].key == -1, "clear removes every key");
        CHECK(c.slot[i].state == EC_EMPTY, "clear leaves every slot empty");
    }
    CHECK(c.hits == 0 && c.misses == 0 && c.bytes_read == 0,
          "clear resets measurement counters");
    CHECK(c.rng == 0x9e3779b9u, "clear resets the eviction sequence");
    CHECK(c.pf_gen == 1 && c.purged == 0,
          "clear resets hint and purge generations");

    waste_ecache_free(&c);
    pthread_cond_destroy(&g.cv);
    pthread_mutex_destroy(&g.mu);
    waste_model_set_lookahead(0);
    if (failed) return 1;
    puts("ECACHE OK");
    return 0;
}
