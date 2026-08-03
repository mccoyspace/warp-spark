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
#include <stdlib.h>
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

static void age_env(const char *value)
{
#if defined(_WIN32)
    _putenv_s("WASTE_LFRU_AGE_TOKENS", value ? value : "");
#else
    if (value) setenv("WASTE_LFRU_AGE_TOKENS", value, 1);
    else unsetenv("WASTE_LFRU_AGE_TOKENS");
#endif
}

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

typedef struct {
    waste_ecache *cache;
    fetch_gate *gate;
    waste_ecache_hold hold;
    const uint8_t *data;
} hold_get;

static void *run_hold_get(void *arg)
{
    hold_get *g = (hold_get *)arg;
    g->data = waste_ecache_get_hold(g->cache, 6, 42, gated_fetch, g->gate,
                                    &g->hold);
    return NULL;
}

int main(void)
{
    age_env(NULL);
    waste_model_set_lookahead(-1);
    CHECK(waste_model_get_lookahead() == 0, "negative lookahead clamps to zero");
    waste_model_set_lookahead(WASTE_PF_MAX + 1);
    CHECK(waste_model_get_lookahead() == WASTE_PF_MAX,
          "lookahead cannot exceed the fixed prediction buffer");

    waste_ecache c;
    CHECK(waste_ecache_init(&c, 4 * REC_BYTES, REC_BYTES, 0) == 0,
          "cache initialization");
    CHECK(waste_ecache_get_lfru_age(&c) == 0,
          "LFRU aging is disabled by default");
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

    /* Two callers requesting the same in-flight record both acquire a real
     * reference. The second waits on the reader mutex instead of observing
     * a half-filled payload or stealing the first caller's lifetime. */
    pthread_mutex_lock(&g.mu);
    g.started = 0;
    g.release = 0;
    pthread_mutex_unlock(&g.mu);
    hold_get hg[2] = {
        { &c, &g, WASTE_ECACHE_HOLD_INIT, NULL },
        { &c, &g, WASTE_ECACHE_HOLD_INIT, NULL }
    };
    pthread_t ht[2];
    const int hrc0 = pthread_create(&ht[0], NULL, run_hold_get, &hg[0]);
    CHECK(hrc0 == 0, "first concurrent hold thread");
    if (!hrc0) {
        pthread_mutex_lock(&g.mu);
        while (!g.started) pthread_cond_wait(&g.cv, &g.mu);
        pthread_mutex_unlock(&g.mu);
    }
    const int hrc1 = pthread_create(&ht[1], NULL, run_hold_get, &hg[1]);
    CHECK(hrc1 == 0, "second concurrent hold thread");
    pthread_mutex_lock(&g.mu);
    g.release = 1;
    pthread_cond_broadcast(&g.cv);
    pthread_mutex_unlock(&g.mu);
    if (!hrc0) pthread_join(ht[0], NULL);
    if (!hrc1) pthread_join(ht[1], NULL);
    if (!hrc0 && !hrc1) {
        CHECK(hg[0].data != NULL && hg[0].data == hg[1].data,
              "concurrent holds share the completed resident record");
        CHECK(hg[0].hold.slot == hg[1].hold.slot &&
              c.slot[hg[0].hold.slot].holds == 2,
              "concurrent holds are reference-counted");
    }
    waste_ecache_release_hold(&c, &hg[0].hold);
    waste_ecache_release_hold(&c, &hg[1].hold);
    waste_ecache_clear(&c);
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

    /* A hotlist's raw hit counts are retained until the opt-in decode
     * clock reaches its half-life. Only resident, completed records age;
     * an in-flight reader still owns its metadata and payload. */
    c.slot[0].state = EC_READY;    c.slot[0].hits = 1;
    c.slot[1].state = EC_READY;    c.slot[1].hits = 2;
    c.slot[2].state = EC_READY;    c.slot[2].hits = 3;
    c.slot[3].state = EC_INFLIGHT; c.slot[3].hits = 8;
    waste_ecache_decode_tick(&c);
    CHECK(c.slot[1].hits == 2 && c.slot[2].hits == 3,
          "default-off decode ticks do not age frequencies");
    waste_ecache_set_lfru_age(&c, 4);
    for (int i = 0; i < 3; i++) waste_ecache_decode_tick(&c);
    CHECK(c.slot[0].hits == 1 && c.slot[1].hits == 2 &&
          c.slot[2].hits == 3 && c.slot[3].hits == 8,
          "aging waits for the configured decode-token interval");
    waste_ecache_decode_tick(&c);
    CHECK(c.slot[0].hits == 1 && c.slot[1].hits == 1 &&
          c.slot[2].hits == 1 && c.slot[3].hits == 8,
          "aging halves READY frequencies with a minimum of one");
    CHECK(waste_ecache_lfru_age_events(&c) == 1,
          "a completed interval records one aging event");

    /* clear() is also the sweep arm boundary: keep the selected half-life,
     * but restart its phase so the following arm cannot inherit a partial
     * interval from the previous one. */
    waste_ecache_clear(&c);
    CHECK(waste_ecache_get_lfru_age(&c) == 4 && c.lfru_age_phase == 0 &&
          waste_ecache_lfru_age_events(&c) == 0,
          "clear retains the setting and resets the aging phase");
    c.slot[0].state = EC_READY; c.slot[0].hits = 8;
    waste_ecache_decode_tick(&c);
    waste_ecache_decode_tick(&c);
    waste_ecache_clear(&c);
    c.slot[0].state = EC_READY; c.slot[0].hits = 8;
    waste_ecache_decode_tick(&c);
    waste_ecache_decode_tick(&c);
    CHECK(c.slot[0].hits == 8,
          "a cleared partial interval does not age the next arm early");
    waste_ecache_decode_tick(&c);
    waste_ecache_decode_tick(&c);
    CHECK(c.slot[0].hits == 4, "the reset arm ages on its own fourth tick");
    c.policy = 1;
    waste_ecache_set_lfru_age(&c, 1);
    c.slot[0].hits = 8;
    waste_ecache_decode_tick(&c);
    CHECK(c.slot[0].hits == 8 && c.lfru_age_phase == 0,
          "the LFRU lever does not alter an LRU cache");
    c.policy = 0;
    waste_ecache_set_lfru_age(&c, 0);
    waste_ecache_clear(&c);

    /* Explicit holds survive later get() calls and drain(), but leave the
     * slot available immediately after release. Fill the whole cache with
     * held records so eviction has no candidate at all. */
    waste_ecache_hold h[4] = {
        WASTE_ECACHE_HOLD_INIT, WASTE_ECACHE_HOLD_INIT,
        WASTE_ECACHE_HOLD_INIT, WASTE_ECACHE_HOLD_INIT
    };
    const uint8_t *held[4];
    for (int i = 0; i < 4; i++) {
        held[i] = waste_ecache_get_hold(&c, 2, 10 + i, gated_fetch, &g, &h[i]);
        CHECK(held[i] != NULL, "get-and-hold returns a record");
        CHECK(h[i].slot >= 0 && c.slot[h[i].slot].holds == 1,
              "get-and-hold pins its slot");
    }
    waste_ecache_drain(&c);
    for (int i = 0; i < 4; i++)
        CHECK(c.slot[h[i].slot].holds == 1, "drain preserves an active hold");

    waste_ecache_hold blocked = WASTE_ECACHE_HOLD_INIT;
    CHECK(waste_ecache_get_hold(&c, 2, 99, gated_fetch, &g, &blocked) == NULL,
          "a cache containing only held slots cannot evict one");
    CHECK(blocked.slot == -1 && blocked.epoch == 0,
          "failed get-and-hold leaves an invalid handle");

    waste_ecache_release_hold(&c, &h[1]);
    CHECK(h[1].slot == -1 && h[1].epoch == 0,
          "release invalidates its handle");
    CHECK(waste_ecache_get_hold(&c, 2, 99, gated_fetch, &g, &blocked) != NULL,
          "a released slot is evictable again");

    /* clear() is the sweep-level lifetime boundary. Old handles must not be
     * able to decrement a new hold that happens to reuse their slot. */
    waste_ecache_hold stale = h[0];
    waste_ecache_clear(&c);
    waste_ecache_hold fresh = WASTE_ECACHE_HOLD_INIT;
    CHECK(waste_ecache_get_hold(&c, 3, 7, gated_fetch, &g, &fresh) != NULL,
          "get-and-hold works after clear");
    CHECK(fresh.slot >= 0 && c.slot[fresh.slot].holds == 1,
          "new post-clear hold is active");
    waste_ecache_release_hold(&c, &stale);
    CHECK(c.slot[fresh.slot].holds == 1,
          "a stale pre-clear handle cannot release a new record");
    waste_ecache_release_hold(&c, &fresh);
    CHECK(fresh.slot == -1, "fresh post-clear handle releases normally");
    for (int i = 0; i < 4; i++) waste_ecache_release_hold(&c, &h[i]);
    waste_ecache_release_hold(&c, &blocked);

    /* Free invalidates the cache before releasing storage; cleanup code may
     * still safely release an owned handle afterward. */
    waste_ecache_hold at_free = WASTE_ECACHE_HOLD_INIT;
    CHECK(waste_ecache_get_hold(&c, 4, 1, gated_fetch, &g, &at_free) != NULL,
          "active hold before free");
    waste_ecache_free(&c);
    waste_ecache_release_hold(&c, &at_free);
    CHECK(at_free.slot == -1, "release after free is harmless");

    waste_ecache from_env;
    age_env("4");
    CHECK(waste_ecache_init(&from_env, REC_BYTES, REC_BYTES, 0) == 0,
          "environment-configured aging cache initializes");
    CHECK(waste_ecache_get_lfru_age(&from_env) == 4,
          "WASTE_LFRU_AGE_TOKENS initializes the per-cache setting");
    waste_ecache_free(&from_env);
    age_env(NULL);
    pthread_cond_destroy(&g.cv);
    pthread_mutex_destroy(&g.mu);
    waste_model_set_lookahead(0);
    if (failed) return 1;
    puts("ECACHE OK");
    return 0;
}
