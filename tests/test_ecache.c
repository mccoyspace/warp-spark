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
#else
#include <unistd.h>
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

static void prior_env(const char *value)
{
#if defined(_WIN32)
    _putenv_s("WASTE_LFRU_PRIOR_LOG2", value ? value : "");
#else
    if (value) setenv("WASTE_LFRU_PRIOR_LOG2", value, 1);
    else unsetenv("WASTE_LFRU_PRIOR_LOG2");
#endif
}

static int immediate_fetch(void *user, int layer, int expert, uint8_t *dst)
{
    (void)user;
    memset(dst, (layer + expert) & 0xff, REC_BYTES);
    return 0;
}

typedef struct {
    int layer, expert;
    uint32_t hits;
} usage_fixture;

static int temp_path(char *path, size_t cap)
{
#if defined(_WIN32)
    char dir[MAX_PATH], name[MAX_PATH];
    const DWORD n = GetTempPathA(MAX_PATH, dir);
    if (!n || n >= MAX_PATH || !GetTempFileNameA(dir, "wst", 0, name)) return -1;
    if (strlen(name) + 1 > cap) { DeleteFileA(name); return -1; }
    memcpy(path, name, strlen(name) + 1);
    return 0;
#else
    const char pattern[] = "/tmp/waste-ecache-XXXXXX";
    if (sizeof pattern > cap) return -1;
    memcpy(path, pattern, sizeof pattern);
    const int fd = mkstemp(path);
    if (fd < 0) return -1;
    close(fd);
    return 0;
#endif
}

static int write_usage_fixture(const char *path, const usage_fixture *entry,
                               int n)
{
    waste_ecache source;
    if (waste_ecache_init(&source, (size_t)n * REC_BYTES, REC_BYTES, 0)) return -1;
    for (int i = 0; i < n; i++) {
        source.slot[i].key = (entry[i].layer << 16) | entry[i].expert;
        source.slot[i].state = EC_READY;
        source.slot[i].hits = entry[i].hits;
        source.slot[i].last = (uint64_t)i + 1;
    }
    const int rc = waste_ecache_save_usage(&source, path, (uint64_t)n);
    waste_ecache_free(&source);
    return rc;
}

static int find_slot(const waste_ecache *c, int layer, int expert)
{
    const int32_t key = (layer << 16) | expert;
    for (int i = 0; i < c->n_slots; i++)
        if (c->slot[i].key == key && c->slot[i].state == EC_READY) return i;
    return -1;
}

static void check_prior_log2(void)
{
    char usage_a[512] = {0}, usage_b[512] = {0}, usage_zero[512] = {0};
    const usage_fixture a[] = {
        { 4, 1, 8 }, { 3, 2, 63 }, { 5, 3, 0 }
    };
    const usage_fixture b[] = { { 6, 4, 7 } };
    const usage_fixture zero[] = { { 7, 5, 0 } };
    if (temp_path(usage_a, sizeof usage_a) ||
        temp_path(usage_b, sizeof usage_b) ||
        temp_path(usage_zero, sizeof usage_zero)) {
        CHECK(0, "temporary usage paths");
        goto done;
    }
    if (write_usage_fixture(usage_a, a, 3) ||
        write_usage_fixture(usage_b, b, 1) ||
        write_usage_fixture(usage_zero, zero, 1)) {
        CHECK(0, "usage fixtures");
        goto done;
    }

    waste_ecache hot;
    if (waste_ecache_init(&hot, 2 * REC_BYTES, REC_BYTES, 0)) {
        CHECK(0, "prior cache initialization");
        goto done;
    }
    CHECK(waste_ecache_get_lfru_prior_log2(&hot) == 0,
          "prior compression is disabled by default");
    CHECK(waste_ecache_warm(&hot, usage_a, immediate_fetch, NULL) == 2,
          "raw hotlist fills the bounded cache");
    int s63 = find_slot(&hot, 3, 2), s8 = find_slot(&hot, 4, 1);
    CHECK(s63 >= 0 && s8 >= 0 && find_slot(&hot, 5, 3) < 0,
          "raw counts select the hottest entries before compression");
    CHECK(s63 >= 0 && hot.slot[s63].hits == 63 &&
          s8 >= 0 && hot.slot[s8].hits == 8,
          "control warm retains raw prior counts");
    CHECK(waste_ecache_lfru_prior_events(&hot) == 0 &&
          waste_ecache_lfru_prior_entries(&hot) == 0,
          "control warm records no compression");

    waste_ecache_clear(&hot);
    waste_ecache_set_lfru_prior_log2(&hot, 1);
    CHECK(waste_ecache_get_lfru_prior_log2(&hot) == 1,
          "prior compression setter enables the mode");
    const int warmed = waste_ecache_warm(&hot, usage_a, immediate_fetch, NULL);
    s63 = find_slot(&hot, 3, 2); s8 = find_slot(&hot, 4, 1);
    CHECK(warmed == 2 && s63 >= 0 && s8 >= 0,
          "compressed warm keeps raw-count membership");
    CHECK(hot.slot[s63].hits == 6, "63 imported hits compress to six");
    CHECK(hot.slot[s8].hits == 4, "eight imported hits compress to four");
    CHECK(waste_ecache_lfru_prior_events(&hot) == 1 &&
          waste_ecache_lfru_prior_entries(&hot) == (uint64_t)warmed,
          "one nonempty warm counts every transformed entry");
    CHECK(waste_ecache_get(&hot, 4, 1, immediate_fetch, NULL) != NULL &&
          hot.slot[s8].hits == 5,
          "current prompt and decode hits remain linear");

    CHECK(waste_ecache_warm(&hot, usage_a, immediate_fetch, NULL) == 0 &&
          waste_ecache_lfru_prior_events(&hot) == 1 &&
          waste_ecache_lfru_prior_entries(&hot) == 2,
          "an empty repeated warm does not create an event");
    CHECK(waste_ecache_warm(&hot, usage_b, immediate_fetch, NULL) == 1 &&
          waste_ecache_lfru_prior_events(&hot) == 2 &&
          waste_ecache_lfru_prior_entries(&hot) == 3,
          "a second nonempty warm accumulates one event and entry");

    waste_ecache_clear(&hot);
    CHECK(waste_ecache_get_lfru_prior_log2(&hot) == 1 &&
          waste_ecache_lfru_prior_events(&hot) == 0 &&
          waste_ecache_lfru_prior_entries(&hot) == 0,
          "clear retains the mode and resets prior counters");
    CHECK(waste_ecache_warm(&hot, usage_zero, immediate_fetch, NULL) == 1,
          "zero-count defensive fixture warms");
    const int szero = find_slot(&hot, 7, 5);
    CHECK(szero >= 0 && hot.slot[szero].hits == 0 &&
          waste_ecache_lfru_prior_events(&hot) == 1 &&
          waste_ecache_lfru_prior_entries(&hot) == 1,
          "zero has a defined zero-bit score and is still counted");

    waste_ecache_clear(&hot);
    hot.policy = 1;
    CHECK(waste_ecache_warm(&hot, usage_a, immediate_fetch, NULL) == 2,
          "LRU hotlist warms normally");
    s63 = find_slot(&hot, 3, 2); s8 = find_slot(&hot, 4, 1);
    CHECK(s63 >= 0 && hot.slot[s63].hits == 63 &&
          s8 >= 0 && hot.slot[s8].hits == 8,
          "LRU ignores prior compression");
    CHECK(waste_ecache_lfru_prior_events(&hot) == 0 &&
          waste_ecache_lfru_prior_entries(&hot) == 0,
          "LRU records no prior compression");
    waste_ecache_free(&hot);

done:
    if (usage_a[0]) remove(usage_a);
    if (usage_b[0]) remove(usage_b);
    if (usage_zero[0]) remove(usage_zero);
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
    prior_env(NULL);
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
    CHECK(waste_ecache_get_lfru_prior_log2(&c) == 0,
          "LFRU prior compression is disabled by default");
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

    check_prior_log2();

    waste_ecache from_env;
    age_env("4");
    prior_env("1");
    CHECK(waste_ecache_init(&from_env, REC_BYTES, REC_BYTES, 0) == 0,
          "environment-configured aging cache initializes");
    CHECK(waste_ecache_get_lfru_age(&from_env) == 4,
          "WASTE_LFRU_AGE_TOKENS initializes the per-cache setting");
    CHECK(waste_ecache_get_lfru_prior_log2(&from_env) == 1,
          "WASTE_LFRU_PRIOR_LOG2 initializes the per-cache setting");
    waste_ecache_free(&from_env);
    age_env(NULL);
    prior_env(NULL);
    pthread_cond_destroy(&g.cv);
    pthread_mutex_destroy(&g.mu);
    waste_model_set_lookahead(0);
    if (failed) return 1;
    puts("ECACHE OK");
    return 0;
}
