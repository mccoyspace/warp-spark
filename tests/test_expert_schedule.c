/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 *
 * Typed whole-expert scheduler acceptance on the synthetic container:
 * exact memory accounting, runtime fallbacks, bit identity, unchanged
 * routing and LA0 demand traffic, state identity, and clean error paths.
 */

#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/model.h"
#include "../src/waste.h"

#ifdef _WIN32
static int set_env(const char *name, const char *value)
{
    return _putenv_s(name, value);
}
#else
static int set_env(const char *name, const char *value)
{
    return setenv(name, value, 1);
}
#endif

static int same_file(const char *a, const char *b)
{
    FILE *fa = fopen(a, "rb"), *fb = fopen(b, "rb");
    if (!fa || !fb) { if (fa) fclose(fa); if (fb) fclose(fb); return 0; }
    int same = 1;
    for (;;) {
        unsigned char xa[4096], xb[4096];
        const size_t na = fread(xa, 1, sizeof xa, fa);
        const size_t nb = fread(xb, 1, sizeof xb, fb);
        if (na != nb || memcmp(xa, xb, na)) { same = 0; break; }
        if (na < sizeof xa) break;
    }
    fclose(fa); fclose(fb);
    return same;
}

static int fail(const char *what)
{
    fprintf(stderr, "expert schedule: %s\n", what);
    return 1;
}

typedef struct { size_t bytes; int fail_expert; } fetch_fixture;

static int fixture_fetch(void *user, int layer, int expert, uint8_t *dst)
{
    fetch_fixture *f = (fetch_fixture *)user;
    (void)layer;
    if (expert == f->fail_expert) return -1;
    memset(dst, expert, f->bytes);
    return 0;
}

static int batch_ref_cleanup(void)
{
    waste_ecache c;
    const size_t rec = 4096;
    fetch_fixture f = { rec, 1 };
    if (waste_ecache_init(&c, 8 * rec, rec, 0))
        return fail("cache fixture init failed");
    if (waste_ecache_io_start(&c, fixture_fetch, &f, 2, 2)) {
        waste_ecache_free(&c);
        return fail("cache fixture readers failed");
    }
    const int bad_ids[2] = { 0, 1 };
    const uint8_t *records[2] = { 0 };
    waste_ecache_hint(&c, 2, bad_ids, 2);
    if (waste_ecache_acquire_many(
            &c, 2, bad_ids, 2, fixture_fetch, &f, records) == 0) {
        waste_ecache_release_many(&c, 2, bad_ids, 2);
        waste_ecache_free(&c);
        return fail("failed batch unexpectedly acquired");
    }
    for (int i = 0; i < c.n_slots; i++)
        if (c.slot[i].refs) {
            waste_ecache_free(&c);
            return fail("failed batch leaked a record reference");
        }

    f.fail_expert = -1;
    const int good_ids[2] = { 2, 3 };
    waste_ecache_hint(&c, 2, good_ids, 2);
    if (waste_ecache_acquire_many(
            &c, 2, good_ids, 2, fixture_fetch, &f, records)) {
        waste_ecache_free(&c);
        return fail("cache was not reusable after a failed batch");
    }
    waste_ecache_release_many(&c, 2, good_ids, 2);
    for (int i = 0; i < c.n_slots; i++)
        if (c.slot[i].refs) {
            waste_ecache_free(&c);
            return fail("successful batch leaked a record reference");
        }
    waste_ecache_free(&c);
    return 0;
}

static int public_contract(const char *model)
{
    waste_memplan row, whole;
    if (waste_plan_memory(model, 64, &row) != WASTE_OK)
        return fail("row plan failed");
    if (waste_plan_memory_for_schedule(
            model, 64, WASTE_EXPERT_SCHEDULE_WHOLE, &whole) != WASTE_OK)
        return fail("whole plan failed");

    const uint64_t want = waste_model_whole_expert_scratch_bytes(
        2, 64, 128, 8, 3, 256);
    if (want != 51200u || whole.scratch_bytes - row.scratch_bytes != want ||
        whole.floor_bytes - row.floor_bytes != want ||
        whole.recommended_bytes - row.recommended_bytes != want)
        return fail("planner scratch delta is not exact");
    if (waste_model_whole_expert_scratch_bytes(0, 64, 128, 8, 3, 256) !=
            UINT64_MAX ||
        waste_model_whole_expert_scratch_bytes(
            INT_MAX, INT_MAX, INT_MAX, 1, INT_MAX, INT_MAX) != UINT64_MAX)
        return fail("scratch overflow was not rejected");
    if (waste_plan_memory_for_schedule(
            model, 64, (waste_expert_schedule)99, &whole) != WASTE_E_ARG)
        return fail("planner accepted an invalid schedule");

    waste_cfg cfg;
    waste_cfg_init(&cfg);
    cfg.ctx_tokens = 64;
    cfg.expert_schedule = (waste_expert_schedule)99;
    waste_ctx *ctx = (waste_ctx *)(uintptr_t)1;
    if (waste_open(model, &cfg, &ctx) != WASTE_E_ARG || ctx != NULL)
        return fail("open accepted an invalid schedule");

    waste_cfg_init(&cfg);
    cfg.ctx_tokens = 64;
    cfg.expert_schedule = WASTE_EXPERT_SCHEDULE_WHOLE;
    cfg.ram_budget_bytes = whole.floor_bytes - 1;
    ctx = (waste_ctx *)(uintptr_t)1;
    if (waste_open(model, &cfg, &ctx) != WASTE_E_RAM_BUDGET || ctx != NULL)
        return fail("whole floor-minus-one did not fail cleanly");

    /* The minimum cache is four slots for top-2. A complete hint may pin
     * only one quarter of the cache, so WHOLE must fall back at the floor. */
    cfg.ram_budget_bytes = whole.floor_bytes;
    if (waste_open(model, &cfg, &ctx) != WASTE_OK)
        return fail("geometry-fallback open failed");
    waste_expert_schedule got = WASTE_EXPERT_SCHEDULE_WHOLE;
    if (waste_get_expert_schedule(ctx, &got) != WASTE_OK ||
        got != WASTE_EXPERT_SCHEDULE_ROW) {
        waste_close(ctx);
        return fail("small cache did not fall back to row");
    }
    waste_close(ctx);

    /* Two minimum working sets are eight slots, enough to pin top-2. */
    cfg.ram_budget_bytes = whole.floor_bytes + whole.min_expert_cache;
    if (waste_open(model, &cfg, &ctx) != WASTE_OK)
        return fail("whole open failed");
    got = WASTE_EXPERT_SCHEDULE_ROW;
    if (waste_get_expert_schedule(ctx, &got) != WASTE_OK ||
        got != WASTE_EXPERT_SCHEDULE_WHOLE) {
        waste_close(ctx);
        return fail("whole schedule was not effective");
    }
    waste_memplan used;
    if (waste_memory_used(ctx, &used) != WASTE_OK ||
        used.scratch_bytes != whole.scratch_bytes ||
        used.min_expert_cache != 2 * whole.min_expert_cache) {
        waste_close(ctx);
        return fail("allocator and typed plan disagree");
    }
    waste_close(ctx);

    if (set_env("WASTE_IO_THREADS", "0"))
        return fail("could not disable reader threads");
    if (waste_open(model, &cfg, &ctx) != WASTE_OK)
        return fail("reader-fallback open failed");
    got = WASTE_EXPERT_SCHEDULE_WHOLE;
    if (waste_get_expert_schedule(ctx, &got) != WASTE_OK ||
        got != WASTE_EXPERT_SCHEDULE_ROW) {
        waste_close(ctx);
        return fail("disabled read-ahead did not fall back to row");
    }
    waste_close(ctx);
    if (set_env("WASTE_IO_THREADS", "2"))
        return fail("could not restore reader threads");
    return 0;
}

static int dense_fallback(const char *model)
{
    waste_memplan row, whole;
    if (waste_plan_memory(model, 64, &row) != WASTE_OK ||
        waste_plan_memory_for_schedule(
            model, 64, WASTE_EXPERT_SCHEDULE_WHOLE, &whole) != WASTE_OK)
        return fail("dense scheduler plans failed");
    if (memcmp(&row, &whole, sizeof row))
        return fail("dense WHOLE request reserved unusable scratch");

    waste_cfg cfg;
    waste_cfg_init(&cfg);
    cfg.ctx_tokens = 64;
    cfg.ram_budget_bytes = whole.floor_bytes;
    cfg.expert_schedule = WASTE_EXPERT_SCHEDULE_WHOLE;
    waste_ctx *ctx = NULL;
    if (waste_open(model, &cfg, &ctx) != WASTE_OK)
        return fail("dense WHOLE request did not open");

    waste_expert_schedule got = WASTE_EXPERT_SCHEDULE_WHOLE;
    if (waste_get_expert_schedule(ctx, &got) != WASTE_OK ||
        got != WASTE_EXPERT_SCHEDULE_ROW) {
        waste_close(ctx);
        return fail("dense WHOLE request did not normalize to row");
    }
    const int32_t token = 3;
    const float *logits = NULL;
    size_t vocab = 0;
    if (waste_eval(ctx, &token, 1, &logits, &vocab) != WASTE_OK ||
        !logits || !vocab) {
        waste_close(ctx);
        return fail("dense fallback context did not evaluate");
    }
    waste_close(ctx);
    return 0;
}

static int identity(const char *model, const char *tmp)
{
    waste_memplan plan;
    if (waste_plan_memory(model, 64, &plan) != WASTE_OK)
        return fail("identity plan failed");

    waste_load_opts ro = {0}, wo = {0};
    ro.cache_bytes = wo.cache_bytes = (size_t)(2 * plan.min_expert_cache);
    ro.direct_io = wo.direct_io = 1;
    wo.expert_schedule = WASTE_EXPERT_SCHEDULE_WHOLE;
    waste_model row, whole;
    if (waste_model_load(&row, model, 64, &ro)) {
        waste_model_free(&row);
        return fail("row model load failed");
    }
    if (waste_model_load(&whole, model, 64, &wo)) {
        waste_model_free(&whole);
        waste_model_free(&row);
        return fail("whole model load failed");
    }
    if (row.expert_schedule_effective || !whole.expert_schedule_effective) {
        waste_model_free(&whole); waste_model_free(&row);
        return fail("direct-load effective modes are wrong");
    }

    static const int tokens[] = { 3, 7, 11, 5, 9, 13, 2, 17 };
    const size_t nroutes = (size_t)row.cfg.n_layers * row.cfg.top_k;
    int rr[WASTE_MAX_LAYERS * 64], wr[WASTE_MAX_LAYERS * 64];
    for (size_t t = 0; t < sizeof tokens / sizeof *tokens; t++) {
        memset(rr, 0xff, nroutes * sizeof *rr);
        memset(wr, 0xff, nroutes * sizeof *wr);
        const float *rl = waste_model_step(&row, tokens[t], (int)t, rr);
        const float *wl = waste_model_step(&whole, tokens[t], (int)t, wr);
        if (!rl || !wl ||
            memcmp(rl, wl, (size_t)row.cfg.vocab * sizeof(float)) ||
            memcmp(rr, wr, nroutes * sizeof *rr)) {
            waste_model_free(&whole); waste_model_free(&row);
            return fail("logits or routes changed");
        }
    }

    if (row.cache.hits != whole.cache.hits ||
        row.cache.misses != whole.cache.misses ||
        row.cache.bytes_read != whole.cache.bytes_read ||
        row.cache.evictions != whole.cache.evictions ||
        row.expert_reads != whole.expert_reads) {
        waste_model_free(&whole); waste_model_free(&row);
        return fail("cache traffic changed");
    }

    char rp[1024], wp[1024];
    snprintf(rp, sizeof rp, "%s/schedule-row.state", tmp);
    snprintf(wp, sizeof wp, "%s/schedule-whole.state", tmp);
    if (waste_model_state_save(&row, rp, (int)(sizeof tokens / sizeof *tokens)) ||
        waste_model_state_save(&whole, wp, (int)(sizeof tokens / sizeof *tokens)) ||
        !same_file(rp, wp)) {
        waste_model_free(&whole); waste_model_free(&row);
        return fail("state changed");
    }

    waste_model_free(&whole);
    waste_model_free(&row);
    return 0;
}

int main(int argc, char **argv)
{
    if (argc != 4) {
        fprintf(stderr, "usage: %s tiny.waste dense.waste tmpdir\n", argv[0]);
        return 2;
    }
    if (set_env("WASTE_IO_THREADS", "2") ||
        set_env("WASTE_IO_DEPTH", "2") ||
        set_env("WASTE_LOOKAHEAD", "0"))
        return fail("could not set deterministic environment");
    if (batch_ref_cleanup() || public_contract(argv[1]) ||
        dense_fallback(argv[2]) || identity(argv[1], argv[3]))
        return 1;
    puts("EXPERT SCHEDULE OK");
    return 0;
}
