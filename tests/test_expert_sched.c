/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/* Whole-expert scheduling must change ownership, not model semantics. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <process.h>
#define getpid _getpid
#else
#include <unistd.h>
#endif

#include "../src/model.h"
#include "../src/waste.h"

static int same_traffic(const waste_model *a, const waste_model *b)
{
    return a->expert_reads == b->expert_reads &&
           a->cache.hits == b->cache.hits &&
           a->cache.misses == b->cache.misses &&
           a->cache.bytes_read == b->cache.bytes_read;
}

int main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(stderr, "usage: %s MODEL\n", argv[0]);
        return 2;
    }

    unsetenv("WASTE_EXPERT_SCHED");
    waste_memplan row_plan, whole_plan;
    if (waste_plan_memory(argv[1], 64, &row_plan) != WASTE_OK) return 1;

    waste_load_opts opt;
    memset(&opt, 0, sizeof opt);
    opt.cache_bytes = 16u << 20;
    opt.n_threads = 4;
    opt.direct_io = 0;
    opt.io_backend = WASTE_IO_PREAD_BATCH;
    opt.io_queue_depth = 1;

    waste_model row, whole;
    if (waste_model_load(&row, argv[1], 64, &opt)) {
        fprintf(stderr, "row load failed\n");
        return 1;
    }
    setenv("WASTE_EXPERT_SCHED", "whole", 1);
    if (waste_plan_memory(argv[1], 64, &whole_plan) != WASTE_OK ||
        waste_model_load(&whole, argv[1], 64, &opt)) {
        fprintf(stderr, "whole load failed\n");
        waste_model_free(&row);
        return 1;
    }

    const int lat = row.cfg.latent_dim ? row.cfg.latent_dim : row.cfg.hidden;
    const uint64_t expected = waste_model_whole_expert_scratch_bytes(
        row.cfg.top_k, row.cfg.moe_inter, lat, row.vec_dim, row.stages,
        row.cb_entries);
    if (expected == UINT64_MAX ||
        whole_plan.scratch_bytes - row_plan.scratch_bytes != expected ||
        whole_plan.floor_bytes - row_plan.floor_bytes != expected ||
        whole_plan.recommended_bytes - row_plan.recommended_bytes != expected ||
        whole_plan.min_expert_cache != row_plan.min_expert_cache) {
        fprintf(stderr, "whole scratch is not exactly budgeted\n");
        return 1;
    }
    if (row.expert_sched_whole || !whole.expert_sched_whole ||
        whole.io_backend != WASTE_IO_PREAD_BATCH || !whole.whole_gate ||
        !whole.whole_up || !whole.whole_down_lut || !whole.whole_out) {
        fprintf(stderr, "whole scheduler was not selected\n");
        return 1;
    }

    /* Meta reports the effective schedule, not just the env request.  With
     * no cache a complete routed set cannot be staged, so the request must
     * remain visible while effective ownership falls back to row.  Read the
     * still-open trace immediately to keep the request-boundary flush honest. */
    {
        waste_model no_cache;
        char trace_path[160], line[512];
        snprintf(trace_path, sizeof trace_path,
                 "/tmp/waste_expert_sched_meta_%ld.jsonl", (long)getpid());
        opt.cache_bytes = 0;
        opt.trace_path = trace_path;
        if (waste_model_load(&no_cache, argv[1], 64, &opt)) {
            fprintf(stderr, "no-cache schedule load failed\n");
            return 1;
        }
        FILE *trace = fopen(trace_path, "r");
        const int meta_ok = trace && fgets(line, sizeof line, trace) &&
            strstr(line, "\"expert_schedule_requested\":\"whole\"") &&
            strstr(line, "\"expert_schedule\":\"row\"");
        if (trace) fclose(trace);
        remove(trace_path);
        waste_model_free(&no_cache);
        opt.trace_path = NULL;
        opt.cache_bytes = 16u << 20;
        if (!meta_ok) {
            fprintf(stderr, "effective schedule trace metadata is wrong\n");
            return 1;
        }
    }

    static const int tokens[] = {3, 7, 11, 5, 9, 13, 2, 17};
    const size_t route_n = (size_t)row.cfg.n_layers * row.cfg.top_k;
    int *row_routes = (int *)malloc(route_n * sizeof(int));
    int *whole_routes = (int *)malloc(route_n * sizeof(int));
    if (!row_routes || !whole_routes) return 1;
    for (size_t t = 0; t < sizeof tokens / sizeof *tokens; t++) {
        memset(row_routes, 0xff, route_n * sizeof(int));
        memset(whole_routes, 0xff, route_n * sizeof(int));
        const float *a = waste_model_step(&row, tokens[t], (int)t, row_routes);
        const float *b = waste_model_step(&whole, tokens[t], (int)t,
                                          whole_routes);
        if (!a || !b || memcmp(a, b, (size_t)row.cfg.vocab * sizeof(float)) ||
            memcmp(row_routes, whole_routes, route_n * sizeof(int)) ||
            !same_traffic(&row, &whole)) {
            fprintf(stderr, "row/whole mismatch at token %zu\n", t);
            return 1;
        }
    }

    size_t row_n = 0, whole_n = 0, row_written = 0, whole_written = 0;
    const int pos = (int)(sizeof tokens / sizeof *tokens);
    if (waste_model_state_size(&row, pos, &row_n) ||
        waste_model_state_size(&whole, pos, &whole_n) || row_n != whole_n)
        return 1;
    unsigned char *row_state = (unsigned char *)malloc(row_n);
    unsigned char *whole_state = (unsigned char *)malloc(whole_n);
    if (!row_state || !whole_state ||
        waste_model_state_export(&row, pos, row_state, row_n, &row_written) ||
        waste_model_state_export(&whole, pos, whole_state, whole_n,
                                 &whole_written) ||
        row_written != row_n || whole_written != whole_n ||
        memcmp(row_state, whole_state, row_n)) {
        fprintf(stderr, "row/whole state mismatch\n");
        return 1;
    }

    free(row_state); free(whole_state);
    free(row_routes); free(whole_routes);
    waste_model_free(&row); waste_model_free(&whole);
    unsetenv("WASTE_EXPERT_SCHED");
    puts("PASS whole-expert scheduler is budgeted and bit-identical");
    return 0;
}
