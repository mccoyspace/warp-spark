/* SPDX-License-Identifier: Apache-2.0 */
/* API-2 identity and guard checks for the two enlarged public structures. */

#include "../src/waste.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define BEFORE 0x13579bdf2468ace0ULL
#define AFTER  0xfedcba9876543210ULL
#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "ABI check failed at line %d: %s\n", __LINE__, #x); \
    return 1; \
} } while (0)

_Static_assert(WASTE_API_VERSION == 2, "this test is for the API-2 layout");
_Static_assert(offsetof(waste_cfg, host_reserved_bytes) + sizeof(uint64_t) ==
               sizeof(waste_cfg), "host reservation must remain the cfg tail");
_Static_assert(offsetof(waste_memplan, host_reserved_bytes) + sizeof(uint64_t) ==
               sizeof(waste_memplan), "host reservation must remain the plan tail");

typedef struct { uint64_t before; waste_cfg value; uint64_t after; } cfg_guard;
typedef struct { uint64_t before; waste_memplan value; uint64_t after; } plan_guard;

static int cfg_intact(const cfg_guard *g)
{
    return g->before == BEFORE && g->after == AFTER;
}

static int plan_intact(const plan_guard *g)
{
    return g->before == BEFORE && g->after == AFTER;
}

int main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(stderr, "usage: %s MODEL\n", argv[0]);
        return 2;
    }
    CHECK(waste_api_version() == WASTE_API_VERSION);
    CHECK(waste_sizeof_cfg() == sizeof(waste_cfg));
    CHECK(waste_sizeof_memplan() == sizeof(waste_memplan));

    cfg_guard cfg = { BEFORE, {0}, AFTER };
    waste_cfg_init(&cfg.value);
    CHECK(cfg_intact(&cfg));
    cfg.value.ctx_tokens = 512;

    plan_guard planned = { BEFORE, {0}, AFTER };
    CHECK(waste_plan_memory(argv[1], cfg.value.ctx_tokens, &planned.value) ==
          WASTE_OK);
    CHECK(plan_intact(&planned));
    CHECK(planned.value.floor_bytes > 0);
    cfg.value.host_reserved_bytes = 4096;
    CHECK(planned.value.recommended_bytes <= UINT64_MAX -
          cfg.value.host_reserved_bytes);
    cfg.value.ram_budget_bytes = planned.value.recommended_bytes +
                                 cfg.value.host_reserved_bytes;

    waste_ctx *ctx = NULL;
    CHECK(waste_open(argv[1], &cfg.value, &ctx) == WASTE_OK);
    CHECK(ctx != NULL && cfg_intact(&cfg));

    plan_guard used = { BEFORE, {0}, AFTER };
    CHECK(waste_memory_used(ctx, &used.value) == WASTE_OK);
    CHECK(plan_intact(&used));
    CHECK(used.value.floor_bytes == planned.value.floor_bytes);
    CHECK(used.value.host_reserved_bytes == cfg.value.host_reserved_bytes);
    waste_close(ctx);

    puts("ABI OK");
    return 0;
}
