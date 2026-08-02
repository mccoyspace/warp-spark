/* Print the memory plan without loading the model. */
#include "waste.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

static double gib(uint64_t n)
{
    return (double)n / (double)(1ULL << 30);
}

int main(int argc, char **argv)
{
    if (argc < 2 || argc > 3) {
        fprintf(stderr, "usage: %s MODEL [CONTEXT_TOKENS]\n", argv[0]);
        return 2;
    }

    const uint32_t context = argc == 3 ? (uint32_t)strtoul(argv[2], NULL, 10)
                                       : 4096;
    waste_memplan plan;
    const waste_status st = waste_plan_memory(argv[1], context, &plan);
    if (st != WASTE_OK) {
        fprintf(stderr, "plan: %s\n", waste_strerror(st));
        return 1;
    }

    printf("context:     %" PRIu32 " tokens\n", context);
    printf("trunk:       %.2f GiB\n", gib(plan.trunk_bytes));
    printf("state:       %.2f GiB\n", gib(plan.state_bytes));
    printf("scratch:     %.2f GiB\n", gib(plan.scratch_bytes));
    printf("floor:       %.2f GiB\n", gib(plan.floor_bytes));
    printf("recommended: %.2f GiB\n", gib(plan.recommended_bytes));
    printf("vision:      %.2f GiB\n", gib(plan.vision_bytes));
    return 0;
}
