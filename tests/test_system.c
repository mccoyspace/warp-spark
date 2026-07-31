/* SPDX-License-Identifier: Apache-2.0 */
#define _GNU_SOURCE
#include "../src/waste.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__linux__)
#include <sched.h>
#include <sys/stat.h>
#include <unistd.h>

static int put(const char *path, const char *text)
{
    FILE *f = fopen(path, "w");
    if (!f) return -1;
    const int rc = fputs(text, f) < 0 || fclose(f);
    return rc ? -1 : 0;
}
#endif

int main(int argc, char **argv)
{
    if (argc != 2) return 2;
#if !defined(__linux__)
    (void)argv;
#endif
    char resolved[256];
    if (waste_set_cpu_affinity("none", resolved, sizeof resolved) != WASTE_OK ||
        strcmp(resolved, "none")) return 1;
#if defined(__linux__)
    char p[1024], cg[512];
    snprintf(p, sizeof p, "%s/meminfo", argv[1]);
    if (put(p, "MemTotal: 99999 kB\nMemAvailable: 12345 kB\n")) return 1;
    setenv("WASTE_MEMINFO_PATH", p, 1);
    if (waste_available_ram() != 12345ULL * 1024) return 1;

    snprintf(cg, sizeof cg, "%s/cgroup", argv[1]);
    if (mkdir(cg, 0700)) return 1;
    snprintf(p, sizeof p, "%s/memory.max", cg);
    if (put(p, "1000000\n")) return 1;
    snprintf(p, sizeof p, "%s/memory.current", cg);
    if (put(p, "250000\n")) return 1;
    setenv("WASTE_CGROUP_DIR", cg, 1);
    if (waste_cgroup_available_ram() != 750000) return 1;
    snprintf(p, sizeof p, "%s/memory.current", cg);
    if (put(p, "1000001\n")) return 1;
    if (waste_cgroup_available_ram() != 1) return 1;

    cpu_set_t before;
    if (sched_getaffinity(0, sizeof before, &before)) return 1;
    const waste_status perf = waste_set_cpu_affinity("performance", resolved,
                                                      sizeof resolved);
    if (perf != WASTE_OK && perf != WASTE_E_UNSUPPORTED) return 1;
    if (perf == WASTE_OK && !resolved[0]) return 1;
    if (sched_setaffinity(0, sizeof before, &before)) return 1;
    int first = -1;
    for (int c = 0; c < CPU_SETSIZE; c++)
        if (CPU_ISSET(c, &before)) { first = c; break; }
    if (first < 0) return 1;
    char spec[32]; snprintf(spec, sizeof spec, "%d", first);
    if (waste_set_cpu_affinity(spec, resolved, sizeof resolved) != WASTE_OK ||
        strcmp(resolved, spec)) return 1;
    if (sched_setaffinity(0, sizeof before, &before)) return 1;
    unsetenv("WASTE_MEMINFO_PATH"); unsetenv("WASTE_CGROUP_DIR");
#else
    if (waste_available_ram() != 0 || waste_cgroup_available_ram() != 0)
        return 1;
#endif
    puts("PASS system");
    return 0;
}
