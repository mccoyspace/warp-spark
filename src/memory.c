/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/* Linux memory accounting for automatic budgets. This is deliberately
 * independent of model loading: policy tests should take milliseconds and
 * synthetic proc files, not a multi-gigabyte container. */

#include "memory.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t smaller_nonzero(uint64_t a, uint64_t b)
{
    if (!a) return b;
    if (!b) return a;
    return a < b ? a : b;
}

uint64_t waste_memory_safe_ceiling(uint64_t physical,
                                   const waste_memory_pressure *pressure)
{
    uint64_t cap = physical ? physical - physical / 8 : 0;
    if (!pressure) return cap;

    if (pressure->available_known) {
        /* MemAvailable is current host headroom. Preserve one eighth of the
         * host, not one eighth of the fluctuating headroom: that is the same
         * reserve the physical-RAM rule promised before current pressure was
         * visible. With no physical figure, retain one eighth of the only
         * capacity figure available. */
        const uint64_t reserve = physical ? physical / 8
                                          : pressure->available_bytes / 8;
        const uint64_t available_cap = pressure->available_bytes > reserve
                                     ? pressure->available_bytes - reserve : 1;
        cap = smaller_nonzero(cap, available_cap);
    }
    if (pressure->cgroup_limited)
        cap = smaller_nonzero(cap, pressure->cgroup_ceiling_bytes
                                 ? pressure->cgroup_ceiling_bytes : 1);
    return cap;
}

uint64_t waste_memory_auto_budget(uint64_t floor, uint64_t recommended,
                                  uint64_t ceiling)
{
    if (recommended < floor || (ceiling && floor > ceiling)) return 0;
    const uint64_t working_set = (recommended - floor) / 3;
    uint64_t budget = floor;
    for (int k = 3; k >= 1; k--) {
        const uint64_t n = (uint64_t)k;
        if (working_set && working_set <= (UINT64_MAX - floor) / n) {
            const uint64_t candidate = floor + working_set * n;
            if (!ceiling || candidate <= ceiling) {
                budget = candidate;
                break;
            }
        }
    }
    return budget;
}

#if defined(__linux__)

enum read_result { READ_BAD = 0, READ_VALUE = 1, READ_UNLIMITED = 2 };

static enum read_result read_u64_file(const char *path, uint64_t *out)
{
    FILE *f = fopen(path, "r");
    if (!f) return READ_BAD;
    char b[128];
    const int got = fgets(b, sizeof b, f) != NULL;
    fclose(f);
    if (!got) return READ_BAD;

    char *p = b;
    while (*p == ' ' || *p == '\t') p++;
    if (!strncmp(p, "max", 3) && (p[3] == 0 || p[3] == '\n' ||
                                  p[3] == '\r' || p[3] == ' ' ||
                                  p[3] == '\t'))
        return READ_UNLIMITED;
    errno = 0;
    char *end = NULL;
    const unsigned long long v = strtoull(p, &end, 10);
    if (errno || end == p) return READ_BAD;
    while (*end == ' ' || *end == '\t' || *end == '\r' || *end == '\n') end++;
    if (*end) return READ_BAD;
    *out = (uint64_t)v;
    return READ_VALUE;
}

static int read_mem_available(const char *path, uint64_t *out)
{
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    char line[256];
    int found = 0;
    while (fgets(line, sizeof line, f)) {
        if (strncmp(line, "MemAvailable:", 13)) continue;
        char *p = line + 13, *end = NULL;
        while (*p == ' ' || *p == '\t') p++;
        errno = 0;
        const unsigned long long kib = strtoull(p, &end, 10);
        if (!errno && end != p && kib <= UINT64_MAX / 1024) {
            while (*end == ' ' || *end == '\t') end++;
            if (!strncmp(end, "kB", 2) &&
                (end[2] == 0 || end[2] == '\r' || end[2] == '\n')) {
                *out = (uint64_t)kib * 1024;
                found = 1;
            }
        }
        break;
    }
    fclose(f);
    return found;
}

static int unified_cgroup_path(const char *path, char *out, size_t cap)
{
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    char line[1024];
    int found = 0;
    while (fgets(line, sizeof line, f)) {
        if (strncmp(line, "0::", 3)) continue;
        const char *rel = line + 3;
        const size_t n = strcspn(rel, "\r\n");
        /* The kernel emits an absolute cgroup path. Reject parent traversal
         * as well: the injected paths used by tests must not turn this reader
         * into a way to escape the supplied cgroup root. */
        if (!n || rel[0] != '/' || n >= cap) break;
        int bad = 0;
        for (size_t i = 0; i + 1 < n; i++)
            if (rel[i] == '.' && rel[i + 1] == '.' &&
                (i == 0 || rel[i - 1] == '/') &&
                (i + 2 == n || rel[i + 2] == '/')) bad = 1;
        if (bad) break;
        memcpy(out, rel, n);
        out[n] = 0;
        found = 1;
        break;
    }
    fclose(f);
    return found;
}

static uint64_t cgroup_safe_ceiling(uint64_t physical,
                                    const char *self_cgroup_path,
                                    const char *root, int *limited)
{
    char rel[900];
    if (!self_cgroup_path || !root ||
        !unified_cgroup_path(self_cgroup_path, rel, sizeof rel)) return 0;

    char dir[1400];
    size_t root_n = strlen(root);
    while (root_n > 1 && root[root_n - 1] == '/') root_n--;
    const int n = snprintf(dir, sizeof dir, "%.*s%s",
                           (int)root_n, root, !strcmp(rel, "/") ? "" : rel);
    if (n < 0 || (size_t)n >= sizeof dir) return 0;

    uint64_t cap = 0;
    for (;;) {
        char max_path[1600], current_path[1600];
        const int a = snprintf(max_path, sizeof max_path, "%s/memory.max", dir);
        const int b = snprintf(current_path, sizeof current_path,
                               "%s/memory.current", dir);
        if (a > 0 && (size_t)a < sizeof max_path &&
            b > 0 && (size_t)b < sizeof current_path) {
            uint64_t max = 0, current = 0;
            const enum read_result mr = read_u64_file(max_path, &max);
            const enum read_result cr = read_u64_file(current_path, &current);
            if (mr == READ_VALUE && cr == READ_VALUE) {
                /* A small cgroup is a small machine, not a slice of the
                 * host that owes the host's entire reserve. Retain one eighth
                 * of min(cgroup limit, physical RAM). This fixes the earlier
                 * rule where an 8 GiB group on a 128 GiB host lost 16 GiB to
                 * reserve and could never open even a tiny model. */
                const uint64_t effective = physical && physical < max
                                         ? physical : max;
                const uint64_t reserve = effective / 8;
                const uint64_t headroom = max > current ? max - current : 0;
                const uint64_t here = headroom > reserve
                                    ? headroom - reserve : 1;
                cap = smaller_nonzero(cap, here);
                *limited = 1;
            }
        }

        /* memory.max is hierarchical. A leaf can say "max" while a parent
         * is finite, so walk to the injected mount root and retain the most
         * restrictive ancestor rather than trusting only the leaf. */
        const size_t len = strlen(dir);
        if (len <= root_n) break;
        char *slash = strrchr(dir, '/');
        if (!slash || (size_t)(slash - dir) < root_n) break;
        if ((size_t)(slash - dir) == root_n) dir[root_n] = 0;
        else *slash = 0;
    }
    return cap;
}

#endif /* __linux__ */

void waste_memory_pressure_read(waste_memory_pressure *out, uint64_t physical,
                                const char *meminfo_path,
                                const char *self_cgroup_path,
                                const char *cgroup_root)
{
    if (!out) return;
    memset(out, 0, sizeof *out);
#if defined(__linux__)
    if (meminfo_path)
        out->available_known = read_mem_available(meminfo_path,
                                                   &out->available_bytes);
    out->cgroup_ceiling_bytes = cgroup_safe_ceiling(physical,
                                                     self_cgroup_path,
                                                     cgroup_root,
                                                     &out->cgroup_limited);
#else
    (void)physical;
    (void)meminfo_path;
    (void)self_cgroup_path;
    (void)cgroup_root;
#endif
}
