/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/* Stable capacity and current-pressure accounting for automatic budgets.
 * All readers take paths as parameters, so cgroup namespace shapes and
 * malformed telemetry are tested without model loading or host mutation. */

#include "memory.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum read_result { READ_BAD = 0, READ_VALUE = 1, READ_UNLIMITED = 2 };

static uint64_t smaller_nonzero(uint64_t a, uint64_t b)
{
    if (!a) return b;
    if (!b) return a;
    return a < b ? a : b;
}

static enum read_result read_u64_path(const char *path, uint64_t *out)
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

static enum read_result read_u64_at(const char *dir, const char *leaf,
                                    uint64_t *out)
{
    char path[1600];
    const int n = snprintf(path, sizeof path, "%s/%s", dir, leaf);
    if (n < 0 || (size_t)n >= sizeof path) return READ_BAD;
    return read_u64_path(path, out);
}

/* A numeric zero is a known exhausted limit, represented as one byte so it
 * cannot be confused with the public 0 = unknown/unlimited convention. */
static uint64_t limits_at(const char *dir)
{
    uint64_t cap = 0, value = 0;
    if (read_u64_at(dir, "memory.max", &value) == READ_VALUE)
        cap = value ? value : 1;
    if (read_u64_at(dir, "memory.high", &value) == READ_VALUE)
        cap = smaller_nonzero(cap, value ? value : 1);
    return cap;
}

/* The unified hierarchy line is "0::" followed by an absolute path. Reject
 * parent traversal because tests supply these paths and must not escape the
 * cgroup root they were given. */
static int unified_path(const char *path, char *out, size_t cap)
{
    if (!path) return 0;
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    char line[1024];
    int found = 0;
    while (fgets(line, sizeof line, f)) {
        if (strncmp(line, "0::", 3)) continue;
        const char *rel = line + 3;
        const size_t n = strcspn(rel, "\r\n");
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

static int walk_start(const char *self_cgroup_path, const char *cgroup_root,
                      char *root, size_t root_cap, size_t *root_n_out,
                      char *dir, size_t dir_cap)
{
    if (!cgroup_root || !*cgroup_root) return 0;
    size_t root_n = strlen(cgroup_root);
    while (root_n > 1 && cgroup_root[root_n - 1] == '/') root_n--;
    if (root_n >= root_cap) return 0;
    memcpy(root, cgroup_root, root_n);
    root[root_n] = 0;

    char rel[512];
    int n = -1;
    if (unified_path(self_cgroup_path, rel, sizeof rel)) {
        if (!strcmp(rel, "/")) n = snprintf(dir, dir_cap, "%s", root);
        else if (root_n == 1 && root[0] == '/')
            n = snprintf(dir, dir_cap, "%s", rel);
        else n = snprintf(dir, dir_cap, "%s%s", root, rel);
    }
    /* A private namespace, a mount containing only the leaf, cgroup v1, or
     * missing /proc all converge on the mounted root. Reading that root is
     * what makes the fallback correct rather than "unlimited." */
    if (n < 0 || (size_t)n >= dir_cap)
        n = snprintf(dir, dir_cap, "%s", root);
    if (n < 0 || (size_t)n >= dir_cap) return 0;
    *root_n_out = root_n;
    return 1;
}

static int walk_parent(char *dir, size_t root_n)
{
    const size_t len = strlen(dir);
    if (len <= root_n) return 0;
    char *slash = strrchr(dir, '/');
    if (!slash) return 0;
    const size_t at = (size_t)(slash - dir);
    if (root_n == 1) {
        if (at == 0) dir[1] = 0;
        else *slash = 0;
        return 1;
    }
    if (at < root_n) return 0;
    if (at == root_n) dir[root_n] = 0;
    else *slash = 0;
    return 1;
}

uint64_t waste_cgroup_limit(const char *self_cgroup_path,
                            const char *cgroup_root)
{
    char root[1024], dir[1600];
    size_t root_n = 0;
    if (!walk_start(self_cgroup_path, cgroup_root, root, sizeof root,
                    &root_n, dir, sizeof dir)) return 0;

    uint64_t cap = 0;
    for (;;) {
        cap = smaller_nonzero(cap, limits_at(dir));
        if (!walk_parent(dir, root_n)) break;
    }
    return cap;
}

static int read_mem_available(const char *path, uint64_t *out)
{
    if (!path) return 0;
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

static uint64_t cgroup_headroom_ceiling(uint64_t physical,
                                        const char *self_cgroup_path,
                                        const char *cgroup_root,
                                        int *known)
{
    char root[1024], dir[1600];
    size_t root_n = 0;
    *known = 0;
    if (!walk_start(self_cgroup_path, cgroup_root, root, sizeof root,
                    &root_n, dir, sizeof dir)) return 0;

    uint64_t cap = 0;
    for (;;) {
        const uint64_t limit = limits_at(dir);
        uint64_t current = 0;
        if (limit && read_u64_at(dir, "memory.current", &current) == READ_VALUE) {
            const uint64_t effective = physical && physical < limit
                                     ? physical : limit;
            const uint64_t reserve = effective / 8;
            const uint64_t headroom = limit > current ? limit - current : 0;
            const uint64_t here = headroom > reserve
                                ? headroom - reserve : 1;
            cap = smaller_nonzero(cap, here);
            *known = 1;
        }
        if (!walk_parent(dir, root_n)) break;
    }
    return cap;
}

void waste_memory_pressure_read(waste_memory_pressure *out, uint64_t physical,
                                const char *meminfo_path,
                                const char *self_cgroup_path,
                                const char *cgroup_root)
{
    if (!out) return;
    memset(out, 0, sizeof *out);
    out->available_known = read_mem_available(meminfo_path,
                                               &out->available_bytes);
    out->cgroup_ceiling_bytes = cgroup_headroom_ceiling(
        physical, self_cgroup_path, cgroup_root, &out->cgroup_limited);
}

uint64_t waste_memory_safe_ceiling(uint64_t usable_capacity,
                                   const waste_memory_pressure *pressure)
{
    uint64_t cap = usable_capacity
                 ? usable_capacity - usable_capacity / 8 : 0;
    if (!pressure) return cap;

    if (pressure->available_known) {
        const uint64_t reserve = usable_capacity
                               ? usable_capacity / 8
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
