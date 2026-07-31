/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/* Linux host accounting and CPU placement. Kept outside the model so the
 * policy is testable without loading weights. */

#define _GNU_SOURCE
#include "waste.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__linux__)
#include <sched.h>
#include <unistd.h>
#endif

#if defined(__linux__)
static uint64_t read_u64(const char *path, int kib)
{
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    char b[128];
    const int ok = fgets(b, sizeof b, f) != NULL;
    fclose(f);
    if (!ok || !strncmp(b, "max", 3)) return 0;
    char *end = NULL;
    errno = 0;
    const unsigned long long v = strtoull(b, &end, 10);
    if (errno || end == b) return 0;
    if (kib && v > UINT64_MAX / 1024) return 0;
    return kib ? (uint64_t)v * 1024 : (uint64_t)v;
}
#endif

uint64_t waste_available_ram(void)
{
#if defined(__linux__)
    const char *path = getenv("WASTE_MEMINFO_PATH");
    if (!path || !*path) path = "/proc/meminfo";
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    char line[256];
    uint64_t out = 0;
    while (fgets(line, sizeof line, f)) {
        unsigned long long kib = 0;
        if (sscanf(line, "MemAvailable: %llu kB", &kib) == 1) {
            if (kib <= UINT64_MAX / 1024) out = (uint64_t)kib * 1024;
            break;
        }
    }
    fclose(f);
    return out;
#else
    return 0;
#endif
}

uint64_t waste_cgroup_available_ram(void)
{
#if defined(__linux__)
    char dir[1024];
    const char *forced = getenv("WASTE_CGROUP_DIR");
    if (forced && *forced) {
        snprintf(dir, sizeof dir, "%s", forced);
    } else {
        FILE *f = fopen("/proc/self/cgroup", "r");
        if (!f) return 0;
        char line[1024], rel[900] = "";
        while (fgets(line, sizeof line, f)) {
            if (!strncmp(line, "0::", 3)) {
                const size_t n = strcspn(line + 3, "\r\n");
                if (n >= sizeof rel) { fclose(f); return 0; }
                memcpy(rel, line + 3, n);
                rel[n] = 0;
                break;
            }
        }
        fclose(f);
        if (!*rel) return 0;
        snprintf(dir, sizeof dir, "/sys/fs/cgroup%s", rel);
    }
    char pmax[1200], pcur[1200];
    snprintf(pmax, sizeof pmax, "%s/memory.max", dir);
    snprintf(pcur, sizeof pcur, "%s/memory.current", dir);
    const uint64_t max = read_u64(pmax, 0);
    const uint64_t cur = read_u64(pcur, 0);
    /* Zero means unknown or unlimited to the caller.  A finite cgroup that
     * is already at its limit must therefore return a nonzero sentinel,
     * otherwise it would accidentally disable the safety ceiling. */
    return max ? (max > cur ? max - cur : 1) : 0;
#else
    return 0;
#endif
}

#if defined(__linux__)
static int parse_cpulist(const char *s, cpu_set_t *set)
{
    CPU_ZERO(set);
    if (!s || !*s) return -1;
    const char *p = s;
    while (*p) {
        char *end = NULL;
        errno = 0;
        const long first = strtol(p, &end, 10);
        if (errno || end == p || first < 0 || first >= CPU_SETSIZE) return -1;
        long last = first;
        p = end;
        if (*p == '-') {
            p++;
            errno = 0;
            last = strtol(p, &end, 10);
            if (errno || end == p || last < first || last >= CPU_SETSIZE) return -1;
            p = end;
        }
        for (long c = first; c <= last; c++) CPU_SET((int)c, set);
        if (!*p) break;
        if (*p++ != ',') return -1;
    }
    return CPU_COUNT(set) ? 0 : -1;
}

static void format_cpulist(const cpu_set_t *set, char *out, size_t cap)
{
    if (!out || !cap) return;
    size_t used = 0;
    out[0] = 0;
    for (int c = 0; c < CPU_SETSIZE; c++) {
        if (!CPU_ISSET(c, set)) continue;
        int last = c;
        while (last + 1 < CPU_SETSIZE && CPU_ISSET(last + 1, set)) last++;
        const int n = last > c
            ? snprintf(out + used, cap > used ? cap - used : 0,
                       "%s%d-%d", used ? "," : "", c, last)
            : snprintf(out + used, cap > used ? cap - used : 0,
                       "%s%d", used ? "," : "", c);
        if (n < 0 || (size_t)n >= (cap > used ? cap - used : 0)) {
            out[cap - 1] = 0;
            return;
        }
        used += (size_t)n;
        c = last;
    }
}

static uint64_t cpu_metric(const char *root, int cpu, const char *leaf)
{
    char path[1024];
    snprintf(path, sizeof path, "%s/cpu%d/%s", root, cpu, leaf);
    return read_u64(path, 0);
}

static int performance_set(cpu_set_t *out, const cpu_set_t *allowed)
{
    const char *root = getenv("WASTE_SYS_CPU_ROOT");
    if (!root || !*root) root = "/sys/devices/system/cpu";
    uint64_t max = 0;
    const char *leaf = "cpufreq/cpuinfo_max_freq";
    for (int c = 0; c < CPU_SETSIZE; c++)
        if (CPU_ISSET(c, allowed)) {
            const uint64_t v = cpu_metric(root, c, leaf);
            if (v > max) max = v;
        }
    if (!max) {
        leaf = "cpu_capacity";
        for (int c = 0; c < CPU_SETSIZE; c++)
            if (CPU_ISSET(c, allowed)) {
                const uint64_t v = cpu_metric(root, c, leaf);
                if (v > max) max = v;
            }
    }
    if (!max) return -1;
    CPU_ZERO(out);
    for (int c = 0; c < CPU_SETSIZE; c++) {
        if (!CPU_ISSET(c, allowed)) continue;
        const uint64_t v = cpu_metric(root, c, leaf);
        /* Capacity values can vary slightly per CPU on the same core class;
         * maximum frequency is normally exact. Ninety percent separates the
         * GN100's X925 and A725 classes without selecting only one X925. */
        if (v && v >= (max * 9) / 10) CPU_SET(c, out);
    }
    return CPU_COUNT(out) ? 0 : -1;
}
#endif

waste_status waste_set_cpu_affinity(const char *spec,
                                    char *resolved, size_t resolved_cap)
{
    if (resolved && resolved_cap) resolved[0] = 0;
    if (!spec || !*spec || !strcmp(spec, "none")) {
        if (resolved && resolved_cap) snprintf(resolved, resolved_cap, "none");
        return WASTE_OK;
    }
#if defined(__linux__)
    cpu_set_t allowed, chosen;
    if (sched_getaffinity(0, sizeof allowed, &allowed) != 0) return WASTE_E_IO;
    if (!strcmp(spec, "performance")) {
        if (performance_set(&chosen, &allowed)) return WASTE_E_UNSUPPORTED;
    } else {
        if (parse_cpulist(spec, &chosen)) return WASTE_E_ARG;
        for (int c = 0; c < CPU_SETSIZE; c++)
            if (CPU_ISSET(c, &chosen) && !CPU_ISSET(c, &allowed)) return WASTE_E_ARG;
    }
    if (sched_setaffinity(0, sizeof chosen, &chosen) != 0) return WASTE_E_IO;
    format_cpulist(&chosen, resolved, resolved_cap);
    return WASTE_OK;
#else
    (void)spec;
    return WASTE_E_UNSUPPORTED;
#endif
}
