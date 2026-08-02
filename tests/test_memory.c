/* SPDX-License-Identifier: Apache-2.0 */
#include "../src/memory.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#if defined(__linux__)
#include <sys/stat.h>

static int put(const char *path, const char *text)
{
    FILE *f = fopen(path, "w");
    if (!f) return -1;
    const int bad = fputs(text, f) < 0 || fclose(f);
    return bad ? -1 : 0;
}

static int mkdir_one(const char *path)
{
    return mkdir(path, 0700) && errno != EEXIST ? -1 : 0;
}
#endif

#define GiB (1ULL << 30)
#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "memory check failed at line %d: %s\n", __LINE__, #x); \
    return 1; \
} } while (0)

int main(int argc, char **argv)
{
    if (argc != 2) return 2;

    waste_memory_pressure p;
    memset(&p, 0, sizeof p);
    CHECK(waste_memory_safe_ceiling(128 * GiB, &p) == 112 * GiB);

    p.available_known = 1;
    p.available_bytes = 116 * GiB;
    CHECK(waste_memory_safe_ceiling(128 * GiB, &p) == 100 * GiB);
    p.cgroup_limited = 1;
    p.cgroup_ceiling_bytes = 6 * GiB;
    CHECK(waste_memory_safe_ceiling(128 * GiB, &p) == 6 * GiB);

    memset(&p, 0, sizeof p);
    p.available_known = 1;
    p.available_bytes = 8 * GiB;
    CHECK(waste_memory_safe_ceiling(0, &p) == 7 * GiB);

    /* K3-shaped whole-set arithmetic: 30 GiB floor + three 17 GiB sets. */
    CHECK(waste_memory_auto_budget(30 * GiB, 81 * GiB, 100 * GiB) == 81 * GiB);
    CHECK(waste_memory_auto_budget(30 * GiB, 81 * GiB, 75 * GiB) == 64 * GiB);
    CHECK(waste_memory_auto_budget(30 * GiB, 81 * GiB, 50 * GiB) == 47 * GiB);
    CHECK(waste_memory_auto_budget(30 * GiB, 81 * GiB, 40 * GiB) == 30 * GiB);
    CHECK(waste_memory_auto_budget(30 * GiB, 81 * GiB, 29 * GiB) == 0);
    CHECK(waste_memory_auto_budget(30 * GiB, 29 * GiB, 0) == 0);

#if defined(__linux__)
    char meminfo[1024], self[1024], root[1024];
    char parent[1200], child[1400], path[1600];
    snprintf(meminfo, sizeof meminfo, "%s/meminfo", argv[1]);
    snprintf(self, sizeof self, "%s/self.cgroup", argv[1]);
    snprintf(root, sizeof root, "%s/cgroup", argv[1]);
    CHECK(!mkdir_one(root));
    snprintf(parent, sizeof parent, "%s/parent", root);
    snprintf(child, sizeof child, "%s/child", parent);
    CHECK(!mkdir_one(parent));
    CHECK(!mkdir_one(child));

    CHECK(!put(meminfo, "MemTotal: 134217728 kB\nMemAvailable: 121634816 kB\n"));
    CHECK(!put(self, "0::/parent/child\n"));
    snprintf(path, sizeof path, "%s/memory.max", child);
    CHECK(!put(path, "8589934592\n"));       /* 8 GiB */
    snprintf(path, sizeof path, "%s/memory.current", child);
    CHECK(!put(path, "1073741824\n"));       /* 1 GiB */
    snprintf(path, sizeof path, "%s/memory.max", parent);
    CHECK(!put(path, "max\n"));
    snprintf(path, sizeof path, "%s/memory.current", parent);
    CHECK(!put(path, "1073741824\n"));

    waste_memory_pressure_read(&p, 128 * GiB, meminfo, self, root);
    CHECK(p.available_known && p.available_bytes == 121634816ULL * 1024);
    CHECK(p.cgroup_limited && p.cgroup_ceiling_bytes == 6 * GiB);
    /* The old implementation subtracted the host's 16 GiB reserve from
     * this 7 GiB headroom and returned one byte. The group now keeps its
     * own 1 GiB reserve and leaves 6 GiB usable. */

    /* A finite ancestor constrains a leaf whose own memory.max is "max". */
    snprintf(path, sizeof path, "%s/memory.max", parent);
    CHECK(!put(path, "17179869184\n"));       /* 16 GiB */
    snprintf(path, sizeof path, "%s/memory.current", parent);
    CHECK(!put(path, "13958643712\n"));       /* 13 GiB */
    waste_memory_pressure_read(&p, 128 * GiB, meminfo, self, root);
    CHECK(p.cgroup_limited && p.cgroup_ceiling_bytes == 1 * GiB);

    /* At or above a finite limit is a known exhausted constraint, not the
     * zero value used for unknown/unlimited. */
    snprintf(path, sizeof path, "%s/memory.current", child);
    CHECK(!put(path, "9663676416\n"));        /* 9 GiB > 8 GiB */
    waste_memory_pressure_read(&p, 128 * GiB, meminfo, self, root);
    CHECK(p.cgroup_limited && p.cgroup_ceiling_bytes == 1);

    CHECK(!put(meminfo, "MemTotal: 1024 kB\nMemAvailable: broken kB\n"));
    CHECK(!put(self, "0::/../outside\n"));
    waste_memory_pressure_read(&p, 128 * GiB, meminfo, self, root);
    CHECK(!p.available_known && !p.cgroup_limited);
#else
    (void)argv;
    waste_memory_pressure_read(&p, 128 * GiB, NULL, NULL, NULL);
    CHECK(!p.available_known && !p.cgroup_limited);
#endif

    puts("PASS memory budget");
    return 0;
}
