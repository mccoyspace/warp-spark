/* SPDX-License-Identifier: Apache-2.0 */
/* Stable cgroup capacity, current headroom, and automatic-budget arithmetic. */

#include "../src/memory.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#ifdef _WIN32
#include <direct.h>
#define make_dir(p) _mkdir(p)
#else
#include <sys/stat.h>
#define make_dir(p) mkdir((p), 0700)
#endif

#define GiB (1ULL << 30)
#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "memory check failed at line %d: %s\n", __LINE__, #x); \
    return 1; \
} } while (0)

static char ROOT[512], SELF[640], MEMINFO[640], PARENT[768], CHILD[896];

static int mk(const char *path)
{
    return make_dir(path) && errno != EEXIST ? -1 : 0;
}

static int put(const char *dir, const char *leaf, const char *text)
{
    char path[1100];
    snprintf(path, sizeof path, "%s/%s", dir, leaf);
    FILE *f = fopen(path, "w");
    if (!f) return -1;
    return fputs(text, f) < 0 || fclose(f) ? -1 : 0;
}

static int put_path(const char *path, const char *text)
{
    FILE *f = fopen(path, "w");
    if (!f) return -1;
    return fputs(text, f) < 0 || fclose(f) ? -1 : 0;
}

static int rm_leaf(const char *dir, const char *leaf)
{
    char path[1100];
    snprintf(path, sizeof path, "%s/%s", dir, leaf);
    return remove(path) && errno != ENOENT ? -1 : 0;
}

static int self_says(const char *line)
{
    return put_path(SELF, line);
}

static int policy_cases(void)
{
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

    /* K3-shaped whole-set arithmetic: 30 GiB floor plus three 17 GiB sets. */
    CHECK(waste_memory_auto_budget(30 * GiB, 81 * GiB, 100 * GiB) == 81 * GiB);
    CHECK(waste_memory_auto_budget(30 * GiB, 81 * GiB, 75 * GiB) == 64 * GiB);
    CHECK(waste_memory_auto_budget(30 * GiB, 81 * GiB, 50 * GiB) == 47 * GiB);
    CHECK(waste_memory_auto_budget(30 * GiB, 81 * GiB, 40 * GiB) == 30 * GiB);
    CHECK(waste_memory_auto_budget(30 * GiB, 81 * GiB, 29 * GiB) == 0);
    CHECK(waste_memory_auto_budget(30 * GiB, 29 * GiB, 0) == 0);
    return 0;
}

static int hierarchy_cases(const char *tmp)
{
    snprintf(ROOT, sizeof ROOT, "%s/cgroup", tmp);
    snprintf(SELF, sizeof SELF, "%s/self.cgroup", tmp);
    snprintf(MEMINFO, sizeof MEMINFO, "%s/meminfo", tmp);
    snprintf(PARENT, sizeof PARENT, "%s/parent", ROOT);
    snprintf(CHILD, sizeof CHILD, "%s/child", PARENT);
    CHECK(!mk(ROOT) && !mk(PARENT) && !mk(CHILD));
    CHECK(!put_path(MEMINFO,
                    "MemTotal: 134217728 kB\nMemAvailable: 121634816 kB\n"));
    CHECK(!self_says("0::/parent/child\n"));

    /* No finite max/high anywhere is an unconfined hierarchy. */
    CHECK(waste_cgroup_limit(SELF, ROOT) == 0);

    /* Empty 8 GiB group: reserve exactly 1 GiB once, not once in the stable
     * capacity and again in headroom. */
    CHECK(!put(CHILD, "memory.max", "8589934592\n"));
    CHECK(!put(CHILD, "memory.current", "0\n"));
    CHECK(!put(PARENT, "memory.max", "max\n"));
    CHECK(!put(PARENT, "memory.current", "0\n"));
    CHECK(waste_cgroup_limit(SELF, ROOT) == 8 * GiB);
    waste_memory_pressure p;
    waste_memory_pressure_read(&p, 128 * GiB, MEMINFO, SELF, ROOT);
    CHECK(p.available_known && p.available_bytes == 121634816ULL * 1024);
    CHECK(p.cgroup_limited && p.cgroup_ceiling_bytes == 7 * GiB);
    CHECK(waste_memory_safe_ceiling(8 * GiB, &p) == 7 * GiB);

    /* One GiB is already resident, so safe new allocation is 6 GiB. */
    CHECK(!put(CHILD, "memory.current", "1073741824\n"));
    waste_memory_pressure_read(&p, 128 * GiB, MEMINFO, SELF, ROOT);
    CHECK(p.cgroup_limited && p.cgroup_ceiling_bytes == 6 * GiB);

    /* A finite ancestor constrains a leaf; each level uses its own current. */
    CHECK(!put(PARENT, "memory.max", "17179869184\n"));
    CHECK(!put(PARENT, "memory.current", "13958643712\n"));
    waste_memory_pressure_read(&p, 128 * GiB, MEMINFO, SELF, ROOT);
    CHECK(p.cgroup_limited && p.cgroup_ceiling_bytes == 1 * GiB);

    /* memory.high participates in the same ancestor minimum. */
    CHECK(!put(PARENT, "memory.max", "max\n"));
    CHECK(!put(CHILD, "memory.max", "4294967296\n"));
    CHECK(!put(CHILD, "memory.high", "2147483648\n"));
    CHECK(!put(CHILD, "memory.current", "1073741824\n"));
    CHECK(waste_cgroup_limit(SELF, ROOT) == 2 * GiB);
    waste_memory_pressure_read(&p, 128 * GiB, MEMINFO, SELF, ROOT);
    CHECK(p.cgroup_limited && p.cgroup_ceiling_bytes == 3 * GiB / 4);

    /* At or above a finite limit is known exhaustion, not unknown. */
    CHECK(!put(CHILD, "memory.current", "3221225472\n"));
    waste_memory_pressure_read(&p, 128 * GiB, MEMINFO, SELF, ROOT);
    CHECK(p.cgroup_limited && p.cgroup_ceiling_bytes == 1);

    /* Missing current does not erase stable capacity. The dynamic layer has
     * no cgroup headroom sample, so its caller still starts from 7/8 usable. */
    CHECK(!rm_leaf(CHILD, "memory.current"));
    waste_memory_pressure_read(&p, 128 * GiB, MEMINFO, SELF, ROOT);
    CHECK(!p.cgroup_limited);
    CHECK(waste_cgroup_limit(SELF, ROOT) == 2 * GiB);
    CHECK(waste_memory_safe_ceiling(2 * GiB, &p) == 7 * GiB / 4);

    /* A mount containing only the leaf falls back to its mounted root. The
     * same covers private namespace 0::/ and unreadable /proc. */
    CHECK(!rm_leaf(CHILD, "memory.max") && !rm_leaf(CHILD, "memory.high"));
    CHECK(!rm_leaf(PARENT, "memory.max") && !rm_leaf(PARENT, "memory.current"));
    CHECK(!put(ROOT, "memory.max", "34359738368\n"));
    CHECK(!put(ROOT, "memory.current", "4294967296\n"));
    CHECK(!self_says("0::/system.slice/docker-0123456789ab.scope\n"));
    CHECK(waste_cgroup_limit(SELF, ROOT) == 32 * GiB);
    waste_memory_pressure_read(&p, 128 * GiB, MEMINFO, SELF, ROOT);
    CHECK(p.cgroup_limited && p.cgroup_ceiling_bytes == 24 * GiB);
    CHECK(waste_cgroup_limit("/nonexistent/self.cgroup", ROOT) == 32 * GiB);
    CHECK(waste_cgroup_limit(NULL, ROOT) == 32 * GiB);
    CHECK(!self_says("0::/\n"));
    CHECK(waste_cgroup_limit(SELF, ROOT) == 32 * GiB);

    /* Malformed telemetry claims nothing rather than inventing a number. */
    CHECK(!put(ROOT, "memory.max", "not-a-number\n"));
    CHECK(waste_cgroup_limit(SELF, ROOT) == 0);
    CHECK(!put(ROOT, "memory.max", "8589934592 trailing\n"));
    CHECK(waste_cgroup_limit(SELF, ROOT) == 0);

    /* Parent traversal in the supplied cgroup path is refused, then root
     * fallback still sees the mounted limit. */
    CHECK(!put(ROOT, "memory.max", "34359738368\n"));
    CHECK(!self_says("0::/../outside\n"));
    CHECK(waste_cgroup_limit(SELF, ROOT) == 32 * GiB);
    CHECK(!self_says("garbage with no unified line\n"));
    CHECK(waste_cgroup_limit(SELF, ROOT) == 32 * GiB);

    char slashed[520];
    snprintf(slashed, sizeof slashed, "%s/", ROOT);
    CHECK(!self_says("0::/parent/child\n"));
    CHECK(waste_cgroup_limit(SELF, slashed) == 32 * GiB);

    /* Unknown physical RAM still uses the finite group for its reserve. */
    CHECK(!put(ROOT, "memory.max", "8589934592\n"));
    CHECK(!put(ROOT, "memory.current", "1073741824\n"));
    waste_memory_pressure_read(&p, 0, MEMINFO, SELF, ROOT);
    CHECK(p.cgroup_limited && p.cgroup_ceiling_bytes == 6 * GiB);

    CHECK(!put_path(MEMINFO, "MemTotal: 1024 kB\nMemAvailable: broken kB\n"));
    CHECK(!put(ROOT, "memory.max", "broken\n"));
    waste_memory_pressure_read(&p, 128 * GiB, MEMINFO, SELF, ROOT);
    CHECK(!p.available_known && !p.cgroup_limited);
    return 0;
}

int main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(stderr, "usage: test_memory TMPDIR\n");
        return 2;
    }
    CHECK(waste_cgroup_limit("/proc/self/cgroup", NULL) == 0);
    CHECK(waste_cgroup_limit("/proc/self/cgroup", "") == 0);
    if (policy_cases() || hierarchy_cases(argv[1])) return 1;

    /* Production paths either report a plausible limit or no limit. */
    const uint64_t live = waste_cgroup_limit("/proc/self/cgroup",
                                             "/sys/fs/cgroup");
    CHECK(live == 0 || live > (1 << 20));
    puts("PASS memory budget and cgroup limits");
    return 0;
}
