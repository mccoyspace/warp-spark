/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/* Memory capacity and current-pressure accounting. Kept independent of model
 * loading so policy tests can use synthetic files and finish in milliseconds. */

#ifndef WASTE_MEMORY_H
#define WASTE_MEMORY_H

#include <stdint.h>

typedef struct {
    uint64_t available_bytes;       /* Linux MemAvailable                 */
    uint64_t cgroup_ceiling_bytes;  /* safe new allocation inside cgroup */
    int available_known;
    int cgroup_limited;
} waste_memory_pressure;

/* The smallest finite cgroup-v2 memory.max or memory.high applying to this
 * process, or 0 when none can be determined. Both paths are parameters so
 * hierarchy and namespace shapes can be checked with synthetic files on every
 * platform. Production passes /proc/self/cgroup and /sys/fs/cgroup. */
uint64_t waste_cgroup_limit(const char *self_cgroup_path,
                            const char *cgroup_root);

/* Read current memory constraints. The paths are parameters for deterministic
 * tests. Production uses /proc/meminfo, /proc/self/cgroup and /sys/fs/cgroup;
 * missing paths simply contribute no current-pressure constraint. */
void waste_memory_pressure_read(waste_memory_pressure *out, uint64_t physical,
                                const char *meminfo_path,
                                const char *self_cgroup_path,
                                const char *cgroup_root);

/* Apply the one-eighth reserve to stable usable capacity, then further bound
 * it by current host and cgroup headroom. A known exhausted source is encoded
 * as a one-byte ceiling; 0 means no ceiling could be determined. */
uint64_t waste_memory_safe_ceiling(uint64_t usable_capacity,
                                   const waste_memory_pressure *pressure);

/* Apply WASTE's whole-working-set rule. Returns 0 when the known ceiling
 * cannot hold even the floor. */
uint64_t waste_memory_auto_budget(uint64_t floor, uint64_t recommended,
                                  uint64_t ceiling);

#endif /* WASTE_MEMORY_H */
