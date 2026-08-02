/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/* Host memory pressure, kept private so the automatic-budget policy can be
 * tested without allocating a model or depending on the test host's /proc. */

#ifndef WASTE_MEMORY_H
#define WASTE_MEMORY_H

#include <stdint.h>

typedef struct {
    uint64_t available_bytes;       /* Linux MemAvailable                 */
    uint64_t cgroup_ceiling_bytes;  /* safe new allocation inside cgroup */
    int available_known;
    int cgroup_limited;
} waste_memory_pressure;

/* Read Linux's current memory constraints. The three paths are parameters
 * for deterministic tests; production passes /proc/meminfo,
 * /proc/self/cgroup and /sys/fs/cgroup. Other platforms return no current
 * constraints and keep the physical-RAM rule unchanged. */
void waste_memory_pressure_read(waste_memory_pressure *out, uint64_t physical,
                                const char *meminfo_path,
                                const char *self_cgroup_path,
                                const char *cgroup_root);

/* 0 means that no ceiling could be determined. A known exhausted source is
 * represented by a one-byte ceiling, which cannot be mistaken for unknown. */
uint64_t waste_memory_safe_ceiling(uint64_t physical,
                                   const waste_memory_pressure *pressure);

/* Apply WASTE's whole-working-set rule. Returns 0 when the known ceiling
 * cannot hold even the floor. */
uint64_t waste_memory_auto_budget(uint64_t floor, uint64_t recommended,
                                  uint64_t ceiling);

#endif /* WASTE_MEMORY_H */
