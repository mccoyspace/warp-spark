/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/*
 * ecache.h — bounded expert cache over the per-layer banks.
 *
 * This is the piece the whole K3 argument rests on: the model does not fit
 * in RAM, so experts are streamed from disk and a bounded cache decides how
 * often that costs a read. Gate 2 measured the hit rates in simulation;
 * this implements them for real.
 *
 * Policy is LFRU — frequency first, recency as tiebreak, victims chosen
 * from a small random sample rather than a full scan (the same
 * approximation the Gate 2 simulator used).
 * Gate 2 also showed why: at small cache fractions plain LRU collapses to
 * 5% where LFRU still gets 29%.
 *
 * Reads bypass the page cache (F_NOCACHE on macOS, O_DIRECT on Linux,
 * FILE_FLAG_NO_BUFFERING on Windows). That is deliberate:
 * with a 17 GB container on a 64 GB machine the kernel would cache
 * everything and the hit rate we measure would be a fiction. K3's ~900 GB
 * gets no such help, so the engine must not depend on it.
 */

#ifndef WASTE_ECACHE_H
#define WASTE_ECACHE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int32_t  key;        /* layer<<16 | expert, or -1 when empty            */
    uint32_t hits;       /* LFRU frequency term                             */
    uint64_t last;       /* LFRU recency term                               */
    uint8_t *data;
    int      pinned;     /* transient: a batch still needs this slot       */
} waste_eslot;

typedef struct {
    waste_eslot *slot;
    int32_t *hash;       /* open addressing: hash -> slot index, -1 empty   */
    int n_slots, hash_mask;
    size_t rec_bytes, budget_bytes;
    uint64_t clock, hits, misses, bytes_read, evictions;
    unsigned rng;
    int policy;          /* 0 = LFRU, 1 = LRU                               */
} waste_ecache;

/* O_DIRECT requires the destination buffer to be aligned to the device's
 * logical block size, so record buffers come from here rather than malloc.
 * Expert records are whole 4 KiB pages by construction, which covers both
 * the 512- and 4096-byte cases; on macOS the alignment is merely harmless. */
/* 16 KiB, the Apple Silicon page: O_DIRECT wants 512 or 4096, and Metal's
 * newBufferWithBytesNoCopy wants a whole page — one alignment serves both,
 * which is what lets the GPU read trunk weights the CPU already has. */
#define WASTE_DIO_ALIGN 16384
void *waste_dio_alloc(size_t n);
void  waste_dio_free(void *p);

/* budget_bytes 0 disables caching (every access reads). Returns 0 on ok. */
int  waste_ecache_init(waste_ecache *c, size_t budget_bytes, size_t rec_bytes,
                       int policy);
void waste_ecache_free(waste_ecache *c);

/* Returns a pointer to the expert's record bytes, reading it through
 * `fetch` on a miss. `fetch(user, layer, expert, dst)` must fill rec_bytes
 * and return 0 on success. NULL on failure. */
typedef int (*waste_fetch_fn)(void *user, int layer, int expert, uint8_t *dst);
const uint8_t *waste_ecache_get(waste_ecache *c, int layer, int expert,
                                waste_fetch_fn fetch, void *user);

/* Batch form used by the Linux async backend. All requests name one layer,
 * as one router produces them. Existing hits and newly reserved miss slots
 * are pinned until fetch_many completes, so another miss in the same batch
 * cannot overwrite a record whose pointer the caller has not consumed yet.
 * Returns 0 and fills out[] on success; -1 when caching is disabled or a
 * fetch fails. */
typedef int (*waste_fetch_many_fn)(void *user, int n, const int *layers,
                                   const int *experts, uint8_t *const *dst);
int waste_ecache_get_many(waste_ecache *c, int layer, const int *experts,
                          int n, waste_fetch_many_fn fetch_many, void *user,
                          const uint8_t **out);

/* Persist / restore which experts this workload actually uses. Gate 5
 * measured a 0% hit rate until the cache exceeds one token's working set;
 * a warm start does not change that floor, but it does remove the cold
 * ramp at the beginning of every run. */
int waste_ecache_save_usage(const waste_ecache *c, const char *path,
                            uint64_t tokens);
int waste_ecache_warm(waste_ecache *c, const char *path,
                      waste_fetch_fn fetch, void *user);

static inline double waste_ecache_hit_rate(const waste_ecache *c)
{
    const uint64_t t = c->hits + c->misses;
    return t ? (double)c->hits / (double)t : 0.0;
}

#ifdef __cplusplus
}
#endif
#endif /* WASTE_ECACHE_H */
