/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/* ecache.c — see ecache.h. */

#include "ecache.h"

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "platform.h"
#include "waste_format.h"

#define EC_SAMPLE 16      /* victims sampled per eviction (Redis-style)     */

/* Rounding the length up as well as the address is not decoration: both
 * O_DIRECT and FILE_FLAG_NO_BUFFERING refuse a transfer whose length is
 * not a whole number of sectors, and the tail of a record buffer is the
 * one the caller does not think about. */
void *waste_dio_alloc(size_t n)
{
    const size_t pad = (n + WASTE_DIO_ALIGN - 1) / WASTE_DIO_ALIGN * WASTE_DIO_ALIGN;
    return waste_aligned_alloc(WASTE_DIO_ALIGN, pad);
}

void waste_dio_free(void *p) { waste_aligned_free(p); }

static int32_t ec_key(int layer, int expert) { return (layer << 16) | expert; }

static uint32_t ec_hash(int32_t k)
{
    uint32_t x = (uint32_t)k;
    x ^= x >> 16; x *= 0x7feb352du;
    x ^= x >> 15; x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

int waste_ecache_init(waste_ecache *c, size_t budget_bytes, size_t rec_bytes,
                      int policy)
{
    memset(c, 0, sizeof *c);
    c->rec_bytes = rec_bytes;
    c->budget_bytes = budget_bytes;
    c->policy = policy;
    c->rng = 0x9e3779b9u;
    if (!rec_bytes || budget_bytes < rec_bytes) return 0;   /* no cache */

    const size_t slots = budget_bytes / rec_bytes;
    /* n_slots and the open-addressing table size are ints. This limit leaves
     * room to double the slot count and round the hash table up to a power of
     * two without signed overflow. Real models are thousands of slots, not
     * hundreds of millions; an impossible budget is an argument error, not
     * permission to wrap an allocation size. */
    if (slots > (size_t)INT_MAX / 4) return -1;
    c->n_slots = (int)slots;
    int hs = 1;
    while (hs < c->n_slots * 2) hs <<= 1;
    c->hash_mask = hs - 1;

    c->slot = (waste_eslot *)calloc((size_t)c->n_slots, sizeof *c->slot);
    c->hash = (int32_t *)malloc((size_t)hs * sizeof *c->hash);
    if (!c->slot || !c->hash) { waste_ecache_free(c); return -1; }
    memset(c->hash, 0xff, (size_t)hs * sizeof *c->hash);   /* all -1 */

    for (int i = 0; i < c->n_slots; i++) {
        c->slot[i].key = -1;
        c->slot[i].data = (uint8_t *)waste_dio_alloc(rec_bytes);
        if (!c->slot[i].data) { waste_ecache_free(c); return -1; }
    }
    return 0;
}

void waste_ecache_free(waste_ecache *c)
{
    if (c->slot) {
        for (int i = 0; i < c->n_slots; i++) waste_dio_free(c->slot[i].data);
        free(c->slot);
    }
    free(c->hash);
    c->slot = NULL; c->hash = NULL; c->n_slots = 0;
}

static int ec_lookup(waste_ecache *c, int32_t key)
{
    uint32_t h = ec_hash(key) & (uint32_t)c->hash_mask;
    for (int probe = 0; probe <= c->hash_mask; probe++) {
        const int32_t si = c->hash[h];
        if (si < 0) return -1;
        if (c->slot[si].key == key) return si;
        h = (h + 1) & (uint32_t)c->hash_mask;
    }
    return -1;
}

static void ec_insert(waste_ecache *c, int32_t key, int slot)
{
    uint32_t h = ec_hash(key) & (uint32_t)c->hash_mask;
    while (c->hash[h] >= 0) h = (h + 1) & (uint32_t)c->hash_mask;
    c->hash[h] = slot;
}

/* Rebuilding is simpler and safe: open addressing with deletions needs
 * tombstones, and evictions are rare relative to lookups. */
static void ec_rehash(waste_ecache *c)
{
    memset(c->hash, 0xff, ((size_t)c->hash_mask + 1) * sizeof *c->hash);
    for (int i = 0; i < c->n_slots; i++)
        if (c->slot[i].key >= 0) ec_insert(c, c->slot[i].key, i);
}

static int ec_victim(waste_ecache *c)
{
    /* free slot first */
    for (int i = 0; i < c->n_slots; i++)
        if (c->slot[i].key < 0 && !c->slot[i].pinned) return i;

    int best = -1;
    uint32_t best_h = 0;
    uint64_t best_l = 0;
    for (int s = 0; s < EC_SAMPLE; s++) {
        c->rng = c->rng * 1664525u + 1013904223u;
        const int i = (int)(c->rng % (uint32_t)c->n_slots);
        const waste_eslot *sl = &c->slot[i];
        if (sl->pinned) continue;
        int better;
        if (c->policy == 1)                       /* LRU */
            better = (best < 0) || sl->last < best_l;
        else                                      /* LFRU */
            better = (best < 0) || sl->hits < best_h ||
                     (sl->hits == best_h && sl->last < best_l);
        if (better) { best = i; best_h = sl->hits; best_l = sl->last; }
    }
    /* A small batch can make all 16 random samples land on pinned slots.
     * That is not an out-of-cache condition; find the best unpinned slot
     * deterministically rather than failing or overwriting live data. */
    if (best < 0) {
        for (int i = 0; i < c->n_slots; i++) {
            const waste_eslot *sl = &c->slot[i];
            if (sl->pinned) continue;
            const int better = c->policy == 1
                ? (best < 0 || sl->last < best_l)
                : (best < 0 || sl->hits < best_h ||
                   (sl->hits == best_h && sl->last < best_l));
            if (better) { best = i; best_h = sl->hits; best_l = sl->last; }
        }
    }
    return best;
}

const uint8_t *waste_ecache_get(waste_ecache *c, int layer, int expert,
                                waste_fetch_fn fetch, void *user)
{
    const int32_t key = ec_key(layer, expert);
    c->clock++;

    if (c->n_slots > 0) {
        const int si = ec_lookup(c, key);
        if (si >= 0) {
            c->hits++;
            c->slot[si].hits++;
            c->slot[si].last = c->clock;
            return c->slot[si].data;
        }
    }

    c->misses++;
    c->bytes_read += c->rec_bytes;

    if (c->n_slots == 0) return NULL;      /* caller falls back to its own buf */

    const int vi = ec_victim(c);
    const int had = c->slot[vi].key >= 0;
    if (fetch(user, layer, expert, c->slot[vi].data) != 0) {
        c->slot[vi].key = -1;
        if (had) { c->evictions++; ec_rehash(c); }
        return NULL;
    }
    if (had) { c->evictions++; c->slot[vi].key = key; ec_rehash(c); }
    else { c->slot[vi].key = key; ec_insert(c, key, vi); }
    c->slot[vi].hits = 1;
    c->slot[vi].last = c->clock;
    return c->slot[vi].data;
}

int waste_ecache_get_many(waste_ecache *c, int layer, const int *experts,
                          int n, waste_fetch_many_fn fetch_many, void *user,
                          const uint8_t **out)
{
    if (!c || !experts || !fetch_many || !out || n <= 0 || n > 64 ||
        c->n_slots <= 0)
        return -1;
    int initial[64], slots[64], miss_at[64], miss_layers[64],
        miss_experts[64];
    uint8_t *miss_dst[64];

    /* Snapshot and pin the hits before choosing a victim. Without this,
     * an early miss could evict a later hit and overwrite its destination
     * while the batch I/O is in flight. */
    for (int i = 0; i < n; i++) {
        initial[i] = ec_lookup(c, ec_key(layer, experts[i]));
        slots[i] = -1;
        if (initial[i] >= 0) c->slot[initial[i]].pinned = 1;
    }

    int nm = 0, failed = 0;
    for (int i = 0; i < n; i++) {
        c->clock++;
        int si = initial[i];
        if (si < 0) {
            /* A duplicate later in the same batch is a hit on the slot we
             * just reserved; routers normally return distinct ids, but the
             * cache API need not depend on that. */
            for (int j = 0; j < i; j++)
                if (experts[j] == experts[i]) { si = slots[j]; break; }
        }
        if (si >= 0) {
            c->hits++;
            c->slot[si].hits++;
            c->slot[si].last = c->clock;
            c->slot[si].pinned = 1;
            slots[i] = si;
            out[i] = c->slot[si].data;
            continue;
        }

        c->misses++;
        c->bytes_read += c->rec_bytes;
        const int vi = ec_victim(c);
        if (vi < 0) { failed = 1; break; }
        const int had = c->slot[vi].key >= 0;
        c->slot[vi].key = ec_key(layer, experts[i]);
        c->slot[vi].hits = 1;
        c->slot[vi].last = c->clock;
        c->slot[vi].pinned = 1;
        if (had) { c->evictions++; ec_rehash(c); }
        else ec_insert(c, c->slot[vi].key, vi);
        slots[i] = vi;
        out[i] = c->slot[vi].data;
        miss_at[nm] = i;
        miss_layers[nm] = layer;
        miss_experts[nm] = experts[i];
        miss_dst[nm] = c->slot[vi].data;
        nm++;
    }

    if (!failed && nm && fetch_many(user, nm, miss_layers, miss_experts,
                                    miss_dst) != 0)
        failed = 1;
    if (failed) {
        for (int k = 0; k < nm; k++) {
            const int si = slots[miss_at[k]];
            if (si >= 0) c->slot[si].key = -1;
        }
        ec_rehash(c);
    }
    for (int i = 0; i < c->n_slots; i++) c->slot[i].pinned = 0;
    return failed ? -1 : 0;
}

/* ---- learned hotlist ---------------------------------------------------- */

int waste_ecache_save_usage(const waste_ecache *c, const char *path,
                            uint64_t tokens)
{
    if (!c->slot || !path) return -1;
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    waste_usage_hdr h = { WASTE_MAGIC_USAGE, 1, tokens };
    int rc = fwrite(&h, sizeof h, 1, f) == 1 ? 0 : -1;
    for (int i = 0; i < c->n_slots && !rc; i++) {
        if (c->slot[i].key < 0) continue;
        waste_usage_ent e;
        e.layer = (uint16_t)(c->slot[i].key >> 16);
        e.expert_id = (uint16_t)(c->slot[i].key & 0xFFFF);
        e.hits = c->slot[i].hits;
        e.last_seen = (uint32_t)c->slot[i].last;
        e.next_layer_top = 0;
        if (fwrite(&e, sizeof e, 1, f) != 1) rc = -1;
    }
    fclose(f);
    if (rc) remove(path);
    return rc;
}

/* Descending by hit count, ties by (layer, expert) so a given usage file
 * warms the same way every run whatever qsort does with equal keys. */
static int hits_desc(const void *a, const void *b)
{
    const waste_usage_ent *x = (const waste_usage_ent *)a;
    const waste_usage_ent *y = (const waste_usage_ent *)b;
    if (x->hits != y->hits) return x->hits > y->hits ? -1 : 1;
    if (x->layer != y->layer) return x->layer < y->layer ? -1 : 1;
    if (x->expert_id != y->expert_id) return x->expert_id < y->expert_id ? -1 : 1;
    return 0;
}

/* Preload the hottest recorded experts, best first, until the cache is
 * full. Returns how many were loaded, or -1. */
int waste_ecache_warm(waste_ecache *c, const char *path,
                      waste_fetch_fn fetch, void *user)
{
    if (!c->slot || !path) return -1;
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    waste_usage_hdr h;
    if (fread(&h, sizeof h, 1, f) != 1 || h.magic != WASTE_MAGIC_USAGE) {
        fclose(f);
        return -1;
    }

    long start = ftell(f);
    fseek(f, 0, SEEK_END);
    const long n = (ftell(f) - start) / (long)sizeof(waste_usage_ent);
    fseek(f, start, SEEK_SET);
    if (n <= 0) { fclose(f); return 0; }

    waste_usage_ent *ent = (waste_usage_ent *)malloc((size_t)n * sizeof *ent);
    if (!ent || fread(ent, sizeof *ent, (size_t)n, f) != (size_t)n) {
        free(ent); fclose(f); return -1;
    }
    fclose(f);

    /* Hottest first. This was a partial selection over the first n_slots,
     * which is O(n_slots * n) — with K3's 46 GB cache that is 46000 by
     * 46000, seconds of a cold open spent sorting before a single expert
     * is read. The whole array sorted properly is n log n. */
    qsort(ent, (size_t)n, sizeof *ent, hits_desc);
    const int want = (int)n < c->n_slots ? (int)n : c->n_slots;

    int loaded = 0;
    for (int i = 0; i < want; i++) {
        const int32_t key = ((int32_t)ent[i].layer << 16) | ent[i].expert_id;
        if (ec_lookup(c, key) >= 0) continue;
        const int vi = ec_victim(c);
        if (vi < 0) break;
        if (fetch(user, ent[i].layer, ent[i].expert_id, c->slot[vi].data) != 0) continue;
        const int had = c->slot[vi].key >= 0;
        c->slot[vi].key = key;
        c->slot[vi].hits = ent[i].hits;
        c->slot[vi].last = ++c->clock;
        if (had) ec_rehash(c); else ec_insert(c, key, vi);
        loaded++;
    }
    free(ent);
    return loaded;
}
