/* SPDX-License-Identifier: Apache-2.0 */
#include "../src/ecache.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int fetch_many(void *user, int n, const int *layers,
                      const int *experts, uint8_t *const *dst)
{
    (void)user;
    for (int i = 0; i < n; i++) {
        memset(dst[i], experts[i], 64);
        dst[i][0] = (uint8_t)layers[i];
    }
    return 0;
}

static int check(const uint8_t *p, int layer, int expert)
{
    if (!p || p[0] != layer) return -1;
    for (int i = 1; i < 64; i++) if (p[i] != expert) return -1;
    return 0;
}

int main(void)
{
    waste_ecache c;
    if (waste_ecache_init(&c, 2 * 64, 64, 0)) return 1;
    const uint8_t *out[3];
    const int first[2] = { 1, 2 };
    if (waste_ecache_get_many(&c, 7, first, 2, fetch_many, NULL, out) ||
        check(out[0], 7, 1) || check(out[1], 7, 2)) return 1;

    /* Expert 2 is pinned while expert 3 chooses a victim, so its pointer
     * cannot be overwritten before this batch is consumed. */
    const int second[2] = { 2, 3 };
    if (waste_ecache_get_many(&c, 7, second, 2, fetch_many, NULL, out) ||
        check(out[0], 7, 2) || check(out[1], 7, 3)) return 1;

    const int duplicate[3] = { 4, 4, 3 };
    if (waste_ecache_get_many(&c, 7, duplicate, 3, fetch_many, NULL, out) ||
        out[0] != out[1] || check(out[0], 7, 4) || check(out[2], 7, 3))
        return 1;
    if (c.misses != 4 || c.hits != 3) return 1;
    waste_ecache_free(&c);
    puts("PASS batched expert cache");
    return 0;
}
