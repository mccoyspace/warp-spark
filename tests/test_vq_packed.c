/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/* The first VQ4P gate: the three-byte stream is the same size as VQ3R's
 * three whole-byte indices, so a bit-order mistake is silent and plausible.
 * Exhaust the complete 4x6-bit space before any numerical test is allowed. */
#include <stdint.h>
#include <stdio.h>

#include "../src/vq_packed.h"

int main(void)
{
    for (unsigned packed = 0; packed < (1u << 24); packed++) {
        volatile unsigned input = packed; /* keep the exhaustive gate real */
        const uint8_t b0 = (uint8_t)input;
        const uint8_t b1 = (uint8_t)(input >> 8);
        const uint8_t b2 = (uint8_t)(input >> 16);
        const unsigned j0 = input & 0x3f;
        const unsigned j1 = (input >> 6) & 0x3f;
        const unsigned j2 = (input >> 12) & 0x3f;
        const unsigned j3 = (input >> 18) & 0x3f;
        if (WASTE_P6_J0(b0, b1, b2) != j0 ||
            WASTE_P6_J1(b0, b1, b2) != j1 ||
            WASTE_P6_J2(b0, b1, b2) != j2 ||
            WASTE_P6_J3(b0, b1, b2) != j3) {
            fprintf(stderr, "VQ4P unpack mismatch at 0x%06x\n", packed);
            return 1;
        }
    }
    puts("VQ4P PACK OK");
    return 0;
}
