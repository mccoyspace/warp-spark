/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/*
 * simd_sve.c — SVE expert-VQ table application.
 *
 * Built separately with -march=armv8.2-a+sve.  Nothing here may run until
 * Linux HWCAP has reported SVE; backend.c owns that check.  Rows are vector
 * lanes, so each lane retains the scalar arithmetic order across stages and
 * vector positions.  The implementation is vector-length agnostic even
 * though the first measured target has a 128-bit SVE vector length.
 */

#include "simd.h"
#include "waste_backend.h"

#if defined(__ARM_FEATURE_SVE)

#include <arm_sve.h>
#include <string.h>

void waste_vq_rows_sve(int b, int e, void *p)
{
    vq_arg *a = (vq_arg *)p;
    const int nv = a->nv, st = a->stages, en = a->entries;

    /* VQ2 and any future stage count retain the established implementation.
     * This spike changes only the three-stage expert format used by K3. */
    if (st != 3) {
        waste_vq_rows_cpu(b, e, p);
        return;
    }

    float acc[WASTE_VQ_RANGE];

    for (int r0 = b; r0 < e; r0 += WASTE_VQ_RANGE) {
        const int rows = (r0 + WASTE_VQ_RANGE < e) ? WASTE_VQ_RANGE : e - r0;
        const int nblk = (rows + WASTE_VQ_TILE - 1) / WASTE_VQ_TILE;
        memset(acc, 0, (size_t)rows * sizeof(float));

        for (int v = 0; v < nv; v++) {
            const float *b0 = a->lut + (size_t)v * st * en;
            const float *b1 = b0 + en;
            const float *b2 = b1 + en;

            for (int j = 0; j < nblk; j++) {
                const int nr = (j + 1) * WASTE_VQ_TILE <= rows ? WASTE_VQ_TILE
                                                               : rows - j * WASTE_VQ_TILE;
                const uint8_t *ix = a->idx +
                    ((size_t)(r0 / WASTE_VQ_TILE + j) * nv + v) *
                    WASTE_VQ_TILE * st;
                float *ac = acc + (size_t)j * WASTE_VQ_TILE;

                for (int r = 0; r < nr; r += (int)svcntw()) {
                    const uint64_t left = (uint64_t)(nr - r);
                    const uint64_t lanes = left < svcntw() ? left : svcntw();
                    const svbool_t pg8 = svwhilelt_b8((uint64_t)0, lanes);
                    const svbool_t pg32 = svwhilelt_b32((uint64_t)r,
                                                        (uint64_t)nr);

                    /* One structured load deinterleaves stage bytes.  Two
                     * lower-half unpacks expand the active row bytes to the
                     * u32 indices required by an f32 gather. */
                    const svuint8x3_t q = svld3_u8(pg8, ix + (size_t)r * st);
                    const svuint32_t i0 = svunpklo_u32(
                        svunpklo_u16(svget3_u8(q, 0)));
                    const svuint32_t i1 = svunpklo_u32(
                        svunpklo_u16(svget3_u8(q, 1)));
                    const svuint32_t i2 = svunpklo_u32(
                        svunpklo_u16(svget3_u8(q, 2)));

                    const svfloat32_t g0 = svld1_gather_u32index_f32(pg32, b0, i0);
                    const svfloat32_t g1 = svld1_gather_u32index_f32(pg32, b1, i1);
                    const svfloat32_t g2 = svld1_gather_u32index_f32(pg32, b2, i2);

                    /* Do not reassociate: this is the scalar per-row order,
                     * (stage 0 + stage 1) + stage 2, followed by acc += t. */
                    svfloat32_t t = svadd_f32_x(pg32, g0, g1);
                    t = svadd_f32_x(pg32, t, g2);
                    t = svadd_f32_x(pg32, svld1_f32(pg32, ac + r), t);
                    svst1_f32(pg32, ac + r, t);
                }
            }
        }

        for (int r = 0; r < rows; r++)
            a->y[r0 + r] = acc[r] * waste_f16(a->scale[r0 + r]);
    }
}

const char *waste_register_sve(waste_kernels *t)
{
    t->vq_rows = waste_vq_rows_sve;
    return "SVE";
}

#endif /* __ARM_FEATURE_SVE */
