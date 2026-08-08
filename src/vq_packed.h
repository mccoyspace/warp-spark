/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/* WQ_VQ4P's little-endian four-six-bit-indices-in-three-bytes contract.
 * Kept in one header so the converter-facing unit test, CPU apply and CUDA
 * apply cannot acquire three subtly different interpretations of the same
 * byte stream. */
#ifndef WASTE_VQ_PACKED_H
#define WASTE_VQ_PACKED_H

#define WASTE_P6_J0(b0, b1, b2) ((unsigned)(b0) & 0x3fu)
#define WASTE_P6_J1(b0, b1, b2) \
    ((((unsigned)(b0) >> 6) | ((unsigned)(b1) << 2)) & 0x3fu)
#define WASTE_P6_J2(b0, b1, b2) \
    ((((unsigned)(b1) >> 4) | ((unsigned)(b2) << 4)) & 0x3fu)
#define WASTE_P6_J3(b0, b1, b2) (((unsigned)(b2) >> 2) & 0x3fu)

#endif
