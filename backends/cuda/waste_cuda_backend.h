/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
#ifndef WASTE_CUDA_BACKEND_H
#define WASTE_CUDA_BACKEND_H

#include "waste.h"

#ifdef __cplusplus
extern "C" {
#endif

#define WASTE_CUDA_BACKEND_CONFIG_VERSION 1u

typedef enum {
    /* Fast promoted Q4G kernel. Its reduction tree is not the CPU tree. */
    WASTE_CUDA_Q4_FAST = 1,
    /* Four-lane group ordering retained for diagnostic comparisons. */
    WASTE_CUDA_Q4_CPU_ORDER = 2,
} waste_cuda_q4_mode;

typedef struct {
    size_t struct_size;
    uint32_t config_version;
    int32_t device_ordinal;       /* v1 accepts ordinal 0; qualified GB10 */
    uint32_t capabilities;        /* 0 = MATVEC | VQ */
    uint32_t q4_mode;             /* 0 = WASTE_CUDA_Q4_FAST */
} waste_cuda_backend_config;

/* Pass this object and an optional waste_cuda_backend_config to
 * waste_open_with_backend. The object has static lifetime. */
extern const waste_backend_v1 waste_cuda_backend_provider;

#ifdef __cplusplus
}
#endif

#endif
