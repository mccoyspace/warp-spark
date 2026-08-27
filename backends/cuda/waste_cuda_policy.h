/* SPDX-License-Identifier: Apache-2.0 */
#ifndef WASTE_CUDA_POLICY_H
#define WASTE_CUDA_POLICY_H

#include "waste_cuda_backend.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
    WASTE_CUDA_MODEL_NONE = 0,
    WASTE_CUDA_MODEL_K2 = 2,
    WASTE_CUDA_MODEL_K3 = 3,
};

typedef struct {
    int model_kind;
    int device_ordinal;
    uint32_t capabilities;
    uint32_t q4_mode;
    size_t capacity;
    size_t vq_y_capacity;
    size_t vq_lut_values[3];
    size_t book_values;
    uint64_t reserved_bytes;
} waste_cuda_policy;

waste_status waste_cuda_policy_resolve(
    const waste_backend_model_info *model, const void *backend_cfg,
    waste_cuda_policy *out);

int waste_cuda_policy_q4_tensor(const waste_backend_tensor *tensor);

#ifdef __cplusplus
}
#endif

#endif
