/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/*
 * kda_neon.c — ARM NEON specialization of the KDA kernels.
 *
 * Compiled only on ARM; registers over the CPU baseline at init. Must
 * produce the same results as kda.c (tools/kda_ref.py checks both against
 * the reference with WASTE_BACKEND=cpu / auto).
 */

#if defined(__ARM_NEON) || defined(__aarch64__)

#include "kda.h"
#include "waste_backend.h"

#include <arm_neon.h>
#include <math.h>
#include <string.h>

static float l2_rnorm_neon(const float *x, int n, float eps)
{
    float32x4_t acc = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t v = vld1q_f32(x + i);
        acc = vfmaq_f32(acc, v, v);
    }
    float s = vaddvq_f32(acc);
    for (; i < n; i++) s += x[i] * x[i];
    return 1.0f / sqrtf(s + eps);
}

static void kda_step_neon_ex(int H, int K, int V,
                             const float *q, const float *k, const float *v,
                             const float *g_log, const float *beta,
                             float l2_eps,
                             float *S, float *o, float *u)
{
    const float qscale = 1.0f / sqrtf((float)K);

    for (int h = 0; h < H; h++) {
        const float *qh = q + (size_t)h * K;
        const float *kh = k + (size_t)h * K;
        const float *vh = v + (size_t)h * V;
        const float *gh = g_log + (size_t)h * K;
        float *Sh = S + (size_t)h * K * V;
        float *oh = o + (size_t)h * V;
        const float b = beta[h];

        const float qn = l2_rnorm_neon(qh, K, l2_eps) * qscale;
        const float kn = l2_rnorm_neon(kh, K, l2_eps);

        memset(u, 0, (size_t)V * sizeof(float));

        /* pass A: decay rows, accumulate u = S'^T k */
        for (int kk = 0; kk < K; kk++) {
            float *row = Sh + (size_t)kk * V;
            const float32x4_t vd = vdupq_n_f32(expf(gh[kk]));
            const float32x4_t vk = vdupq_n_f32(kh[kk] * kn);
            int i = 0;
            for (; i + 4 <= V; i += 4) {
                float32x4_t r = vmulq_f32(vld1q_f32(row + i), vd);
                vst1q_f32(row + i, r);
                vst1q_f32(u + i, vfmaq_f32(vld1q_f32(u + i), r, vk));
            }
            const float d = expf(gh[kk]), kv = kh[kk] * kn;
            for (; i < V; i++) { row[i] *= d; u[i] += row[i] * kv; }
        }

        /* delta */
        {
            const float32x4_t vb = vdupq_n_f32(b);
            int i = 0;
            for (; i + 4 <= V; i += 4) {
                float32x4_t d = vsubq_f32(vld1q_f32(vh + i), vld1q_f32(u + i));
                vst1q_f32(u + i, vmulq_f32(vb, d));
            }
            for (; i < V; i++) u[i] = b * (vh[i] - u[i]);
        }

        /* pass B: rank-1 update, accumulate o = S^T q */
        memset(oh, 0, (size_t)V * sizeof(float));
        for (int kk = 0; kk < K; kk++) {
            float *row = Sh + (size_t)kk * V;
            const float32x4_t vk = vdupq_n_f32(kh[kk] * kn);
            const float32x4_t vq = vdupq_n_f32(qh[kk] * qn);
            int i = 0;
            for (; i + 4 <= V; i += 4) {
                float32x4_t r = vfmaq_f32(vld1q_f32(row + i), vld1q_f32(u + i), vk);
                vst1q_f32(row + i, r);
                vst1q_f32(oh + i, vfmaq_f32(vld1q_f32(oh + i), r, vq));
            }
            const float kv = kh[kk] * kn, qv = qh[kk] * qn;
            for (; i < V; i++) { row[i] += u[i] * kv; oh[i] += row[i] * qv; }
        }
    }
}

static void kda_step_neon(int H, int K, int V,
                          const float *q, const float *k, const float *v,
                          const float *g_log, const float *beta,
                          float *S, float *o, float *u)
{
    kda_step_neon_ex(H, K, V, q, k, v, g_log, beta, 1e-12f,
                     S, o, u);
}

const char *waste_kda_register_neon(waste_kernels *t)
{
    t->kda_step = kda_step_neon;
    t->kda_step_ex = kda_step_neon_ex;
    /* short_conv_step and rmsnorm_gated stay on the CPU baseline until
     * they show up in a profile — partial override is the whole point. */
    /* The name reports what this build *uses*, not what the CPU offers.
     * waste_cpu_features() detects dotprod and i8mm, and neither drives a
     * kernel yet — no SDOT or SMMLA is emitted anywhere in the engine — so
     * naming them here only made `waste version` overstate the binary.
     * Add the suffix back in the same commit that adds the kernel. */
    return "NEON";
}

#endif /* ARM */
