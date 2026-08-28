/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/*
 * kda.h — Kimi Delta Attention kernels.
 *
 * Recurrence (matches fla's naive_recurrent_kda, the reference shipped
 * with Kimi-Linear; note the decay is applied BEFORE the delta term):
 *
 *     S' = Diag(exp(g_t)) . S_{t-1}          row-scale along the K axis
 *     S_t = S' + beta_t * k_t (v_t - S'^T k_t)^T
 *     o_t = S_t^T q_t
 *
 * with q, k L2-normalized per head and q then scaled by K^-0.5
 * (use_qk_l2norm_in_kernel=True + default scale in the reference).
 *
 * Layout: S is [H][K][V] row-major — rows indexed by the key dimension, so
 * the rank-1 update writes contiguous rows and both GEMVs read them
 * sequentially. State streams through cache twice per token.
 */

#ifndef WASTE_KDA_H
#define WASTE_KDA_H

#ifdef __cplusplus
extern "C" {
#endif

/* One decode step for all heads. Updates S in place, writes o.
 *   q, k, g_log : [H][K]     v, o : [H][V]     beta : [H]
 *   S           : [H][K][V]
 * scratch must hold at least V floats. */
void waste_kda_step(int H, int K, int V,
                    const float *q, const float *k, const float *v,
                    const float *g_log, const float *beta,
                    float *S, float *o, float *scratch);

/* Same recurrence with an explicit q/k L2-normalization epsilon.  The
 * original entry point remains the 1e-12 Kimi/K3 contract. */
void waste_kda_step_ex(int H, int K, int V,
                       const float *q, const float *k, const float *v,
                       const float *g_log, const float *beta,
                       float l2_eps,
                       float *S, float *o, float *scratch);


/* Sequential prefill: T steps of the above. Inputs are [T][H][*]
 * (time-major, as the reference lays them out). Used for validation and
 * short prompts; the chunked path lands later. */
void waste_kda_forward(int T, int H, int K, int V,
                       const float *q, const float *k, const float *v,
                       const float *g_log, const float *beta,
                       float *S, float *o, float *scratch);

/* Short causal conv (kernel size KS, SiLU) over one projection, decode
 * step: ring holds the last KS-1 inputs, x is the new one, y is output.
 * w is [C][KS] with w[c][KS-1] applied to the newest sample. */
void waste_short_conv_step(int C, int KS, const float *w, const float *bias,
                           float *ring, const float *x, float *y);

/* Gated RMSNorm with sigmoid activation on the gate (fla's
 * FusedRMSNormGated): y = rmsnorm(x) * weight * sigmoid(gate). */
void waste_rmsnorm_gated(int C, const float *x, const float *gate,
                         const float *weight, float eps, float *y);

#ifdef __cplusplus
}
#endif
#endif /* WASTE_KDA_H */
