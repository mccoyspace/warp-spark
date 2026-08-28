/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/*
 * test_k3parts.c — exercise the three K3 components whose maths is new,
 * and dump inputs + outputs so tools/k3parts_ref.py can recompute them
 * from the released reference source and diff.
 *
 * These cannot be validated end to end until the weights land, so they are
 * validated in isolation now, on synthetic inputs.
 *
 *   cc -O2 -o test_k3parts tests/test_k3parts.c libwaste.a -lm -lpthread
 *   ./test_k3parts out.bin
 *
 * Layout of out.bin (little-endian f32 unless noted):
 *   [situ]   i32 n, f32 beta, f32 linear_beta, gate[n], up[n], out[n]
 *   [gate]   i32 H, i32 D, f32 lower_bound, g_in[H*D], A_log[H],
 *            dt[H*D], g_k3[H*D], g_linear[H*D]
 *   [ares]   i32 nb, i32 hid, f32 eps, blocks[nb*hid], prefix[hid],
 *            norm_w[hid], proj_w[hid], out[hid]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../src/model.h"
#include "../src/kda.h"

static uint64_t rng = 0x243F6A8885A308D3ULL;
static float frand(void)
{
    rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
    return (float)((double)((rng >> 11) & 0x1FFFFFFFFFFFFFULL) / 9007199254740992.0
                   * 2.0 - 1.0);
}

static void wr(FILE *f, const float *v, int n) { fwrite(v, sizeof(float), (size_t)n, f); }
static void wi(FILE *f, int v) { fwrite(&v, sizeof(int), 1, f); }
static void wf(FILE *f, float v) { fwrite(&v, sizeof(float), 1, f); }

int main(int argc, char **argv)
{
    const char *out = argc > 1 ? argv[1] : "k3parts.bin";
    FILE *f = fopen(out, "wb");
    if (!f) { perror("open"); return 1; }

    /* The historical KDA entry point remains exactly the 1e-12 contract;
     * GLM-5.3 selects its 1e-6 epsilon only through the explicit entry. */
    {
        enum { H = 1, K = 2, V = 2 };
        const float q[K] = { 1e-4f, -2e-4f };
        const float k[K] = { -3e-4f, 1e-4f };
        const float v[V] = { .375f, -.625f };
        const float g[K] = { -.2f, -.35f };
        const float beta[H] = { .4f };
        float s_default[K * V] = { 0 }, s_legacy[K * V] = { 0 };
        float s_glm53[K * V] = { 0 }, o_default[V], o_legacy[V], o_glm53[V];
        float u_default[V], u_legacy[V], u_glm53[V];
        waste_kda_step(H, K, V, q, k, v, g, beta,
                       s_default, o_default, u_default);
        waste_kda_step_ex(H, K, V, q, k, v, g, beta, 1e-12f,
                          s_legacy, o_legacy, u_legacy);
        waste_kda_step_ex(H, K, V, q, k, v, g, beta, 1e-6f,
                          s_glm53, o_glm53, u_glm53);
        if (memcmp(s_default, s_legacy, sizeof s_default) ||
            memcmp(o_default, o_legacy, sizeof o_default) ||
            !memcmp(o_default, o_glm53, sizeof o_default)) {
            fprintf(stderr, "KDA epsilon dispatch contract differs\n");
            return 1;
        }
    }

    /* A zero-affine two-stream mHC has pre=0.5, post=1 and a uniform
     * doubly-stochastic comb.  It is a small closed-form check of flattening,
     * collapse, Sinkhorn orientation and comb-transpose merge semantics. */
    {
        enum { hc = 2, hid = 2, mix = (2 + hc) * hc };
        const float streams[4] = { 1, 2, 3, 4 };
        float fn[mix * hc * hid], base[mix], scale[3] = { 1, 1, 1 };
        float post[hc], comb[hc * hc], collapsed[hid], merged[hc * hid];
        float scratch[hc * hid + mix];
        memset(fn, 0, sizeof fn);
        memset(base, 0, sizeof base);
        if (waste_mhc_f32(hc, hid, 3, 0.0f, 1e-5f, streams, fn,
                          base, scale, post, comb, collapsed, scratch)) {
            fprintf(stderr, "mHC map refused valid fixture\n");
            return 1;
        }
        const float sub[2] = { 10, 20 };
        waste_mhc_merge(hc, hid, streams, sub, post, comb, merged);
        if (fabsf(collapsed[0] - 2.0f) > 1e-6f ||
            fabsf(collapsed[1] - 3.0f) > 1e-6f ||
            fabsf(merged[0] - 12.0f) > 1e-6f ||
            fabsf(merged[1] - 23.0f) > 1e-6f ||
            fabsf(merged[2] - 12.0f) > 1e-6f ||
            fabsf(merged[3] - 23.0f) > 1e-6f) {
            fprintf(stderr, "mHC closed-form fixture differs\n");
            return 1;
        }
    }

    /* Independent oracle from the pinned official Transformers source:
     * exact release hc_mult=4 and all 20 Sinkhorn iterations.  Its comb is
     * non-symmetric, so using comb rather than comb.T cannot slip through. */
    {
        enum { hc = 4, hid = 2, mix = (2 + hc) * hc };
        const float streams[8] = {
            .25f, -.5f, 1.f, .75f, -.25f, .5f, .125f, -1.25f
        };
        float fn[mix * hc * hid], base[mix];
        const float scale[3] = { .75f, -.625f, .5f };
        float post[hc], comb[hc * hc], collapsed[hid], merged[hc * hid];
        float scratch[hc * hid + mix];
        for (int r = 0; r < mix; r++)
            for (int d = 0; d < hc * hid; d++)
                fn[r * hc * hid + d] =
                    (float)((((r + 1) * 11 + (d + 1) * 7) % 23) - 11) /
                    32.0f;
        for (int r = 0; r < mix; r++)
            base[r] = (float)(((r * 5 + 3) % 13) - 6) / 16.0f;
        if (waste_mhc_f32(hc, hid, 20, 1e-6f, 1e-5f, streams, fn,
                          base, scale, post, comb, collapsed, scratch)) {
            fprintf(stderr, "mHC official fixture refused\n");
            return 1;
        }
        const float sub[2] = { .375f, -.625f };
        const float post_ref[4] = {
            1.266113043f, .804597199f, 1.337715149f, 1.082338572f
        };
        const float comb_ref[16] = {
            .180506617f, .282190681f, .180506691f, .356795043f,
            .248299286f, .255138665f, .248299316f, .248261735f,
            .173448190f, .352339715f, .173448309f, .300762802f,
            .397744894f, .110329978f, .397744656f, .0941794664f
        };
        const float collapsed_ref[2] = { .675257564f, -.275566131f };
        const float merged_ref[8] = {
            .774574399f, -1.105806589f,
            .553116560f, -.414357215f,
            .801425159f, -1.150557518f,
            .679919183f, -.636005759f
        };
        const float final_ref[2] = { .702258825f, -.826681733f };
        waste_mhc_merge(hc, hid, streams, sub, post, comb, merged);
        for (int i = 0; i < hc; i++)
            if (fabsf(post[i] - post_ref[i]) > 3e-6f) {
                fprintf(stderr, "mHC post oracle differs at %d\n", i);
                return 1;
            }
        for (int i = 0; i < hc * hc; i++)
            if (fabsf(comb[i] - comb_ref[i]) > 3e-6f) {
                fprintf(stderr, "mHC Sinkhorn oracle differs at %d\n", i);
                return 1;
            }
        for (int i = 0; i < hid; i++)
            if (fabsf(collapsed[i] - collapsed_ref[i]) > 3e-6f) {
                fprintf(stderr, "mHC collapse oracle differs at %d\n", i);
                return 1;
            }
        for (int i = 0; i < hc * hid; i++)
            if (fabsf(merged[i] - merged_ref[i]) > 3e-6f) {
                fprintf(stderr, "mHC transposed merge oracle differs at %d\n", i);
                return 1;
            }
        for (int d = 0; d < hid; d++) {
            float mean = 0.0f;
            for (int i = 0; i < hc; i++) mean += merged[i * hid + d];
            mean /= (float)hc;
            if (fabsf(mean - final_ref[d]) > 3e-6f) {
                fprintf(stderr, "mHC final mean oracle differs at %d\n", d);
                return 1;
            }
        }
    }

    /* ---- 1. SiTU ------------------------------------------------------ */
    {
        const int n = 512;
        const float beta = 4.0f, lbeta = 25.0f;    /* K3's config values */
        float *g = malloc((size_t)n * 4), *u = malloc((size_t)n * 4),
              *o = malloc((size_t)n * 4);
        for (int i = 0; i < n; i++) { g[i] = frand() * 30.0f; u[i] = frand() * 30.0f; }
        for (int i = 0; i < n; i++) o[i] = waste_situ_pair(g[i], u[i], beta, lbeta);
        wi(f, n); wf(f, beta); wf(f, lbeta);
        wr(f, g, n); wr(f, u, n); wr(f, o, n);
        free(g); free(u); free(o);
    }

    /* ---- 2. decay gate, both forms ------------------------------------ */
    {
        const int H = 8, D = 16, n = H * D;
        const float lb = -5.0f;
        float *gin = malloc((size_t)n * 4), *A = malloc((size_t)H * 4),
              *dt = malloc((size_t)n * 4), *g1 = malloc((size_t)n * 4),
              *g2 = malloc((size_t)n * 4);
        for (int i = 0; i < n; i++) { gin[i] = frand() * 4.0f; dt[i] = frand(); }
        for (int i = 0; i < H; i++) A[i] = frand();
        memcpy(g1, gin, (size_t)n * 4);
        waste_kda_decay_gate(g1, A, dt, H, D, lb);          /* K3 form */
        memcpy(g2, gin, (size_t)n * 4);
        waste_kda_decay_gate(g2, A, dt, H, D, 0.0f);        /* Kimi-Linear form */
        wi(f, H); wi(f, D); wf(f, lb);
        wr(f, gin, n); wr(f, A, H); wr(f, dt, n); wr(f, g1, n); wr(f, g2, n);
        free(gin); free(A); free(dt); free(g1); free(g2);
    }

    /* ---- 3. attention residuals --------------------------------------- */
    {
        const int nb = 5, hid = 128;
        const float eps = 1e-5f;
        waste_model m;
        memset(&m, 0, sizeof m);
        m.cfg.hidden = hid;
        m.cfg.eps = eps;
        m.blockres = malloc((size_t)nb * hid * 4);
        m.att = malloc(4096 * 4);
        float *ps = malloc((size_t)hid * 4), *nw = malloc((size_t)hid * 4),
              *pw = malloc((size_t)hid * 4), *o = malloc((size_t)hid * 4);
        for (int i = 0; i < nb * hid; i++) m.blockres[i] = frand() * 2.0f;
        for (int i = 0; i < hid; i++) { ps[i] = frand() * 2.0f; nw[i] = frand(); pw[i] = frand(); }
        waste_apply_attn_res(&m, m.blockres, nb, ps, nw, pw, o);
        wi(f, nb); wi(f, hid); wf(f, eps);
        wr(f, m.blockres, nb * hid); wr(f, ps, hid);
        wr(f, nw, hid); wr(f, pw, hid); wr(f, o, hid);
        free(m.blockres); free(m.att); free(ps); free(nw); free(pw); free(o);
    }

    fclose(f);
    printf("wrote %s\n", out);
    return 0;
}
