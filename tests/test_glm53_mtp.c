/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/* Integrated GLM-5.3 NextN/MTP state-and-alignment gate.
 *
 *   ./test_glm53_mtp BASE MTP [MALFORMED ...]
 *
 * BASE and MTP are built from the same tiny fixture seed.  This makes the
 * target decoder itself an exact control while the appended layer exercises
 * the real loader, DSA cache, router, separate expert bank and public hooks.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/model.h"

#define REQUIRE(expr) do {                                                   \
    if (!(expr)) {                                                           \
        fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #expr);             \
        goto fail;                                                           \
    }                                                                        \
} while (0)

static void load_opts(waste_load_opts *opts)
{
    memset(opts, 0, sizeof *opts);
    opts->cache_bytes = (size_t)8 << 20;
    opts->n_threads = 1;
    opts->direct_io = 0;
}

static int run_control(const char *path, const int *tokens, int n_prompt,
                       float **prompt_logits, float **next_logits,
                       int *vocab_out)
{
    waste_load_opts opts;
    waste_model model;
    const float *logits = NULL;
    float *prompt = NULL, *next = NULL;
    int loaded = 0;

    load_opts(&opts);
    if (waste_model_load(&model, path, 8, &opts)) {
        fprintf(stderr, "control load failed: %s\n", path);
        return -1;
    }
    loaded = 1;
    if (waste_model_mtp_available(&model) ||
        waste_model_mtp_enabled(&model) ||
        waste_model_mtp_cache_pos(&model) != -1 ||
        waste_model_mtp_set_enabled(&model, 1) == 0) {
        fprintf(stderr, "ordinary fixture unexpectedly exposed MTP\n");
        goto fail;
    }

    const size_t bytes = (size_t)model.cfg.vocab * sizeof(float);
    prompt = (float *)malloc(bytes);
    next = (float *)malloc(bytes);
    if (!prompt || !next) goto fail;
    logits = waste_model_prefill(&model, tokens, n_prompt, 0);
    if (!logits) goto fail;
    memcpy(prompt, logits, bytes);
    logits = waste_model_step(&model, tokens[n_prompt], n_prompt, NULL);
    if (!logits) goto fail;
    memcpy(next, logits, bytes);

    *prompt_logits = prompt;
    *next_logits = next;
    *vocab_out = model.cfg.vocab;
    waste_model_free(&model);
    return 0;

fail:
    free(prompt);
    free(next);
    if (loaded) waste_model_free(&model);
    return -1;
}

static int malformed_rejected(const char *path)
{
    waste_load_opts opts;
    waste_model model;
    load_opts(&opts);
    if (waste_model_load(&model, path, 8, &opts) == 0) {
        fprintf(stderr, "malformed MTP container loaded: %s\n", path);
        waste_model_free(&model);
        return -1;
    }
    return 0;
}

static int invalid_vq2_env_rejected(const char *path)
{
    const char *prior = getenv("WASTE_MTP_VQ2");
    char *saved = NULL;
    waste_load_opts opts;
    waste_model model;
    int loaded = 0;

    if (prior) {
        const size_t n = strlen(prior) + 1;
        saved = (char *)malloc(n);
        if (!saved) return -1;
        memcpy(saved, prior, n);
    }
    load_opts(&opts);
    if (setenv("WASTE_MTP_VQ2", "2", 1)) {
        free(saved);
        return -1;
    }
    loaded = waste_model_load(&model, path, 8, &opts) == 0;
    if (saved) setenv("WASTE_MTP_VQ2", saved, 1);
    else unsetenv("WASTE_MTP_VQ2");
    free(saved);
    if (loaded) {
        waste_model_free(&model);
        return -1;
    }
    return 0;
}

static int corrupt_record_rejected(const char *path)
{
    static const int tokens[] = {3, 7};
    waste_load_opts opts;
    waste_model model;
    int loaded = 0, rc = -1;
    load_opts(&opts);
    /* Header identity is checked when the record is fetched, not while the
     * bank is opened.  The fixture changes only the per-record codebook id,
     * to another globally valid book, so this specifically gates equality
     * with the manifest bank's declared base. */
    if (waste_model_load(&model, path, 8, &opts)) return -1;
    loaded = 1;
    if (waste_model_mtp_set_enabled(&model, 1)) goto done;
    if (waste_model_prefill(&model, tokens, 2, 0) != NULL) goto done;
    int layer = -1, expert = -1;
    const char *why = waste_model_read_error(&model, &layer, &expert);
    if (!why || layer != model.mtp_layer || expert < 0 ||
        strcmp(why, "record header is not what the bank index describes"))
        goto done;
    rc = 0;
done:
    if (loaded) waste_model_free(&model);
    return rc;
}

static int embedding_failure_is_sticky(const char *path)
{
    waste_load_opts opts;
    waste_model model;
    float *row = NULL;
    int loaded = 0, rc = -1;
    load_opts(&opts);
    if (waste_model_load(&model, path, 8, &opts)) return -1;
    loaded = 1;
    row = (float *)malloc((size_t)model.cfg.hidden * sizeof(float));
    if (!row) goto done;

    /* Keep ownership of the real descriptor for model_free, but make this
     * one row read hit EBADF.  The forward-facing embedding helper must
     * return failure and publish it on the sticky read-error channel. */
    const int trunk_fd = model.trunk_fd;
    model.trunk_fd = -1;
    if (waste_embed_row(&model, 7, row) == 0) goto restore;
    int layer = 0, token = -1;
    const char *why = waste_model_read_error(&model, &layer, &token);
    if (!why || layer != -1 || token != 7 ||
        strcmp(why, "embedding row short read"))
        goto restore;
    waste_model_clear_read_error(&model);
    if (waste_model_read_error(&model, NULL, NULL)) goto restore;
    rc = 0;
restore:
    model.trunk_fd = trunk_fd;
done:
    free(row);
    if (loaded) waste_model_free(&model);
    return rc;
}

static int logits_argmax(const float *logits, int n)
{
    int best = 0;
    for (int i = 1; i < n; i++)
        if (logits[i] > logits[best]) best = i;
    return best;
}

static int prepare_oracle_model(waste_model *model, const char *path,
                                const int *prompt, int n_prompt, int token0,
                                int *draft_token1)
{
    waste_load_opts opts;
    load_opts(&opts);
    if (waste_model_load(model, path, 8, &opts)) return -1;
    if (waste_model_mtp_set_enabled(model, 1) ||
        !waste_model_prefill(model, prompt, n_prompt, 0))
        goto fail;
    int hidden_pos = -1;
    if (!waste_model_mtp_target_hidden(model, &hidden_pos) ||
        hidden_pos != n_prompt - 1 ||
        waste_model_mtp_cache_pos(model) != n_prompt - 1)
        goto fail;
    const float *proposal = waste_model_mtp_propose(
        model, token0, n_prompt - 1, NULL);
    if (!proposal || waste_model_mtp_cache_pos(model) != n_prompt) goto fail;
    *draft_token1 = logits_argmax(proposal, model->cfg.vocab);
    return 0;
fail:
    waste_model_free(model);
    return -1;
}

/* Mathematical/persistent state only. Cache heat and physical-work counters
 * are intentionally excluded: rejected verification really did perform
 * those reads and launches. MLA bytes after n_kv are dead by contract. */
static int same_oracle_state(const waste_model *a, const waste_model *b)
{
    const waste_config *c = &a->cfg;
    if (memcmp(c, &b->cfg, sizeof *c) ||
        a->mtp_active != b->mtp_active ||
        a->mtp_target_hidden_pos != b->mtp_target_hidden_pos ||
        a->mtp_alignment_error != b->mtp_alignment_error ||
        a->mtp_last_pos != b->mtp_last_pos ||
        a->mtp_last_token != b->mtp_last_token ||
        a->mtp_target_token != b->mtp_target_token ||
        a->mtp_shadow_argmax != b->mtp_shadow_argmax ||
        a->mtp_steps != b->mtp_steps ||
        a->mtp_shadow_steps != b->mtp_shadow_steps ||
        a->mtp_shadow_matches != b->mtp_shadow_matches ||
        a->n_blockres != b->n_blockres || a->media_used != b->media_used ||
        a->ctx_full != b->ctx_full || a->read_error != b->read_error ||
        a->cuda_kda_state_dirty != b->cuda_kda_state_dirty ||
        a->cuda_kda_failed != b->cuda_kda_failed ||
        a->mtp_oracle_open || b->mtp_oracle_open ||
        memcmp(a->x, b->x, (size_t)c->hidden * sizeof(float)) ||
        memcmp(a->logits, b->logits,
               (size_t)c->vocab * sizeof(float)) ||
        memcmp(a->mtp_target_hidden, b->mtp_target_hidden,
               (size_t)c->hidden * sizeof(float)) ||
        memcmp(a->mtp_hidden, b->mtp_hidden,
               (size_t)c->hidden * sizeof(float)) ||
        memcmp(a->mtp_input_embed, b->mtp_input_embed,
               (size_t)c->hidden * sizeof(float)) ||
        memcmp(a->mtp_logits, b->mtp_logits,
               (size_t)c->vocab * sizeof(float)))
        return 0;

    const size_t H = (size_t)c->kda_heads, D = (size_t)c->kda_dim;
    const size_t C = H * D;
    for (int L = 0; L < c->n_layers; L++) {
        if (a->n_kv[L] != b->n_kv[L]) return 0;
        if (c->kda_layer[L]) {
            if (memcmp(a->S[L], b->S[L], H * D * D * sizeof(float)) ||
                memcmp(a->conv[L], b->conv[L],
                       (size_t)3 * C * (size_t)(c->conv_k - 1) *
                       sizeof(float)))
                return 0;
        } else {
            const size_t n = (size_t)a->n_kv[L] *
                             (size_t)(c->kv_lora + c->qk_rope);
            if (n && memcmp(a->latcache[L], b->latcache[L],
                            n * sizeof(float)))
                return 0;
        }
    }
    const int M = a->mtp_layer;
    if (a->n_kv[M] != b->n_kv[M]) return 0;
    {
        const size_t n = (size_t)a->n_kv[M] *
                         (size_t)(c->kv_lora + c->qk_rope);
        if (n && memcmp(a->latcache[M], b->latcache[M],
                        n * sizeof(float)))
            return 0;
    }
    if (c->attn_res_block && a->n_blockres > 0 &&
        memcmp(a->blockres, b->blockres,
               (size_t)a->n_blockres * c->hidden * sizeof(float)))
        return 0;
    return 1;
}

static int prepare_chain_model(waste_model *model, const char *path,
                               const int *prompt, int n_prompt)
{
    waste_load_opts opts;
    load_opts(&opts);
    if (waste_model_load(model, path, 8, &opts)) return -1;
    if (waste_model_mtp_set_enabled(model, 1) ||
        !waste_model_prefill(model, prompt, n_prompt, 0))
        goto fail;
    int hidden_pos = -1;
    if (!waste_model_mtp_target_hidden(model, &hidden_pos) ||
        hidden_pos != n_prompt - 1 ||
        waste_model_mtp_cache_pos(model) != n_prompt - 1)
        goto fail;
    return 0;
fail:
    waste_model_free(model);
    return -1;
}

static int run_recursive_chain_contract(const char *path)
{
    static const int prompt[] = {3, 7, 11, 5};
    enum { N_PROMPT = 4, N_MODELS = 5 };
    const int token0 = 9, target_pos = N_PROMPT - 1;
    waste_model model[N_MODELS];
    int loaded = 0;
    int draft1[3] = {-1, -1, -1};
    int draft2[3] = {-1, -1, -1};
    int draft3a[3] = {-1, -1, -1};
    int draft3b[3] = {-1, -1, -1};
    double seconds1 = 0.0, seconds2 = 0.0;
    double seconds3a = 0.0, seconds3b = 0.0;

    memset(model, 0, sizeof model);
    for (int i = 0; i < N_MODELS; i++) {
        if (prepare_chain_model(&model[i], path, prompt, N_PROMPT))
            goto fail;
        loaded++;
    }

    /* All argument and complete-chain capacity gates are pre-mutation. */
    int rejected[3] = {101, 102, 103};
    double rejected_seconds = 1.0;
    if (waste_model_mtp_propose_greedy_chain(
            &model[1], token0, target_pos, 0, rejected,
            &rejected_seconds) == 0 || rejected_seconds != 0.0 ||
        memcmp(rejected, (int[3]){101, 102, 103}, sizeof rejected))
        goto fail;
    rejected_seconds = 1.0;
    if (waste_model_mtp_propose_greedy_chain(
            &model[1], token0, target_pos, 4, rejected,
            &rejected_seconds) == 0 || rejected_seconds != 0.0 ||
        memcmp(rejected, (int[3]){101, 102, 103}, sizeof rejected))
        goto fail;
    const int saved_cap = model[1].kv_cap;
    model[1].kv_cap = target_pos + 2;
    rejected_seconds = 1.0;
    if (waste_model_mtp_propose_greedy_chain(
            &model[1], token0, target_pos, 3, rejected,
            &rejected_seconds) == 0 || rejected_seconds != 0.0 ||
        memcmp(rejected, (int[3]){101, 102, 103}, sizeof rejected)) {
        model[1].kv_cap = saved_cap;
        goto fail;
    }
    model[1].kv_cap = saved_cap;
    if (!same_oracle_state(&model[0], &model[1])) goto fail;

    const float *ordinary = waste_model_mtp_propose(
        &model[0], token0, target_pos, NULL);
    if (!ordinary) goto fail;
    const int ordinary_draft = logits_argmax(ordinary, model[0].cfg.vocab);
    if (waste_model_mtp_propose_greedy_chain(
            &model[1], token0, target_pos, 1, draft1, &seconds1) ||
        waste_model_mtp_propose_greedy_chain(
            &model[2], token0, target_pos, 2, draft2, &seconds2) ||
        waste_model_mtp_propose_greedy_chain(
            &model[3], token0, target_pos, 3, draft3a, &seconds3a) ||
        waste_model_mtp_propose_greedy_chain(
            &model[4], token0, target_pos, 3, draft3b, &seconds3b))
        goto fail;
    if (draft1[0] != ordinary_draft || draft2[0] != ordinary_draft ||
        draft3a[0] != ordinary_draft || draft3b[0] != ordinary_draft ||
        draft2[1] != draft3a[1] ||
        memcmp(draft3a, draft3b, sizeof draft3a) ||
        seconds1 <= 0.0 || seconds2 <= 0.0 ||
        seconds3a <= 0.0 || seconds3b <= 0.0)
        goto fail;

    /* Every depth unwinds to the exact state of the grounded first row.
     * A target continuation therefore cannot observe the deeper draft work. */
    for (int i = 1; i < N_MODELS; i++)
        if (!same_oracle_state(&model[0], &model[i])) goto fail;
    const int vocab = model[0].cfg.vocab;
    const float *control = waste_model_step(
        &model[0], token0, N_PROMPT, NULL);
    if (!control) goto fail;
    for (int i = 1; i < N_MODELS; i++) {
        const float *continued = waste_model_step(
            &model[i], token0, N_PROMPT, NULL);
        if (!continued ||
            memcmp(control, continued, (size_t)vocab * sizeof(float)) ||
            !same_oracle_state(&model[0], &model[i]))
            goto fail;
    }

    for (int i = 0; i < loaded; i++) waste_model_free(&model[i]);
    return 0;
fail:
    for (int i = 0; i < loaded; i++) waste_model_free(&model[i]);
    return -1;
}

static int run_oracle_contract(const char *path)
{
    static const int prompt[] = {3, 7, 11, 5};
    enum { N_PROMPT = 4, N_MODELS = 4 };
    const int token0 = 9, pos0 = N_PROMPT;
    waste_model model[N_MODELS];
    int loaded = 0, draft_token1 = -1;
    float *ordinary_logits = NULL, *ordinary_hidden = NULL;
    waste_mtp_verify2_oracle *oracle = NULL;

    memset(model, 0, sizeof model);
    for (int i = 0; i < N_MODELS; i++) {
        int draft = -1;
        if (prepare_oracle_model(&model[i], path, prompt, N_PROMPT,
                                 token0, &draft))
            goto fail;
        loaded++;
        if (i == 0) draft_token1 = draft;
        else if (draft != draft_token1) goto fail;
    }
    const int vocab = model[0].cfg.vocab, hidden = model[0].cfg.hidden;
    ordinary_logits = (float *)malloc((size_t)2 * vocab * sizeof(float));
    ordinary_hidden = (float *)malloc((size_t)2 * hidden * sizeof(float));
    if (!ordinary_logits || !ordinary_hidden) goto fail;

    /* Commit: the oracle's retained rows and complete live state equal two
     * ordinary serial target steps from the same proposed draft. */
    const float *logits = waste_model_step(&model[1], token0, pos0, NULL);
    if (!logits) goto fail;
    memcpy(ordinary_logits, logits, (size_t)vocab * sizeof(float));
    memcpy(ordinary_hidden, model[1].mtp_target_hidden,
           (size_t)hidden * sizeof(float));
    logits = waste_model_step(&model[1], draft_token1, pos0 + 1, NULL);
    if (!logits) goto fail;
    memcpy(ordinary_logits + vocab, logits, (size_t)vocab * sizeof(float));
    memcpy(ordinary_hidden + hidden, model[1].mtp_target_hidden,
           (size_t)hidden * sizeof(float));

    if (waste_model_mtp_verify2_oracle_begin(
            &model[0], token0, draft_token1, pos0, NULL, NULL, &oracle) ||
        !oracle ||
        !waste_model_mtp_verify2_oracle_logits(oracle, 0) ||
        !waste_model_mtp_verify2_oracle_logits(oracle, 1) ||
        waste_model_mtp_verify2_oracle_logits(oracle, 2) ||
        !waste_model_mtp_verify2_oracle_hidden(oracle, 0) ||
        !waste_model_mtp_verify2_oracle_hidden(oracle, 1) ||
        waste_model_mtp_verify2_oracle_hidden(oracle, -1) ||
        memcmp(waste_model_mtp_verify2_oracle_logits(oracle, 0),
               ordinary_logits, (size_t)2 * vocab * sizeof(float)) ||
        memcmp(waste_model_mtp_verify2_oracle_hidden(oracle, 0),
               ordinary_hidden, (size_t)2 * hidden * sizeof(float)) ||
        waste_model_mtp_verify2_oracle_finish(oracle, 2) == 0)
        goto fail;
    if (waste_model_mtp_verify2_oracle_finish(oracle, 1)) goto fail;
    oracle = NULL;
    if (!same_oracle_state(&model[0], &model[1])) goto fail;

    /* The committed state also continues identically, not merely at the two
     * retained output rows. */
    const int token2 = draft_token1 == 17 ? 18 : 17;
    const float *a = waste_model_step(&model[0], token2, pos0 + 2, NULL);
    const float *b = waste_model_step(&model[1], token2, pos0 + 2, NULL);
    if (!a || !b || memcmp(a, b, (size_t)vocab * sizeof(float)) ||
        !same_oracle_state(&model[0], &model[1]))
        goto fail;

    /* Reject: compare against a control that consumed token0 only.  The
     * oracle has really run token1, so this catches every missing recurrent
     * or MTP rollback field. */
    b = waste_model_step(&model[3], token0, pos0, NULL);
    if (!b) goto fail;
    memcpy(ordinary_logits, b, (size_t)vocab * sizeof(float));
    memcpy(ordinary_hidden, model[3].mtp_target_hidden,
           (size_t)hidden * sizeof(float));
    const double reject_mtp_seconds = model[2].mtp_seconds;
    if (waste_model_mtp_verify2_oracle_begin(
            &model[2], token0, draft_token1, pos0, NULL, NULL, &oracle) ||
        !oracle ||
        memcmp(waste_model_mtp_verify2_oracle_logits(oracle, 0),
               ordinary_logits, (size_t)vocab * sizeof(float)) ||
        memcmp(waste_model_mtp_verify2_oracle_hidden(oracle, 0),
               ordinary_hidden, (size_t)hidden * sizeof(float)) ||
        waste_model_mtp_verify2_oracle_finish(oracle, 0))
        goto fail;
    oracle = NULL;
    if (model[2].mtp_seconds != reject_mtp_seconds ||
        !same_oracle_state(&model[2], &model[3]))
        goto fail;

    /* A replacement for the rejected token1 must deterministically match a
     * genuine one-step continuation, including its newly rebuilt MTP row. */
    const int replacement = draft_token1 == 23 ? 24 : 23;
    a = waste_model_step(&model[2], replacement, pos0 + 1, NULL);
    b = waste_model_step(&model[3], replacement, pos0 + 1, NULL);
    if (!a || !b || memcmp(a, b, (size_t)vocab * sizeof(float)) ||
        !same_oracle_state(&model[2], &model[3]))
        goto fail;

    for (int i = 0; i < loaded; i++) waste_model_free(&model[i]);
    free(ordinary_logits); free(ordinary_hidden);
    return 0;
fail:
    /* An open transaction owns heap state but has no separate abort API;
     * finish(false) is the rollback-and-destroy operation. */
    if (oracle) waste_model_mtp_verify2_oracle_finish(oracle, 0);
    for (int i = 0; i < loaded; i++) waste_model_free(&model[i]);
    free(ordinary_logits); free(ordinary_hidden);
    return -1;
}

static int run_fast_verify2_contract(const char *path)
{
    static const int prompt[] = {3, 7, 11, 5};
    enum { N_PROMPT = 4, N_MODELS = 4 };
    const int token0 = 9, pos0 = N_PROMPT;
    waste_model model[N_MODELS];
    int loaded = 0, draft_token1 = -1;
    float *serial_logits = NULL, *serial_hidden = NULL;
    int *serial_routes = NULL, *fast_routes = NULL;
    waste_mtp_verify2 *verify = NULL;

    memset(model, 0, sizeof model);
    for (int i = 0; i < N_MODELS; i++) {
        int draft = -1;
        if (prepare_oracle_model(&model[i], path, prompt, N_PROMPT,
                                 token0, &draft))
            goto fail;
        loaded++;
        if (i == 0) draft_token1 = draft;
        else if (draft != draft_token1) goto fail;
        if (waste_model_mtp_verify2_get_vq2(&model[i]) != 0 ||
            waste_model_mtp_verify2_set_vq2(&model[i], -1) == 0 ||
            waste_model_mtp_verify2_set_vq2(&model[i], 2) == 0 ||
            waste_model_mtp_verify2_set_vq2(&model[i], 1) == 0 ||
            waste_model_mtp_verify2_get_vq2(&model[i]) != 0 ||
            waste_model_mtp_verify2_set_vq2(&model[i], 0) != 0)
            goto fail;
    }
    const int vocab = model[0].cfg.vocab, hidden = model[0].cfg.hidden;
    const size_t route_n = (size_t)model[0].cfg.n_layers *
                           model[0].cfg.top_k;
    serial_logits = (float *)malloc((size_t)2 * vocab * sizeof(float));
    serial_hidden = (float *)malloc((size_t)2 * hidden * sizeof(float));
    serial_routes = (int *)malloc((size_t)2 * route_n * sizeof(int));
    fast_routes = (int *)malloc((size_t)2 * route_n * sizeof(int));
    if (!serial_logits || !serial_hidden || !serial_routes || !fast_routes)
        goto fail;
    memset(serial_routes, 0xa5, (size_t)2 * route_n * sizeof(int));
    memset(fast_routes, 0xa5, (size_t)2 * route_n * sizeof(int));

    /* Accepted fast state, both exposed rows, and one further continuation
     * must be byte-identical to the ordinary serial target. */
    const float *p = waste_model_step(
        &model[1], token0, pos0, serial_routes);
    if (!p) goto fail;
    memcpy(serial_logits, p, (size_t)vocab * sizeof(float));
    memcpy(serial_hidden, model[1].mtp_target_hidden,
           (size_t)hidden * sizeof(float));
    p = waste_model_step(
        &model[1], draft_token1, pos0 + 1, serial_routes + route_n);
    if (!p) goto fail;
    memcpy(serial_logits + vocab, p, (size_t)vocab * sizeof(float));
    memcpy(serial_hidden + hidden, model[1].mtp_target_hidden,
           (size_t)hidden * sizeof(float));

    if (waste_model_mtp_verify2_begin(
            &model[0], token0, draft_token1, pos0,
            fast_routes, fast_routes + route_n, &verify) || !verify ||
        !waste_model_mtp_verify2_logits(verify, 0) ||
        !waste_model_mtp_verify2_logits(verify, 1) ||
        waste_model_mtp_verify2_logits(verify, 2) ||
        !waste_model_mtp_verify2_hidden(verify, 0) ||
        !waste_model_mtp_verify2_hidden(verify, 1) ||
        waste_model_mtp_verify2_hidden(verify, -1) ||
        memcmp(waste_model_mtp_verify2_logits(verify, 0),
               serial_logits, (size_t)2 * vocab * sizeof(float)) ||
        memcmp(waste_model_mtp_verify2_hidden(verify, 0),
               serial_hidden, (size_t)2 * hidden * sizeof(float)) ||
        memcmp(fast_routes, serial_routes,
               (size_t)2 * route_n * sizeof(int)))
        goto fail;
    /* The result views live in model-owned persistent scratch.  While they
     * are open, every semantic mutator and model free must fail/no-op; the
     * transaction must remain finishable and unchanged afterward. */
    size_t snapshot_bytes = 0;
    const int held_nkv = model[0].n_kv[0];
    const uint64_t held_steps = model[0].mtp_steps;
    const int held_target_pos = model[0].mtp_target_hidden_pos;
    waste_model_reset(&model[0]);
    waste_model_clear_read_error(&model[0]);
    waste_model_free(&model[0]);
    const int blocked_token = 31;
    if (waste_model_step(&model[0], blocked_token, pos0 + 2, NULL) ||
        waste_model_prefill(&model[0], &blocked_token, 1, pos0 + 2) ||
        waste_model_mtp_set_enabled(&model[0], 0) == 0 ||
        waste_model_mtp_propose(&model[0], blocked_token,
                                pos0 + 1, NULL) ||
        waste_model_state_size(&model[0], pos0 + 2, &snapshot_bytes) == 0 ||
        waste_model_set_cuda_kda(&model[0], 0) == 0 ||
        waste_model_mtp_verify2_set_vq2(&model[0], 0) == 0 ||
        model[0].n_kv[0] != held_nkv ||
        model[0].mtp_steps != held_steps ||
        model[0].mtp_target_hidden_pos != held_target_pos ||
        memcmp(waste_model_mtp_verify2_logits(verify, 0),
               serial_logits, (size_t)2 * vocab * sizeof(float)) ||
        memcmp(waste_model_mtp_verify2_hidden(verify, 0),
               serial_hidden, (size_t)2 * hidden * sizeof(float)) ||
        waste_model_mtp_verify2_finish(verify, 2) == 0)
        goto fail;
    if (waste_model_mtp_verify2_finish(verify, 1)) goto fail;
    verify = NULL;
    if (!same_oracle_state(&model[0], &model[1])) goto fail;
    const int token2 = draft_token1 == 17 ? 18 : 17;
    const float *a = waste_model_step(&model[0], token2, pos0 + 2, NULL);
    const float *b = waste_model_step(&model[1], token2, pos0 + 2, NULL);
    if (!a || !b || memcmp(a, b, (size_t)vocab * sizeof(float)) ||
        !same_oracle_state(&model[0], &model[1]))
        goto fail;

    /* Rejection is exactly the state after token0, including every KDA
     * ring/state matrix and the predictor's anticipatory proposal row. */
    b = waste_model_step(&model[3], token0, pos0, NULL);
    if (!b) goto fail;
    memcpy(serial_logits, b, (size_t)vocab * sizeof(float));
    memcpy(serial_hidden, model[3].mtp_target_hidden,
           (size_t)hidden * sizeof(float));
    const double mtp_seconds = model[2].mtp_seconds;
    if (waste_model_mtp_verify2_begin(
            &model[2], token0, draft_token1, pos0,
            NULL, NULL, &verify) || !verify ||
        memcmp(waste_model_mtp_verify2_logits(verify, 0),
               serial_logits, (size_t)vocab * sizeof(float)) ||
        memcmp(waste_model_mtp_verify2_hidden(verify, 0),
               serial_hidden, (size_t)hidden * sizeof(float)) ||
        waste_model_mtp_verify2_finish(verify, 0))
        goto fail;
    verify = NULL;
    if (model[2].mtp_seconds != mtp_seconds ||
        !same_oracle_state(&model[2], &model[3]))
        goto fail;
    const int replacement = draft_token1 == 23 ? 24 : 23;
    a = waste_model_step(&model[2], replacement, pos0 + 1, NULL);
    b = waste_model_step(&model[3], replacement, pos0 + 1, NULL);
    if (!a || !b || memcmp(a, b, (size_t)vocab * sizeof(float)) ||
        !same_oracle_state(&model[2], &model[3]))
        goto fail;

    for (int i = 0; i < loaded; i++) waste_model_free(&model[i]);
    free(serial_logits);
    free(serial_hidden);
    free(serial_routes);
    free(fast_routes);
    return 0;
fail:
    if (verify) waste_model_mtp_verify2_finish(verify, 0);
    for (int i = 0; i < loaded; i++) waste_model_free(&model[i]);
    free(serial_logits);
    free(serial_hidden);
    free(serial_routes);
    free(fast_routes);
    return -1;
}

int main(int argc, char **argv)
{
    static const int tokens[] = {3, 7, 11, 5, 9};
    enum { N_PROMPT = 4 };
    waste_load_opts opts;
    waste_model model;
    float *control_prompt = NULL, *control_next = NULL;
    float *first_proposal = NULL, *first_target = NULL;
    unsigned char *snapshot = NULL;
    int vocab = 0, loaded = 0;

    if (argc < 3) {
        fprintf(stderr, "usage: %s BASE MTP [MALFORMED ...]\n", argv[0]);
        return 2;
    }
    if (invalid_vq2_env_rejected(argv[2])) goto fail;
    if (run_control(argv[1], tokens, N_PROMPT, &control_prompt,
                    &control_next, &vocab))
        goto fail;

    load_opts(&opts);
    if (waste_model_load(&model, argv[2], 8, &opts)) {
        fprintf(stderr, "MTP fixture load failed: %s\n", argv[2]);
        goto fail;
    }
    loaded = 1;
    REQUIRE(model.cfg.vocab == vocab);
    const size_t logits_bytes = (size_t)vocab * sizeof(float);
    first_proposal = (float *)malloc(logits_bytes);
    first_target = (float *)malloc(logits_bytes);
    REQUIRE(first_proposal && first_target);

    /* Merely carrying an MTP contract must not disable v1 session state.
     * Once MTP is active, all three v1 entry points fail closed because v1
     * has no appended-layer cache or target-hidden representation. */
    REQUIRE(waste_model_mtp_available(&model));
    REQUIRE(!waste_model_mtp_enabled(&model));
    REQUIRE(waste_model_mtp_cache_pos(&model) == 0);
    REQUIRE(waste_model_mtp_target_hidden(&model, NULL) == NULL);
    REQUIRE(waste_model_mtp_set_enabled(&model, 2) != 0);
    model.media = model.x;
    model.media_n = 1;
    REQUIRE(waste_model_mtp_set_enabled(&model, 1) != 0);
    REQUIRE(!waste_model_mtp_enabled(&model));
    model.media = NULL;
    model.media_n = 0;

    size_t snapshot_bytes = 0, written = 0;
    REQUIRE(waste_model_state_size(&model, 0, &snapshot_bytes) == 0);
    REQUIRE(snapshot_bytes > 0);
    snapshot = (unsigned char *)malloc(snapshot_bytes);
    REQUIRE(snapshot != NULL);
    REQUIRE(waste_model_state_export(&model, 0, snapshot, snapshot_bytes,
                                     &written) == 0);
    REQUIRE(written == snapshot_bytes);

    REQUIRE(waste_model_mtp_set_enabled(&model, 1) == 0);
    REQUIRE(waste_model_mtp_enabled(&model));
    REQUIRE(waste_model_state_size(&model, 0, &written) != 0);
    REQUIRE(waste_model_state_export(&model, 0, snapshot, snapshot_bytes,
                                     &written) != 0);
    int restored_pos = -1;
    REQUIRE(waste_model_state_import(&model, snapshot, snapshot_bytes,
                                     &restored_pos) == -2);
    REQUIRE(waste_model_mtp_set_enabled(&model, 0) == 0);
    REQUIRE(waste_model_state_import(&model, snapshot, snapshot_bytes,
                                     &restored_pos) == 0);
    REQUIRE(restored_pos == 0);
    /* A v1 import deliberately lacks the target hidden needed to resume
     * MTP.  Only an explicit fresh reset may make it eligible again. */
    REQUIRE(waste_model_mtp_set_enabled(&model, 1) != 0);
    waste_model_reset(&model);
    REQUIRE(!waste_model_mtp_enabled(&model));
    REQUIRE(waste_model_mtp_set_enabled(&model, 1) == 0);
    /* Media may be attached after the mode switch, so the step boundary
     * independently fails closed and poisons the mixed recurrent state. */
    model.media = model.x;
    model.media_n = 1;
    REQUIRE(waste_model_step(&model, tokens[0], 0, NULL) == NULL);
    REQUIRE(model.mtp_alignment_error);
    model.media = NULL;
    model.media_n = 0;
    waste_model_reset(&model);
    REQUIRE(waste_model_mtp_enabled(&model));

    /* N target tokens provide N-1 recurrent MTP rows: token zero has no
     * preceding target hidden, then every known token advances one row. */
    const float *target = waste_model_prefill(&model, tokens, N_PROMPT, 0);
    REQUIRE(target != NULL);
    REQUIRE(memcmp(target, control_prompt, logits_bytes) == 0);
    REQUIRE(waste_model_mtp_cache_pos(&model) == N_PROMPT - 1);
    REQUIRE(waste_model_mtp_steps(&model) == N_PROMPT - 1);
    REQUIRE(waste_model_mtp_shadow_steps(&model) == N_PROMPT - 1);
    REQUIRE(waste_model_mtp_shadow_matches(&model) <=
            waste_model_mtp_shadow_steps(&model));
    int hidden_pos = -1;
    REQUIRE(waste_model_mtp_target_hidden(&model, &hidden_pos) != NULL);
    REQUIRE(hidden_pos == N_PROMPT - 1);

    /* A proposal is anchored to the saved target position.  Rejecting a
     * stale position must be side-effect free; the valid proposal advances
     * exactly once.  Consuming the same known token then reuses that row. */
    REQUIRE(waste_model_mtp_propose(&model, tokens[N_PROMPT],
                                    N_PROMPT - 2, NULL) == NULL);
    REQUIRE(waste_model_mtp_cache_pos(&model) == N_PROMPT - 1);
    REQUIRE(waste_model_mtp_steps(&model) == N_PROMPT - 1);
    const float *proposal = waste_model_mtp_propose(
        &model, tokens[N_PROMPT], N_PROMPT - 1, NULL);
    REQUIRE(proposal != NULL);
    memcpy(first_proposal, proposal, logits_bytes);
    REQUIRE(waste_model_mtp_cache_pos(&model) == N_PROMPT);
    REQUIRE(waste_model_mtp_steps(&model) == N_PROMPT);
    REQUIRE(waste_model_mtp_shadow_steps(&model) == N_PROMPT - 1);

    target = waste_model_step(&model, tokens[N_PROMPT], N_PROMPT, NULL);
    REQUIRE(target != NULL);
    memcpy(first_target, target, logits_bytes);
    REQUIRE(memcmp(target, control_next, logits_bytes) == 0);
    REQUIRE(waste_model_mtp_cache_pos(&model) == N_PROMPT);
    REQUIRE(waste_model_mtp_steps(&model) == N_PROMPT);
    REQUIRE(waste_model_mtp_shadow_steps(&model) == N_PROMPT);
    REQUIRE(waste_model_mtp_shadow_matches(&model) <= N_PROMPT);
    REQUIRE(waste_model_mtp_seconds(&model) > 0.0);
    REQUIRE(waste_model_mtp_target_hidden(&model, &hidden_pos) != NULL);
    REQUIRE(hidden_pos == N_PROMPT);
    const uint64_t first_matches = waste_model_mtp_shadow_matches(&model);

    /* Reset preserves the caller's enabled mode but clears every recurrent
     * row, saved hidden, timing and counter.  The complete replay must be
     * bit-identical in both target and proposal streams. */
    waste_model_reset(&model);
    REQUIRE(waste_model_mtp_enabled(&model));
    REQUIRE(waste_model_mtp_cache_pos(&model) == 0);
    REQUIRE(waste_model_mtp_steps(&model) == 0);
    REQUIRE(waste_model_mtp_shadow_steps(&model) == 0);
    REQUIRE(waste_model_mtp_shadow_matches(&model) == 0);
    REQUIRE(waste_model_mtp_seconds(&model) == 0.0);
    REQUIRE(waste_model_mtp_target_hidden(&model, NULL) == NULL);

    target = waste_model_prefill(&model, tokens, N_PROMPT, 0);
    REQUIRE(target != NULL);
    REQUIRE(memcmp(target, control_prompt, logits_bytes) == 0);
    proposal = waste_model_mtp_propose(
        &model, tokens[N_PROMPT], N_PROMPT - 1, NULL);
    REQUIRE(proposal != NULL);
    REQUIRE(memcmp(proposal, first_proposal, logits_bytes) == 0);
    target = waste_model_step(&model, tokens[N_PROMPT], N_PROMPT, NULL);
    REQUIRE(target != NULL);
    REQUIRE(memcmp(target, first_target, logits_bytes) == 0);
    REQUIRE(waste_model_mtp_cache_pos(&model) == N_PROMPT);
    REQUIRE(waste_model_mtp_steps(&model) == N_PROMPT);
    REQUIRE(waste_model_mtp_shadow_steps(&model) == N_PROMPT);
    REQUIRE(waste_model_mtp_shadow_matches(&model) == first_matches);

    waste_model_free(&model);
    loaded = 0;
    REQUIRE(embedding_failure_is_sticky(argv[2]) == 0);
    REQUIRE(run_recursive_chain_contract(argv[2]) == 0);
    REQUIRE(run_oracle_contract(argv[2]) == 0);
    REQUIRE(run_fast_verify2_contract(argv[2]) == 0);
    for (int i = 3; i < argc; i++) {
        const int rejected = strstr(argv[i], "record-codebook")
            ? corrupt_record_rejected(argv[i])
            : malformed_rejected(argv[i]);
        if (rejected) goto fail;
    }

    free(snapshot);
    free(first_proposal);
    free(first_target);
    free(control_prompt);
    free(control_next);
    printf("GLM-5.3 MTP OK — alignment, shadow, replay, verifier transaction, state and load gates\n");
    return 0;

fail:
    if (loaded) waste_model_free(&model);
    free(snapshot);
    free(first_proposal);
    free(first_target);
    free(control_prompt);
    free(control_next);
    return 1;
}
