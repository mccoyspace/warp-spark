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
    for (int i = 3; i < argc; i++)
        if (malformed_rejected(argv[i])) goto fail;

    free(snapshot);
    free(first_proposal);
    free(first_target);
    free(control_prompt);
    free(control_next);
    printf("GLM-5.3 MTP OK — alignment, shadow, replay, state and load gates\n");
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
