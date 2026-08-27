/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/* Public-only exercise of the source-separated backend seam. */

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "../src/waste.h"

enum { FAIL_NONE, FAIL_VQ_BEGIN, FAIL_VQ_GATE_UP, FAIL_VQ_DOWN };

typedef struct {
    int plan_calls, open_calls, close_calls, claim_calls, matvec_calls;
    int begin_calls, gate_up_calls, down_calls;
    int claim_result, fail_vq;
    uint32_t caps;
    uint64_t reserved;
    waste_status plan_status, open_status;
    waste_backend_model_info info;
    const waste_backend_model *model_view;
    uint32_t current_base;
    const uint16_t *current_scales;
    char events[128], detail[128];
    size_t n_events;
    int close_model_valid;
} mock_state;

#define CHECK(expr) do {                                                     \
    if (!(expr)) {                                                           \
        fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #expr);            \
        return 1;                                                            \
    }                                                                        \
} while (0)

static waste_status mock_problem(mock_state *s, const char *detail)
{
    snprintf(s->detail, sizeof s->detail, "%s", detail);
    return WASTE_E_BACKEND;
}

static void mock_event(mock_state *s, char event)
{
    if (s->n_events + 1 < sizeof s->events) {
        s->events[s->n_events++] = event;
        s->events[s->n_events] = '\0';
    }
}

static waste_status mock_plan(const waste_backend_model_info *model,
                              const void *cfg, uint64_t *reserved)
{
    mock_state *s = (mock_state *)cfg;
    if (!s || !model || model->struct_size != sizeof *model ||
        !model->arch || !reserved)
        return WASTE_E_BACKEND;
    s->plan_calls++;
    *reserved = s->reserved;
    return s->plan_status;
}

static waste_status mock_open(const waste_backend_model *model,
                              const void *cfg, void **ctx, uint32_t *caps)
{
    mock_state *s = (mock_state *)cfg;
    if (!s || !model || model->struct_size != sizeof *model ||
        model->info.struct_size != sizeof model->info ||
        !model->tensors || !model->n_tensors || !ctx || !caps)
        return WASTE_E_BACKEND;
    /* The synthetic container is a complete VQ3R tuple. Keeping this check
     * in a public-only mock catches a scheme name inferred from one field. */
    if (model->info.vq_scheme != WASTE_BACKEND_VQ3R ||
        model->info.vq_index_bits != 8 || model->info.vq_stages != 3 ||
        model->info.vq_entries != 256 || model->info.vq_vec_dim != 8 ||
        model->info.vq_index_block != 64)
        return mock_problem(s, "incomplete VQ3R model descriptor");
    s->open_calls++;
    s->info = model->info;
    s->model_view = model;
    *ctx = s;                         /* also tests cleanup on open failure */
    *caps = s->caps;
    return s->open_status;
}

static int mock_claim(void *ctx, const waste_backend_tensor *tensor)
{
    mock_state *s = (mock_state *)ctx;
    s->claim_calls++;
    if (s->claim_result < 0) return -1;
    return s->claim_result && tensor && tensor->bits == 4 && tensor->weights;
}

static waste_status mock_matvec(void *ctx,
                                const waste_backend_tensor *tensor,
                                const float *input, float *output)
{
    mock_state *s = (mock_state *)ctx;
    (void)tensor; (void)input; (void)output;
    s->matvec_calls++;
    return mock_problem(s, "injected projection failure");
}

static waste_status mock_vq_begin(void *ctx, const float *input,
                                  uint32_t cols, uint32_t gate_base,
                                  uint32_t up_base)
{
    mock_state *s = (mock_state *)ctx;
    const uint32_t latent = s->info.latent_dim ? s->info.latent_dim
                                               : s->info.hidden;
    mock_event(s, 'B');
    s->begin_calls++;
    if (!input || cols != latent ||
        up_base != gate_base + s->info.vq_stages)
        return mock_problem(s, "invalid VQ begin descriptor");
    s->current_base = gate_base;
    if (s->fail_vq == FAIL_VQ_BEGIN)
        return mock_problem(s, "injected VQ begin failure");
    return WASTE_OK;
}

static waste_status mock_vq_gate_up(void *ctx,
                                    const waste_backend_vq_matrix *gate,
                                    const waste_backend_vq_matrix *up,
                                    float *gate_output, float *up_output)
{
    mock_state *s = (mock_state *)ctx;
    const uint32_t latent = s->info.latent_dim ? s->info.latent_dim
                                               : s->info.hidden;
    mock_event(s, 'G');
    s->gate_up_calls++;
    const size_t gate_bytes =
        ((gate ? gate->rows : 0) + s->info.vq_index_block - 1) /
        s->info.vq_index_block *
        ((gate ? gate->cols : 0) / s->info.vq_vec_dim) *
        s->info.vq_index_block * s->info.vq_stages;
    const size_t up_bytes =
        ((up ? up->rows : 0) + s->info.vq_index_block - 1) /
        s->info.vq_index_block *
        ((up ? up->cols : 0) / s->info.vq_vec_dim) *
        s->info.vq_index_block * s->info.vq_stages;
    if (!gate || !up || gate->struct_size != sizeof *gate ||
        up->struct_size != sizeof *up || !gate->indices || !up->indices ||
        !gate->channel_scales || !up->channel_scales ||
        !gate_output || !up_output ||
        gate->record_scheme != WASTE_BACKEND_VQ3R ||
        up->record_scheme != WASTE_BACKEND_VQ3R)
        return mock_problem(s, "non-VQ3R expert record rejected");
    if (
        gate->indices_bytes != gate_bytes || up->indices_bytes != up_bytes ||
        gate->n_channel_scales != gate->rows ||
        up->n_channel_scales != up->rows ||
        gate->rows != s->info.moe_inter || up->rows != s->info.moe_inter ||
        gate->cols != latent || up->cols != latent ||
        gate->codebook_base != s->current_base ||
        up->codebook_base != s->current_base + s->info.vq_stages ||
        up->channel_scales != gate->channel_scales + gate->rows)
        return mock_problem(s, "invalid VQ gate/up descriptor");
    s->current_scales = gate->channel_scales;
    if (s->fail_vq == FAIL_VQ_GATE_UP)
        return mock_problem(s, "injected VQ gate/up failure");
    memset(gate_output, 0, (size_t)gate->rows * sizeof *gate_output);
    memset(up_output, 0, (size_t)up->rows * sizeof *up_output);
    return WASTE_OK;
}

static waste_status mock_vq_down(void *ctx, const float *input,
                                 const waste_backend_vq_matrix *down,
                                 float *output)
{
    mock_state *s = (mock_state *)ctx;
    const uint32_t latent = s->info.latent_dim ? s->info.latent_dim
                                               : s->info.hidden;
    mock_event(s, 'D');
    s->down_calls++;
    const size_t down_bytes =
        ((down ? down->rows : 0) + s->info.vq_index_block - 1) /
        s->info.vq_index_block *
        ((down ? down->cols : 0) / s->info.vq_vec_dim) *
        s->info.vq_index_block * s->info.vq_stages;
    if (!input || !down || down->struct_size != sizeof *down ||
        !down->indices || !down->channel_scales || !output ||
        down->record_scheme != WASTE_BACKEND_VQ3R)
        return mock_problem(s, "non-VQ3R expert record rejected");
    if (
        down->indices_bytes != down_bytes ||
        down->n_channel_scales != down->rows ||
        down->rows != latent || down->cols != s->info.moe_inter ||
        down->codebook_base != s->current_base + 2 * s->info.vq_stages ||
        down->channel_scales != s->current_scales + 2 * s->info.moe_inter)
        return mock_problem(s, "invalid VQ down descriptor");
    if (s->fail_vq == FAIL_VQ_DOWN)
        return mock_problem(s, "injected VQ down failure");
    memset(output, 0, (size_t)down->rows * sizeof *output);
    return WASTE_OK;
}

static const char *mock_detail(void *ctx)
{
    mock_state *s = (mock_state *)ctx;
    return s && s->detail[0] ? s->detail : NULL;
}

static void mock_close(void *ctx)
{
    mock_state *s = (mock_state *)ctx;
    s->close_model_valid = s->model_view &&
        s->model_view->struct_size == sizeof *s->model_view &&
        s->model_view->info.struct_size == sizeof s->model_view->info &&
        s->model_view->tensors && s->model_view->n_tensors;
    s->close_calls++;
}

static const waste_backend_v1 mock_backend = {
    .api_version = WASTE_BACKEND_API_VERSION,
    .struct_size = sizeof(waste_backend_v1),
    .name = "public-seam-mock",
    .plan = mock_plan,
    .open = mock_open,
    .claim_matvec = mock_claim,
    .matvec = mock_matvec,
    .vq_begin = mock_vq_begin,
    .vq_gate_up = mock_vq_gate_up,
    .vq_down = mock_vq_down,
    .error_detail = mock_detail,
    .close = mock_close,
};

static waste_status open_with(const char *model, const waste_backend_v1 *backend,
                              mock_state *state, uint64_t budget,
                              waste_ctx **ctx)
{
    waste_cfg cfg;
    waste_cfg_init(&cfg);
    cfg.ctx_tokens = 16;
    cfg.use_direct_io = 0;
    cfg.ram_budget_bytes = budget;
    return waste_open_with_backend(model, &cfg, backend, state, ctx);
}

static int check_rejected_vtable(const char *model, waste_backend_v1 backend)
{
    mock_state state = {0};
    waste_ctx *ctx = (waste_ctx *)1;
    CHECK(open_with(model, &backend, &state, 0, &ctx) == WASTE_E_UNSUPPORTED);
    CHECK(ctx == NULL && state.plan_calls == 0 && state.open_calls == 0);
    return 0;
}

static int check_missing_capability_callback(const char *model,
                                             waste_backend_v1 backend,
                                             uint32_t caps)
{
    mock_state state = { .caps = caps };
    waste_ctx *ctx = (waste_ctx *)1;
    CHECK(open_with(model, &backend, &state, 0, &ctx) == WASTE_E_UNSUPPORTED);
    CHECK(ctx == NULL && state.open_calls == 1 && state.close_calls == 1);
    return 0;
}

int main(int argc, char **argv)
{
    if (argc < 2 || argc > 3 ||
        (argc == 3 && strcmp(argv[2], "--expect-non-vq3r") != 0)) {
        fprintf(stderr, "usage: %s model.waste [--expect-non-vq3r]\n",
                argv[0]);
        return 2;
    }
    const char *model = argv[1];
    const int32_t token = 0;
    waste_ctx *ctx = NULL;

    if (argc == 3) {
        mock_state rejected = { .caps = WASTE_BACKEND_CAP_VQ };
        CHECK(open_with(model, &mock_backend, &rejected, 0, &ctx) == WASTE_OK);
        CHECK(waste_eval(ctx, &token, 1, NULL, NULL) == WASTE_E_BACKEND);
        CHECK(rejected.begin_calls == 1 && rejected.gate_up_calls == 1 &&
              rejected.down_calls == 0);
        CHECK(waste_error_detail(ctx) &&
              strstr(waste_error_detail(ctx), "non-VQ3R expert record"));
        waste_close(ctx);
        puts("PASS: per-record VQ scheme is exposed and rejected");
        return 0;
    }

    /* A provider can claim no tensors and leave the ordinary CPU path intact. */
    mock_state cpu = { .caps = WASTE_BACKEND_CAP_MATVEC };
    CHECK(open_with(model, &mock_backend, &cpu, 0, &ctx) == WASTE_OK && ctx);
    CHECK(waste_eval(ctx, &token, 1, NULL, NULL) == WASTE_OK);
    waste_close(ctx);
    CHECK(cpu.plan_calls == 1 && cpu.open_calls == 1 && cpu.close_calls == 1);
    CHECK(cpu.close_model_valid);
    CHECK(cpu.claim_calls > 0 && cpu.matvec_calls == 0);

    mock_state prefill = {
        .caps = WASTE_BACKEND_CAP_MATVEC, .claim_result = 1
    };
    const int32_t prompt[2] = {0, 0};
    CHECK(open_with(model, &mock_backend, &prefill, 0, &ctx) == WASTE_OK);
    CHECK(waste_eval(ctx, prompt, 2, NULL, NULL) == WASTE_OK);
    CHECK(prefill.claim_calls > 0 && prefill.matvec_calls == 0);
    waste_close(ctx);

    /* Planned backend memory replaces cold cache bytes and is reported. */
    waste_memplan planned, used;
    CHECK(waste_plan_memory(model, 16, &planned) == WASTE_OK);
    mock_state reservation = { .reserved = 4096 };
    CHECK(open_with(model, &mock_backend, &reservation, 0, &ctx) == WASTE_OK);
    CHECK(waste_memory_used(ctx, &used) == WASTE_OK);
    CHECK(used.scratch_bytes == planned.scratch_bytes + reservation.reserved);
    waste_close(ctx);
    CHECK(reservation.close_model_valid);
    mock_state too_large = { .reserved = UINT64_MAX };
    ctx = (waste_ctx *)1;
    CHECK(open_with(model, &mock_backend, &too_large, 0, &ctx) ==
          WASTE_E_RAM_BUDGET);
    CHECK(ctx == NULL && too_large.plan_calls == 1 && !too_large.open_calls);

    /* Malformed public vtables fail before any provider callback. */
    waste_backend_v1 bad = mock_backend;
    bad.api_version++;
    CHECK(check_rejected_vtable(model, bad) == 0);
    bad = mock_backend; bad.struct_size = sizeof bad - 1;
    CHECK(check_rejected_vtable(model, bad) == 0);
    bad = mock_backend; bad.name = NULL;
    CHECK(check_rejected_vtable(model, bad) == 0);
    bad = mock_backend; bad.open = NULL;
    CHECK(check_rejected_vtable(model, bad) == 0);
    bad = mock_backend; bad.close = NULL;
    CHECK(check_rejected_vtable(model, bad) == 0);

    mock_state plan_failed = { .plan_status = WASTE_E_UNSUPPORTED };
    CHECK(open_with(model, &mock_backend, &plan_failed, 0, &ctx) ==
          WASTE_E_UNSUPPORTED);
    CHECK(!ctx && plan_failed.plan_calls == 1 && !plan_failed.open_calls);
    mock_state open_failed = { .open_status = WASTE_E_UNSUPPORTED };
    CHECK(open_with(model, &mock_backend, &open_failed, 0, &ctx) ==
          WASTE_E_UNSUPPORTED);
    CHECK(!ctx && open_failed.open_calls == 1 && open_failed.close_calls == 1);
    CHECK(open_failed.close_model_valid);
    mock_state bad_caps = { .caps = 1u << 31 };
    CHECK(open_with(model, &mock_backend, &bad_caps, 0, &ctx) ==
          WASTE_E_UNSUPPORTED);
    CHECK(!ctx && bad_caps.close_calls == 1);

    bad = mock_backend; bad.claim_matvec = NULL;
    CHECK(check_missing_capability_callback(model, bad,
          WASTE_BACKEND_CAP_MATVEC) == 0);
    bad = mock_backend; bad.matvec = NULL;
    CHECK(check_missing_capability_callback(model, bad,
          WASTE_BACKEND_CAP_MATVEC) == 0);
    mock_state claim_failed = {
        .caps = WASTE_BACKEND_CAP_MATVEC, .claim_result = -1
    };
    CHECK(open_with(model, &mock_backend, &claim_failed, 0, &ctx) ==
          WASTE_E_BACKEND);
    CHECK(!ctx && claim_failed.claim_calls == 1 && claim_failed.close_calls == 1);

    /* Once selected execution fails, no CPU retry or provider retry occurs. */
    mock_state projection_failed = {
        .caps = WASTE_BACKEND_CAP_MATVEC, .claim_result = 1
    };
    CHECK(open_with(model, &mock_backend, &projection_failed, 0, &ctx) ==
          WASTE_OK && ctx);
    CHECK(waste_eval(ctx, &token, 1, NULL, NULL) == WASTE_E_BACKEND);
    CHECK(waste_error_detail(ctx) &&
          strstr(waste_error_detail(ctx), "injected projection failure"));
    CHECK(projection_failed.matvec_calls == 1);
    for (int i = 0; i < 32; i++)
        CHECK(waste_eval(ctx, &token, 1, NULL, NULL) == WASTE_E_BACKEND);
    CHECK(waste_generate(ctx, &token, 1, NULL, NULL, NULL) ==
          WASTE_E_BACKEND);
    CHECK(waste_state_save(ctx, "/not-read-on-terminal-backend") ==
          WASTE_E_BACKEND);
    CHECK(waste_state_load(ctx, "/not-read-on-terminal-backend") ==
          WASTE_E_BACKEND);
    waste_state_reset(ctx);
    CHECK(waste_eval(ctx, &token, 1, NULL, NULL) == WASTE_E_BACKEND);
    CHECK(projection_failed.matvec_calls == 1);
    waste_close(ctx);
    CHECK(projection_failed.close_calls == 1 &&
          projection_failed.close_model_valid);

    /* A projection failure aborts the token before the same provider can be
     * entered through its VQ capability and overwrite the first diagnosis. */
    mock_state mixed_failed = {
        .caps = WASTE_BACKEND_CAP_MATVEC | WASTE_BACKEND_CAP_VQ,
        .claim_result = 1
    };
    CHECK(open_with(model, &mock_backend, &mixed_failed, 0, &ctx) == WASTE_OK);
    CHECK(waste_eval(ctx, &token, 1, NULL, NULL) == WASTE_E_BACKEND);
    CHECK(mixed_failed.matvec_calls == 1 && !mixed_failed.begin_calls &&
          !mixed_failed.gate_up_calls && !mixed_failed.down_calls);
    CHECK(waste_error_detail(ctx) &&
          strstr(waste_error_detail(ctx), "injected projection failure"));
    waste_close(ctx);

    /* The engine owns routing, activation and ordered accumulation. The mock
     * sees one begin per MoE layer and one gate/down pair per routed expert. */
    mock_state vq = { .caps = WASTE_BACKEND_CAP_VQ };
    CHECK(open_with(model, &mock_backend, &vq, 0, &ctx) == WASTE_OK && ctx);
    CHECK(waste_eval(ctx, &token, 1, NULL, NULL) == WASTE_OK);
    CHECK(strcmp(vq.events, "BGDGDBGDGDBGDGD") == 0);
    CHECK(vq.begin_calls == 3 && vq.gate_up_calls == 6 && vq.down_calls == 6);
    CHECK(vq.claim_calls == 0 && vq.matvec_calls == 0);
    waste_close(ctx);
    CHECK(vq.close_calls == 1 && vq.close_model_valid);

    /* Every VQ stage is independently fail-closed and sticky. */
    for (int failure = FAIL_VQ_BEGIN; failure <= FAIL_VQ_DOWN; failure++) {
        mock_state failed = { .caps = WASTE_BACKEND_CAP_VQ,
                              .fail_vq = failure };
        CHECK(open_with(model, &mock_backend, &failed, 0, &ctx) == WASTE_OK);
        CHECK(waste_eval(ctx, &token, 1, NULL, NULL) == WASTE_E_BACKEND);
        CHECK(waste_error_detail(ctx) &&
              strstr(waste_error_detail(ctx), "injected VQ"));
        const size_t calls = failed.n_events;
        CHECK(waste_eval(ctx, &token, 1, NULL, NULL) == WASTE_E_BACKEND);
        CHECK(failed.n_events == calls);
        waste_close(ctx);
        CHECK(failed.close_calls == 1);
    }

    bad = mock_backend; bad.vq_begin = NULL;
    CHECK(check_missing_capability_callback(model, bad,
          WASTE_BACKEND_CAP_VQ) == 0);
    bad = mock_backend; bad.vq_gate_up = NULL;
    CHECK(check_missing_capability_callback(model, bad,
          WASTE_BACKEND_CAP_VQ) == 0);
    bad = mock_backend; bad.vq_down = NULL;
    CHECK(check_missing_capability_callback(model, bad,
          WASTE_BACKEND_CAP_VQ) == 0);

    puts("PASS: public backend lifecycle, VQ handoff and fail-closed execution");
    return 0;
}
