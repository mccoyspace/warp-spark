/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/*
 * cuda_qualify.c -- deterministic public-API capture for an out-of-tree
 * CUDA backend.
 *
 * Build and run this on the CUDA qualification host; see
 * docs/CUDA_GB10_QUALIFICATION.md.  The output contains no model path or host
 * name. It still contains prompt/generated ids, logits and routes and is
 * private evidence; only the comparator's redacted report is publishable.
 */

#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#include "waste.h"
#include "waste_cuda_backend.h"

enum { PATH_CAP = 4096, MAX_PROMPT_TOKENS = 4096 };

typedef enum {
    MODE_CPU,
    MODE_VQ,
    MODE_MATVEC,
    MODE_COMBINED,
} run_mode;

typedef struct {
    uint64_t claim_queries, claimed_tensors, matvec_calls;
    uint64_t vq_begin_calls, vq_gate_up_calls, vq_down_calls;
    uint64_t callback_failures;
    uint32_t effective_capabilities;
} backend_counts;

typedef struct {
    waste_cuda_backend_config provider;
    backend_counts *counts;
    waste_backend_model_info *topology;
    run_mode mode;
} counting_config;

typedef struct {
    void *provider_ctx;
    backend_counts *counts;
} counting_context;

static waste_status counting_plan(const waste_backend_model_info *model,
                                  const void *opaque, uint64_t *reserved)
{
    const counting_config *cfg = (const counting_config *)opaque;
    if (!cfg || !model || !cfg->topology || !reserved) return WASTE_E_ARG;
    *cfg->topology = *model;
    if (cfg->mode == MODE_CPU) {
        *reserved = 0;
        return WASTE_OK;
    }
    return waste_cuda_backend_provider.plan(model, &cfg->provider, reserved);
}

static waste_status counting_open(const waste_backend_model *model,
                                  const void *opaque, void **backend_ctx,
                                  uint32_t *capabilities)
{
    const counting_config *cfg = (const counting_config *)opaque;
    if (!cfg || !cfg->counts || !backend_ctx || !capabilities)
        return WASTE_E_ARG;
    counting_context *ctx = (counting_context *)calloc(1, sizeof *ctx);
    if (!ctx) return WASTE_E_OOM;
    ctx->counts = cfg->counts;
    *cfg->topology = model->info;
    if (cfg->mode == MODE_CPU) {
        *backend_ctx = ctx;
        *capabilities = 0;
        return WASTE_OK;
    }
    const waste_status status = waste_cuda_backend_provider.open(
        model, &cfg->provider, &ctx->provider_ctx, capabilities);
    if (status != WASTE_OK) {
        free(ctx);
        return status;
    }
    ctx->counts->effective_capabilities = *capabilities;
    *backend_ctx = ctx;
    return WASTE_OK;
}

static int counting_claim_matvec(void *opaque,
                                 const waste_backend_tensor *tensor)
{
    counting_context *ctx = (counting_context *)opaque;
    if (!ctx) return -1;
    ctx->counts->claim_queries++;
    const int claimed = waste_cuda_backend_provider.claim_matvec(
        ctx->provider_ctx, tensor);
    if (claimed > 0) ctx->counts->claimed_tensors++;
    return claimed;
}

static waste_status counting_matvec(void *opaque,
                                    const waste_backend_tensor *tensor,
                                    const float *input, float *output)
{
    counting_context *ctx = (counting_context *)opaque;
    if (!ctx) return WASTE_E_ARG;
    ctx->counts->matvec_calls++;
    const waste_status status = waste_cuda_backend_provider.matvec(
        ctx->provider_ctx, tensor, input, output);
    if (status != WASTE_OK) ctx->counts->callback_failures++;
    return status;
}

static waste_status counting_vq_begin(void *opaque, const float *input,
                                      uint32_t cols, uint32_t gate_base,
                                      uint32_t up_base)
{
    counting_context *ctx = (counting_context *)opaque;
    if (!ctx) return WASTE_E_ARG;
    ctx->counts->vq_begin_calls++;
    const waste_status status = waste_cuda_backend_provider.vq_begin(
        ctx->provider_ctx, input, cols, gate_base, up_base);
    if (status != WASTE_OK) ctx->counts->callback_failures++;
    return status;
}

static waste_status counting_vq_gate_up(void *opaque,
                                        const waste_backend_vq_matrix *gate,
                                        const waste_backend_vq_matrix *up,
                                        float *gate_output, float *up_output)
{
    counting_context *ctx = (counting_context *)opaque;
    if (!ctx) return WASTE_E_ARG;
    ctx->counts->vq_gate_up_calls++;
    const waste_status status = waste_cuda_backend_provider.vq_gate_up(
        ctx->provider_ctx, gate, up, gate_output, up_output);
    if (status != WASTE_OK) ctx->counts->callback_failures++;
    return status;
}

static waste_status counting_vq_down(void *opaque, const float *input,
                                     const waste_backend_vq_matrix *down,
                                     float *output)
{
    counting_context *ctx = (counting_context *)opaque;
    if (!ctx) return WASTE_E_ARG;
    ctx->counts->vq_down_calls++;
    const waste_status status = waste_cuda_backend_provider.vq_down(
        ctx->provider_ctx, input, down, output);
    if (status != WASTE_OK) ctx->counts->callback_failures++;
    return status;
}

static const char *counting_error_detail(void *opaque)
{
    const counting_context *ctx = (const counting_context *)opaque;
    if (!ctx) return "qualification wrapper error";
    return ctx->provider_ctx
        ? waste_cuda_backend_provider.error_detail(ctx->provider_ctx)
        : "CPU metadata backend error";
}

static void counting_close(void *opaque)
{
    counting_context *ctx = (counting_context *)opaque;
    if (!ctx) return;
    if (ctx->provider_ctx) waste_cuda_backend_provider.close(ctx->provider_ctx);
    free(ctx);
}

static const waste_backend_v1 counting_provider = {
    .api_version = WASTE_BACKEND_API_VERSION,
    .struct_size = sizeof(waste_backend_v1),
    .name = "qualification-counter(cuda-gb10-q4g-vq3r)",
    .plan = counting_plan,
    .open = counting_open,
    .claim_matvec = counting_claim_matvec,
    .matvec = counting_matvec,
    .vq_begin = counting_vq_begin,
    .vq_gate_up = counting_vq_gate_up,
    .vq_down = counting_vq_down,
    .error_detail = counting_error_detail,
    .close = counting_close,
};

static double monotonic_ms(void)
{
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC, &t)) return 0.0;
    return (double)t.tv_sec * 1000.0 + (double)t.tv_nsec / 1.0e6;
}

static int parse_mode(const char *s, run_mode *mode)
{
    if (!strcmp(s, "cpu")) *mode = MODE_CPU;
    else if (!strcmp(s, "vq")) *mode = MODE_VQ;
    else if (!strcmp(s, "matvec")) *mode = MODE_MATVEC;
    else if (!strcmp(s, "combined")) *mode = MODE_COMBINED;
    else return -1;
    return 0;
}

static const char *mode_string(run_mode mode)
{
    switch (mode) {
    case MODE_CPU:      return "cpu";
    case MODE_VQ:       return "vq";
    case MODE_MATVEC:   return "matvec";
    case MODE_COMBINED: return "combined";
    }
    return "invalid";
}

static uint32_t mode_capabilities(run_mode mode)
{
    switch (mode) {
    case MODE_VQ:       return WASTE_BACKEND_CAP_VQ;
    case MODE_MATVEC:   return WASTE_BACKEND_CAP_MATVEC;
    case MODE_COMBINED: return WASTE_BACKEND_CAP_MATVEC |
                               WASTE_BACKEND_CAP_VQ;
    case MODE_CPU:      return 0;
    }
    return 0;
}

static int parse_uint(const char *s, unsigned long long limit,
                      unsigned long long *out)
{
    char *end = NULL;
    if (!s[0] || s[0] < '0' || s[0] > '9') return -1;
    errno = 0;
    const unsigned long long v = strtoull(s, &end, 10);
    if (errno || !end || *end || v > limit) return -1;
    *out = v;
    return 0;
}

static int parse_prompt(const char *text, int32_t *out, size_t *n_out)
{
    char *copy = strdup(text);
    if (!copy) return -1;
    size_t n = 0;
    char *save = NULL;
    for (char *part = strtok_r(copy, ",", &save); part;
         part = strtok_r(NULL, ",", &save)) {
        char *end = NULL;
        errno = 0;
        const long long v = strtoll(part, &end, 10);
        if (errno || !part[0] || !end || *end || v < 0 || v > INT32_MAX ||
            n == MAX_PROMPT_TOKENS) {
            free(copy);
            return -1;
        }
        out[n++] = (int32_t)v;
    }
    free(copy);
    if (!n) return -1;
    *n_out = n;
    return 0;
}

static int path_join(char out[PATH_CAP], const char *dir, const char *leaf)
{
    const int n = snprintf(out, PATH_CAP, "%s/%s", dir, leaf);
    return n > 0 && n < PATH_CAP ? 0 : -1;
}

static int write_logits(FILE *f, const float *logits, size_t vocab)
{
    return fwrite(logits, sizeof *logits, vocab, f) == vocab ? 0 : -1;
}

static int32_t argmax(const float *logits, size_t vocab)
{
    size_t best = 0;
    for (size_t i = 1; i < vocab; i++)
        if (logits[i] > logits[best]) best = i;
    return (int32_t)best;
}

static void json_string(FILE *f, const char *s)
{
    if (!s) {
        fputs("null", f);
        return;
    }
    fputc('"', f);
    for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
        switch (*p) {
        case '"': fputs("\\\"", f); break;
        case '\\': fputs("\\\\", f); break;
        case '\b': fputs("\\b", f); break;
        case '\f': fputs("\\f", f); break;
        case '\n': fputs("\\n", f); break;
        case '\r': fputs("\\r", f); break;
        case '\t': fputs("\\t", f); break;
        default:
            if (*p < 0x20) fprintf(f, "\\u%04x", (unsigned)*p);
            else fputc(*p, f);
        }
    }
    fputc('"', f);
}

static void json_tokens(FILE *f, const int32_t *tokens, size_t n)
{
    fputc('[', f);
    for (size_t i = 0; i < n; i++)
        fprintf(f, "%s%d", i ? ", " : "", tokens[i]);
    fputc(']', f);
}

static void json_control(FILE *f, const char *name, int comma)
{
    fprintf(f, "    \"%s\": ", name);
    json_string(f, getenv(name));
    fprintf(f, "%s\n", comma ? "," : "");
}

static int write_manifest(const char *path, run_mode mode,
                          const waste_cfg *cfg, const waste_model_info *model,
                          const waste_backend_model_info *topology,
                          const waste_memplan *memory,
                          const backend_counts *backend,
                          const backend_counts *after_prefill,
                          const int32_t *prompt, size_t n_prompt,
                          const int32_t *generated, size_t n_generated,
                          size_t vocab, double load_ms, double prefill_ms,
                          double decode_ms, const waste_stats *stats)
{
    char temporary[PATH_CAP];
    const int temporary_n = snprintf(temporary, sizeof temporary, "%s.tmp", path);
    if (temporary_n <= 0 || temporary_n >= (int)sizeof temporary) return -1;
    FILE *f = fopen(temporary, "wb");
    if (!f) return -1;

    fputs("{\n  \"format\": \"waste-backend-qualification-v1\",\n", f);
    fputs("  \"mode\": ", f); json_string(f, mode_string(mode));
    fputs(",\n  \"provider\": ", f);
    json_string(f, mode == MODE_CPU ? NULL : waste_cuda_backend_provider.name);
    fputs(",\n  \"build_info\": ", f); json_string(f, waste_build_info());
    fputs(",\n  \"model\": {\n    \"arch\": ", f);
    json_string(f, model->arch);
    fputs(",\n    \"quant_summary\": ", f);
    json_string(f, model->quant_summary);
    fprintf(f,
            ",\n    \"n_layers\": %u,\n    \"n_experts\": %u,\n"
            "    \"top_k\": %u,\n    \"first_dense\": %u,\n"
            "    \"hidden\": %u,\n    \"vq_scheme\": %u,\n"
            "    \"vq_stages\": %u,\n    \"vq_entries\": %u,\n"
            "    \"vq_vec_dim\": %u,\n    \"vq_index_bits\": %u,\n"
            "    \"params_total\": %llu,\n    \"params_active\": %llu\n"
            "  },\n",
            model->n_layers, model->n_experts, model->top_k,
            topology->first_dense, model->hidden, topology->vq_scheme,
            topology->vq_stages, topology->vq_entries,
            topology->vq_vec_dim, topology->vq_index_bits,
            (unsigned long long)model->params_total,
            (unsigned long long)model->params_active);
    fputs("  \"capture\": {\n    \"prompt_tokens\": ", f);
    json_tokens(f, prompt, n_prompt);
    fputs(",\n    \"generated_tokens\": ", f);
    json_tokens(f, generated, n_generated);
    fprintf(f,
            ",\n    \"vocab\": %llu,\n    \"logit_steps\": %llu\n  },\n",
            (unsigned long long)vocab, (unsigned long long)n_generated);
    fprintf(f,
            "  \"config\": {\n    \"ctx_tokens\": %u,\n"
            "    \"ram_budget_bytes\": %llu,\n    \"n_threads\": %d,\n"
            "    \"cpu_list\": ",
            cfg->ctx_tokens, (unsigned long long)cfg->ram_budget_bytes,
            cfg->n_threads);
    json_string(f, cfg->cpu_list);
    fprintf(f, ",\n    \"use_direct_io\": %d\n  },\n", cfg->use_direct_io);
    fprintf(f,
            "  \"resolved_memory\": {\n    \"trunk_bytes\": %llu,\n"
            "    \"state_bytes\": %llu,\n    \"scratch_bytes\": %llu,\n"
            "    \"expert_cache_bytes\": %llu,\n"
            "    \"floor_bytes\": %llu,\n"
            "    \"working_set_bytes\": %llu,\n"
            "    \"allocated_bytes\": %llu\n  },\n",
            (unsigned long long)memory->trunk_bytes,
            (unsigned long long)memory->state_bytes,
            (unsigned long long)memory->scratch_bytes,
            (unsigned long long)memory->min_expert_cache,
            (unsigned long long)memory->floor_bytes,
            (unsigned long long)memory->working_set_bytes,
            (unsigned long long)(memory->trunk_bytes + memory->state_bytes +
                                 memory->scratch_bytes +
                                 memory->min_expert_cache));
    fputs("  \"controls\": {\n", f);
    json_control(f, "WASTE_BACKEND", 1);
    json_control(f, "WASTE_THREADS", 1);
    json_control(f, "WASTE_CPUS", 1);
    json_control(f, "WASTE_Q8", 1);
    json_control(f, "WASTE_SDOT", 1);
    json_control(f, "WASTE_LOOKAHEAD", 1);
    json_control(f, "WASTE_XPAR", 1);
    json_control(f, "WASTE_XPAR_BATCH", 1);
    json_control(f, "WASTE_I8MM", 1);
    json_control(f, "WASTE_CCR_LAMBDA", 1);
    json_control(f, "WASTE_IO_THREADS", 1);
    json_control(f, "WASTE_IO_DEPTH", 1);
    json_control(f, "WASTE_P6_CHUNK", 1);
    json_control(f, "WASTE_DIRECT", 1);
    json_control(f, "WASTE_VERIFY", 1);
    json_control(f, "WASTE_MLOCK", 1);
    json_control(f, "WASTE_PURGEABLE", 0);
    fprintf(f,
            "  },\n  \"backend_calls\": {\n"
            "    \"effective_capabilities\": %u,\n"
            "    \"claim_queries\": %llu,\n"
            "    \"claimed_tensors\": %llu,\n"
            "    \"matvec_calls\": %llu,\n"
            "    \"vq_begin_calls\": %llu,\n"
            "    \"vq_gate_up_calls\": %llu,\n"
            "    \"vq_down_calls\": %llu,\n"
            "    \"callback_failures\": %llu,\n"
            "    \"prefill_matvec_calls\": %llu,\n"
            "    \"prefill_vq_begin_calls\": %llu,\n"
            "    \"prefill_vq_gate_up_calls\": %llu,\n"
            "    \"prefill_vq_down_calls\": %llu,\n"
            "    \"prefill_callback_failures\": %llu\n  },\n"
            "  \"timing\": {\n    \"load_ms\": %.6f,\n"
            "    \"prefill_ms\": %.6f,\n    \"decode_ms\": %.6f,\n"
            "    \"decode_tokens\": %llu,\n    \"decode_tok_s\": %.9f\n"
            "  },\n  \"stats\": {\n    \"tokens_generated\": %llu,\n"
            "    \"experts_hit\": %llu,\n    \"experts_missed\": %llu,\n"
            "    \"bytes_read\": %llu,\n    \"sec_total\": %.9f,\n"
            "    \"sec_io\": %.9f,\n    \"direct_io\": %d\n  }\n}\n",
            backend->effective_capabilities,
            (unsigned long long)backend->claim_queries,
            (unsigned long long)backend->claimed_tensors,
            (unsigned long long)backend->matvec_calls,
            (unsigned long long)backend->vq_begin_calls,
            (unsigned long long)backend->vq_gate_up_calls,
            (unsigned long long)backend->vq_down_calls,
            (unsigned long long)backend->callback_failures,
            (unsigned long long)after_prefill->matvec_calls,
            (unsigned long long)after_prefill->vq_begin_calls,
            (unsigned long long)after_prefill->vq_gate_up_calls,
            (unsigned long long)after_prefill->vq_down_calls,
            (unsigned long long)after_prefill->callback_failures,
            load_ms, prefill_ms, decode_ms,
            (unsigned long long)(n_generated ? n_generated - 1 : 0),
            n_generated > 1 && decode_ms > 0.0
                ? (double)(n_generated - 1) * 1000.0 / decode_ms : 0.0,
            (unsigned long long)stats->tokens_generated,
            (unsigned long long)stats->experts_hit,
            (unsigned long long)stats->experts_missed,
            (unsigned long long)stats->bytes_read,
            stats->sec_total, stats->sec_io, stats->direct_io);
    if (fclose(f) || rename(temporary, path)) {
        remove(temporary);
        return -1;
    }
    return 0;
}

static void usage(const char *program)
{
    fprintf(stderr,
            "usage: %s MODEL MODE OUTDIR IDS N_GENERATE "
            "[RAM_BYTES [THREADS [CPU_LIST]]]\n"
            "  MODE: cpu | vq | matvec | combined\n"
            "  IDS:  comma-separated, already-tokenized prompt ids\n"
            "  OUTDIR must not already exist; N_GENERATE must be >= 2\n",
            program);
}

int main(int argc, char **argv)
{
    if (argc < 6 || argc > 9) {
        usage(argv[0]);
        return 2;
    }
    run_mode mode;
    if (parse_mode(argv[2], &mode)) {
        fprintf(stderr, "unknown mode: %s\n", argv[2]);
        return 2;
    }
    int32_t prompt[MAX_PROMPT_TOKENS];
    size_t n_prompt = 0;
    if (parse_prompt(argv[4], prompt, &n_prompt)) {
        fprintf(stderr, "invalid prompt token list\n");
        return 2;
    }
    unsigned long long n_gen_ull = 0;
    if (parse_uint(argv[5], 1024, &n_gen_ull) || n_gen_ull < 2) {
        fprintf(stderr, "N_GENERATE must be in 2..1024\n");
        return 2;
    }
    const size_t n_gen = (size_t)n_gen_ull;

    waste_cfg cfg;
    waste_cfg_init(&cfg);
    cfg.ctx_tokens = (uint32_t)(n_prompt + n_gen + 8);
    if (argc > 6) {
        unsigned long long value = 0;
        if (parse_uint(argv[6], UINT64_MAX, &value)) {
            fprintf(stderr, "invalid RAM_BYTES\n");
            return 2;
        }
        cfg.ram_budget_bytes = (uint64_t)value;
    }
    if (argc > 7) {
        unsigned long long value = 0;
        if (parse_uint(argv[7], INT32_MAX, &value)) {
            fprintf(stderr, "invalid THREADS\n");
            return 2;
        }
        cfg.n_threads = (int)value;
    }
    if (argc > 8 && strcmp(argv[8], "-")) cfg.cpu_list = argv[8];

    (void)umask(0077);
    if (mkdir(argv[3], 0700)) {
        fprintf(stderr, "cannot create fresh output directory %s: %s\n",
                argv[3], strerror(errno));
        return 2;
    }
    char routes_path[PATH_CAP], logits_path[PATH_CAP], steps_path[PATH_CAP];
    char manifest_path[PATH_CAP], usage_path[PATH_CAP];
    if (path_join(routes_path, argv[3], "routes.txt") ||
        path_join(logits_path, argv[3], "logits.f32") ||
        path_join(steps_path, argv[3], "steps.tsv") ||
        path_join(manifest_path, argv[3], "run.json") ||
        path_join(usage_path, argv[3], "no-warm.usage")) {
        fprintf(stderr, "output path is too long\n");
        return 2;
    }
    cfg.usage_path = usage_path;
    if (setenv("WASTE_DUMP_ROUTE", routes_path, 1)) {
        fprintf(stderr, "cannot set route trace path: %s\n", strerror(errno));
        return 1;
    }

    waste_cuda_backend_config cuda_cfg;
    memset(&cuda_cfg, 0, sizeof cuda_cfg);
    cuda_cfg.struct_size = sizeof cuda_cfg;
    cuda_cfg.config_version = WASTE_CUDA_BACKEND_CONFIG_VERSION;
    cuda_cfg.device_ordinal = 0;
    cuda_cfg.capabilities = mode_capabilities(mode);
    cuda_cfg.q4_mode = WASTE_CUDA_Q4_FAST;
    backend_counts backend;
    memset(&backend, 0, sizeof backend);
    waste_backend_model_info topology;
    memset(&topology, 0, sizeof topology);
    counting_config wrapper_cfg = {
        .provider = cuda_cfg,
        .counts = &backend,
        .topology = &topology,
        .mode = mode,
    };

    waste_ctx *ctx = NULL;
    const double load_begin = monotonic_ms();
    waste_status status = waste_open_with_backend(
        argv[1], &cfg, &counting_provider, &wrapper_cfg, &ctx);
    const double load_ms = monotonic_ms() - load_begin;
    if (status != WASTE_OK) {
        fprintf(stderr, "open (%s) failed: %s\n", mode_string(mode),
                waste_strerror(status));
        return 1;
    }

    waste_model_info model;
    if (waste_model_get_info(ctx, &model) != WASTE_OK) {
        fprintf(stderr, "model introspection failed\n");
        waste_close(ctx);
        return 1;
    }
    char arch[256], quant[256];
    snprintf(arch, sizeof arch, "%s", model.arch ? model.arch : "");
    snprintf(quant, sizeof quant, "%s",
             model.quant_summary ? model.quant_summary : "");
    model.arch = arch;
    model.quant_summary = quant;
    if (topology.struct_size < sizeof topology ||
        topology.n_layers != model.n_layers ||
        topology.n_experts != model.n_experts ||
        topology.top_k != model.top_k || topology.first_dense > model.n_layers) {
        fprintf(stderr, "backend topology introspection failed\n");
        waste_close(ctx);
        return 1;
    }
    waste_memplan memory;
    if (waste_memory_used(ctx, &memory) != WASTE_OK) {
        fprintf(stderr, "resolved-memory introspection failed\n");
        waste_close(ctx);
        return 1;
    }

    FILE *logits_file = fopen(logits_path, "wb");
    FILE *steps_file = fopen(steps_path, "wb");
    if (!logits_file || !steps_file) {
        fprintf(stderr, "cannot create capture files: %s\n", strerror(errno));
        if (logits_file) fclose(logits_file);
        if (steps_file) fclose(steps_file);
        waste_close(ctx);
        return 1;
    }
    fputs("step\tphase\tinput_token\toutput_token\telapsed_ms\n", steps_file);

    int32_t *generated = (int32_t *)calloc(n_gen, sizeof *generated);
    if (!generated) {
        fprintf(stderr, "allocation failed\n");
        fclose(logits_file);
        fclose(steps_file);
        waste_close(ctx);
        return 1;
    }
    const float *logits = NULL;
    size_t vocab = 0;
    const double prefill_begin = monotonic_ms();
    status = waste_eval(ctx, prompt, n_prompt, &logits, &vocab);
    const double prefill_ms = monotonic_ms() - prefill_begin;
    if (status != WASTE_OK || !logits || !vocab ||
        write_logits(logits_file, logits, vocab)) {
        fprintf(stderr, "prefill failed: %s%s%s\n", waste_strerror(status),
                waste_error_detail(ctx) ? ": " : "",
                waste_error_detail(ctx) ? waste_error_detail(ctx) : "");
        free(generated);
        fclose(logits_file);
        fclose(steps_file);
        waste_close(ctx);
        return 1;
    }
    generated[0] = argmax(logits, vocab);
    const backend_counts after_prefill = backend;
    fprintf(steps_file, "0\tprefill\t%d\t%d\t%.6f\n",
            prompt[n_prompt - 1], generated[0], prefill_ms);

    double decode_ms = 0.0;
    for (size_t step = 1; step < n_gen; step++) {
        const double begin = monotonic_ms();
        const int32_t input = generated[step - 1];
        size_t step_vocab = 0;
        status = waste_eval(ctx, &input, 1, &logits, &step_vocab);
        const double elapsed = monotonic_ms() - begin;
        decode_ms += elapsed;
        if (status != WASTE_OK || !logits || step_vocab != vocab ||
            write_logits(logits_file, logits, vocab)) {
            fprintf(stderr, "decode step %llu failed: %s%s%s\n",
                    (unsigned long long)step, waste_strerror(status),
                    waste_error_detail(ctx) ? ": " : "",
                    waste_error_detail(ctx) ? waste_error_detail(ctx) : "");
            free(generated);
            fclose(logits_file);
            fclose(steps_file);
            waste_close(ctx);
            return 1;
        }
        generated[step] = argmax(logits, vocab);
        fprintf(steps_file, "%llu\tdecode\t%d\t%d\t%.6f\n",
                (unsigned long long)step, input, generated[step], elapsed);
    }

    waste_stats stats;
    memset(&stats, 0, sizeof stats);
    status = waste_get_stats(ctx, &stats);
    const int logits_close_error = fclose(logits_file);
    const int steps_close_error = fclose(steps_file);
    if (logits_close_error || steps_close_error || status != WASTE_OK ||
        write_manifest(manifest_path, mode, &cfg, &model, &topology, &memory,
                       &backend,
                       &after_prefill,
                       prompt, n_prompt, generated, n_gen, vocab,
                       load_ms, prefill_ms, decode_ms, &stats)) {
        fprintf(stderr, "cannot finalize capture\n");
        free(generated);
        waste_close(ctx);
        return 1;
    }

    printf("%s: %llu generated tokens, %.3f decode tok/s, "
           "%.3f ms prefill -> %s\n",
           mode_string(mode), (unsigned long long)n_gen,
           (double)(n_gen - 1) * 1000.0 / decode_ms, prefill_ms, argv[3]);
    free(generated);
    waste_close(ctx);
    return 0;
}
