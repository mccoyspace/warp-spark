/* Minimal raw text generation through the public C API. */
#include "waste.h"

#include <stdio.h>
#include <stdlib.h>

#define CONTEXT_TOKENS 4096

static int print_token(const waste_token_info *info, const char *piece,
                       void *user)
{
    (void)info;
    (void)user;
    fputs(piece, stdout);
    fflush(stdout);
    return 0;
}

static int report(const char *where, waste_status st, waste_ctx *ctx)
{
    fprintf(stderr, "%s: %s", where, waste_strerror(st));
    if (ctx && waste_error_detail(ctx))
        fprintf(stderr, " (%s)", waste_error_detail(ctx));
    fputc('\n', stderr);
    return 1;
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "usage: %s MODEL PROMPT\n", argv[0]);
        return 2;
    }

    waste_cfg cfg;
    waste_cfg_init(&cfg);
    cfg.ctx_tokens = CONTEXT_TOKENS;

    waste_ctx *ctx = NULL;
    waste_status st = waste_open(argv[1], &cfg, &ctx);
    if (st != WASTE_OK) return report("open", st, ctx);

    int32_t *tokens = malloc(CONTEXT_TOKENS * sizeof(*tokens));
    if (!tokens) {
        waste_close(ctx);
        return report("tokens", WASTE_E_OOM, NULL);
    }

    size_t n_tokens = 0;
    st = waste_tokenize(ctx, argv[2], 1, tokens, CONTEXT_TOKENS, &n_tokens);
    if (st != WASTE_OK) {
        free(tokens);
        const int rc = report("tokenize", st, ctx);
        waste_close(ctx);
        return rc;
    }

    waste_gen_params params;
    waste_gen_params_init(&params);
    params.temperature = 0.0f;
    params.max_tokens = 64;

    st = waste_generate(ctx, tokens, n_tokens, &params, print_token, NULL);
    free(tokens);
    if (st != WASTE_OK) {
        const int rc = report("generate", st, ctx);
        waste_close(ctx);
        return rc;
    }

    fputc('\n', stdout);
    waste_close(ctx);
    return 0;
}
