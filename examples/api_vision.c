/* Kimi K3 image-and-text generation through the public C API. */
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

static waste_status append(waste_ctx *ctx, const char *text, int markup,
                           int add_bos, int32_t *tokens, size_t cap, size_t *n)
{
    size_t added = 0;
    waste_status st;
    if (markup)
        st = waste_tokenize_markup(ctx, text, add_bos, tokens + *n, cap - *n,
                                   &added);
    else
        st = waste_tokenize(ctx, text, add_bos, tokens + *n, cap - *n,
                            &added);
    if (st == WASTE_OK) *n += added;
    return st;
}

int main(int argc, char **argv)
{
    if (argc != 4) {
        fprintf(stderr, "usage: %s K3_MODEL IMAGE PROMPT\n", argv[0]);
        return 2;
    }

    int width = 0, height = 0;
    waste_status st = waste_image_dimensions(argv[2], &width, &height);
    if (st != WASTE_OK) return report("image dimensions", st, NULL);

    waste_cfg cfg;
    waste_cfg_init(&cfg);
    cfg.ctx_tokens = CONTEXT_TOKENS;
    cfg.vision = 1;

    waste_ctx *ctx = NULL;
    st = waste_open(argv[1], &cfg, &ctx);
    if (st != WASTE_OK) return report("open", st, ctx);

    size_t image_tokens = 0;
    st = waste_image_add(ctx, argv[2], &image_tokens);
    if (st != WASTE_OK) {
        const int rc = report("image", st, ctx);
        waste_close(ctx);
        return rc;
    }
    fprintf(stderr, "%s: %zu image tokens\n", argv[2], image_tokens);

    char media[160];
    snprintf(media, sizeof(media),
             "<|media_begin|>image %dx%d<|media_content|>"
             "<|media_pad|><|media_end|>", width, height);

    int32_t *raw = malloc(CONTEXT_TOKENS * sizeof(*raw));
    int32_t *expanded = malloc(CONTEXT_TOKENS * sizeof(*expanded));
    if (!raw || !expanded) {
        free(raw);
        free(expanded);
        waste_close(ctx);
        return report("tokens", WASTE_E_OOM, NULL);
    }

    /* Markup and user content are tokenized separately so user text cannot
     * forge K3 control tokens. */
    size_t n_raw = 0;
    st = append(ctx, "<|open|>message role=\"user\"<|sep|>", 1, 1,
                raw, CONTEXT_TOKENS, &n_raw);
    if (st == WASTE_OK)
        st = append(ctx, media, 1, 0, raw, CONTEXT_TOKENS, &n_raw);
    if (st == WASTE_OK)
        st = append(ctx, argv[3], 0, 0, raw, CONTEXT_TOKENS, &n_raw);
    if (st == WASTE_OK)
        st = append(ctx,
                    "<|close|>message<|sep|><|end_of_msg|>"
                    "<|open|>message role=\"assistant\"<|sep|>"
                    "<|open|>response<|sep|>",
                    1, 0, raw, CONTEXT_TOKENS, &n_raw);
    if (st != WASTE_OK) {
        free(raw);
        free(expanded);
        const int rc = report("tokenize", st, ctx);
        waste_close(ctx);
        return rc;
    }

    size_t n_expanded = 0;
    st = waste_image_expand(ctx, raw, n_raw, expanded, CONTEXT_TOKENS,
                            &n_expanded);
    free(raw);
    if (st != WASTE_OK) {
        free(expanded);
        const int rc = report("expand image", st, ctx);
        waste_close(ctx);
        return rc;
    }

    waste_gen_params params;
    waste_gen_params_init(&params);
    params.temperature = 0.0f;
    params.max_tokens = 64;

    st = waste_generate(ctx, expanded, n_expanded, &params, print_token, NULL);
    free(expanded);
    if (st != WASTE_OK) {
        const int rc = report("generate", st, ctx);
        waste_close(ctx);
        return rc;
    }

    fputc('\n', stdout);
    waste_close(ctx);
    return 0;
}
