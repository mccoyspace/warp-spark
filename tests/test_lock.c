/* SPDX-License-Identifier: Apache-2.0 */
#include "../src/waste.h"

#include <stdio.h>
#include <stdlib.h>

#if !defined(_WIN32)
#include <sys/wait.h>
#include <unistd.h>

static int cross_process(const char *model, int allow)
{
    int p[2];
    if (pipe(p)) return 1;
    const pid_t child = fork();
    if (child < 0) return 1;
    if (child == 0) {
        close(p[1]);
        char go = 0;
        if (read(p[0], &go, 1) != 1) _exit(2);
        waste_cfg cfg; waste_cfg_init(&cfg);
        cfg.allow_concurrent_open = allow;
        waste_ctx *c = NULL;
        const waste_status st = waste_open(model, &cfg, &c);
        if (c) waste_close(c);
        _exit(allow ? (st == WASTE_OK ? 0 : 3)
                    : (st == WASTE_E_BUSY ? 0 : 4));
    }
    close(p[0]);
    waste_cfg cfg; waste_cfg_init(&cfg);
    waste_ctx *c = NULL;
    const waste_status st = waste_open(model, &cfg, &c);
    if (st != WASTE_OK) return 1;
    if (write(p[1], "x", 1) != 1) return 1;
    close(p[1]);
    int ws = 0;
    waitpid(child, &ws, 0);
    waste_close(c);
    return !WIFEXITED(ws) || WEXITSTATUS(ws) != 0;
}
#endif

int main(int argc, char **argv)
{
    if (argc != 2) return 2;
    waste_cfg cfg; waste_cfg_init(&cfg);
    waste_ctx *a = NULL, *b = NULL;
    if (waste_open(argv[1], &cfg, &a) != WASTE_OK) return 1;
    if (waste_open(argv[1], &cfg, &b) != WASTE_OK) return 1;
    waste_close(b); waste_close(a);
#if !defined(_WIN32)
    if (cross_process(argv[1], 0)) return 1;
    if (cross_process(argv[1], 1)) return 1;
#endif
    puts("PASS lock");
    return 0;
}
