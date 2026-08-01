/* SPDX-License-Identifier: Apache-2.0 */
#define _GNU_SOURCE
#include "../src/io_uring.h"

#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main(int argc, char **argv)
{
    if (argc != 2) return 2;
    waste_io_ring ring;
#if !defined(__linux__)
    (void)argv;
    if (waste_io_ring_init(&ring, 4) == 0) return 1;
    puts("PASS io_uring unsupported fallback");
    return 0;
#else
    char path[1024];
    snprintf(path, sizeof path, "%s/io-ring.bin", argv[1]);
    int fd = open(path, O_CREAT | O_TRUNC | O_RDWR, 0600);
    if (fd < 0) return 1;
    uint8_t page[4096];
    for (int p = 0; p < 5; p++) {
        memset(page, 17 + p, sizeof page);
        if (write(fd, page, sizeof page) != sizeof page) return 1;
    }
    if (waste_io_ring_init(&ring, 4)) return 1;
    int fds[6] = { fd, fd, fd, fd, fd, fd };
    void *dst[6];
    size_t len[6] = { 4096, 4096, 4096, 4096, 4096, 4096 };
    int64_t off[6] = { 12288, 0, 16384, 8192, 4096, 20480 }, got[6];
    for (int i = 0; i < 6; i++) {
        if (posix_memalign(&dst[i], 4096, 4096)) return 1;
        memset(dst[i], 0, 4096);
    }
    /* Six requests on a depth-four ring exercise the transport's own
     * batching. The last positional read begins at EOF: collecting that
     * zero-byte CQE is transport success, while the caller remains able to
     * reject it as a short record. */
    if (waste_io_ring_read_many(&ring, 6, fds, dst, len, off, got)) return 1;
    const int expect[5] = { 20, 17, 21, 19, 18 };
    for (int i = 0; i < 5; i++) {
        if (got[i] != 4096) return 1;
        for (int j = 0; j < 4096; j++)
            if (((uint8_t *)dst[i])[j] != expect[i]) return 1;
    }
    if (got[5] != 0) return 1;
    for (int i = 0; i < 6; i++) free(dst[i]);
    waste_io_ring_free(&ring);
    close(fd);
    puts("PASS io_uring qd4 positional reads");
    return 0;
#endif
}
