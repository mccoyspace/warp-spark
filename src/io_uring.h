/* SPDX-License-Identifier: Apache-2.0 */
#ifndef WASTE_IO_URING_H
#define WASTE_IO_URING_H

#include <stddef.h>
#include <stdint.h>

/* A deliberately small raw-syscall wrapper.  WASTE remains free of a
 * liburing runtime dependency, and non-Linux builds compile the same API as
 * an unavailable backend.  One model owns one ring and submits from its
 * serialized inference thread. */
typedef struct {
    int fd, depth;
    void *sq_ring, *cq_ring, *sqes;
    size_t sq_ring_size, cq_ring_size, sqes_size;
    unsigned *sq_head, *sq_tail, *sq_mask, *sq_entries, *sq_array;
    unsigned *cq_head, *cq_tail, *cq_mask, *cq_entries;
    void *cqes;
    int single_mmap;
} waste_io_ring;

int  waste_io_ring_init(waste_io_ring *ring, int depth);
void waste_io_ring_free(waste_io_ring *ring);
int  waste_io_ring_active(const waste_io_ring *ring);

/* Submit positional reads in batches no deeper than ring->depth, wait for
 * every completion, and put each CQE result in results[i].  Returns 0 when
 * all CQEs were collected; individual short/error reads remain visible in
 * results rather than being hidden by this transport layer. */
int waste_io_ring_read_many(waste_io_ring *ring, int n, const int *fds,
                            void *const *dst, const size_t *len,
                            const int64_t *off, int64_t *results);

#endif
