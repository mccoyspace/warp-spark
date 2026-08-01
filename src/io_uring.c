/* SPDX-License-Identifier: Apache-2.0 */
#define _GNU_SOURCE
#include "io_uring.h"

#include <errno.h>
#include <stdatomic.h>
#include <string.h>

#if defined(__linux__)

#include <linux/io_uring.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

static int uring_setup(unsigned entries, struct io_uring_params *p)
{
    return (int)syscall(__NR_io_uring_setup, entries, p);
}

static int uring_enter(int fd, unsigned submit, unsigned complete,
                       unsigned flags)
{
    return (int)syscall(__NR_io_uring_enter, fd, submit, complete, flags,
                        NULL, 0);
}

static unsigned load_acquire(const unsigned *p)
{
    return atomic_load_explicit((const _Atomic unsigned *)p,
                                memory_order_acquire);
}

static unsigned load_relaxed(const unsigned *p)
{
    return atomic_load_explicit((const _Atomic unsigned *)p,
                                memory_order_relaxed);
}

static void store_release(unsigned *p, unsigned v)
{
    atomic_store_explicit((_Atomic unsigned *)p, v, memory_order_release);
}

void waste_io_ring_free(waste_io_ring *r)
{
    if (!r) return;
    if (r->sqes && r->sqes != MAP_FAILED) munmap(r->sqes, r->sqes_size);
    if (r->single_mmap) {
        if (r->sq_ring && r->sq_ring != MAP_FAILED)
            munmap(r->sq_ring, r->sq_ring_size);
    } else {
        if (r->sq_ring && r->sq_ring != MAP_FAILED)
            munmap(r->sq_ring, r->sq_ring_size);
        if (r->cq_ring && r->cq_ring != MAP_FAILED)
            munmap(r->cq_ring, r->cq_ring_size);
    }
    if (r->fd >= 0) close(r->fd);
    memset(r, 0, sizeof *r);
    r->fd = -1;
}

int waste_io_ring_init(waste_io_ring *r, int depth)
{
    if (!r || depth < 2 || depth > 64) return -1;
    memset(r, 0, sizeof *r);
    r->fd = -1;

    struct io_uring_params p;
    memset(&p, 0, sizeof p);
    const int fd = uring_setup((unsigned)depth, &p);
    if (fd < 0) return -1;
    r->fd = fd;
    r->depth = depth < (int)p.sq_entries ? depth : (int)p.sq_entries;
    r->sq_ring_size = p.sq_off.array + p.sq_entries * sizeof(unsigned);
    r->cq_ring_size = p.cq_off.cqes +
                      p.cq_entries * sizeof(struct io_uring_cqe);
    r->single_mmap = (p.features & IORING_FEAT_SINGLE_MMAP) != 0;
    if (r->single_mmap && r->cq_ring_size > r->sq_ring_size)
        r->sq_ring_size = r->cq_ring_size;

    r->sq_ring = mmap(NULL, r->sq_ring_size, PROT_READ | PROT_WRITE,
                      MAP_SHARED | MAP_POPULATE, fd, IORING_OFF_SQ_RING);
    if (r->sq_ring == MAP_FAILED) { waste_io_ring_free(r); return -1; }
    if (r->single_mmap) {
        r->cq_ring = r->sq_ring;
        r->cq_ring_size = r->sq_ring_size;
    } else {
        r->cq_ring = mmap(NULL, r->cq_ring_size, PROT_READ | PROT_WRITE,
                          MAP_SHARED | MAP_POPULATE, fd, IORING_OFF_CQ_RING);
        if (r->cq_ring == MAP_FAILED) { waste_io_ring_free(r); return -1; }
    }
    r->sqes_size = p.sq_entries * sizeof(struct io_uring_sqe);
    r->sqes = mmap(NULL, r->sqes_size, PROT_READ | PROT_WRITE,
                   MAP_SHARED | MAP_POPULATE, fd, IORING_OFF_SQES);
    if (r->sqes == MAP_FAILED) { waste_io_ring_free(r); return -1; }

    char *sq = (char *)r->sq_ring, *cq = (char *)r->cq_ring;
    r->sq_head = (unsigned *)(sq + p.sq_off.head);
    r->sq_tail = (unsigned *)(sq + p.sq_off.tail);
    r->sq_mask = (unsigned *)(sq + p.sq_off.ring_mask);
    r->sq_entries = (unsigned *)(sq + p.sq_off.ring_entries);
    r->sq_array = (unsigned *)(sq + p.sq_off.array);
    r->cq_head = (unsigned *)(cq + p.cq_off.head);
    r->cq_tail = (unsigned *)(cq + p.cq_off.tail);
    r->cq_mask = (unsigned *)(cq + p.cq_off.ring_mask);
    r->cq_entries = (unsigned *)(cq + p.cq_off.ring_entries);
    r->cqes = cq + p.cq_off.cqes;
    return 0;
}

int waste_io_ring_active(const waste_io_ring *r)
{
    return r && r->fd >= 0 && r->depth >= 2;
}

static int read_batch(waste_io_ring *r, int n, const int *fds,
                      void *const *dst, const size_t *len,
                      const int64_t *off, int64_t *results)
{
    const unsigned head = load_acquire(r->sq_head);
    unsigned tail = load_relaxed(r->sq_tail);
    if ((unsigned)n > *r->sq_entries - (tail - head)) return -1;
    struct io_uring_sqe *sqes = (struct io_uring_sqe *)r->sqes;
    for (int i = 0; i < n; i++) {
        const unsigned slot = tail & *r->sq_mask;
        struct io_uring_sqe *sqe = &sqes[slot];
        memset(sqe, 0, sizeof *sqe);
        sqe->opcode = IORING_OP_READ;
        sqe->fd = fds[i];
        sqe->off = (uint64_t)off[i];
        sqe->addr = (uint64_t)(uintptr_t)dst[i];
        sqe->len = (uint32_t)len[i];
        sqe->user_data = (uint64_t)(unsigned)i;
        r->sq_array[slot] = slot;
        tail++;
        results[i] = -EIO;
    }
    store_release(r->sq_tail, tail);

    unsigned left = (unsigned)n;
    while (left) {
        int rc;
        do { rc = uring_enter(r->fd, left, 0, 0); }
        while (rc < 0 && errno == EINTR);
        if (rc < 0) return -1;
        if (rc == 0) break;
        left -= (unsigned)rc;
    }
    if (left) return -1;

    int done = 0;
    while (done < n) {
        unsigned ch = load_relaxed(r->cq_head);
        unsigned ct = load_acquire(r->cq_tail);
        if (ch == ct) {
            int rc;
            do { rc = uring_enter(r->fd, 0, 1, IORING_ENTER_GETEVENTS); }
            while (rc < 0 && errno == EINTR);
            if (rc < 0)
                return -1;
            continue;
        }
        while (ch != ct && done < n) {
            const struct io_uring_cqe *cqe =
                &((const struct io_uring_cqe *)r->cqes)[ch & *r->cq_mask];
            if (cqe->user_data < (uint64_t)(unsigned)n)
                results[cqe->user_data] = cqe->res;
            ch++;
            done++;
        }
        store_release(r->cq_head, ch);
    }
    return 0;
}

int waste_io_ring_read_many(waste_io_ring *r, int n, const int *fds,
                            void *const *dst, const size_t *len,
                            const int64_t *off, int64_t *results)
{
    if (!waste_io_ring_active(r) || n < 0 || !fds || !dst || !len ||
        !off || !results) return -1;
    for (int base = 0; base < n; base += r->depth) {
        const int take = n - base < r->depth ? n - base : r->depth;
        if (read_batch(r, take, fds + base, dst + base, len + base,
                       off + base, results + base)) return -1;
    }
    return 0;
}

#else

int waste_io_ring_init(waste_io_ring *r, int depth)
{
    if (r) { memset(r, 0, sizeof *r); r->fd = -1; }
    (void)depth;
    return -1;
}
void waste_io_ring_free(waste_io_ring *r)
{
    if (r) { memset(r, 0, sizeof *r); r->fd = -1; }
}
int waste_io_ring_active(const waste_io_ring *r) { (void)r; return 0; }
int waste_io_ring_read_many(waste_io_ring *r, int n, const int *fds,
                            void *const *dst, const size_t *len,
                            const int64_t *off, int64_t *results)
{
    (void)r; (void)n; (void)fds; (void)dst; (void)len; (void)off; (void)results;
    return -1;
}

#endif
