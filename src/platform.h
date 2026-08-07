/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/*
 * platform.h — the calls that are not POSIX everywhere.
 *
 * The engine is POSIX apart from seven things, and they are all here
 * rather than spread through the sources as #ifdefs: a positional read, an
 * aligned allocation, the CPU count, a file's size, the physical RAM,
 * opening a file with the page cache out of the way, and binding a thread
 * to a set of CPUs. Each has a Windows implementation and a one-line POSIX
 * one, so every call site reads the same on all three targets. (Physical
 * RAM is the exception that proves it: macOS answers with sysctlbyname and
 * Linux with sysconf, so that one branch stays in waste.c and only the
 * Windows half lives here.)
 *
 * Affinity is the exception in the other direction: it is not a Windows
 * hole in POSIX but a thing POSIX never specified, and macOS has no
 * equivalent at all — its thread affinity tags are a hint on Intel and do
 * nothing on Apple silicon. So waste_bind_thread_cpus is allowed to fail,
 * WASTE_HAVE_AFFINITY says so before anything is attempted, and a caller
 * that asked for a cpuset is refused rather than left believing it got
 * one. See waste_cfg.cpu_list.
 *
 * Two things this fixes that are not "missing functions":
 *
 *   - `long` is 32 bits on Windows (LLP64), so every file offset here is
 *     int64_t. A container is 17 GB at the small end and ~900 GB for K3;
 *     an offset that silently truncates at 2 GB would build fine and read
 *     the wrong expert.
 *   - `_aligned_malloc` needs `_aligned_free`. Passing that pointer to
 *     free() is heap corruption on Windows, not a leak, so the allocation
 *     and its release are a pair and neither is used directly.
 */

#ifndef WASTE_PLATFORM_H
#define WASTE_PLATFORM_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Whether this platform can bind a thread to a set of CPUs at all. Tested
 * before anything is attempted, so a caller who asked for a cpuset is
 * refused at the front door rather than at the far end of a load. */
#if defined(_WIN32) || defined(__linux__)
#define WASTE_HAVE_AFFINITY 1
#else
#define WASTE_HAVE_AFFINITY 0
#endif

/* ---- CPU sets ----------------------------------------------------------
 *
 * A bitmap, one bit per CPU, in the shape Linux wants it: an array of
 * unsigned long. 1024 is what glibc's cpu_set_t holds and more CPUs than
 * the pool can use (it caps at 64 threads), so the bound costs nothing and
 * keeps a parsed cpu list from indexing off the end. */
#define WASTE_MAX_CPUS 1024
#define WASTE_CPU_BITS ((int)(8 * sizeof(unsigned long)))

typedef struct {
    unsigned long w[WASTE_MAX_CPUS / (8 * sizeof(unsigned long))];
} waste_cpumask;

static inline void waste_cpumask_zero(waste_cpumask *m) { memset(m, 0, sizeof *m); }

static inline void waste_cpumask_set(waste_cpumask *m, int cpu)
{
    if (cpu >= 0 && cpu < WASTE_MAX_CPUS)
        m->w[cpu / WASTE_CPU_BITS] |= 1UL << (cpu % WASTE_CPU_BITS);
}

static inline int waste_cpumask_test(const waste_cpumask *m, int cpu)
{
    if (cpu < 0 || cpu >= WASTE_MAX_CPUS) return 0;
    return (int)((m->w[cpu / WASTE_CPU_BITS] >> (cpu % WASTE_CPU_BITS)) & 1UL);
}

static inline int waste_cpumask_count(const waste_cpumask *m)
{
    int n = 0;
    for (int i = 0; i < WASTE_MAX_CPUS; i++) n += waste_cpumask_test(m, i);
    return n;
}

/* ---- cpu lists ---------------------------------------------------------- */

enum {
    WASTE_CPUS_NONE = 0,       /* nobody asked; placement stays the OS's    */
    WASTE_CPUS_OK = 1,
    WASTE_CPUS_BAD = -1,       /* not a cpu list, or names no CPU           */
    WASTE_CPUS_UNSUPPORTED = -2 /* a list, on a platform that cannot bind   */
};

/* "0-5", "0-2,6-8", "3" -> a mask. Returns how many CPUs were named, or
 * WASTE_CPUS_BAD.
 *
 * Strict on purpose. "5-0" and "0-5," and "0,,5" are typos, and a typo
 * that parses to a smaller set than the user meant is the one failure this
 * option cannot afford: it would look exactly like the option not helping.
 * Indices are bounded by WASTE_MAX_CPUS here; whether the machine has that
 * CPU is the OS's answer to give, at bind time, since a cgroup or a
 * container can make the online count and the legal indices disagree. */
static inline int waste_cpuset_parse(const char *s, waste_cpumask *out)
{
    waste_cpumask_zero(out);
    if (!s) return WASTE_CPUS_BAD;
    for (;;) {
        while (*s == ' ' || *s == '\t') s++;
        if (*s < '0' || *s > '9') return WASTE_CPUS_BAD;
        long a = 0, b;
        while (*s >= '0' && *s <= '9') {
            a = a * 10 + (*s++ - '0');
            if (a >= WASTE_MAX_CPUS) return WASTE_CPUS_BAD;
        }
        b = a;
        /* Whitespace separates tokens and never joins them: "0 - 2" is the
         * range a shell user typed with spaces, "0 3" is a missing comma
         * and stays an error. */
        while (*s == ' ' || *s == '\t') s++;
        if (*s == '-') {
            s++;
            while (*s == ' ' || *s == '\t') s++;
            if (*s < '0' || *s > '9') return WASTE_CPUS_BAD;
            b = 0;
            while (*s >= '0' && *s <= '9') {
                b = b * 10 + (*s++ - '0');
                if (b >= WASTE_MAX_CPUS) return WASTE_CPUS_BAD;
            }
            if (b < a) return WASTE_CPUS_BAD;
        }
        for (long i = a; i <= b; i++) waste_cpumask_set(out, (int)i);
        while (*s == ' ' || *s == '\t') s++;
        if (!*s) break;
        if (*s != ',') return WASTE_CPUS_BAD;
        s++;
    }
    const int n = waste_cpumask_count(out);
    return n > 0 ? n : WASTE_CPUS_BAD;
}

/* What the caller asked for, or WASTE_CPUS in the environment when the
 * caller said nothing — the same rule as WASTE_THREADS, an escape hatch
 * for when there is no caller to ask rather than an override.
 *
 * Returns one of the WASTE_CPUS_* codes. Both the validating call in
 * waste_open and the applying call in the loader go through here, so a
 * list that is refused in one is refused in the other. */
static inline int waste_cpus_resolve(const char *list, waste_cpumask *out)
{
    if (!list || !*list) list = getenv("WASTE_CPUS");
    if (!list || !*list) return WASTE_CPUS_NONE;
    if (waste_cpuset_parse(list, out) < 0) return WASTE_CPUS_BAD;
    return WASTE_HAVE_AFFINITY ? WASTE_CPUS_OK : WASTE_CPUS_UNSUPPORTED;
}

#ifdef _WIN32

/* GetActiveProcessorCount and ALL_PROCESSOR_GROUPS are Windows 7. */
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0601
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <fcntl.h>
#include <io.h>
#include <malloc.h>
#include <string.h>

/* Windows opens in text mode unless told otherwise, which turns 0x0D 0x0A
 * into 0x0A in the middle of weights. Every open of a container file ORs
 * this in; on POSIX it is 0. */
#define WASTE_O_BINARY _O_BINARY

/* One positional read. ReadFile with an OVERLAPPED offset does not touch
 * the shared file pointer, which is what pread() is for: the expert cache
 * reads records by offset and nothing wants a seek in between.
 *
 * Reads are capped at 1 GiB because the count is a DWORD. Callers loop
 * (see pread_all) and a short read is legal, so the cap is invisible. */
static inline int64_t waste_pread(int fd, void *dst, size_t n, int64_t off)
{
    const HANDLE h = (HANDLE)_get_osfhandle(fd);
    if (h == INVALID_HANDLE_VALUE || off < 0) return -1;
    if (n > (size_t)1 << 30) n = (size_t)1 << 30;

    OVERLAPPED ov;
    memset(&ov, 0, sizeof ov);
    ov.Offset     = (DWORD)((uint64_t)off & 0xFFFFFFFFu);
    ov.OffsetHigh = (DWORD)((uint64_t)off >> 32);

    DWORD got = 0;
    if (!ReadFile(h, dst, (DWORD)n, &got, &ov)) {
        /* Reading at or past the end is not an error anywhere else. */
        return GetLastError() == ERROR_HANDLE_EOF ? 0 : -1;
    }
    return (int64_t)got;
}

static inline void *waste_aligned_alloc(size_t align, size_t n)
{
    return _aligned_malloc(n, align);
}

static inline void waste_aligned_free(void *p)
{
    _aligned_free(p);            /* NOT free() — see the header comment */
}

/* GetSystemInfo would answer for the current processor group only, so it
 * says 64 on a machine with more. The pool caps at 64 anyway, but the
 * number is also reported to the user. */
static inline int waste_cpu_count(void)
{
    const DWORD n = GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
    if (n > 0) return (int)n;
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return si.dwNumberOfProcessors > 0 ? (int)si.dwNumberOfProcessors : 1;
}

static inline int64_t waste_file_size(int fd)
{
    const HANDLE h = (HANDLE)_get_osfhandle(fd);
    LARGE_INTEGER sz;
    if (h == INVALID_HANDLE_VALUE || !GetFileSizeEx(h, &sz)) return -1;
    return (int64_t)sz.QuadPart;
}

/* Restrict the calling thread to `m`. Returns 0, or -1 if the set cannot
 * be expressed or the OS refused it.
 *
 * A thread's affinity mask is per processor group here, so a CPU past the
 * group's width — 64 on a 64-bit build — is not a mask this call can send.
 * Refusing is the point: the alternative is binding to the low half of
 * what was asked for and reporting success. */
static inline int waste_bind_thread_cpus(const waste_cpumask *m)
{
    const int width = (int)(8 * sizeof(DWORD_PTR));
    DWORD_PTR mask = 0;
    for (int i = 0; i < WASTE_MAX_CPUS; i++) {
        if (!waste_cpumask_test(m, i)) continue;
        if (i >= width) return -1;
        mask |= (DWORD_PTR)1 << i;
    }
    if (!mask) return -1;
    return SetThreadAffinityMask(GetCurrentThread(), mask) ? 0 : -1;
}

static inline uint64_t waste_physical_ram_bytes(void)
{
    MEMORYSTATUSEX st;
    st.dwLength = sizeof st;
    return GlobalMemoryStatusEx(&st) ? (uint64_t)st.ullTotalPhys : 0;
}

static inline int waste_sync_file(FILE *f)
{
    return fflush(f) || _commit(_fileno(f));
}

static inline int waste_replace_file(const char *src, const char *dst)
{
    return MoveFileExA(src, dst, MOVEFILE_REPLACE_EXISTING |
                       MOVEFILE_WRITE_THROUGH) ? 0 : -1;
}

/* The Windows half of the page-cache bypass, which is the part of this
 * port that actually matters: the expert-streaming argument is that the
 * hit rate we measure is ours and not the kernel's.
 *
 * FILE_FLAG_NO_BUFFERING is the O_DIRECT of this platform and carries the
 * same contract — offset, length and destination address must all be
 * multiples of the volume's sector size. Records are whole 4 KiB pages and
 * buffers come from waste_aligned_alloc at 16 KiB, so both hold; the
 * caller checks the record size anyway rather than assuming, because a
 * misaligned read fails outright instead of merely running slow.
 *
 * Without the bypass, FILE_FLAG_RANDOM_ACCESS at least stops Windows
 * reading ahead into pages nothing will ask for — the same fallback as
 * posix_fadvise(POSIX_FADV_RANDOM) on Linux.
 *
 * The handle is adopted by the returned descriptor, so close(fd) closes
 * it and the rest of the engine needs no Windows branch. */
static inline int waste_open_stream(const char *path, int bypass)
{
    const DWORD flags = bypass ? FILE_FLAG_NO_BUFFERING | FILE_FLAG_RANDOM_ACCESS
                               : FILE_FLAG_RANDOM_ACCESS;
    const HANDLE h = CreateFileA(path, GENERIC_READ,
                                 FILE_SHARE_READ | FILE_SHARE_WRITE, NULL,
                                 OPEN_EXISTING, flags, NULL);
    if (h == INVALID_HANDLE_VALUE) return -1;
    const int fd = _open_osfhandle((intptr_t)h, _O_RDONLY | _O_BINARY);
    if (fd < 0) { CloseHandle(h); return -1; }
    return fd;
}

#else /* POSIX */

#include <stdlib.h>
#include <unistd.h>

#define WASTE_O_BINARY 0

static inline int64_t waste_pread(int fd, void *dst, size_t n, int64_t off)
{
    return (int64_t)pread(fd, dst, n, (off_t)off);
}

static inline void *waste_aligned_alloc(size_t align, size_t n)
{
    void *p = NULL;
    return posix_memalign(&p, align, n) == 0 ? p : NULL;
}

static inline void waste_aligned_free(void *p) { free(p); }

static inline int waste_cpu_count(void)
{
    const long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 1 ? (int)n : 1;
}

static inline int64_t waste_file_size(int fd)
{
    const off_t n = lseek(fd, 0, SEEK_END);
    return n < 0 ? -1 : (int64_t)n;
}

#ifdef __linux__
#include <sys/syscall.h>

/* Restrict the calling thread to `m`. Returns 0, or -1 if the OS refused.
 *
 * The syscall rather than pthread_setaffinity_np, and not for fun:
 * the wrapper and the CPU_SET macros are behind _GNU_SOURCE, which has to
 * be defined before the first libc header and so cannot be promised by a
 * header four translation units include in whatever order they like. The
 * syscall takes exactly what the wrapper builds — an array of unsigned
 * long, one bit per CPU — and pid 0 means this thread, not this process,
 * which is the whole point. A length the running kernel does not use is
 * its business: it truncates a longer mask and zero-fills a shorter one. */
static inline int waste_bind_thread_cpus(const waste_cpumask *m)
{
    return syscall(SYS_sched_setaffinity, 0, sizeof m->w, m->w) == 0 ? 0 : -1;
}

#else

static inline int waste_bind_thread_cpus(const waste_cpumask *m)
{
    (void)m;
    return -1;                       /* WASTE_HAVE_AFFINITY is 0 here */
}

#endif

static inline int waste_sync_file(FILE *f)
{
    return fflush(f) || fsync(fileno(f));
}

static inline int waste_replace_file(const char *src, const char *dst)
{
    return rename(src, dst);
}

#endif /* _WIN32 */

#endif /* WASTE_PLATFORM_H */
