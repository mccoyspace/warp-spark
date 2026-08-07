/*
 * diskbench.c — measure the I/O path WASTE actually uses.
 *
 * The engine reads whole experts as ~12 MB records scattered across a huge
 * file, with the page cache bypassed (F_NOCACHE / O_DIRECT), so sequential
 * `dd` numbers are misleading. This measures:
 *   1. sequential write   (how fast the download/conversion can land)
 *   2. sequential read, cache-bypassed
 *   3. random record reads, cache-bypassed  <- the number that sets tok/s
 *   4. the same with N threads (the engine's async pool)
 *
 * Every row says whether the bypass was actually obtained, because on a
 * filesystem that refuses it these are RAM numbers wearing a disk's label.
 *
 * The last argument turns GB/s into tok/s for a model that reads that many
 * GB per token cold. It has no default: the column used to assume K3's 12.5
 * unconditionally, which is ~8x off on a 48B model (1.61 GB/token measured)
 * and says so nowhere. `waste bench` prints the real figure for a container
 * as "disk N GB total, M GB/token"; pass M here, or leave it out and read
 * the GB/s.
 *
 * Build: cc -O2 -o diskbench tools/diskbench.c
 * Usage: ./diskbench /Volumes/WasteDisk/k3/.bench [file_gb] [rec_mb] [threads] [gb_per_token]
 */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>

static double now(void) {
    struct timeval t; gettimeofday(&t, NULL);
    return t.tv_sec + t.tv_usec / 1e6;
}

/* The bypass, one mechanism under three names — the same contract bank_open
 * meets in model.c, and for the same reason: without it the kernel serves the
 * file back out of RAM and this tool reports the page cache. It did exactly
 * that on Linux until now, where O_DIRECT appeared in the comment above and
 * nowhere in the code, so a Gen3 x4 drive benched at 44 GB/s sequential
 * against a 3.9 GB/s link (issue #22 — the same bug LEARNED §14 fixed in the
 * engine, in the one place it was left behind).
 *
 * O_DIRECT wants offset, length and buffer aligned to the device's logical
 * block. All three come free here: buffers are posix_memalign'd to 4096, the
 * record size is masked to a 4096 multiple, and offsets are whole records. */
#define DIO_ALIGN 4096u

/* Cleared by any open that could not get the bypass, including the reader
 * threads'. Every one of them stores the same value and none reads it back,
 * so ordering does not matter — it is atomic to keep the race out of the
 * abstract machine, not because anything here needs to synchronise. */
static _Atomic int g_direct = 1;

static const char *g_path;
static size_t g_rec, g_file;
static int g_reps;

#if defined(__linux__) && defined(O_DIRECT)
#define DIO_FLAG O_DIRECT
/* O_DIRECT and FILE_FLAG_NO_BUFFERING both accept the open and then fail
 * every misaligned transfer, so eligibility is necessary and not sufficient:
 * tmpfs takes the flag and refuses the read, and so would a device wanting a
 * bigger block than we align to. The engine confirms with a transfer rather
 * than with the open; do the same, or a refusing filesystem turns into
 * "short read -1" and a table of zeroes with no cause given. */
static int dio_probe(int fd, int writing) {
    void *buf = NULL;
    if (posix_memalign(&buf, DIO_ALIGN, DIO_ALIGN)) return 0;
    memset(buf, 0, DIO_ALIGN);
    const ssize_t got = writing ? pwrite(fd, buf, DIO_ALIGN, 0)
                                : pread(fd, buf, DIO_ALIGN, 0);
    free(buf);
    if (writing && got == (ssize_t)DIO_ALIGN) ftruncate(fd, 0);  /* undo the probe */
    return got == (ssize_t)DIO_ALIGN;
}
#endif

/* Open the working file with the page cache out of the way, and fall back
 * rather than fail when the filesystem will not have it — a bench that
 * quietly measures something else is worse than one that says it could not
 * (LEARNED §14). Falling back clears g_direct, and the rows say so.
 *
 * The write goes through the same door as the reads, and that is not
 * cosmetic: F_NOCACHE only stops *new* pages being cached, it does not evict
 * what is already resident, so a buffered write leaves the whole file in the
 * UBC and every read row below then measures RAM. Measured on an M5 Pro, 1 GB
 * file: 8.07 GB/s sequential read with the write bypassed, 26.04 GB/s with it
 * buffered. Linux would have survived the same mistake, since an O_DIRECT
 * read writes back and invalidates the range first — which is exactly why it
 * has to be a rule here and not a judgement call per platform. */
static int open_path(int writing) {
    const int rw = writing ? (O_WRONLY | O_CREAT | O_TRUNC) : O_RDONLY;
#if defined(__linux__) && defined(O_DIRECT)
    int dfd = open(g_path, rw | DIO_FLAG, 0644);
    if (dfd >= 0) {
        if (dio_probe(dfd, writing)) return dfd;
        close(dfd);
    }
    g_direct = 0;
#endif
    int fd = open(g_path, rw, 0644);
    if (fd < 0) return fd;
#ifdef __APPLE__
    /* F_NOCACHE has no alignment contract, so this is the whole bypass on
     * macOS and its failure is the difference between the number the row
     * claims and the number it prints. */
    if (fcntl(fd, F_NOCACHE, 1) < 0) g_direct = 0;
    fcntl(fd, F_RDAHEAD, 0);
#elif defined(__linux__)
    /* No bypass to be had: at least stop the kernel reading ahead into pages
     * nothing will ask for, which is what the engine settles for too. */
    posix_fadvise(fd, 0, 0, POSIX_FADV_RANDOM);
#endif
    return fd;
}

static const char *bypass_note(void) {
    return g_direct ? "cache bypassed" : "PAGE CACHE, not the disk";
}

typedef struct { int id, nthreads; double bytes; } targ;

static void *rand_reader(void *p) {
    targ *a = (targ *)p;
    int fd = open_path(0);
    if (fd < 0) { perror("open"); return NULL; }
    void *buf = NULL;
    if (posix_memalign(&buf, DIO_ALIGN, g_rec)) { close(fd); return NULL; }
    size_t nrec = g_file / g_rec;
    unsigned seed = 12345u + a->id * 7919u;
    double got = 0;
    for (int i = 0; i < g_reps; i++) {
        seed = seed * 1103515245u + 12345u;
        off_t off = (off_t)(seed % nrec) * g_rec;
        ssize_t r = pread(fd, buf, g_rec, off);
        if (r != (ssize_t)g_rec) { fprintf(stderr, "short read %zd: %s\n", r, strerror(errno)); break; }
        got += r;
    }
    a->bytes = got;
    free(buf); close(fd);
    return NULL;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s PATH [file_gb] [rec_mb] [threads] [gb_per_token]\n"
                        "  gb_per_token: from `waste bench` on the container you are\n"
                        "  sizing for; omitted, the tok/s column is omitted with it\n", argv[0]);
        return 1;
    }
    g_path = argv[1];
    double file_gb = argc > 2 ? atof(argv[2]) : 8.0;
    double rec_mb  = argc > 3 ? atof(argv[3]) : 12.0;
    int maxthreads = argc > 4 ? atoi(argv[4]) : 8;
    /* No default, deliberately. This was K3's 12.5 hardcoded into the format
     * string, so every run printed a K3 answer whatever the container being
     * sized — off by ~8x on a 48B model, and silent about it. A column that
     * cannot be derived from what was measured does not get printed. */
    double gb_tok = argc > 5 ? atof(argv[5]) : 0.0;
    g_file = (size_t)(file_gb * (1u << 30));
    g_rec  = (size_t)(rec_mb * (1u << 20)) & ~4095UL;
    /* A record under one page rounds to zero and divides by it two screens
     * further down, as a crash rather than as a usage error. */
    if (!g_rec || g_file < g_rec) {
        fprintf(stderr, "record must be >= 4 KiB and file must hold at least one\n");
        return 1;
    }

    printf("file %.1f GB, record %.1f MB, path %s\n", file_gb, rec_mb, g_path);

    /* 1. sequential write */
    int fd = open_path(1);
    if (fd < 0) { perror("open write"); return 1; }
    void *buf;
    if (posix_memalign(&buf, DIO_ALIGN, g_rec)) return 1;
    memset(buf, 0xA5, g_rec);
    double t0 = now(); size_t written = 0;
    while (written < g_file) {
        ssize_t w = write(fd, buf, g_rec);
        if (w <= 0) { perror("write"); break; }
        written += w;
    }
    fsync(fd); close(fd);
    double dt = now() - t0;
    printf("seq write   : %6.2f GB/s  (%s)\n", written / dt / (1u << 30), bypass_note());

    /* 2. sequential read, cache-bypassed */
    fd = open_path(0);
    if (fd < 0) { perror("open read"); return 1; }
    t0 = now(); size_t rd = 0; ssize_t r;
    while ((r = read(fd, buf, g_rec)) > 0) rd += r;
    if (r < 0) perror("read");   /* a failed read otherwise just ends the loop and shortens the row */
    dt = now() - t0; close(fd);
    printf("seq read    : %6.2f GB/s  (%s)\n", rd / dt / (1u << 30), bypass_note());

    /* 3+4. random record reads, 1..maxthreads */
    g_reps = 40;
    for (int nt = 1; nt <= maxthreads; nt *= 2) {
        pthread_t th[64]; targ ta[64];
        t0 = now();
        for (int i = 0; i < nt; i++) {
            ta[i].id = i; ta[i].nthreads = nt; ta[i].bytes = 0;
            pthread_create(&th[i], NULL, rand_reader, &ta[i]);
        }
        double tot = 0;
        for (int i = 0; i < nt; i++) { pthread_join(th[i], NULL); tot += ta[i].bytes; }
        dt = now() - t0;
        double gbs = tot / dt / (1u << 30);
        if (gb_tok > 0)
            printf("rand %2d thr : %6.2f GB/s  -> %.2f tok/s at %.4g GB/token cold  (%s)\n",
                   nt, gbs, gbs / gb_tok, gb_tok, bypass_note());
        else
            printf("rand %2d thr : %6.2f GB/s  (%s)\n", nt, gbs, bypass_note());
    }

    if (!g_direct) {
        fflush(stdout);   /* or the diagnostic overtakes the table it is about */
        fprintf(stderr,
                "\nthe page cache could not be bypassed on this filesystem, so the rows\n"
                "above are partly the kernel serving RAM back and are not the disk. The\n"
                "engine falls back the same way and reports it as direct_io=0; move the\n"
                "working file to the filesystem the container will live on.\n");
    }

    free(buf);
    unlink(g_path);
    return 0;
}
