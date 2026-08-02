// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 SQLite Cloud, Inc.
/* Standalone benchmark for WASTE's real Q4G/group-128 trunk projections.
 *
 * The benchmark reads one matrix directly from trunk.bin, so it measures the
 * format and memory path the engine would use rather than a synthetic float
 * matrix.  The pageable path is the integration candidate: registering or
 * duplicating the 29 GiB trunk would violate the model/cache memory budget.
 *
 * Example (K3 layer 1 q_proj):
 *
 *   nvcc -O3 -std=c++17 -arch=native -fmad=false \
 *     -Xcompiler=-ffp-contract=off -Xcompiler=-pthread \
 *     -Xcompiler=-mcpu=native -o cuda_q4_matvec_bench \
 *     tools/cuda_q4_matvec_bench.cu
 *   ./cuda_q4_matvec_bench trunk.bin 3226180096 3270220288 12288 7168 40 5
 *
 * Arguments are TRUNK Q_OFFSET SCALE_OFFSET OUT IN [ITERATIONS] [WARMUP].
 */

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <thread>
#include <unistd.h>
#include <vector>

constexpr int GROUP = 128;
constexpr int THREADS = 128;

static void ok(cudaError_t status, const char *what)
{
    if (status == cudaSuccess) return;
    std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(status));
    std::exit(1);
}

static uint64_t number(const char *text, const char *what)
{
    char *end = nullptr;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (!text[0] || !end || *end) {
        std::fprintf(stderr, "invalid %s: %s\n", what, text);
        std::exit(2);
    }
    return (uint64_t)value;
}

static void pread_all(int fd, void *dst, size_t bytes, uint64_t offset)
{
    uint8_t *p = static_cast<uint8_t *>(dst);
    while (bytes) {
        const ssize_t got = pread(fd, p, bytes, (off_t)offset);
        if (got <= 0) {
            std::perror("pread");
            std::exit(1);
        }
        p += got;
        bytes -= (size_t)got;
        offset += (uint64_t)got;
    }
}

static void *aligned_alloc_or_die(size_t bytes)
{
    void *p = nullptr;
    if (posix_memalign(&p, 4096, bytes ? bytes : 4096) || !p) {
        std::fprintf(stderr, "could not allocate %zu bytes\n", bytes);
        std::exit(1);
    }
    return p;
}

static inline float half_to_float_cpu(uint16_t h)
{
    const uint32_t sign = (uint32_t)(h >> 15) << 31;
    const uint32_t exponent = (h >> 10) & 0x1f;
    const uint32_t mantissa = h & 0x3ff;
    if (exponent == 0) {
        const float value = (float)mantissa * 5.9604644775390625e-08f;
        return sign ? -value : value;
    }
    const uint32_t bits = sign | ((exponent + 112u) << 23) | (mantissa << 13);
    float value;
    std::memcpy(&value, &bits, sizeof value);
    return value;
}

__device__ static float half_to_float_device(uint16_t h)
{
    const uint32_t sign = (uint32_t)(h >> 15) << 31;
    const uint32_t exponent = (h >> 10) & 0x1f;
    const uint32_t mantissa = h & 0x3ff;
    if (exponent == 0) {
        const float value = (float)mantissa * 5.9604644775390625e-08f;
        return sign ? -value : value;
    }
    return __uint_as_float(sign | ((exponent + 112u) << 23) |
                           (mantissa << 13));
}

static inline int q4_value(const uint8_t *row, int index)
{
    const uint8_t packed = row[index >> 1];
    return ((index & 1) ? (packed >> 4) : (packed & 0x0f)) - 8;
}

static float cpu_row(const uint8_t *row, const uint16_t *scale,
                     const float *x, int in)
{
    const int groups = (in + GROUP - 1) / GROUP;
    float total = 0.0f;
    for (int group = 0; group < groups; group++) {
        const int begin = group * GROUP;
        const int limit = std::min(GROUP, in - begin);
        /* NVCC cannot parse GCC's full arm_neon.h. Four scalar accumulators
         * reproduce the engine's vfmaq lane order and pairwise vaddvq
         * reduction without including that header in a CUDA translation
         * unit. The engine profile, not this emulation, is the performance
         * control used by the whole-layer gate. */
        float lane[4] = {};
        int i = 0;
        for (; i + 8 <= limit; i += 8) {
            for (int j = 0; j < 4; j++)
                lane[j] = std::fma((float)q4_value(row, begin + i + j),
                                   x[begin + i + j], lane[j]);
            for (int j = 0; j < 4; j++)
                lane[j] = std::fma((float)q4_value(row, begin + i + 4 + j),
                                   x[begin + i + 4 + j], lane[j]);
        }
        float part = (lane[0] + lane[1]) + (lane[2] + lane[3]);
        for (; i < limit; i++)
            part += (float)q4_value(row, begin + i) * x[begin + i];
        total += half_to_float_cpu(scale[group]) * part;
    }
    return total;
}

static void cpu_rows(int begin, int end, const uint8_t *weights,
                     const uint16_t *scales, const float *x, float *y,
                     int in, size_t rowbytes, int iterations)
{
    const int groups = (in + GROUP - 1) / GROUP;
    for (int iteration = 0; iteration < iterations; iteration++)
        for (int row = begin; row < end; row++)
            y[row] = cpu_row(weights + (size_t)row * rowbytes,
                             scales + (size_t)row * groups, x, in);
}

static double cpu_bench(int iterations, int threads, const uint8_t *weights,
                        const uint16_t *scales, const float *x, float *y,
                        int out, int in, size_t rowbytes)
{
    const auto begin = std::chrono::steady_clock::now();
    std::vector<std::thread> workers;
    for (int thread = 0; thread < threads; thread++) {
        const int row0 = out * thread / threads;
        const int row1 = out * (thread + 1) / threads;
        workers.emplace_back(cpu_rows, row0, row1, weights, scales, x, y,
                             in, rowbytes, iterations);
    }
    for (auto &worker : workers) worker.join();
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - begin).count() /
           (double)iterations;
}

__device__ static int q4_device(const uint8_t *row, int index)
{
    const uint8_t packed = row[index >> 1];
    return ((index & 1) ? (packed >> 4) : (packed & 0x0f)) - 8;
}

/* Throughput path: one block owns a row and reduces 128 independent partial
 * sums. Scaling each element before the reduction is mathematically equal
 * to WASTE's per-group scaling, but deliberately changes association. */
__global__ static void q4_fast_kernel(const uint8_t *weights,
                                      const uint16_t *scales,
                                      const float *x, float *y,
                                      int out, int in, size_t rowbytes)
{
    const int row_index = (int)blockIdx.x;
    const int lane = (int)threadIdx.x;
    if (row_index >= out || lane >= THREADS) return;
    const int groups = (in + GROUP - 1) / GROUP;
    const uint8_t *row = weights + (size_t)row_index * rowbytes;
    const uint16_t *row_scales = scales + (size_t)row_index * groups;
    float sum = 0.0f;
    for (int i = lane; i < in; i += THREADS) {
        const float scale = half_to_float_device(row_scales[i / GROUP]);
        sum = fmaf(scale * (float)q4_device(row, i), x[i], sum);
    }
    __shared__ float partial[THREADS];
    partial[lane] = sum;
    __syncthreads();
    for (int stride = THREADS / 2; stride; stride >>= 1) {
        if (lane < stride) partial[lane] += partial[lane + stride];
        __syncthreads();
    }
    if (lane == 0) y[row_index] = partial[0];
}

/* Equivalence path: four active lanes reproduce the four FP32 accumulators
 * in model.c's ARM NEON loop. Groups and their scale application remain in
 * CPU order. It intentionally gives up parallelism to test the price of the
 * strongest arithmetic contract. */
__global__ static void q4_neon_order_kernel(const uint8_t *weights,
                                            const uint16_t *scales,
                                            const float *x, float *y,
                                            int out, int in, size_t rowbytes)
{
    const int row_index = (int)blockIdx.x;
    const int lane = (int)threadIdx.x;
    if (row_index >= out || lane >= 4) return;
    const int groups = (in + GROUP - 1) / GROUP;
    const uint8_t *row = weights + (size_t)row_index * rowbytes;
    const uint16_t *row_scales = scales + (size_t)row_index * groups;
    __shared__ float partial[4];
    __shared__ float total;
    if (lane == 0) total = 0.0f;
    __syncwarp(0x0fu);
    for (int group = 0; group < groups; group++) {
        const int begin = group * GROUP;
        const int limit = min(GROUP, in - begin);
        float sum = 0.0f;
        for (int i = lane; i < limit; i += 4)
            sum = fmaf((float)q4_device(row, begin + i), x[begin + i], sum);
        partial[lane] = sum;
        __syncwarp(0x0fu);
        if (lane == 0) {
            const float part = (partial[0] + partial[1]) +
                               (partial[2] + partial[3]);
            total += half_to_float_device(row_scales[group]) * part;
        }
        __syncwarp(0x0fu);
    }
    if (lane == 0) y[row_index] = total;
}

__global__ static void signal_kernel(volatile unsigned *flag)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        __threadfence_system();
        *flag = 1;
    }
}

enum class Kernel { Fast, NeonOrder };

static void launch(Kernel kernel, const uint8_t *weights,
                   const uint16_t *scales, const float *x, float *y,
                   int out, int in, size_t rowbytes, cudaStream_t stream)
{
    if (kernel == Kernel::Fast)
        q4_fast_kernel<<<out, THREADS, 0, stream>>>(weights, scales, x, y,
                                                    out, in, rowbytes);
    else
        q4_neon_order_kernel<<<out, 32, 0, stream>>>(weights, scales, x, y,
                                                     out, in, rowbytes);
}

static double event_bench(Kernel kernel, int iterations,
                          const uint8_t *weights, const uint16_t *scales,
                          const float *x, float *y, int out, int in,
                          size_t rowbytes, cudaStream_t stream)
{
    cudaEvent_t begin, end;
    ok(cudaEventCreate(&begin), "cudaEventCreate(begin)");
    ok(cudaEventCreate(&end), "cudaEventCreate(end)");
    ok(cudaEventRecord(begin, stream), "cudaEventRecord(begin)");
    for (int i = 0; i < iterations; i++)
        launch(kernel, weights, scales, x, y, out, in, rowbytes, stream);
    ok(cudaGetLastError(), "Q4 benchmark launch");
    ok(cudaEventRecord(end, stream), "cudaEventRecord(end)");
    ok(cudaEventSynchronize(end), "cudaEventSynchronize(end)");
    float elapsed = 0.0f;
    ok(cudaEventElapsedTime(&elapsed, begin, end), "cudaEventElapsedTime");
    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    return elapsed / (double)iterations;
}

static double host_sync_bench(int iterations, const uint8_t *weights,
                              const uint16_t *scales, const float *x, float *y,
                              int out, int in, size_t rowbytes,
                              cudaStream_t stream)
{
    const auto begin = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; i++) {
        launch(Kernel::Fast, weights, scales, x, y, out, in, rowbytes, stream);
        ok(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    }
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - begin).count() /
           (double)iterations;
}

static void compare(const char *name, const float *cpu, const float *gpu,
                    size_t count)
{
    float max_abs = 0.0f;
    double total_abs = 0.0;
    size_t nonfinite = 0;
    for (size_t i = 0; i < count; i++) {
        if (!std::isfinite(cpu[i]) || !std::isfinite(gpu[i])) nonfinite++;
        const float difference = std::fabs(cpu[i] - gpu[i]);
        max_abs = std::max(max_abs, difference);
        total_abs += difference;
    }
    std::printf("correctness=%s elements=%zu byte_exact=%d max_abs=%.9g "
                "mean_abs=%.9g nonfinite=%zu\n",
                name, count, std::memcmp(cpu, gpu, count * sizeof(float)) == 0,
                max_abs, total_abs / (double)count, nonfinite);
}

static double gib_per_second(size_t bytes, double milliseconds)
{
    return milliseconds > 0.0
         ? (double)bytes / (1024.0 * 1024.0 * 1024.0) /
           (milliseconds / 1000.0)
         : 0.0;
}

int main(int argc, char **argv)
{
    if (argc < 6 || argc > 8) {
        std::fprintf(stderr, "usage: %s TRUNK Q_OFFSET SCALE_OFFSET OUT IN "
                             "[ITERATIONS] [WARMUP]\n", argv[0]);
        return 2;
    }
    const char *path = argv[1];
    const uint64_t q_offset = number(argv[2], "Q_OFFSET");
    const uint64_t scale_offset = number(argv[3], "SCALE_OFFSET");
    const int out = (int)number(argv[4], "OUT");
    const int in = (int)number(argv[5], "IN");
    const int iterations = argc > 6 ? (int)number(argv[6], "ITERATIONS") : 40;
    const int warmup = argc > 7 ? (int)number(argv[7], "WARMUP") : 5;
    if (out <= 0 || in <= 0 || iterations <= 0 || warmup < 0) {
        std::fprintf(stderr, "dimensions and iteration count must be positive\n");
        return 2;
    }
    const int groups = (in + GROUP - 1) / GROUP;
    const size_t rowbytes = ((size_t)in + 1) / 2;
    const size_t q_bytes = (size_t)out * rowbytes;
    const size_t scale_bytes = (size_t)out * groups * sizeof(uint16_t);
    const size_t traffic = q_bytes + scale_bytes +
                           ((size_t)in + out) * sizeof(float);
    if (q_offset + q_bytes > scale_offset) {
        std::fprintf(stderr, "Q payload overlaps scale offset\n");
        return 2;
    }

    const int fd = open(path, O_RDONLY);
    if (fd < 0) { std::perror(path); return 1; }
    uint8_t *weights = static_cast<uint8_t *>(aligned_alloc_or_die(q_bytes));
    uint16_t *scales = static_cast<uint16_t *>(aligned_alloc_or_die(scale_bytes));
    pread_all(fd, weights, q_bytes, q_offset);
    pread_all(fd, scales, scale_bytes, scale_offset);
    close(fd);

    ok(cudaSetDeviceFlags(cudaDeviceMapHost), "cudaSetDeviceFlags");
    ok(cudaSetDevice(0), "cudaSetDevice");
    cudaDeviceProp properties{};
    ok(cudaGetDeviceProperties(&properties, 0), "cudaGetDeviceProperties");
    int pageable = 0, host_tables = 0;
    ok(cudaDeviceGetAttribute(&pageable, cudaDevAttrPageableMemoryAccess, 0),
       "pageable-memory attribute");
    ok(cudaDeviceGetAttribute(&host_tables,
                              cudaDevAttrPageableMemoryAccessUsesHostPageTables,
                              0), "host-page-table attribute");
    if (!pageable || !host_tables) {
        std::fprintf(stderr, "device cannot directly use pageable host pointers\n");
        return 1;
    }

    float *x = nullptr, *gpu_y = nullptr;
    volatile unsigned *host_flag = nullptr;
    ok(cudaHostAlloc((void **)&x, (size_t)in * sizeof(float), cudaHostAllocMapped),
       "cudaHostAlloc(x)");
    ok(cudaHostAlloc((void **)&gpu_y, (size_t)out * sizeof(float),
                     cudaHostAllocMapped), "cudaHostAlloc(y)");
    ok(cudaHostAlloc((void **)&host_flag, sizeof(*host_flag), cudaHostAllocMapped),
       "cudaHostAlloc(flag)");
    float *device_x = nullptr, *device_y = nullptr;
    volatile unsigned *device_flag = nullptr;
    ok(cudaHostGetDevicePointer((void **)&device_x, x, 0),
       "cudaHostGetDevicePointer(x)");
    ok(cudaHostGetDevicePointer((void **)&device_y, gpu_y, 0),
       "cudaHostGetDevicePointer(y)");
    ok(cudaHostGetDevicePointer((void **)&device_flag, (void *)host_flag, 0),
       "cudaHostGetDevicePointer(flag)");

    uint32_t state = 1;
    for (int i = 0; i < in; i++) {
        state = state * 1664525u + 1013904223u;
        x[i] = ((float)((state >> 8) & 0x00ffffffu) / 8388608.0f - 1.0f) *
               0.125f;
    }
    std::vector<float> cpu_y((size_t)out), fast_y((size_t)out),
                       strict_y((size_t)out), cpu_bench_y((size_t)out);
    cpu_rows(0, out, weights, scales, x, cpu_y.data(), in, rowbytes, 1);

    cudaStream_t stream;
    ok(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
       "cudaStreamCreate");
    launch(Kernel::Fast, weights, scales, device_x, device_y,
           out, in, rowbytes, stream);
    ok(cudaStreamSynchronize(stream), "fast correctness synchronize");
    std::memcpy(fast_y.data(), gpu_y, (size_t)out * sizeof(float));
    launch(Kernel::NeonOrder, weights, scales, device_x, device_y,
           out, in, rowbytes, stream);
    ok(cudaStreamSynchronize(stream), "strict correctness synchronize");
    std::memcpy(strict_y.data(), gpu_y, (size_t)out * sizeof(float));

    std::printf("device=%s compute=%d.%d pageable=%d host_page_tables=%d\n",
                properties.name, properties.major, properties.minor,
                pageable, host_tables);
    std::printf("shape=%dx%d group=%d q_bytes=%zu scale_bytes=%zu "
                "traffic_bytes=%zu iterations=%d warmup=%d\n",
                out, in, GROUP, q_bytes, scale_bytes, traffic,
                iterations, warmup);
    compare("fast", cpu_y.data(), fast_y.data(), (size_t)out);
    compare("neon-order", cpu_y.data(), strict_y.data(), (size_t)out);

    for (int i = 0; i < warmup; i++) {
        launch(Kernel::Fast, weights, scales, device_x, device_y,
               out, in, rowbytes, stream);
        launch(Kernel::NeonOrder, weights, scales, device_x, device_y,
               out, in, rowbytes, stream);
    }
    ok(cudaStreamSynchronize(stream), "warmup synchronize");

    const double cpu_ms = cpu_bench(iterations, 8, weights, scales, x,
                                    cpu_bench_y.data(), out, in, rowbytes);
    const double fast_ms = event_bench(Kernel::Fast, iterations, weights,
                                       scales, device_x, device_y, out, in,
                                       rowbytes, stream);
    const double strict_ms = event_bench(Kernel::NeonOrder, iterations,
                                         weights, scales, device_x, device_y,
                                         out, in, rowbytes, stream);
    const double host_ms = host_sync_bench(iterations, weights, scales,
                                           device_x, device_y, out, in,
                                           rowbytes, stream);
    std::printf("path=cpu-8thread-neon-order-emulation kernel_ms=%.6f "
                "effective_GiB_s=%.3f\n",
                cpu_ms, gib_per_second(traffic, cpu_ms));
    std::printf("path=pageable-fast-event kernel_ms=%.6f speedup=%.3f "
                "effective_GiB_s=%.3f\n", fast_ms, cpu_ms / fast_ms,
                gib_per_second(traffic, fast_ms));
    std::printf("path=pageable-neon-order-event kernel_ms=%.6f speedup=%.3f "
                "effective_GiB_s=%.3f\n", strict_ms, cpu_ms / strict_ms,
                gib_per_second(traffic, strict_ms));
    std::printf("path=pageable-fast-host-sync kernel_ms=%.6f speedup=%.3f\n",
                host_ms, cpu_ms / host_ms);

    /* One fixed graph, including the mapped completion flag used by a CPU
     * handoff. This is the launch economy proposed for the layer graph. */
    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;
    *host_flag = 0;
    ok(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
       "cudaStreamBeginCapture");
    launch(Kernel::Fast, weights, scales, device_x, device_y,
           out, in, rowbytes, stream);
    signal_kernel<<<1, 1, 0, stream>>>(device_flag);
    ok(cudaStreamEndCapture(stream, &graph), "cudaStreamEndCapture");
    ok(cudaGraphInstantiate(&graph_exec, graph, 0), "cudaGraphInstantiate");
    const auto poll_begin = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; i++) {
        *host_flag = 0;
        std::atomic_thread_fence(std::memory_order_seq_cst);
        ok(cudaGraphLaunch(graph_exec, stream), "cudaGraphLaunch");
        while (!*host_flag) {
#if defined(__aarch64__)
            __asm__ volatile("yield");
#else
            std::this_thread::yield();
#endif
        }
        std::atomic_thread_fence(std::memory_order_seq_cst);
    }
    const auto poll_end = std::chrono::steady_clock::now();
    const double poll_ms =
        std::chrono::duration<double, std::milli>(poll_end - poll_begin).count() /
        (double)iterations;
    ok(cudaStreamSynchronize(stream), "graph final synchronize");
    std::printf("path=pageable-fast-graph-poll kernel_ms=%.6f speedup=%.3f\n",
                poll_ms, cpu_ms / poll_ms);

    /* Registration is an upper-bound screen only. The integrated path must
     * not pin all trunk pages behind the expert cache's back. */
    cudaError_t reg_w = cudaHostRegister(weights, q_bytes, cudaHostRegisterMapped);
    cudaError_t reg_s = reg_w == cudaSuccess
                      ? cudaHostRegister(scales, scale_bytes, cudaHostRegisterMapped)
                      : reg_w;
    if (reg_w == cudaSuccess && reg_s == cudaSuccess) {
        uint8_t *registered_w = nullptr;
        uint16_t *registered_s = nullptr;
        ok(cudaHostGetDevicePointer((void **)&registered_w, weights, 0),
           "registered weight pointer");
        ok(cudaHostGetDevicePointer((void **)&registered_s, scales, 0),
           "registered scale pointer");
        const double registered_ms = event_bench(
            Kernel::Fast, iterations, registered_w, registered_s,
            device_x, device_y, out, in, rowbytes, stream);
        std::printf("path=registered-fast-event kernel_ms=%.6f speedup=%.3f "
                    "effective_GiB_s=%.3f\n", registered_ms,
                    cpu_ms / registered_ms,
                    gib_per_second(traffic, registered_ms));
        cudaHostUnregister(scales);
        cudaHostUnregister(weights);
    } else {
        std::printf("path=registered-fast-event unavailable=%s\n",
                    cudaGetErrorString(reg_s));
        if (reg_w == cudaSuccess) cudaHostUnregister(weights);
        cudaGetLastError();
    }

    cudaGraphExecDestroy(graph_exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    cudaFreeHost((void *)host_flag);
    cudaFreeHost(gpu_y);
    cudaFreeHost(x);
    std::free(scales);
    std::free(weights);
    return 0;
}
