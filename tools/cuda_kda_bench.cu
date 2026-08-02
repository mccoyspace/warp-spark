// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 SQLite Cloud, Inc.
/* Standalone K3 recurrence correctness and handoff benchmark.
 *
 * This is not the end-to-end CUDA target: the measured recurrence is only
 * 0.4% of accounted decode. It establishes the kernel mapping, arithmetic,
 * graph-launch cost, and mapped-flag handoff needed by a whole-layer graph.
 *
 *   nvcc -O3 -std=c++17 -arch=native -fmad=false \
 *     -Xcompiler=-ffp-contract=off -Xcompiler=-pthread \
 *     -o cuda_kda_bench tools/cuda_kda_bench.cu -pthread
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
#include <thread>
#include <vector>

constexpr int H = 96;
constexpr int K = 128;
constexpr int V = 128;
constexpr size_t STATE_N = (size_t)H * K * V;

static void ok(cudaError_t st, const char *what)
{
    if (st == cudaSuccess) return;
    std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(st));
    std::exit(1);
}

static cudaMemLocation device_location()
{
    cudaMemLocation loc{};
    loc.type = cudaMemLocationTypeDevice;
    loc.id = 0;
    return loc;
}

static void prefetch(void *p, size_t bytes, cudaStream_t stream = 0)
{
    ok(cudaMemPrefetchAsync(p, bytes, device_location(), 0, stream),
       "cudaMemPrefetchAsync");
}

static uint32_t rng_state = 1;
static float random_float(float scale)
{
    rng_state = rng_state * 1664525u + 1013904223u;
    const float unit = (float)((rng_state >> 8) & 0x00ffffffu) / 16777216.0f;
    return (2.0f * unit - 1.0f) * scale;
}

static void cpu_heads(int h0, int h1, const float *q, const float *k,
                      const float *v, const float *g, const float *beta,
                      float *state, float *out)
{
    const float qscale = 1.0f / std::sqrt((float)K);
    for (int h = h0; h < h1; h++) {
        const float *qh = q + (size_t)h * K;
        const float *kh = k + (size_t)h * K;
        const float *vh = v + (size_t)h * V;
        const float *gh = g + (size_t)h * K;
        float *sh = state + (size_t)h * K * V;
        float *oh = out + (size_t)h * V;
        float qs = 0.0f, ks = 0.0f;
        for (int i = 0; i < K; i++) {
            qs += qh[i] * qh[i];
            ks += kh[i] * kh[i];
        }
        const float qn = (1.0f / sqrtf(qs + 1e-12f)) * qscale;
        const float kn = 1.0f / sqrtf(ks + 1e-12f);
        float u[V]{};
        for (int kk = 0; kk < K; kk++) {
            float *row = sh + (size_t)kk * V;
            const float decay = expf(gh[kk]);
            const float kv = kh[kk] * kn;
            for (int col = 0; col < V; col++) {
                row[col] *= decay;
                u[col] += row[col] * kv;
            }
        }
        for (int col = 0; col < V; col++) u[col] = beta[h] * (vh[col] - u[col]);
        std::memset(oh, 0, V * sizeof(float));
        for (int kk = 0; kk < K; kk++) {
            float *row = sh + (size_t)kk * V;
            const float kv = kh[kk] * kn;
            const float qv = qh[kk] * qn;
            for (int col = 0; col < V; col++) {
                row[col] += u[col] * kv;
                oh[col] += row[col] * qv;
            }
        }
    }
}

__global__ static void kda_kernel(const float *q, const float *k,
                                  const float *v, const float *g,
                                  const float *beta, float *state, float *out)
{
    const int h = (int)blockIdx.x;
    const int col = (int)threadIdx.x;
    if (h >= H || col >= V) return;
    __shared__ float qn, kn;
    __shared__ float decay[K];
    const float *qh = q + (size_t)h * K;
    const float *kh = k + (size_t)h * K;
    const float *vh = v + (size_t)h * V;
    const float *gh = g + (size_t)h * K;
    float *sh = state + (size_t)h * K * V;
    if (col == 0) {
        float qs = 0.0f, ks = 0.0f;
        for (int i = 0; i < K; i++) {
            qs += qh[i] * qh[i];
            ks += kh[i] * kh[i];
        }
        qn = (1.0f / sqrtf(qs + 1e-12f)) * (1.0f / sqrtf((float)K));
        kn = 1.0f / sqrtf(ks + 1e-12f);
    }
    decay[col] = expf(gh[col]);
    __syncthreads();

    float u = 0.0f;
    for (int kk = 0; kk < K; kk++) {
        float *cell = sh + (size_t)kk * V + col;
        *cell *= decay[kk];
        u += *cell * (kh[kk] * kn);
    }
    const float delta = beta[h] * (vh[col] - u);
    float o = 0.0f;
    for (int kk = 0; kk < K; kk++) {
        float *cell = sh + (size_t)kk * V + col;
        const float kv = kh[kk] * kn;
        *cell += delta * kv;
        o += *cell * (qh[kk] * qn);
    }
    out[(size_t)h * V + col] = o;
}

__global__ static void signal_kernel(volatile unsigned *flag, unsigned value)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        __threadfence_system();
        *flag = value;
    }
}

static void compare(const char *name, const float *cpu, const float *gpu, size_t n)
{
    double mean = 0.0;
    float max_abs = 0.0f;
    size_t nonfinite = 0;
    for (size_t i = 0; i < n; i++) {
        if (!std::isfinite(cpu[i]) || !std::isfinite(gpu[i])) nonfinite++;
        const float d = std::fabs(cpu[i] - gpu[i]);
        mean += d;
        max_abs = std::max(max_abs, d);
    }
    std::printf("correctness=%s elements=%zu max_abs=%.9g mean_abs=%.9g nonfinite=%zu\n",
                name, n, max_abs, mean / (double)n, nonfinite);
}

static double cpu_bench(int iterations, int threads, const float *q,
                        const float *k, const float *v, const float *g,
                        const float *beta, float *state, float *out)
{
    const auto begin = std::chrono::steady_clock::now();
    std::vector<std::thread> workers;
    for (int t = 0; t < threads; t++) {
        const int h0 = H * t / threads, h1 = H * (t + 1) / threads;
        workers.emplace_back([=]() {
            for (int i = 0; i < iterations; i++)
                cpu_heads(h0, h1, q, k, v, g, beta, state, out);
        });
    }
    for (auto &worker : workers) worker.join();
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - begin).count() / iterations;
}

static double event_bench(int iterations, cudaStream_t stream, cudaGraphExec_t graph,
                          const float *q, const float *k, const float *v,
                          const float *g, const float *beta, float *state, float *out)
{
    cudaEvent_t begin, end;
    ok(cudaEventCreate(&begin), "cudaEventCreate");
    ok(cudaEventCreate(&end), "cudaEventCreate");
    ok(cudaEventRecord(begin, stream), "cudaEventRecord(begin)");
    for (int i = 0; i < iterations; i++) {
        if (graph) ok(cudaGraphLaunch(graph, stream), "cudaGraphLaunch");
        else kda_kernel<<<H, V, 0, stream>>>(q, k, v, g, beta, state, out);
    }
    ok(cudaGetLastError(), "event benchmark launch");
    ok(cudaEventRecord(end, stream), "cudaEventRecord(end)");
    ok(cudaEventSynchronize(end), "cudaEventSynchronize");
    float ms = 0.0f;
    ok(cudaEventElapsedTime(&ms, begin, end), "cudaEventElapsedTime");
    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    return ms / iterations;
}

static double sync_bench(int iterations, cudaStream_t stream, cudaGraphExec_t graph,
                         const float *q, const float *k, const float *v,
                         const float *g, const float *beta, float *state, float *out)
{
    const auto begin = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; i++) {
        if (graph) ok(cudaGraphLaunch(graph, stream), "cudaGraphLaunch");
        else kda_kernel<<<H, V, 0, stream>>>(q, k, v, g, beta, state, out);
        ok(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    }
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - begin).count() / iterations;
}

int main(int argc, char **argv)
{
    const int event_iterations = argc > 1 ? std::atoi(argv[1]) : 500;
    const int sync_iterations = argc > 2 ? std::atoi(argv[2]) : 100;
    if (event_iterations < 1 || sync_iterations < 1) return 2;
    ok(cudaSetDeviceFlags(cudaDeviceMapHost), "cudaSetDeviceFlags");
    ok(cudaSetDevice(0), "cudaSetDevice");

    const size_t qk_bytes = (size_t)H * K * sizeof(float);
    const size_t vo_bytes = (size_t)H * V * sizeof(float);
    const size_t state_bytes = STATE_N * sizeof(float);
    float *q, *k, *v, *g, *beta, *state, *out;
    ok(cudaMallocManaged(&q, qk_bytes), "cudaMallocManaged(q)");
    ok(cudaMallocManaged(&k, qk_bytes), "cudaMallocManaged(k)");
    ok(cudaMallocManaged(&v, vo_bytes), "cudaMallocManaged(v)");
    ok(cudaMallocManaged(&g, qk_bytes), "cudaMallocManaged(g)");
    ok(cudaMallocManaged(&beta, H * sizeof(float)), "cudaMallocManaged(beta)");
    ok(cudaMallocManaged(&state, state_bytes), "cudaMallocManaged(state)");
    ok(cudaMallocManaged(&out, vo_bytes), "cudaMallocManaged(out)");

    std::vector<float> cpu_state(STATE_N), cpu_out((size_t)H * V);
    for (size_t i = 0; i < (size_t)H * K; i++) {
        q[i] = random_float(1.0f);
        k[i] = random_float(1.0f);
        g[i] = -0.001f - std::fabs(random_float(0.05f));
    }
    for (size_t i = 0; i < (size_t)H * V; i++) v[i] = random_float(1.0f);
    for (int h = 0; h < H; h++) beta[h] = 0.5f + random_float(0.49f);
    for (size_t i = 0; i < STATE_N; i++) state[i] = cpu_state[i] = random_float(0.1f);

    cpu_heads(0, H, q, k, v, g, beta, cpu_state.data(), cpu_out.data());
    prefetch(q, qk_bytes); prefetch(k, qk_bytes); prefetch(v, vo_bytes);
    prefetch(g, qk_bytes); prefetch(beta, H * sizeof(float));
    prefetch(state, state_bytes); prefetch(out, vo_bytes);
    kda_kernel<<<H, V>>>(q, k, v, g, beta, state, out);
    ok(cudaGetLastError(), "correctness launch");
    ok(cudaDeviceSynchronize(), "correctness synchronize");
    compare("output", cpu_out.data(), out, (size_t)H * V);
    compare("state", cpu_state.data(), state, STATE_N);

    cudaStream_t stream;
    ok(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");
    volatile unsigned *host_flag = nullptr;
    ok(cudaHostAlloc((void **)&host_flag, sizeof(*host_flag), cudaHostAllocMapped),
       "cudaHostAlloc(flag)");
    volatile unsigned *device_flag = nullptr;
    ok(cudaHostGetDevicePointer((void **)&device_flag, (void *)host_flag, 0),
       "cudaHostGetDevicePointer(flag)");

    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;
    ok(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
       "cudaStreamBeginCapture");
    kda_kernel<<<H, V, 0, stream>>>(q, k, v, g, beta, state, out);
    signal_kernel<<<1, 1, 0, stream>>>(device_flag, 1);
    ok(cudaStreamEndCapture(stream, &graph), "cudaStreamEndCapture");
    ok(cudaGraphInstantiate(&graph_exec, graph, 0), "cudaGraphInstantiate");

    std::vector<float> bench_state(STATE_N), bench_out((size_t)H * V);
    for (size_t i = 0; i < STATE_N; i++) bench_state[i] = random_float(0.1f);
    const double cpu_ms = cpu_bench(event_iterations, 8, q, k, v, g, beta,
                                    bench_state.data(), bench_out.data());
    prefetch(q, qk_bytes, stream); prefetch(k, qk_bytes, stream);
    prefetch(v, vo_bytes, stream); prefetch(g, qk_bytes, stream);
    prefetch(beta, H * sizeof(float), stream); prefetch(state, state_bytes, stream);
    prefetch(out, vo_bytes, stream);
    ok(cudaStreamSynchronize(stream), "benchmark prefetch synchronize");
    const double direct_event = event_bench(event_iterations, stream, nullptr,
                                            q, k, v, g, beta, state, out);
    const double graph_event = event_bench(event_iterations, stream, graph_exec,
                                           q, k, v, g, beta, state, out);
    const double direct_sync = sync_bench(sync_iterations, stream, nullptr,
                                          q, k, v, g, beta, state, out);
    const double graph_sync = sync_bench(sync_iterations, stream, graph_exec,
                                         q, k, v, g, beta, state, out);

    const auto poll_begin = std::chrono::steady_clock::now();
    for (int i = 0; i < sync_iterations; i++) {
        __atomic_store_n((unsigned *)host_flag, 0u, __ATOMIC_RELEASE);
        ok(cudaGraphLaunch(graph_exec, stream), "poll cudaGraphLaunch");
        while (__atomic_load_n((unsigned *)host_flag, __ATOMIC_ACQUIRE) != 1u)
            __asm__ volatile("yield" ::: "memory");
    }
    const auto poll_end = std::chrono::steady_clock::now();
    const double graph_poll =
        std::chrono::duration<double, std::milli>(poll_end - poll_begin).count() /
        sync_iterations;
    ok(cudaStreamSynchronize(stream), "final synchronize");

    std::printf("cpu_8thread_ms=%.6f\n", cpu_ms);
    std::printf("gpu_direct_event_ms=%.6f speedup=%.3f\n",
                direct_event, cpu_ms / direct_event);
    std::printf("gpu_graph_event_ms=%.6f speedup=%.3f\n",
                graph_event, cpu_ms / graph_event);
    std::printf("gpu_direct_host_sync_ms=%.6f speedup=%.3f\n",
                direct_sync, cpu_ms / direct_sync);
    std::printf("gpu_graph_host_sync_ms=%.6f speedup=%.3f\n",
                graph_sync, cpu_ms / graph_sync);
    std::printf("gpu_graph_mapped_poll_ms=%.6f speedup=%.3f\n",
                graph_poll, cpu_ms / graph_poll);

    cudaGraphExecDestroy(graph_exec);
    cudaGraphDestroy(graph);
    cudaFreeHost((void *)host_flag);
    cudaStreamDestroy(stream);
    cudaFree(out); cudaFree(state); cudaFree(beta); cudaFree(g);
    cudaFree(v); cudaFree(k); cudaFree(q);
    return 0;
}
