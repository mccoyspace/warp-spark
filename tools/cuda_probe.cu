// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 SQLite Cloud, Inc.
/*
 * GB10 CUDA/HMM probe. This is deliberately standalone: compile it with
 * nvcc before teaching the dependency-free engine build about CUDA.
 *
 *   nvcc -O3 -std=c++17 -arch=native -o cuda_probe tools/cuda_probe.cu -lcuda
 */

#include <cuda.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static void runtime_ok(cudaError_t st, const char *what)
{
    if (st == cudaSuccess) return;
    std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(st));
    std::exit(1);
}

static void driver_ok(CUresult st, const char *what)
{
    if (st == CUDA_SUCCESS) return;
    const char *name = nullptr, *text = nullptr;
    cuGetErrorName(st, &name);
    cuGetErrorString(st, &text);
    std::fprintf(stderr, "%s: %s (%s)\n", what,
                 name ? name : "CUDA driver error", text ? text : "?");
    std::exit(1);
}

static int attr(cudaDeviceAttr key)
{
    int value = 0;
    runtime_ok(cudaDeviceGetAttribute(&value, key, 0), "cudaDeviceGetAttribute");
    return value;
}

static int driver_attr(CUdevice_attribute key)
{
    CUdevice dev;
    int value = 0;
    driver_ok(cuDeviceGet(&dev, 0), "cuDeviceGet");
    driver_ok(cuDeviceGetAttribute(&value, key, dev), "cuDeviceGetAttribute");
    return value;
}

__global__ static void state_pass(float *state, size_t n, float scale, float add)
{
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (size_t)blockDim.x * gridDim.x)
        state[i] = fmaf(state[i], scale, add);
}

static float run_kernel(const char *name, float *device_ptr, float *host_ptr,
                        size_t count, int iterations)
{
    cudaEvent_t begin, end;
    runtime_ok(cudaEventCreate(&begin), "cudaEventCreate(begin)");
    runtime_ok(cudaEventCreate(&end), "cudaEventCreate(end)");
    runtime_ok(cudaDeviceSynchronize(), "pre-benchmark synchronize");
    runtime_ok(cudaEventRecord(begin), "cudaEventRecord(begin)");
    for (int i = 0; i < iterations; i++)
        state_pass<<<256, 256>>>(device_ptr, count, 0.999999f, 0.000001f);
    runtime_ok(cudaGetLastError(), "state_pass launch");
    runtime_ok(cudaEventRecord(end), "cudaEventRecord(end)");
    runtime_ok(cudaEventSynchronize(end), "cudaEventSynchronize(end)");
    float ms = 0.0f;
    runtime_ok(cudaEventElapsedTime(&ms, begin, end), "cudaEventElapsedTime");
    const double bytes = 2.0 * (double)count * sizeof(float) * iterations;
    const double gib_s = bytes / (ms / 1000.0) / (1024.0 * 1024.0 * 1024.0);
    const float sample = host_ptr ? host_ptr[0] : NAN;
    std::printf("path=%-18s iterations=%d kernel_ms=%.3f traffic_GiB_s=%.3f"
                " host_sample=%.9g\n", name, iterations, ms, gib_s, sample);
    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    return ms;
}

int main(int argc, char **argv)
{
    const int iterations = argc > 1 ? std::atoi(argv[1]) : 100;
    if (iterations < 1) {
        std::fprintf(stderr, "iterations must be positive\n");
        return 2;
    }
    runtime_ok(cudaSetDeviceFlags(cudaDeviceMapHost), "cudaSetDeviceFlags");
    runtime_ok(cudaSetDevice(0), "cudaSetDevice");
    driver_ok(cuInit(0), "cuInit");

    cudaDeviceProp prop{};
    runtime_ok(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");
    int runtime_version = 0, driver_version = 0;
    runtime_ok(cudaRuntimeGetVersion(&runtime_version), "cudaRuntimeGetVersion");
    runtime_ok(cudaDriverGetVersion(&driver_version), "cudaDriverGetVersion");
    std::printf("device=%s compute=%d.%d runtime=%d driver=%d\n",
                prop.name, prop.major, prop.minor, runtime_version, driver_version);
    std::printf("managed_memory=%d\n", attr(cudaDevAttrManagedMemory));
    std::printf("pageable_memory_access=%d\n", attr(cudaDevAttrPageableMemoryAccess));
    std::printf("pageable_uses_host_page_tables=%d\n",
                attr(cudaDevAttrPageableMemoryAccessUsesHostPageTables));
    std::printf("concurrent_managed_access=%d\n",
                attr(cudaDevAttrConcurrentManagedAccess));
    std::printf("direct_managed_access_from_host=%d\n",
                attr(cudaDevAttrDirectManagedMemAccessFromHost));
    std::printf("host_native_atomics=%d\n",
                attr(cudaDevAttrHostNativeAtomicSupported));
    std::printf("registered_host_same_pointer=%d\n",
                attr(cudaDevAttrCanUseHostPointerForRegisteredMem));
    std::printf("stream_mem_ops_v1=%d\n",
                driver_attr(CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS_V1));
    std::printf("stream_mem_ops_64=%d\n",
                driver_attr(CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS));

    /* One K3 layer: 96 heads x 128 x 128 f32 recurrent state = 6 MiB. */
    constexpr size_t count = (size_t)96 * 128 * 128;
    constexpr size_t bytes = count * sizeof(float);
    std::printf("state_bytes=%zu state_MiB=%.3f\n", bytes,
                bytes / (1024.0 * 1024.0));

    float *pageable = (float *)std::calloc(count, sizeof(float));
    if (!pageable) { std::perror("calloc"); return 1; }
    for (size_t i = 0; i < count; i++) pageable[i] = 1.0f;
    if (attr(cudaDevAttrPageableMemoryAccess))
        run_kernel("pageable-calloc", pageable, pageable, count, iterations);

    void *raw = nullptr;
    if (posix_memalign(&raw, 16384, bytes)) { std::perror("posix_memalign"); return 1; }
    float *aligned = (float *)raw;
    for (size_t i = 0; i < count; i++) aligned[i] = 1.0f;
    if (attr(cudaDevAttrPageableMemoryAccess))
        run_kernel("pageable-aligned", aligned, aligned, count, iterations);

    float *registered = nullptr;
    runtime_ok(cudaHostRegister(aligned, bytes, cudaHostRegisterMapped),
               "cudaHostRegister(mapped)");
    runtime_ok(cudaHostGetDevicePointer((void **)&registered, aligned, 0),
               "cudaHostGetDevicePointer");
    run_kernel("host-registered", registered, aligned, count, iterations);
    runtime_ok(cudaHostUnregister(aligned), "cudaHostUnregister");

    float *managed = nullptr;
    runtime_ok(cudaMallocManaged(&managed, bytes), "cudaMallocManaged");
    for (size_t i = 0; i < count; i++) managed[i] = 1.0f;
    runtime_ok(cudaMemPrefetchAsync(managed, bytes, 0), "managed prefetch to device");
    run_kernel("managed-prefetch", managed, managed, count, iterations);

    float *device = nullptr;
    runtime_ok(cudaMalloc(&device, bytes), "cudaMalloc");
    runtime_ok(cudaMemcpy(device, pageable, bytes, cudaMemcpyHostToDevice), "initial H2D");
    run_kernel("device-resident", device, nullptr, count, iterations);
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < 10; i++) {
        runtime_ok(cudaMemcpy(device, pageable, bytes, cudaMemcpyHostToDevice), "H2D");
        state_pass<<<256, 256>>>(device, count, 0.999999f, 0.000001f);
        runtime_ok(cudaGetLastError(), "copy-roundtrip launch");
        runtime_ok(cudaMemcpy(pageable, device, bytes, cudaMemcpyDeviceToHost), "D2H");
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double roundtrip_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count() / 10.0;
    std::printf("path=explicit-roundtrip per_iteration_ms=%.3f\n", roundtrip_ms);

    cudaFree(device);
    cudaFree(managed);
    std::free(aligned);
    std::free(pageable);
    return 0;
}
