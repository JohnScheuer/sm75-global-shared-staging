#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define CHECK_CUDA(call)                                   \
    do {                                                   \
        cudaError_t err = call;                            \
        if (err != cudaSuccess) {                          \
            std::cerr << "CUDA Error: "                    \
                      << cudaGetErrorString(err)           \
                      << std::endl;                        \
            exit(1);                                       \
        }                                                  \
    } while (0)

// ============================================================
// 🔴 SCALAR VERSION
// ============================================================

template<int BLOCK_SIZE, bool USE_BARRIER>
__global__ void kernel_scalar(const float* __restrict__ in,
                              float* __restrict__ out,
                              int N)
{
    __shared__ float smem[BLOCK_SIZE];

    int tid = threadIdx.x;
    int elements_per_block = BLOCK_SIZE;
    int num_tiles = N / elements_per_block;

    float acc = 0.0f;

    for (int tile = blockIdx.x; tile < num_tiles; tile += gridDim.x)
    {
        int idx = tile * elements_per_block + tid;

        smem[tid] = in[idx];

        if constexpr (USE_BARRIER)
            __syncthreads();

        acc += smem[tid] * 1.0001f;

        if constexpr (USE_BARRIER)
            __syncthreads();
    }

    int gid = blockIdx.x * blockDim.x + tid;
    if (gid < N)
        out[gid] = acc;
}

// ============================================================
// 🔥 FLOAT4 VERSION
// ============================================================

template<int BLOCK_SIZE, bool USE_BARRIER>
__global__ void kernel_vec4(const float* __restrict__ in,
                            float* __restrict__ out,
                            int N)
{
    __shared__ float smem[BLOCK_SIZE * 4];

    int tid = threadIdx.x;
    int elements_per_block = BLOCK_SIZE * 4;
    int num_tiles = N / elements_per_block;

    float acc = 0.0f;

    for (int tile = blockIdx.x; tile < num_tiles; tile += gridDim.x)
    {
        int base = tile * elements_per_block;
        int idx4 = (base / 4) + tid;

        float4 v = reinterpret_cast<const float4*>(in)[idx4];

        int smem_base = tid * 4;

        smem[smem_base + 0] = v.x;
        smem[smem_base + 1] = v.y;
        smem[smem_base + 2] = v.z;
        smem[smem_base + 3] = v.w;

        if constexpr (USE_BARRIER)
            __syncthreads();

        acc += smem[smem_base] * 1.0001f;

        if constexpr (USE_BARRIER)
            __syncthreads();
    }

    int gid = blockIdx.x * blockDim.x + tid;
    if (gid < N)
        out[gid] = acc;
}

// ============================================================
// Benchmark helper
// ============================================================

template<typename Kernel>
float benchmark(Kernel kernel,
                const float* d_in,
                float* d_out,
                int N,
                dim3 grid,
                dim3 block)
{
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));

    for (int i = 0; i < 50; i++)
        kernel<<<grid, block>>>(d_in, d_out, N);

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    return ms / 50.0f;
}

// ============================================================
// Runner
// ============================================================

template<int BLOCK_SIZE, bool USE_BARRIER, bool USE_VEC4>
void run_case(const float* d_in,
              float* d_out,
              int N,
              int numSM,
              size_t bytes)
{
    dim3 block(BLOCK_SIZE);
    dim3 grid(numSM * 8);

    float time_ms;

    if constexpr (USE_VEC4)
    {
        time_ms = benchmark(
            kernel_vec4<BLOCK_SIZE, USE_BARRIER>,
            d_in, d_out,
            N, grid, block);
    }
    else
    {
        time_ms = benchmark(
            kernel_scalar<BLOCK_SIZE, USE_BARRIER>,
            d_in, d_out,
            N, grid, block);
    }

    double gbps = bytes / (time_ms / 1000.0) / 1e9;

    std::cout << BLOCK_SIZE << ","
              << (USE_VEC4 ? "vec4" : "scalar") << ","
              << (USE_BARRIER ? "barrier" : "no_barrier") << ","
              << gbps << "\n";
}

// ============================================================

int main()
{
    int N = 1 << 26;   // 67M floats
    size_t bytes = N * sizeof(float);

    std::vector<float> h_in(N, 1.0f);
    std::vector<float> h_out(N);

    float *d_in, *d_out;
    CHECK_CUDA(cudaMalloc(&d_in, bytes));
    CHECK_CUDA(cudaMalloc(&d_out, bytes));

    CHECK_CUDA(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

    int numSM;
    CHECK_CUDA(cudaDeviceGetAttribute(&numSM,
                                      cudaDevAttrMultiProcessorCount,
                                      0));

    std::cout << "BLOCK_SIZE,MODE,BARRIER,GB/s\n";

    // Sweep
    run_case<128,false,false>(d_in,d_out,N,numSM,bytes);
    run_case<128,true ,false>(d_in,d_out,N,numSM,bytes);
    run_case<128,false,true >(d_in,d_out,N,numSM,bytes);
    run_case<128,true ,true >(d_in,d_out,N,numSM,bytes);

    run_case<256,false,false>(d_in,d_out,N,numSM,bytes);
    run_case<256,true ,false>(d_in,d_out,N,numSM,bytes);
    run_case<256,false,true >(d_in,d_out,N,numSM,bytes);
    run_case<256,true ,true >(d_in,d_out,N,numSM,bytes);

    run_case<512,false,false>(d_in,d_out,N,numSM,bytes);
    run_case<512,true ,false>(d_in,d_out,N,numSM,bytes);
    run_case<512,false,true >(d_in,d_out,N,numSM,bytes);
    run_case<512,true ,true >(d_in,d_out,N,numSM,bytes);

    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}