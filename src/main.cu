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

constexpr int TILE = 256;
constexpr int BLOCK_SIZE = 256;

// ------------------------------------------------------------
// 🔥 Vectorized Global → Shared Staging (float4)
// ------------------------------------------------------------
__global__ void kernel_staging_vec4(const float* __restrict__ in,
                                    float* __restrict__ out,
                                    int N)
{
    __shared__ float smem[BLOCK_SIZE * 4];

    int tid = threadIdx.x;
    int num_tiles = N / (BLOCK_SIZE * 4);

    float acc = 0.0f;

    for (int tile = blockIdx.x; tile < num_tiles; tile += gridDim.x)
    {
        int base = tile * BLOCK_SIZE * 4;

        // índice em float4
        int idx4 = (base / 4) + tid;

        float4 v = reinterpret_cast<const float4*>(in)[idx4];

        int smem_base = tid * 4;

        smem[smem_base + 0] = v.x;
        smem[smem_base + 1] = v.y;
        smem[smem_base + 2] = v.z;
        smem[smem_base + 3] = v.w;

        __syncthreads();

        // Compute leve
        acc += smem[smem_base] * 1.0001f;

        __syncthreads();
    }

    int gid = blockIdx.x * blockDim.x + tid;
    if (gid < N)
        out[gid] = acc;
}

// ------------------------------------------------------------
// Benchmark helper
// ------------------------------------------------------------
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

// ------------------------------------------------------------

int main()
{
    int N = 1 << 26;   // 67M elements (divisível por 4)
    size_t bytes = N * sizeof(float);

    std::vector<float> h_in(N, 1.0f);
    std::vector<float> h_out(N);

    float *d_in, *d_out;
    CHECK_CUDA(cudaMalloc(&d_in, bytes));
    CHECK_CUDA(cudaMalloc(&d_out, bytes));

    CHECK_CUDA(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

    dim3 block(BLOCK_SIZE);

    int numSM;
    CHECK_CUDA(cudaDeviceGetAttribute(&numSM,
                                      cudaDevAttrMultiProcessorCount,
                                      0));

    dim3 grid(numSM * 8);

    float time_ms = benchmark(kernel_staging_vec4,
                              d_in, d_out,
                              N, grid, block);

    double total_bytes = bytes;
    double gbps = total_bytes / (time_ms / 1000.0) / 1e9;

    std::cout << "Staging Vectorized float4 Bandwidth: "
              << gbps << " GB/s\n";

    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}