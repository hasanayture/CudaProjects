#include <cuda_runtime.h>
#include <iostream>

#define RADIUS 1
#define BLOCK_SIZE 256

// ============================================================================
// 1. KERNEL: Using Shared Memory to Cache Data and Eliminate VRAM Latency
// ============================================================================
__global__ void smoothFilterKernel(const float* __restrict__ d_in, float* __restrict__ d_out, int n) {
    // Allocate dynamic shared memory per block (Thread count + Padding for neighbors)
    __shared__ float shared_cache[BLOCK_SIZE + 2 * RADIUS];

    int global_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int local_idx  = threadIdx.x + RADIUS; // Offset local index past left padding

    // Boundary guard check
    if (global_idx >= n) return;

    // A. Cooperative Fetching: All threads load their primary value into fast cache
    shared_cache[local_idx] = d_in[global_idx];

    // B. Halo/Padding Fetching: Handle the left and right neighborhood edges
    if (threadIdx.x < RADIUS) {
        // Left halo boundary logic
        int left_neighbor = global_idx - RADIUS;
        shared_cache[threadIdx.x] = (left_neighbor >= 0) ? d_in[left_neighbor] : 0.0f;

        // Right halo boundary logic
        int right_neighbor = global_idx + BLOCK_SIZE;
        if (right_neighbor < n) {
            shared_cache[local_idx + BLOCK_SIZE] = d_in[right_neighbor];
        } else {
            shared_cache[local_idx + BLOCK_SIZE] = 0.0f;
        }
    }

    // C. Execution Barrier: Wait until all threads finish populating shared memory
    __syncthreads();

    // D. Compute step using ultra-fast on-chip Shared Memory registers
    float sum = shared_cache[local_idx - 1] +
                shared_cache[local_idx]     +
                shared_cache[local_idx + 1];
                
    d_out[global_idx] = sum / 3.0f;
}

// ============================================================================
// 2. HOST SYSTEM PIPELINE
// ============================================================================
int main() {
    const int N = 10000;
    const size_t bytes = N * sizeof(float);

    // Topic: Page-Locked (Pinned) Host Allocation
    // Essential for high-speed Async transfers. Bypasses OS virtual memory paging.
    float *h_in, *h_out;
    cudaMallocHost(&h_in, bytes);
    cudaMallocHost(&h_out, bytes);

    // Initialize sample step data
    for (int i = 0; i < N; ++i) h_in[i] = static_cast<float>(i);

    // Allocate standard Device (VRAM) Memory pools
    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);

    // Topic: Asynchronous Streams Creation
    // Spawns an execution stream to overlap PCIe memory transfers with compute.
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Topic: Host-Device Async Data Staging
    // Non-blocking asynchronous copy. Control returns to CPU instantly.
    cudaMemcpyAsync(d_in, h_in, bytes, cudaMemcpyHostToDevice, stream);

    // Topic: Custom Kernel Execution Configuration
    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid   = (N + threadsPerBlock - 1) / threadsPerBlock;

    std::cout << "Launching Kernel Configuration:\n";
    std::cout << "Grid Size: " << blocksPerGrid << " blocks | "
              << "Block Size: " << threadsPerBlock << " threads/block\n\n";

    // Launch execution pipeline directed into the custom async stream queue
    smoothFilterKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_in, d_out, N);

    // Safely copy data back from Device VRAM asynchronously inside the same queue
    cudaMemcpyAsync(h_out, d_out, bytes, cudaMemcpyDeviceToHost, stream);

    // Topic: Hardware Stream Synchronization
    // CPU stalls here ONLY until this specific data execution stream completes.
    cudaStreamSynchronize(stream);

    // Validate edge result calculations
    std::cout << "--- Pipeline Output Proof ---\n";
    std::cout << "Input[5]: " << h_in[5] << " | Smoothed Output[5]: " << h_out[5] << "\n";
    std::cout << "Math: (4.0 + 5.0 + 6.0) / 3.0 = " << h_out[5] << "\n";

    // Context cleanup
    cudaStreamDestroy(stream);
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFreeHost(h_in);
    cudaFreeHost(h_out);

    return 0;
}

