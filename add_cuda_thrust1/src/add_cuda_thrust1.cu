

#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <iostream>
#include <cuda_runtime.h>

class CudaStream {
public:
    cudaStream_t stream;
    CudaStream() { cudaStreamCreate(&stream); }
    ~CudaStream() { cudaStreamDestroy(stream); }
    operator cudaStream_t() const { return stream; }
};

__global__ void add_kernel(int *data, int add_val, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) data[idx] += add_val;
}

int main() {
    cudaDeviceProp device_property;
    int device;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&device_property, device);

    std::cout << "Max threads per block: " << device_property.maxThreadsPerBlock << "\n";
    std::cout << "Warp size: " << device_property.warpSize << "\n";
    std::cout << "Max block dimensions: "
              << device_property.maxThreadsDim[0] << " "
              << device_property.maxThreadsDim[1] << " "
              << device_property.maxThreadsDim[2] << "\n";

    const int N = 1024;
    const int initial_val = 5;
    const int add_val = 32;

    CudaStream stream;
    thrust::device_vector<int> d_vec(N);

    // Fill device vector using Thrust
    thrust::fill(thrust::cuda::par.on(stream), d_vec.begin(), d_vec.end(), initial_val);

    // Compute optimal block size using occupancy API
    int minGridSize, optimalBlockSize;
    cudaOccupancyMaxPotentialBlockSize(
        &minGridSize,
        &optimalBlockSize,
        add_kernel,
        0,  // dynamic shared memory per block
        0   // maximum threads per block (0 = device max)
    );

    int blocks = (N + optimalBlockSize - 1) / optimalBlockSize;

    std::cout << "Optimal threads per block: " << optimalBlockSize << "\n";
    std::cout << "Blocks to launch: " << blocks << "\n";

    // Launch custom kernel on same stream with optimal parameters
    auto ptr = thrust::raw_pointer_cast(d_vec.data());
    add_kernel<<<blocks, optimalBlockSize, 0, stream>>>(ptr, add_val, N);

    // Synchronize stream before host access
    cudaStreamSynchronize(stream);

    // Compute actual sum
    int actual_sum = thrust::reduce(d_vec.begin(), d_vec.end());

    // Compute expected sum
    int expected_sum = N * (initial_val + add_val);

    std::cout << "Expected sum: " << expected_sum << "\n";
    std::cout << "Actual sum:   " << actual_sum << "\n";
    std::cout << "\nCode executed on GPU: " << device_property.name << "\n";
}
