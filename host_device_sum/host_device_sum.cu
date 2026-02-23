#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/copy.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <chrono>

constexpr unsigned long long N = 8'000'000;

__device__ int device_square(int element_value) {
    return element_value * element_value;
}

__global__ void kernel_square_array(int* device_vector_pointer, unsigned long long total_elements) {
    unsigned long long thread_index = threadIdx.x + blockIdx.x * blockDim.x;
    if(thread_index < total_elements) {
        device_vector_pointer[thread_index] = device_square(device_vector_pointer[thread_index]);
    }
}

struct SquarePlusOneFunctor {
    __host__ __device__
    int operator()(int element_value) const {
        return element_value * element_value + 1;
    }
};

struct MultiplyByTwoFunctor {
    __host__ __device__
    int operator()(int element_value) const {
        return element_value * 2;
    }
};

void run_host_example() {
    std::vector<int> host_vector(N);
    for(unsigned long long i = 0; i < N; i++) host_vector[i] = i + 1;

    auto start = std::chrono::high_resolution_clock::now();

    for(unsigned long long i = 0; i < N; i++) host_vector[i] *= host_vector[i];
    for(unsigned long long i = 0; i < N; i++) host_vector[i] *= 2;

    long long total_sum = std::accumulate(host_vector.begin(), host_vector.end(), 0LL);

    auto end = std::chrono::high_resolution_clock::now();
    double duration_ms = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "Host sum: " << total_sum << "\n";
    std::cout << "Host duration: " << duration_ms << " ms\n";
}

void run_device_example() {
    thrust::host_vector<int> host_vector(N);
    for(unsigned long long i = 0; i < N; i++) host_vector[i] = i + 1;

    thrust::device_vector<int> device_vector = host_vector;
    int* raw_ptr = thrust::raw_pointer_cast(device_vector.data());

    auto start = std::chrono::high_resolution_clock::now();

    unsigned long long threads_per_block = 256;
    unsigned long long num_blocks = (N + threads_per_block - 1) / threads_per_block;
    kernel_square_array<<<num_blocks, threads_per_block>>>(raw_ptr, N);
    cudaDeviceSynchronize();

    thrust::transform(device_vector.begin(), device_vector.end(), device_vector.begin(), MultiplyByTwoFunctor());
    long long total_sum = thrust::reduce(device_vector.begin(), device_vector.end(), 0LL, thrust::plus<long long>());

    auto end = std::chrono::high_resolution_clock::now();
    double duration_ms = std::chrono::duration<double, std::milli>(end - start).count();

    thrust::copy(device_vector.begin(), device_vector.end(), host_vector.begin());

    std::cout << "Device sum: " << total_sum << "\n";
    std::cout << "Device duration: " << duration_ms << " ms\n";
}

int main() {
    run_host_example();
    run_device_example();
}
