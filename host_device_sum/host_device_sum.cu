#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/copy.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <chrono>

struct SquareThenDouble {
    __host__ __device__
    unsigned long long operator()(unsigned long long x) const {
        return x * x * 2;
    }
};

void run_host_example(const unsigned long long N) {
    std::vector<unsigned long long> host_vector(N);
    for(unsigned long long i = 0; i < N; i++) host_vector[i] = i + 1;

    //Warm up
    volatile unsigned long long tmp{0ULL};
    for(unsigned long long i = 0; i < N; i++)
        tmp = host_vector[i] * host_vector[i] * 2;

    auto start = std::chrono::high_resolution_clock::now();

    for(unsigned long long i = 0; i < N; i++)
        host_vector[i] = host_vector[i] * host_vector[i] * 2;

    unsigned long long total_sum = std::accumulate(host_vector.begin(), host_vector.end(), 0ULL);

    auto end = std::chrono::high_resolution_clock::now();
    double duration_ms = std::chrono::duration<double, std::milli>(end - start).count();

    //Print tmp variable for it to be not optimized
    std::cout << "tmp: " << tmp << "\n";
    std::cout << "Host sum: " << total_sum << "\n";
    std::cout << "Host duration: " << duration_ms << " ms\n";
}

void run_device_example(const unsigned long long N) {
    thrust::host_vector<unsigned long long> host_vector(N);
    for(unsigned long long i = 0; i < N; i++) host_vector[i] = i + 1;

    thrust::device_vector<unsigned long long> device_vector = host_vector;

    //Warm up
    thrust::transform(device_vector.begin(), device_vector.end(), device_vector.begin(), SquareThenDouble());
    thrust::reduce(device_vector.begin(), device_vector.end(), 0ULL, thrust::plus<unsigned long long>());

    auto start = std::chrono::high_resolution_clock::now();

    thrust::transform(device_vector.begin(), device_vector.end(), device_vector.begin(), SquareThenDouble());
    unsigned long long total_sum = thrust::reduce(device_vector.begin(), device_vector.end(), 0ULL, thrust::plus<unsigned long long>());

    auto end = std::chrono::high_resolution_clock::now();
    double duration_ms = std::chrono::duration<double, std::milli>(end - start).count();

    thrust::copy(device_vector.begin(), device_vector.end(), host_vector.begin());

    std::cout << "Device sum: " << total_sum << "\n";
    std::cout << "Device duration: " << duration_ms << " ms\n";
}

int main() {
    constexpr unsigned long long N = 100000000; // 100 million elements
    run_host_example(N);
    run_device_example(N);
    return 0;
}
