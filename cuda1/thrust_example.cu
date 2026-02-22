#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <iostream>

// Functor for addition
struct add_functor
{
    __host__ __device__
    float operator()(float a, float b) const
    {
        return a + b;
    }
};

int main()
{
    const int N = 10;

    // Host vectors
    thrust::host_vector<float> h_A(N, 1.0f);
    thrust::host_vector<float> h_B(N, 2.0f);

    // Copy to device (GPU)
    thrust::device_vector<float> d_A = h_A;
    thrust::device_vector<float> d_B = h_B;
    thrust::device_vector<float> d_C(N);

    // Perform vector addition on GPU
    thrust::transform(d_A.begin(), d_A.end(),
                      d_B.begin(),
                      d_C.begin(),
                      add_functor());

    // Copy result back to host
    thrust::host_vector<float> h_C = d_C;

    // Print results
    for (int i = 0; i < N; i++)
    {
        std::cout << h_A[i] << " + "
                  << h_B[i] << " = "
                  << h_C[i] << std::endl;
    }

    return 0;
}
