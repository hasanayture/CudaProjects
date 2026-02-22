#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/for_each.h>
#include <thrust/count.h>
#include <thrust/sort.h>
#include <iostream>

//////////////////////////////////////////////////////////////
// Functors (CUDA 8 safe)
//////////////////////////////////////////////////////////////

struct square_functor {
    __host__ __device__
    float operator()(float x) const {
        return x * x;
    }
};

struct add_functor {
    __host__ __device__
    int operator()(int a, int b) const {
        return a + b;
    }
};

struct multiply10_functor {
    __host__ __device__
    void operator()(int &x) const {
        x *= 10;
    }
};

struct even_functor {
    __host__ __device__
    bool operator()(int x) const {
        return (x % 2) == 0;
    }
};

//////////////////////////////////////////////////////////////
// 1️⃣ Unary Transform
//////////////////////////////////////////////////////////////
void example_transform_unary()
{
    std::cout << "\n=== Unary Transform (square) ===\n";

    const int N = 5;
    thrust::device_vector<float> A(N, 3.0f);
    thrust::device_vector<float> B(N);

    thrust::transform(A.begin(), A.end(), B.begin(), square_functor());

    for (int i = 0; i < N; i++)
        std::cout << B[i] << " ";
    std::cout << "\n";
}

//////////////////////////////////////////////////////////////
// 2️⃣ Binary Transform
//////////////////////////////////////////////////////////////
void example_transform_binary()
{
    std::cout << "\n=== Binary Transform (A+B) ===\n";

    const int N = 5;
    thrust::device_vector<int> A(N, 1);
    thrust::device_vector<int> B(N, 2);
    thrust::device_vector<int> C(N);

    thrust::transform(A.begin(), A.end(),
                      B.begin(),
                      C.begin(),
                      add_functor());

    for (int i = 0; i < N; i++)
        std::cout << C[i] << " ";
    std::cout << "\n";
}

//////////////////////////////////////////////////////////////
// 3️⃣ Reduce
//////////////////////////////////////////////////////////////
void example_reduce()
{
    std::cout << "\n=== Reduce (sum) ===\n";

    thrust::device_vector<int> d(5);
    d[0]=1; d[1]=2; d[2]=3; d[3]=4; d[4]=5;

    int sum = thrust::reduce(d.begin(), d.end(), 0, add_functor());

    std::cout << "Sum = " << sum << "\n";
}

//////////////////////////////////////////////////////////////
// 4️⃣ for_each
//////////////////////////////////////////////////////////////
void example_for_each()
{
    std::cout << "\n=== for_each (multiply by 10) ===\n";

    thrust::device_vector<int> d(5);
    d[0]=1; d[1]=2; d[2]=3; d[3]=4; d[4]=5;

    thrust::for_each(d.begin(), d.end(), multiply10_functor());

    for (int i = 0; i < d.size(); i++)
        std::cout << d[i] << " ";
    std::cout << "\n";
}

//////////////////////////////////////////////////////////////
// 5️⃣ count_if
//////////////////////////////////////////////////////////////
void example_count_if()
{
    std::cout << "\n=== count_if (even numbers) ===\n";

    thrust::device_vector<int> d(6);
    d[0]=1; d[1]=2; d[2]=3;
    d[3]=4; d[4]=5; d[5]=6;

    int count = thrust::count_if(d.begin(), d.end(), even_functor());

    std::cout << "Even count = " << count << "\n";
}

//////////////////////////////////////////////////////////////
// 6️⃣ sort
//////////////////////////////////////////////////////////////
void example_sort()
{
    std::cout << "\n=== sort ===\n";

    thrust::device_vector<int> d(5);
    d[0]=4; d[1]=1; d[2]=3; d[3]=5; d[4]=2;

    thrust::sort(d.begin(), d.end());

    for (int i = 0; i < d.size(); i++)
        std::cout << d[i] << " ";
    std::cout << "\n";
}

//////////////////////////////////////////////////////////////
// MAIN
//////////////////////////////////////////////////////////////
int main()
{
    example_transform_unary();
    example_transform_binary();
    example_reduce();
    example_for_each();
    example_count_if();
    example_sort();
    return 0;
}
