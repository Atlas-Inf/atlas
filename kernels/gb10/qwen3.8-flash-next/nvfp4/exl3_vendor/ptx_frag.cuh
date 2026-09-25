#pragma once
#include <cuda/atomic>

// Tensor core fragments

template <typename T, int n>
struct Vec
{
    T elems[n];
    __device__ T& operator[](int i) { return elems[i]; }
};

using FragA = Vec<half2, 4>;
using FragB = Vec<half2, 2>;
using FragC = Vec<float, 4>;
using FragC_h = Vec<half2, 2>;
