#include "kernels.cuh"
#include <cuda_runtime.h>
#include <iostream>

#define TILE_SIZE 16

// Standard kernel
template <typename T>
__global__ void matrixMulStandard(T* c, const T* a, const T* b, int size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < size && col < size) {
        T sum = 0;
        for (int k = 0; k < size; ++k) {
            sum += a[row * size + k] * b[k * size + col];
        }
        c[row * size + col] = sum;
    }
}

// Shared memory kernel
template <typename T>
__global__ void matrixMulShared(T* c, const T* a, const T* b, int size) {
    __shared__ T sharedA[TILE_SIZE][TILE_SIZE];
    __shared__ T sharedB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    T sum = 0;

    for (int t = 0; t < (size + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        if (row < size && t * TILE_SIZE + threadIdx.x < size)
            sharedA[threadIdx.y][threadIdx.x] = a[row * size + t * TILE_SIZE + threadIdx.x];
        else
            sharedA[threadIdx.y][threadIdx.x] = 0;

        if (col < size && t * TILE_SIZE + threadIdx.y < size)
            sharedB[threadIdx.y][threadIdx.x] = b[(t * TILE_SIZE + threadIdx.y) * size + col];
        else
            sharedB[threadIdx.y][threadIdx.x] = 0;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += sharedA[threadIdx.y][k] * sharedB[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < size && col < size) {
        c[row * size + col] = sum;
    }
}

// DualThread kernel
template <typename T>
__global__ void matrixMulDualThread(T* c, const T* a, const T* b, int size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = (blockIdx.x * blockDim.x + threadIdx.x) * 2;

    if (row < size && col < size) {
        T sum1 = 0, sum2 = 0;
        for (int k = 0; k < size; ++k) {
            T a_val = a[row * size + k];
            sum1 += a_val * b[k * size + col];
            if (col + 1 < size)
                sum2 += a_val * b[k * size + col + 1];
        }
        c[row * size + col] = sum1;
        if (col + 1 < size)
            c[row * size + col + 1] = sum2;
    }
}

// DualThreadShared kernel
template <typename T>
__global__ void matrixMulDualThreadShared(T* c, const T* a, const T* b, int size) {
    __shared__ T sharedA[TILE_SIZE][TILE_SIZE];
    __shared__ T sharedB[TILE_SIZE][TILE_SIZE * 2];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = (blockIdx.x * TILE_SIZE * 2) + threadIdx.x * 2;

    T sum1 = 0, sum2 = 0;

    for (int t = 0; t < (size + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        int tiledCol = t * TILE_SIZE + threadIdx.x;

        sharedA[threadIdx.y][threadIdx.x] =
            (row < size && tiledCol < size) ? a[row * size + tiledCol] : 0;

        int b_row = t * TILE_SIZE + threadIdx.y;
        sharedB[threadIdx.y][threadIdx.x * 2] =
            (col < size && b_row < size) ? b[b_row * size + col] : 0;
        sharedB[threadIdx.y][threadIdx.x * 2 + 1] =
            (col + 1 < size && b_row < size) ? b[b_row * size + col + 1] : 0;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            T a_val = sharedA[threadIdx.y][k];
            sum1 += a_val * sharedB[k][threadIdx.x * 2];
            sum2 += a_val * sharedB[k][threadIdx.x * 2 + 1];
        }
        __syncthreads();
    }

    if (row < size && col < size)
        c[row * size + col] = sum1;
    if (row < size && col + 1 < size)
        c[row * size + col + 1] = sum2;
}

// Host launcher
template <typename T>
void RunGpuMatrixMulGeneral(T* c, const T* a, const T* b, int size, KernelType kernelType) {
    T* d_a, * d_b, * d_c;
    size_t bytes = size * size * sizeof(T);

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice);

    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((size + blockDim.x * ((kernelType == KernelType::DualThread || kernelType == KernelType::DualThreadShared) ? 2 : 1) - 1) / (blockDim.x * ((kernelType == KernelType::DualThread || kernelType == KernelType::DualThreadShared) ? 2 : 1)),
        (size + blockDim.y - 1) / blockDim.y);

    switch (kernelType) {
    case KernelType::Standard:
        matrixMulStandard << <gridDim, blockDim >> > (d_c, d_a, d_b, size);
        break;
    case KernelType::Shared:
        matrixMulShared << <gridDim, blockDim >> > (d_c, d_a, d_b, size);
        break;
    case KernelType::DualThread:
        matrixMulDualThread << <gridDim, blockDim >> > (d_c, d_a, d_b, size);
        break;
    case KernelType::DualThreadShared:
        matrixMulDualThreadShared << <gridDim, blockDim >> > (d_c, d_a, d_b, size);
        break;
    }

    cudaMemcpy(c, d_c, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

// Explicit instantiations
template void RunGpuMatrixMulGeneral<float>(float*, const float*, const float*, int, KernelType);
template void RunGpuMatrixMulGeneral<int>(int*, const int*, const int*, int, KernelType);
