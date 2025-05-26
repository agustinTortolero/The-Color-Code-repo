#include "kernels.cuh"
#include <iostream>
#include <vector>
#include <cstdlib>
#include <chrono>
#include <cuda_runtime.h>

template <typename T>
void FillMatrix(std::vector<T>& mat, int size) {
    for (auto& v : mat) {
        v = static_cast<T>(rand() % 10);
    }
}

template <typename T>
void PrintMatrix(const std::vector<T>& mat, int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            std::cout << mat[i * size + j] << " ";
        }
        std::cout << "\n";
    }
}

template <typename T>
void BenchmarkKernel(std::vector<T> a, std::vector<T> b, std::vector<T> c, int size, KernelType kernelType, const std::string& label) {

    cudaEvent_t start, stop;
    float elapsed = 0.0f;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    std::cout << "\nRunning kernel: " << label << "\n";

    auto host_start = std::chrono::high_resolution_clock::now();

    cudaEventRecord(start);

    RunGpuMatrixMulGeneral(c.data(), a.data(), b.data(), size, kernelType);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);

    auto host_end = std::chrono::high_resolution_clock::now();
    auto host_elapsed = std::chrono::duration<double, std::milli>(host_end - host_start).count();

    std::cout << "GPU Time:  " << elapsed << " ms\n";
    std::cout << "Total Time (host+device): " << host_elapsed << " ms\n";

    if (size <= 8) {
        std::cout << "\nMatrix A:\n"; PrintMatrix(a, size);
        std::cout << "\nMatrix B:\n"; PrintMatrix(b, size);
        std::cout << "\nMatrix C:\n"; PrintMatrix(c, size);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    const int size = 1000; 


    std::vector<int> a(size * size);
    std::vector<int> b(size * size);
    std::vector<int> c(size * size);

    FillMatrix(a, size);
    FillMatrix(b, size);

    BenchmarkKernel<int>(a, b, c, size, KernelType::Standard, "Standard");
    //BenchmarkKernel<int>(a, b, c, size, KernelType::Shared, "Shared");
    //BenchmarkKernel<int>(a, b, c, size, KernelType::DualThread, "DualThread");
    //BenchmarkKernel<int>(a, b, c, size, KernelType::DualThreadShared, "DualThreadShared");

    return 0;
}
