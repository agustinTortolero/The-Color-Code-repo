#pragma once

enum class KernelType {
    Standard,
    Shared,
    DualThread,
    DualThreadShared
};

template <typename T>
void RunGpuMatrixMulGeneral(T* c, const T* a, const T* b, int size, KernelType kernelType);
