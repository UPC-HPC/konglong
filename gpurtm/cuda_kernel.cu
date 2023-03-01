#include "cuda_kernel.h"
#include "cuda_related.h"
inline __host__ __device__ unsigned int iDivUp(unsigned int a, unsigned int b)
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

inline __host__ __device__ unsigned int iAlignUp(unsigned int a, unsigned int b)
{
    return (a % b != 0) ?  (a - a % b + b) : a;
}

void printGpuMemInfo () {
    size_t freeMem, totalMem;
    checkCudaErrors(cudaMemGetInfo (&freeMem, &totalMem));
    printf("Free: %llu \t Total: %llu\n",
                (unsigned long long)freeMem,
                (unsigned long long)totalMem );
    return ;
}

