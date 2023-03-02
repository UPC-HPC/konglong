#include "Wavefield.h"
#include "common_util.h"
#include "string.h"
#include "stdlib.h"


Wavefield::Wavefield(int nx, int ny, int nz) : nx_(nx), ny_(ny), nz_(nz)
{

    w0 = NULL;
    w1 = NULL;
    wb = NULL;
    wx = NULL;
    wy = NULL;
    wz = NULL;
    wr = NULL;

    cpuInitialized = false;

    d_w0 = NULL;
    d_w1 = NULL;
    d_wb = NULL;
    d_wx = NULL;
    d_wy = NULL;
    d_wz = NULL;
    d_wr = NULL;

    gpuInitialized = false;

    allocate_mem();
    
}

Wavefield::~Wavefield() {
  deallocate_mem();
}

/*
 * Allocate memory for TTI propagation (multithreaded)
 */

void Wavefield::allocate_mem()
{
    allocate_cpu_mem();
    allocate_gpu_mem();
}

void Wavefield::allocate_cpu_mem() {


  if(!cpuInitialized) {
    size_t nxyz = (size_t)nx_ * ny_ * nz_;
    size_t memSize = nxyz*sizeof(float);

    w0 = (float *)malloc(memSize);
    w1 = (float *)malloc(memSize);
    wb = (float *)malloc(memSize);
    wx = (float *)malloc(memSize);
    wy = (float *)malloc(memSize);
    wz = (float *)malloc(memSize);
    wr = (float *)malloc(memSize);
  }
    
  cpuInitialized = true;
}
void Wavefield::allocate_gpu_mem() {

  if(!gpuInitialized) {
    size_t nxyz = (size_t)nx_ * ny_ * nz_;
    size_t memSize = nxyz*sizeof(float);

    checkCudaErrors(cudaMalloc( (void **) &d_w0, memSize));
    checkCudaErrors(cudaMalloc( (void **) &d_w1, memSize));
    checkCudaErrors(cudaMalloc( (void **) &d_wb, memSize));
    checkCudaErrors(cudaMalloc( (void **) &d_wx, memSize));
    checkCudaErrors(cudaMalloc( (void **) &d_wy, memSize));
    checkCudaErrors(cudaMalloc( (void **) &d_wz, memSize));
    checkCudaErrors(cudaMalloc( (void **) &d_wr, memSize));
  }
    
  gpuInitialized = true;
}


/*
 * Free all memory allocated by Wavefield instance
 */
void Wavefield::deallocate_mem() 
{
    deallocate_cpu_mem();
    deallocate_gpu_mem();
    return;
}

void Wavefield::deallocate_cpu_mem() {

    if(cpuInitialized) 
    {
        free(w0);
        free(w1);
        free(wb);
        free(wx);
        free(wy);
        free(wz);
        free(wr);
    }

    cpuInitialized = false;
}
void Wavefield::deallocate_gpu_mem() {

    if(gpuInitialized) 
    {
        checkCudaErrors(cudaFree(d_w0));
        checkCudaErrors(cudaFree(d_w1));
        checkCudaErrors(cudaFree(d_wb));
        checkCudaErrors(cudaFree(d_wx));
        checkCudaErrors(cudaFree(d_wy));
        checkCudaErrors(cudaFree(d_wz));
        checkCudaErrors(cudaFree(d_wr));
    }

    gpuInitialized = false;
}

/*
 * Zero all memory allocated by Wavefield instance (multithreaded)
 */
void Wavefield::clean_cpu_mem() {
    if(cpuInitialized) 
    {
        size_t nxyz = (size_t)nx_ * ny_ * nz_; 
        size_t memSize = nxyz*sizeof(float);
        memset(w0, 0, memSize);
        memset(w1, 0, memSize);
        memset(wb, 0, memSize);
        memset(wx, 0, memSize);
        memset(wy, 0, memSize);
        memset(wz, 0, memSize);
        memset(wr, 0, memSize);
    }
}
void Wavefield::set_data() {
    if(cpuInitialized) 
    {
        set_data_3d(w0,nx_,ny_,nz_,0);
        set_data_3d(w1,nx_,ny_,nz_,1);
        set_data_3d(wb,nx_,ny_,nz_,2);
        set_data_3d(wx,nx_,ny_,nz_,3);
        set_data_3d(wy,nx_,ny_,nz_,4);
        set_data_3d(wz,nx_,ny_,nz_,5);
        set_data_3d(wr,nx_,ny_,nz_,6);
    }
    else
    {
        printf("ERROR: Memory has not been allocated!\n");
    }

    copy_host2dev();
}

void Wavefield::copy_host2dev() 
{
    size_t nxyz = (size_t)nx_ * ny_ * nz_;
    size_t memSize = nxyz*sizeof(float);
    if(cpuInitialized&&gpuInitialized)
    {
        checkCudaErrors(cudaMemcpy(d_w0, w0, memSize,cudaMemcpyDefault));
        checkCudaErrors(cudaMemcpy(d_w1, w1, memSize,cudaMemcpyDefault));
        checkCudaErrors(cudaMemcpy(d_wb, wb, memSize,cudaMemcpyDefault));
        checkCudaErrors(cudaMemcpy(d_wx, wx, memSize,cudaMemcpyDefault));
        checkCudaErrors(cudaMemcpy(d_wy, wy, memSize,cudaMemcpyDefault));
        checkCudaErrors(cudaMemcpy(d_wz, wz, memSize,cudaMemcpyDefault));
        checkCudaErrors(cudaMemcpy(d_wr, wr, memSize,cudaMemcpyDefault));
    }
    else
    {
    
        printf("ERROR: Either CPU or GPU Memory has not been allocated!\n");
    }
} 

