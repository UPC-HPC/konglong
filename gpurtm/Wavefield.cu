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

void Wavefield::allocate_cpu_mem()
{
    printf("Start Wavefield::allocate_cpu_mem\n");

    if(!cpuInitialized) 
    {
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
    printf("End Wavefield::allocate_cpu_mem\n");
}
void Wavefield::allocate_gpu_mem() 
{

    printf("Start Wavefield::allocate_gpu_mem\n");
    if(!gpuInitialized) 
    {
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
    printf("End Wavefield::allocate_cpu_mem\n");
}


/*
 * Free all memory allocated by Wavefield instance
 */
void Wavefield::deallocate_mem() 
{
    printf("Start Wavefield::deallocate_mem\n");
    deallocate_cpu_mem();
    deallocate_gpu_mem();
    printf("End Wavefield::deallocate_mem\n");
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
void Wavefield::clean_cpu_mem()
{
    printf("Start Wavefield::clean_cpu_mem\n");
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
    printf("End Wavefield::clean_cpu_mem\n");
}
void Wavefield::set_data() 
{
    printf("Start Wavefield::set_data\n");
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
    printf("End Wavefield::set_data\n");
}

void Wavefield::copy_host2dev() 
{
    printf("Start Wavefield::copy_host2dev\n");
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
        printf("CPU Wavefield data are copied to GPU\n");
    }
    else
    {
        printf("ERROR: Either CPU or GPU Memory has not been allocated!\n");
    }
    printf("End Wavefield::copy_host2dev\n");
} 

void Wavefield::copy_dev2host() 
{
    printf("Start Wavefield::copy_dev2host\n");
    size_t nxyz = (size_t)nx_ * ny_ * nz_;
    size_t memSize = nxyz*sizeof(float);
    if(cpuInitialized&&gpuInitialized)
    {
        checkCudaErrors(cudaMemcpy(w0, d_w0, memSize,cudaMemcpyDefault));
        checkCudaErrors(cudaMemcpy(w1, d_w1, memSize,cudaMemcpyDefault));
        checkCudaErrors(cudaMemcpy(wb, d_wb, memSize,cudaMemcpyDefault));
        checkCudaErrors(cudaMemcpy(wx, d_wx, memSize,cudaMemcpyDefault));
        checkCudaErrors(cudaMemcpy(wy, d_wy, memSize,cudaMemcpyDefault));
        checkCudaErrors(cudaMemcpy(wz, d_wz, memSize,cudaMemcpyDefault));
        checkCudaErrors(cudaMemcpy(wr, d_wr, memSize,cudaMemcpyDefault));
    }
    else
    {
        printf("ERROR: Either CPU or GPU Memory has not been allocated!\n");
    }
    printf("End Wavefield::copy_dev2host\n");
} 

int Wavefield::compare_host_dev() 
{
    printf("Start Wavefield::compare_host_dev\n");

    if( compare_data_3d(w0,d_w0,nx_,ny_,nz_)+
        compare_data_3d(w1,d_w1,nx_,ny_,nz_)+
        compare_data_3d(wb,d_wb,nx_,ny_,nz_)+
        compare_data_3d(wx,d_wx,nx_,ny_,nz_)+
        compare_data_3d(wy,d_wy,nx_,ny_,nz_)+
        compare_data_3d(wz,d_wz,nx_,ny_,nz_)+
        compare_data_3d(wr,d_wr,nx_,ny_,nz_))
    {   
        printf("ERROR: Found Mismatch\n");
        return 1;
    }

    printf("Successful CPU GPU data comparison\n");
    printf("End Wavefield::compare_host_dev\n");
    return 0;

} 

