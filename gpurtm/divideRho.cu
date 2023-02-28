#include"cuda_runtime.h"
#include "device_launch_parameters.h"
#include<stdio.h>
using namespace std;

__global__ void divideRhoOnGPU(float* rho, float* wx, float* wy, float* wz, int nx, int ny, int nxyz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nxyz)
    {
        if (nx > 1) wx[i] /= rho[i];
        if (ny > 1) wy[i] /= rho[i];
        wz[i] /= rho[i];
    }
}


void Propagator::divideRho(Wavefield* wf){ // ATTN: check the vectorization
    int nx = myGrid->nx, ny = myGrid->ny, nz = myGrid->nz;

    float* __restrict rho = volModel[RHO];
    float* __restrict wx = wf->wx;
    float* __restrict wy = wf->wy;
    float* __restrict wz = wf->wz;

    //malloc device global memory
    float* d_rho, * d_wx, * d_wy, * d_wz;
    int nxyz = nx * ny * nz;
    int nBytes = nxyz * sizeof(float);
    
    cudaMalloc((float**)&d_rho, nBytes);
    cudaMalloc((float**)&d_wx, nBytes);
    cudaMalloc((float**)&d_wy, nBytes);
    cudaMalloc((float**)&d_wz, nBytes);

    //transfer data from host to device
    cudaMemcpy(d_rho, rho, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_wx, wx, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_wy, wy, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_wz, wz, nBytes, cudaMemcpyHostToDevice);

    //invoke kernel at host side
    dim3 block(128,1,1);
    dim3 grid(nxyz % 128 = 0 ? nxyz / 128 : nxyz / 128 + 1, 1, 1);

    divideRhoOnGPU << <grid, block >> > (d_rho, d_wx, d_wy, d_wz, nx, ny, nxyz);
    
    //copy kernel result back to host side
    cudaMemcpy(rho, d_rho, nBytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(wx, d_wx, nBytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(wy, d_wy, nBytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(wz, d_wz, nBytes, cudaMemcpyDeviceToHost);

    //free device global memory
    cudaFree(d_rho);
    cudaFree(d_wx);
    cudaFree(d_wy);
    cudaFree(d_wz);
}