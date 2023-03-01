#include "cuda_runtime.h"
#include <stdio.h>




__global__ void destretchzGPU(float* volapply, float* jacobz) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    //float*  vola = volapply + i;
    volapply[i] *= jacobz[i];

}







void Derivative::getGradient(Wavefield* myWavefield) {
    if (nx > 1) {
        this->dx1(myWavefield->w1, myWavefield->wx, 1);
        if (bnd) bnd->applyX(myWavefield->wx, 1);
    }

    if (ny > 1) {
        this->dy1(myWavefield->w1, myWavefield->wy, 1);
        if (bnd) bnd->applyY(myWavefield->wy, 1);
    }

    this->dz1(myWavefield->w1, myWavefield->wz, 1);
    if (bnd) bnd->applyZ(myWavefield->wz, 1);










    if (gridType != RECTANGLE) {

        
        float* wz = myWavefield->wz;
        float* d_wz;
        float* d_jacobz = jacobz;
        int nxyz = nx * ny * nz;
        int nBytes = nxyz * sizeof(float);

        cudaMalloc((float**)&d_wz, nBytes);
        cudaMemcpy(d_wz, wz, nBytes, cudaMemcpyHostToDevice);

        dim3 block(128);
        dim3 grid((nxyz + block.x - 1) % block.x);

        destretchzGPU << <grid, block >> > (d_wz, d_jacobz);
        











        if ((gridType == YPYRAMID) || (gridType == XYPYRAMID)) {
            dePyramidy(myWavefield->wy, myWavefield->wz);
        }

        if ((gridType == XPYRAMID) || (gridType == XYPYRAMID)) {
            dePyramidx(myWavefield->wx, myWavefield->wz);
        }
    }
}