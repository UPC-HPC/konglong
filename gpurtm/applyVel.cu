#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <xmmintrin.h>
#include <stddef.h>

__global__ void applyVel(float *wb,const float *vel,const int nxyz){
    int ix = threadIdx.x+blockIdx.x*blockDim.x;
    if(ix < nxyz)
        wb[ix]*=vel[ix];
}


int main(){
    int nx = myGrid->nx, ny = myGrid->ny, nz = myGrid->nz;
    int size=nx*ny*nz;
    size_t nBytes=size*sizeof(float);

    float *__restrict vel = volModel[VEL];

    float *mywb=myLocalWavefield->wb;

    float *d_wb;
    float *d_vel;

    cudaMalloc((void **)&d_wb,nBytes);
    cudaMalloc((void **)&d_vel,nBytes);

    cudaMemcpy(d_vel, vel, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_wb, wb, nBytes, cudaMemcpyHostToDevice);

     
    int dimx = 128;
    dim3 block(dimx);
    dim3 grid((size+block.x-1)/block.x);

    applyVel<<<grid,block>>>(d_wb,d_vel,size);

    cudaMemcpy(mywb,d_wb,nBytes,cudaMemcpyDeviceToHost);

    cudaFree(d_wb);
    cudaFree(d_vel);

    free(wb);
    free(cpu_wb);

}
