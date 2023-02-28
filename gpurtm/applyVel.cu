#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <conio.h>
#include <xmmintrin.h>
#include <stddef.h>

_global_ void applyVel(float* wb,const float *vel,const int nxyz){
    int ix = threadIdx.x+blockIdx.x*blockDim.x;
    //int iy = threadIdx.y+blockIdx.y*blockDim.y;
    //int iz = threadIdx.z+blockIdx.z*blockDim.z;
    //size_t ixz = ix * nz + iz;
    //size_t nxz = nx * nz;
    if(ix < nxyz){
        wb[ix]*=vel[i];
       // size_t i = iy * nxz+ixz;
      //  float v2 = vel[i];
       // float vectx = myLocalWavefield->wb[i];
       // myLocalWavefield->wb[i] = v2 * vectx;
    }
}



int main(){
    int nx = myGrid->nx, ny = myGrid->ny, nz = myGrid->nz;
    
    int size=nx*ny*nz;

    float *__restrict h_vel = volModel[VEL];
    float *__restrict d_vel = NULL;
    
    float *wb = (float*) malloc(size*sizeof(float));

    for(int i=0; i<size;i++)
    {
        wb[i]=0.1f*i+0.5f;
    }

    float *d_wb;
    cudaMalloc((void **)&d_wb, size*sizeof(float));

    cudaMemcpy(d_vel,h_vel,nBytes,cudaMemcpyDefault);
    cudaMemcpy(d_wb,wb,size*sizeof(float),cudaMemcpyHostToDevice)
    


    int dimx = 128;
    dim3 block(dimx);
    dim3 grid((size+block.x-1)/block.x);

    applyVel<<<grid,block>>>(d_wb,d_vel,size);

    cudaMemcpy(myLocalWavefield,d_myLocalWavefield,Wavefield,cudaMemcpyDeviceToHost)
}