#include<stdio.h>
#include<stdlib.h>
#include<conio.h>
#include<xmmintrin.h>
#include<stddef.h>
_gloval_ void applyVel(Wavefield *myLocalWavefield,int nx,int ny,int nz){
    int ix = threadIdx.x+blockIdx.x*blockDim.x
    int iy = threadIdx.y+blockIdx.y*blockDim.y
    int iz = threadIdx.z+blockIdx.z*blockDim.z
    size_t ixz = ix * nz + iz;
    size_t nxz = nx * nz;
    if(iy < ny && ixz < nxz){
        size_t i = iy * nxz+ixz;
        float v2 = vel[i]；
        float vectx = myLocalWavefield->wb[i]
        myLocalWavefield->wb[i] = v2 * vectx;
    }
}



int main(){
    int nx = myGrid->nx, ny = myGrid->ny, nz = myGrid->nz;
    
    int size=nx*ny*nz;

    float *__restrict h_vel = volModel[VEL];
    float *__restrict d_vel = NULL;
    

    cudaMemcpy(d_vel,h_vel,nBytes,cudaMemcpyHostToDevice);
    cudaMemcpy(d_myLocalWavefield,myLocalWavefield,Wavefield,cudaMemcpyHostToDevice)
    cudaDeviceSynchronize();


    int dimx = 128;
    int dimy = 128;
    int dimz = 128;
    dim3 block(dimx,dimy,dimz);
    dim3 grid((nx+block.x-1)/block.x,(ny+block.y-1)/block.y,(nz+block.z-1)/block.z)

    applyVel<<<grid,block>>>(d_myLocalWavefield,d_vel,nx,ny,nz);
    cudaDeviceSynchronize();
    cudaMemcpy(myLocalWavefield,d_myLocalWavefield,cudaMemcpyDeviceToHost)
}