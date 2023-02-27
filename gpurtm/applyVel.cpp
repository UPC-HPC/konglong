#include<stdio.h>
#include<stdlib.h>
#include<conio.h>
#include<xmmintrin.h>
#include<stddef.h>
_gloval_ void applyVel(Wavefield *myLocalWavefield,int nx,int ny,int nz){
  int nx = myGrid->nx, ny = myGrid->ny, nz = myGrid->nz;
  float *__restrict vel = volModel[VEL];
  size_t nxz = nx * nz;
#pragma omp parallel for num_threads(nThreads) schedule(static)
  for(int iy = 0; iy < ny; iy++) {
    for(size_t ixz = 0; ixz < nxz; ixz += SSEsize) {
      size_t i = iy * nxz + ixz;
      __m128 v2 = _mm_load_ps(vel + i);
      __m128 vectx = _mm_load_ps(myLocalWavefield->wb + i);
      _mm_store_ps(myLocalWavefield->wb + i, _mm_mul_ps(vectx, v2));
    }
  }
}



int main(){
    int nx = myGrid->nx, ny = myGrid->ny, nz = myGrid->nz;
    
    int size=nx*ny*nz;

    float *__restrict h_vel = volModel[VEL];
    float *__restrict d_vel = NULL;
    
    cudaMemcpy(d_vel,h_vel,bytes,cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();


    int dimx = 128;
    int dimy = 128;
    int dimz = 128;
    dim3 block(dimx,dimy,dimz);
    dim3 grid((nx+block.x-1)/block.x,(ny+block.y-1)/block.y,(nz+block.z-1)/block.z)

    applyVel<<<grid,block>>>(myLocalWavefield,nx,ny,nz);
}