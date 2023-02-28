#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <conio.h>
#include <xmmintrin.h>
#include <stddef.h>

_global_ void applyVel(float* wb,const float *vel,const int nxyz){
    int ix = threadIdx.x+blockIdx.x*blockDim.x;
    if(ix < nxyz)
        wb[ix]*=vel[i];
}

void checkResult(float *cpu_wb,float *gpu_wb,const int size){
    double epsilon=1.0E-8;
    bool match=1;
    for(int i=0;i<size;i++){
        if(abs(cpu_wb[i]-gpu_wb[i]>epsilon)){
            match=0;
            printf("Arrays do not match!\n")
            printf("cpu %5.2f gpu %5.2f at current %d\n",cpu_wb[i],gpu_wb[i],i);
        }
    }
}


int main(){
    int nx = myGrid->nx, ny = myGrid->ny, nz = myGrid->nz;
    
    size_t size=nx*ny*nz;
    size_t nBytes=size*sizeof(float);

    float *__restrict vel = volModel[VEL];

    float *wb = (float*) malloc(nBytes);
    float *gpu_wb = (float*) malloc(nBytes);
    float *cpu_wb=(float*) malloc(nBytes);

    for(int i=0; i<size;i++)
    {
        wb[i]=0.1f*i+0.5f;
    }

    float *d_wb;
    float *d_vel;
    cudaMalloc((void **) &d_wb, nBytes);
    cudaMalloc((void **) &d_vel, nBytes);

    cudaMemcpy(d_vel, vel, nBytes, cudaMemcpyDefault);
    cudaMemcpy(d_wb, wb, nBytes, cudaMemcpyDefault);
    
    
     
    int dimx = 128;
    dim3 block(dimx);
    dim3 grid((size+block.x-1)/block.x);

    applyVel<<<grid,block>>>(d_wb,d_vel,size);

    cudaMemcpy(gpu_wb,d_wb,nBytes,cudaMemcpyDeviceToHost);
    for(i=0;i<size;i++)
    cpu_wb[i]=wb[i]*vel[i];

    checkResult(cpu_wb,gpu_wb,size);

    cudaFree(d_wb);
    cudaFree(d_vel);

    free(wb);
    free(cpu_wb);
    free(gpu_wb);

}