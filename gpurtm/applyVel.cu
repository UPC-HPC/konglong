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
void checkResult(const float *cpu_wb,const float *gpu_wb,const int size){
    double epsilon=1.0E-8;
    for(int i=0;i<size;i++){
        if(abs(cpu_wb[i]-gpu_wb[i]>epsilon)){
            printf("Arrays do not match!\n");
            printf("cpu %5.2f gpu %5.2f at current %d\n",cpu_wb[i],gpu_wb[i],i);
            break;
        }
    }
}


int main(){
    printf("0\n");
    fflush(0);
    //int nx = myGrid->nx, ny = myGrid->ny, nz = myGrid->nz;
    int nx = 1000, ny = 1000, nz =1000;
    int size=nx*ny*nz;
    size_t nBytes=size*sizeof(float);
    printf("00\n");
    fflush(0);
    //float *__restrict vel = volModel[VEL];
    float  *vel=(float*)malloc(nBytes);
    for(int i=0;i<size;i++){
        vel[i]=1.0f*i;
    }
    printf("000\n");
    fflush(0);
    float *wb = (float *)malloc(nBytes);
    float *gpu_wb = (float *)malloc(nBytes);
    float *cpu_wb=(float *)malloc(nBytes);

    memset(wb,0,nBytes);
    memset(gpu_wb,0,nBytes);
    memset(cpu_wb,0,nBytes);
    printf("-0001\n");
    fflush(0);
    for(int i=0; i<size;i++)
    {
        wb[i]=0.1f*i+0.5f;
    }
    printf("-3000\n");
    fflush(0);
    float *d_wb;
    float *d_vel;

    cudaMalloc((void **)&d_wb,nBytes);
    cudaMalloc((void **)&d_vel,nBytes);
    printf("-2000\n");
    fflush(0);
    cudaMemcpy(d_vel, vel, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_wb, wb, nBytes, cudaMemcpyHostToDevice);
    printf("0001\n");
    fflush(0);
    
     
    int dimx = 128;
    dim3 block(dimx);
    dim3 grid((size+block.x-1)/block.x);
    printf("0002\n");
    fflush(0);
    applyVel<<<grid,block>>>(d_wb,d_vel,size);

    cudaMemcpy(gpu_wb,d_wb,nBytes,cudaMemcpyDeviceToHost);
    for(int i=0;i<size;i++){
        cpu_wb[i]=wb[i]*vel[i];
    }
    printf("0003\n");
    fflush(0);
    checkResult(cpu_wb,gpu_wb,size);
    printf("0004\n");
    fflush(0);
    cudaFree(d_wb);
    cudaFree(d_vel);
    printf("0005\n");
    fflush(0);
    free(wb);
    free(cpu_wb);
    free(gpu_wb);

}
