#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <xmmintrin.h>
#include <stddef.h>

__global__ void update2nd1(float *w0,const float *w1,const int nxyz){
    int ix = threadIdx.x+blockIdx.x*blockDim.x;
    if(ix < nxyz)
        w0[ix]=w1[ix]+w1[ix]-w0[ix];
}
__global__ void update2nd2(const float *invQ,
                           float *w0,
                           const float *w1,
                           float *d0,
                           const float *d1,
                           const float *cl,
                           const float *el,
                           const float cqsum,
                           const int order,
                           const int nxyz){
    int ix = threadIdx.x+blockIdx.x*blockDim.x;

    if(ix<nxyz){
        w0[ix] = w1[ix] * (2 + invQ[ix] * cqsum) - w0[ix];
        float c,e;
        for(int l=0;l<order;l++){
            c=cl[l],e=el[l];
            w0[ix]+=invQ[ix]*((1+e)*d0[ix]-d1[ix]*2);
            d0[ix]=c*w1[ix]+e*d0[ix];
        }
    }
}
int main(){
    //int nx = myGrid->nx, ny = myGrid->ny, nz = myGrid->nz;
    int nx=1000,ny=1000,nz=1000;
    int size = nx * ny * nz;
    size_t nBytes=size*sizeof(float);
    
    if(!myModel->useQ)
    {
        float *w0=myLocalWavefield->w0;
        float *w1=myLocalWavefield->w1;
        
        float *d_w0;
        float *d_w1;

        cudaMalloc((void **)&d_w0,nBytes);
        cudaMalloc((void **)&d_w1,nBytes);
        
        cudaMemcpy(d_w0, w0, nBytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_w1, w1, nBytes, cudaMemcpyHostToDevice);
        
        int dimx=128;
        dim3 block(dimx);
        dim3 grid((size+block.x-1)/block.x);
        
        update2nd1<<<grid,block>>>(d_w0,d_w1,size);
        
        cudaMemcpy(w0,d_w0,nBytes,cudaMemcpyDeviceToHost);
        
        
        cudaFree(d_w0);
        cudaFree(d_w1);
    }
    else
    {//报错 w0：cpu 13011726.00000000 gpu 13011725.00000000 at current 15254
        float *invQ = volModel[Q];
        float *w0=myLocalWavefield->w0;
        float *w1=myLocalWavefield->w1;
        float cqsum=Q::cqsum;
        int   order=Q::order;
        float *d0 = myWavefield->wq[myWavefield->iq0][0];
        float *d1 = myWavefield->wq[myWavefield->iq1][0];
        float *cl = Q::cq,*el = Q::wq;
        
        float *d_invQ;
        float *d_w0;
        float *d_w1;
        float *d_d0;
        float *d_d1;
        float *d_cl;
        float *d_el;

        float orderBytes=order*sizeof(float);
        
        cudaMalloc((void **)&d_invQ,nBytes);
        cudaMalloc((void **)&d_w0,nBytes);
        cudaMalloc((void **)&d_w1,nBytes);
        cudaMalloc((void **)&d_d0,nBytes);
        cudaMalloc((void **)&d_d1,nBytes);
        cudaMalloc((void **)&d_cl,orderBytes);
        cudaMalloc((void **)&d_el,orderBytes);
        
        cudaMemcpy(d_invQ, invQ, nBytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_w0, w0, nBytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_w1, w1, nBytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_d0, d0, nBytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_d1, d1, nBytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_cl, cl, orderBytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_el, el, orderBytes, cudaMemcpyHostToDevice);
        
        int dimx=128;
        dim3 block(dimx);
        dim3 grid((size+block.x-1)/block.x);

        update2nd2<<<grid,block>>>(d_invQ,d_w0,d_w1,d_d0,d_d1,d_cl,d_el,cqsum,order,size);
    
        
        cudaMemcpy(w0,d_w0,nBytes,cudaMemcpyDeviceToHost);
        cudaMemcpy(d0,d_d0,nBytes,cudaMemcpyDeviceToHost);



        cudaFree(d_invQ);
        cudaFree(d_w0);
        cudaFree(d_w1);
        cudaFree(d_d0);
        cudaFree(d_d1);
        cudaFree(d_cl);
        cudaFree(d_el);
    }
}