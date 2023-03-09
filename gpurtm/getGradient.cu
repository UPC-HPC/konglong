#include "cuda_runtime.h"
#include <stdio.h>



__global__ void transposeGpu(int nx, int ny, int dimIn, int dimOut, float* in, float* out) {

	

	return;
}



__global__ void applyGPU(float* pwav, int n, int n1, int n2, int n3, int nxz, int nxbnd1, int nxbnd2, float* coef, float* pmlBuf1, float* pmlBuf2) {


	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = i / (n2 * n3);
	int k = (i % n1) / n3;
	int l = i % n3;

	if (nxbnd1 > 1) {

		float* p = pwav + j * nxz + k * n;
		float* q = pmlBuf1 + j * n2 * nxbnd1 + k * n3;
		float g = coef[l] * 1.0f;
		float qi = (1.0f - g) / (1.0f + g) * q[l] - g * 2.0f / (1.0f - g * g) * p[l];
		q[l] = qi;
		p[l] = qi + p[l] / (1.0f - g);

	}

	if (nxbnd2 > 1) {

		float* p = pwav + j * nxz + n - n3 + k * n;
		float* q = pmlBuf2 + j * n2 * nxbnd2 + k * n3;
		float g = coef[l] * 1.0f;
		float qi = (1.0f - g) / (1.0f + g) * q[l] - g * 2.0f / (1.0f - g * g) * p[l];
		q[l] = qi;
		p[l] = qi + p[l] / (1.0f - g);

	}

}


__global__ void destretchzGPU(float* volapply, float* jacobz, int nxyz) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < nxyz) {
		volapply[i] *= jacobz[i];
	}
}

__global__ void dePyramidyGPU(float* outy, float* outz, float slopey, float* jacoby, int nx, int ny, int nz, int nxyz) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < nxyz) {
		int iy = i / (nx * nz);
		int iz = i % nz;
		float scaly = -slopey * (iy - 0.5f * ny);
		float ddi = outy[i];
		float ddo = outz[i];
		float jji = jacoby[iz];
		ddi *= jji;
		outz[i] = ddo + scaly * ddi;
		outy[i] = ddi;
	}
}

__global__ void dePyramidxGPU(float* outy, float* outz, float slopex, float* jacobx, int nx, int ny, int nz, int nxyz) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < nxyz) {
		int iy = i / (nx * nz);
		int ix = i / nz % nx;
		float scalx = -slopex * (ix - 0.5f * nx);
		float ddi = outy[i];
		float ddo = outz[i];
		float jji = jacobx[ix];
		ddi *= jji;
		outz[i] = ddo + scalx * ddi;
		outy[i] = ddi;
	}
}



void Derivative::getGradient(Wavefield* myWavefield) {

	float* wx = myWavefield->wx;
	float* wy = myWavefield->wy;
	float* wz = myWavefield->wz;
	float* d_wx;
	float* d_wy;
	float* d_wz;
	int nxyz = nx * ny * nz;
	int nyznpml = ny * nz * npml;
	int nxznpml = nx * nz * npml;
	int nxynpml = nx * ny * npml;
	int nBytes = nxyz * sizeof(float);
	int BLOCK_SIZE = 32;

	if (nx > 1) {
		this->dx1(myWavefield->w1, myWavefield->wx, 1);
		if (bnd) {

			cudamalloc((float**)&d_wx, nbytes);
			cudamalloc((float**)&d_trwx, nbytes);
			cudamemcpy(d_wx, wx, nbytes, cudamemcpyhosttodevice);

			//transpose



			dim3 block1(128);
			dim3 grid1((nyznpml + block1.x - 1) / block1.x);

			applyGpu << < grid1, block1 >> > (d_trwx, n, ny, nz, npml, nxz, nxbnd1, nxbnd2, coef, pmlBuf[X][TOP][ROUND1][XX], pmlBuf[X][BOT][ROUND1][XX]);


			//untranspose

		}


	}

	if (ny > 1) {
		this->dy1(myWavefield->w1, myWavefield->wy, 1);
		if (bnd) {

			cudamalloc((float**)&d_wy, nbytes);
			cudamalloc((float**)&d_trwy, nbytes);
			cudamemcpy(d_wy, wy, nbytes, cudamemcpyhosttodevice);

			//transpose



			dim3 block1(128);
			dim3 grid1((nxznpml + block1.x - 1) / block1.x);

			applyGpu << < grid1, block1 >> > (d_trwy, n, nx, nz, npml, nyz, nxbnd1, nxbnd2, coef, pmlBuf[Y][TOP][ROUND1][YY], pmlBuf[Y][TOP][ROUND1][YY]);


			//untranspose

		}
	}

	this->dz1(myWavefield->w1, myWavefield->wz, 1);
	if (bnd) {

		cudamalloc((float**)&d_wz, nbytes);
		cudamemcpy(d_wz, wz, nbytes, cudamemcpyhosttodevice);


		dim3 block1(128);
		dim3 grid1((nxynpml + block1.x - 1) / block1.x);

		applyGpu << < grid1, block1 >> > (d_wz, n, ny, nx, npml, nxz, nxbnd1, nxbnd2, coef, pmlBuf[Z][TOP][ROUND1][ZZ], pmlBuf[Z][BOT][ROUND1][ZZ]);


		cudamemcpy(wz, d_wz, nbytes, cudamemcpydevicetohost);
		cudafree(d_wz);
	}


	if (gridType != RECTANGLE) {

		cudaMalloc((float**)&d_wz, nBytes);
		cudaMemcpy(d_wz, wz, nBytes, cudaMemcpyHostToDevice);
		dim3 block(128);
		dim3 grid((nxyz + block.x - 1) / block.x);
		destretchzGPU << <grid, block >> > (d_wz, jacobz, nxyz);
		cudaMemcpy(wz, d_wz, nBytes, cudaMemcpyDeviceToHost);
		cudaFree(d_wz);



		if ((gridType == YPYRAMID) || (gridType == XYPYRAMID)) {

			cudaMalloc((float**)&d_wy, nBytes);
			cudaMalloc((float**)&d_wz, nBytes);
			cudaMemcpy(d_wy, wy, nBytes, cudaMemcpyHostToDevice);
			cudaMemcpy(d_wz, wz, nBytes, cudaMemcpyHostToDevice);

			dim3 block(128);
			dim3 grid((nxyz + block.x - 1) / block.x);

			dePyramidyGPU << < grid, block >> > (d_wy, d_wz, slopey, jacoby, nx, ny, nz, nxyz);

			cudaMemcpy(wy, d_wy, nBytes, cudaMemcpyDeviceToHost);
			cudaMemcpy(wz, d_wz, nBytes, cudaMemcpyDeviceToHost);
			cudaFree(d_wy);
			cudaFree(d_wz);
		}

		if ((gridType == XPYRAMID) || (gridType == XYPYRAMID)) {

			cudaMalloc((float**)&d_wx, nBytes);
			cudaMalloc((float**)&d_wz, nBytes);
			cudaMemcpy(d_wx, wx, nBytes, cudaMemcpyHostToDevice);
			cudaMemcpy(d_wz, wz, nBytes, cudaMemcpyHostToDevice);

			dim3 block(128);
			dim3 grid((nxyz + block.x - 1) / block.x);

			dePyramidyGPU << < grid, block >> > (d_wx, d_wz, slopex, jacobx, nx, ny, nz, nxyz);

			cudaMemcpy(wx, d_wx, nBytes, cudaMemcpyDeviceToHost);
			cudaMemcpy(wz, d_wz, nBytes, cudaMemcpyDeviceToHost);
			cudaFree(d_wx);
			cudaFree(d_wz);
		}
	}
}