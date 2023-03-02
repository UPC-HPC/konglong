#ifndef VOLUME_FILTER_H
#define VOLUME_FILTER_H

#include <omp.h>
#include <emmintrin.h>

#ifdef  __cplusplus
extern "C" {
#endif

void avgVolume(float *a, float *b, int nny, int nxz, int sizea);
void avgVolume2(float *a, float *b, int nny, int nxz, int sizea);
void a3dVolume(float *a, float *b, int nx, int ny, int nz, int sizea);
void avgVolumeXY(float *a, float *b, int nx, int ny, int nz, int sizea);
void a3dVolume3(float *a, float *b, int nx, int ny, int nz, int sizex, int sizey, int sizez);

void avgVolume3D(float *a, int nx, int ny, int nz, int sizex, int sizey, int sizez);
void avgVolume2D(float *a, int nx, int nz, int sizex, int sizez);
void avgVolume1D(float *a, float *b, int n1, int n2, int sizeh);

void mdmVolume(float *a, float *b, int nny, int nxz, int sizea);
void m3dVolume(float *a, float *b, int nx, int ny, int nz, int sizea);
void m33Volume(float *a, float *b, int nny, int nnz); // only for 3 point

void aa2dVolume2(float *a, float *b, int nx, int ny);
void am2dVolume3(float *a, float *b, int nx, int ny);
void aa2dVolume3(float *a, float *b, int nx, int ny);

// 3 point operation
void average_2d(float *a, float *b, int nx, int nz);
void medium3_2d(float *a, float *b, int nx, int nz);
void average_sl(float *a1, float *a2, float *a3, float *b, int nxz);  // three slice for one output
void medium3_sl(float *a1, float *a2, float *a3, float *b, int nxz);
void average_s5(float *a1, float *a2, float *a3, float *a4, float *a5, float *b, int nxz);


#ifdef  __cplusplus
}
#endif


#endif

