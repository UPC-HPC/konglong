#include "volumefilter.h"
#include "libFFTV/transpose.h"
#include "Util.h"

void a3dVolume(float *a, float *b, int nx, int ny, int nz, int sizea) {
  int nxz = nx * nz;
  avgVolume2(a, b, ny, nxz, sizea);   // do it for y direction
  int nThreads = omp_get_max_threads();
#pragma omp parallel for num_threads(nThreads) schedule(static)
  for(int iy = 0; iy < ny; iy++) {
    float *aa = b + iy * nxz;
    float *buffer0 = new float[2 * nxz + 128];
    float *buffer1 = buffer0 + nxz;
    avgVolume(aa, buffer0, nx, nz, sizea);
    //transpose(buffer0, nz, nx);
    //avgVolume(buffer0, buffer1, nz, nx, sizea);
    //transpose(buffer1, nx, nz);
    //memcpy(aa, buffer1, nxz*sizeof(float));
    //memcpy(aa, buffer0, nxz*sizeof(float));
    libfftv::ssetranspose(nz, nx, buffer0, nz, buffer1, nx);
    avgVolume(buffer1, buffer0, nz, nx, sizea);
    libfftv::ssetranspose(nx, nz, buffer0, nx, aa, nz);
    delete[] buffer0;
  }
}

void a3dVolume3(float *a, float *b, int nx, int ny, int nz, int sizex, int sizey, int sizez) {
  int nxz = nx * nz;
  avgVolume2(a, b, ny, nxz, sizey);   // do it for y direction
  int nThreads = omp_get_max_threads();
#pragma omp parallel for num_threads(nThreads) schedule(static)
  for(int iy = 0; iy < ny; iy++) {
    float *buffer0 = new float[2 * nxz + 128];
    float *buffer1 = buffer0 + nxz;
    avgVolume(b + iy * nxz, buffer0, nx, nz, sizex);            // do it for X direction
    libfftv::ssetranspose(nz, nx, buffer0, nz, buffer1, nx);
    avgVolume(buffer1, buffer0, nz, nx, sizez);           // do it for Z direction
    libfftv::ssetranspose(nx, nz, buffer0, nx, a + iy * nxz, nz);
    delete[] buffer0;
  }
}

void avgVolumeXY(float *a, float *b, int nx, int ny, int nz, int sizea) {
  size_t nxz = (size_t)nx * (size_t)nz;
  avgVolume2(a, b, ny, nxz, sizea);   // do it for y direction
  int nThreads = omp_get_max_threads();
#pragma omp parallel for num_threads(nThreads) schedule(static)
  for(int iy = 0; iy < ny; iy++) {
    float *buffer0 = new float[2 * nxz + 128];
    float *buffer1 = buffer0 + nxz;
    avgVolume(b + iy * nxz, buffer0, nx, nz, sizea);
    libfftv::ssetranspose(nz, nx, buffer0, nz, buffer1, nx);
    m33Volume(buffer1, buffer0, nz, nx);
    libfftv::ssetranspose(nx, nz, buffer0, nx, a + iy * nxz, nz);
    delete[] buffer0;
  }
}

void avgVolume3D(float *a, int nx, int ny, int nz, int sizex, int sizey, int sizez) {

  size_t nxz = (size_t)nx * (size_t)nz;
  size_t nyz = (size_t)ny * (size_t)nz;
  size_t bufSize = max(nxz, nyz);

  int nThreads = omp_get_max_threads();
#pragma omp parallel num_threads(nThreads)
  {

    float *buffer0 = new float[2 * bufSize + 128];
    float *buffer1 = buffer0 + bufSize;

#pragma omp for schedule(static)
    for(int ix = 0; ix < nx; ix++) {
      for(int iy = 0; iy < ny; iy++) {
        size_t ioffset = (size_t)(iy * nx + ix) * (size_t)nz;
        memcpy(buffer0 + iy * nz, a + ioffset, nz * sizeof(float));
      }
      avgVolume1D(buffer0, buffer1, ny, nz, sizey);
      for(int iy = 0; iy < ny; iy++) {
        size_t ioffset = (size_t)(iy * nx + ix) * (size_t)nz;
        memcpy(a + ioffset, buffer1 + iy * nz, nz * sizeof(float));
      }
    }

#pragma omp for schedule(static)
    for(int iy = 0; iy < ny; iy++) {
      avgVolume1D(a + iy * nxz, buffer0, nx, nz, sizex);
      libfftv::ssetranspose(nz, nx, buffer0, nz, buffer1, nx);
      avgVolume1D(buffer1, buffer0, nz, nx, sizez);
      libfftv::ssetranspose(nx, nz, buffer0, nx, a + iy * nxz, nz);
    }

    delete[] buffer0;
  }
}

void avgVolume2D(float *a, int nx, int nz, int sizex, int sizez) {

  size_t bufSize = (size_t)nx * (size_t)nz;
  ;

  float *buffer0 = new float[2 * bufSize + 128];
  float *buffer1 = buffer0 + bufSize;

  avgVolume1D(a, buffer0, nx, nz, sizex);
  libfftv::ssetranspose(nz, nx, buffer0, nz, buffer1, nx);
  avgVolume1D(buffer1, buffer0, nz, nx, sizez);
  libfftv::ssetranspose(nx, nz, buffer0, nx, a, nz);

  delete[] buffer0;
}

void avgVolume1D(float *a, float *b, int n1, int n2, int sizeh) {

  float *bufm = new float[n2];
  float scaler = 1.0f / (2.0f * sizeh + 1.0);
  float sizea1 = sizeh + 1.0f;

  for(ssize_t i = 0; i < n2; i++)
    bufm[i] = sizea1 * a[i];
  for(int j = 1; j <= sizeh; j++)
    for(ssize_t i = 0; i < n2; i++)
      bufm[i] += a[j * n2 + i];
  for(ssize_t i = 0; i < n2; i++)
    b[i] = bufm[i] * scaler;

  for(int k = 1; k < n1; k++) {
    int k1 = max(0, (k - sizeh - 1));
    int k2 = min((n1 - 1), (k + sizeh));
    for(ssize_t i = 0; i < n2; i++)
      bufm[i] += (a[i + k2 * n2] - a[i + k1 * n2]);
    for(ssize_t i = 0; i < n2; i++)
      b[i + k * n2] = bufm[i] * scaler;
  }
  delete[] bufm;
  return;
}

void avgVolume2(float *a, float *b, int nny, int nnz, int sizeh) {
  int nThreads = omp_get_max_threads();
  ssize_t nxz = ssize_t(nnz);
  float *bufm = new float[nxz + 128];
  float scaler = 1.0f / (2.0f * sizeh + 1.0);
  __m128 mscale = _mm_set1_ps(scaler);
  float sizea1 = sizeh + 1.0f;
  //*
  __m128 msizea1 = _mm_set1_ps(sizea1);
  for(ssize_t i = 0; i < nxz; i += SSEsize)
    _mm_store_ps(bufm + i, _mm_mul_ps(_mm_load_ps(a + i), msizea1));
  for(int j = 1; j <= sizeh; j++)
    for(ssize_t i = 0; i < nxz; i += SSEsize)
      _mm_store_ps(bufm + i, _mm_add_ps(_mm_load_ps(bufm + i), _mm_load_ps(a + j * nxz + i)));
  for(ssize_t i = 0; i < nxz; i += SSEsize)
    _mm_store_ps(b + i, _mm_mul_ps(_mm_load_ps(bufm + i), mscale));
  //*/

  //for(size_t i=0; i<nxz;  i++ ) bufm[i] = sizea1*a[i];
  //for(int j=1; j<=sizeh; j++) for(size_t i=0; i<nxz; i++) bufm[i]+= a[j*nxz+i];
  //for(size_t i=0; i<nxz;  i++ ) b[i] = bufm[i]*scaler;
  for(int k = 1; k < nny; k++) {
    int k1 = max(0, (k - sizeh - 1));
    int k2 = min((nny - 1), (k + sizeh));
    //*
    float *__restrict a2 = a + k2 * nxz;
    float *__restrict a1 = a + k1 * nxz;
    float *__restrict bu = bufm;
    float *__restrict bo = b + k * nxz;
#pragma omp parallel for num_threads(nThreads) schedule(static)
    for(ssize_t i = 0; i < nxz; i += SSEsize) {
      __m128 a2m = _mm_load_ps(a2 + i);
      __m128 a1m = _mm_load_ps(a1 + i);
      __m128 bbb = _mm_load_ps(bu + i);
      __m128 bum = _mm_add_ps(bbb, _mm_sub_ps(a2m, a1m));
      _mm_store_ps(bu + i, bum);
      _mm_store_ps(bo + i, _mm_mul_ps(bum, mscale));
    }
    //*/
    //for(size_t i=0; i<nxz; i++) bufm[i] += (a[i+k2*nxz] - a[i+k1*nxz]);
    //for(size_t i=0; i<nxz; i++) b[i+k*nxz] = bufm[i]*scaler;
  }
  delete[] bufm;
  return;
}

void avgVolume(float *a, float *b, int nny, int nnz, int sizeh) {
  size_t nxz = size_t(nnz);
  float *bufm = new float[nxz + 128];
  float scaler = 1.0f / (2.0f * sizeh + 1.0);
  __m128 mscale = _mm_set1_ps(scaler);
  float sizea1 = sizeh + 1.0f;
  //*
  __m128 msizea1 = _mm_set1_ps(sizea1);
  for(size_t i = 0; i < nxz; i += SSEsize)
    _mm_store_ps(bufm + i, _mm_mul_ps(_mm_load_ps(a + i), msizea1));
  for(int j = 1; j <= sizeh; j++)
    for(size_t i = 0; i < nxz; i += SSEsize)
      _mm_store_ps(bufm + i, _mm_add_ps(_mm_load_ps(bufm + i), _mm_load_ps(a + j * nxz + i)));
  for(size_t i = 0; i < nxz; i += SSEsize)
    _mm_store_ps(b + i, _mm_mul_ps(_mm_load_ps(bufm + i), mscale));
  //*/

  //for(size_t i=0; i<nxz;  i++ ) bufm[i] = sizea1*a[i];
  //for(int j=1; j<=sizeh; j++) for(size_t i=0; i<nxz; i++) bufm[i]+= a[j*nxz+i];
  //for(size_t i=0; i<nxz;  i++ ) b[i] = bufm[i]*scaler;
  for(int k = 1; k < nny; k++) {
    int k1 = max(0, (k - sizeh - 1));
    int k2 = min((nny - 1), (k + sizeh));
    //*
    float *__restrict a2 = a + k2 * nxz;
    float *__restrict a1 = a + k1 * nxz;
    float *__restrict bu = bufm;
    float *__restrict bo = b + k * nxz;
    for(size_t i = 0; i < nxz; i += SSEsize) {
      __m128 a2m = _mm_load_ps(a2 + i);
      __m128 a1m = _mm_load_ps(a1 + i);
      __m128 bbb = _mm_load_ps(bu + i);
      __m128 bum = _mm_add_ps(bbb, _mm_sub_ps(a2m, a1m));
      _mm_store_ps(bu + i, bum);
      _mm_store_ps(bo + i, _mm_mul_ps(bum, mscale));
    }
    //*/
    //for(size_t i=0; i<nxz; i++) bufm[i] += (a[i+k2*nxz] - a[i+k1*nxz]);
    //for(size_t i=0; i<nxz; i++) b[i+k*nxz] = bufm[i]*scaler;
  }
  delete[] bufm;
  return;
}

void m3dVolume(float *a, float *b, int nx, int ny, int nz, int sizea) {
  int nxz = nx * nz;
  m33Volume(a, b, ny, nxz);   // do it for y direction
  for(int iy = 0; iy < ny; iy++) {
    float *buffer0 = new float[2 * nxz + 128];
    float *buffer1 = buffer0 + nxz;
    float *aa = b + iy * nxz;
    m33Volume(aa, buffer0, nx, nz);
    libfftv::ssetranspose(nz, nx, buffer0, nz, buffer1, nx);
    m33Volume(buffer1, buffer0, nz, nx);
    libfftv::ssetranspose(nx, nz, buffer0, nx, aa, nz);
    delete[] buffer0;
  }
  //*/
}

void medium3_2d(float *a, float *b, int nx, int nz) {
  memcpy(b, a, nz * sizeof(float));
  memcpy(b + (nx - 1) * nz, a + (nx - 1) * nz, nz * sizeof(float));
  for(int ix = 1; ix < nx - 1; ix++) {
    float *__restrict x0 = a + (ix - 1) * nz;
    float *__restrict y0 = a + ix * nz;
    float *__restrict z0 = a + (ix + 1) * nz;
    float *__restrict ut = b + ix * nz; // output
    for(int i = 0; i < nz; i += SSEsize) {
      __m128 a00 = _mm_load_ps(x0 + i);
      __m128 a01 = _mm_load_ps(y0 + i);
      __m128 a02 = _mm_load_ps(z0 + i);
      __m128 aaa = _mm_add_ps(_mm_add_ps(a00, a01), a02);
      __m128 amn = _mm_min_ps(_mm_min_ps(a00, a01), a02);
      __m128 amx = _mm_max_ps(_mm_max_ps(a00, a01), a02);
      _mm_store_ps(ut + i, _mm_sub_ps(_mm_sub_ps(aaa, amn), amx));
    }
  }
}

void medium3_sl(float *a1, float *a2, float *a3, float *b, int nxz) {
  float *__restrict x0 = a1;
  float *__restrict y0 = a2;
  float *__restrict z0 = a3;
  float *__restrict ut = b;  // output
  for(int i = 0; i < nxz; i += SSEsize) {
    __m128 a00 = _mm_load_ps(x0 + i);
    __m128 a01 = _mm_load_ps(y0 + i);
    __m128 a02 = _mm_load_ps(z0 + i);
    __m128 aaa = _mm_add_ps(_mm_add_ps(a00, a01), a02);
    __m128 amn = _mm_min_ps(_mm_min_ps(a00, a01), a02);
    __m128 amx = _mm_max_ps(_mm_max_ps(a00, a01), a02);
    _mm_store_ps(ut + i, _mm_sub_ps(_mm_sub_ps(aaa, amn), amx));
  }
}

void average_2d(float *a, float *b, int nx, int nz) {
  //__m128 middle  = _mm_set1_ps(1.0/3.0f);
  //__m128 lrside  = _mm_set1_ps(1.0/3.0f);

  __m128 middle = _mm_set1_ps(0.60f);
  __m128 lrside = _mm_set1_ps(0.20f);
  for(int k = 0; k < nx; k++) {
    int k1 = max(0, (k - 1));
    int k2 = min((nx - 1), (k + 1));
    float *__restrict b2 = a + k2 * nz;
    float *__restrict b1 = a + k1 * nz;
    float *__restrict b0 = a + k * nz;
    float *__restrict ao = b + k * nz;
    for(int i = 0; i < nz; i += SSEsize) {
      __m128 a2m = _mm_add_ps(_mm_mul_ps(_mm_add_ps(_mm_load_ps(b2 + i), _mm_load_ps(b1 + i)), lrside),
                              _mm_mul_ps(_mm_load_ps(b0 + i), middle));
      _mm_store_ps(ao + i, a2m);
    }
  }
}

void average_sl(float *a1, float *a2, float *a3, float *b, int nxz) {
  float scaler = 1.0f / 3.0;
  __m128 mscale = _mm_set1_ps(scaler);
  float *__restrict b2 = a1;
  float *__restrict b1 = a2;
  float *__restrict b0 = a3;
  float *__restrict ao = b;
  for(int i = 0; i < nxz; i += SSEsize) {
    __m128 a2m = _mm_add_ps(_mm_add_ps(_mm_load_ps(b2 + i), _mm_load_ps(b1 + i)), _mm_load_ps(b0 + i));
    _mm_store_ps(ao + i, _mm_mul_ps(a2m, mscale));
  }
}
void average_s5(float *a1, float *a2, float *a3, float *a4, float *a5, float *b, int nxz) {
  // 0.6 0.16 0.04; 0.7 0.1275 0.0225

  //float  scaler0  = 1.0f/3.0f;
  //float  scaler1  = 2.0f/9.0f;
  //float  scaler2  = 1.0f/9.0f;
  float scaler0 = 0.60f;
  float scaler1 = 0.16f;
  float scaler2 = 0.04f;
  __m128 mscale0 = _mm_set1_ps(scaler0);
  __m128 mscale1 = _mm_set1_ps(scaler1);
  __m128 mscale2 = _mm_set1_ps(scaler2);
  float *__restrict b1 = a1;
  float *__restrict b2 = a2;
  float *__restrict b3 = a3;
  float *__restrict b4 = a4;
  float *__restrict b5 = a5;
  float *__restrict ao = b;
  for(int i = 0; i < nxz; i += SSEsize) {
    __m128 a2m = _mm_mul_ps(_mm_add_ps(_mm_load_ps(b1 + i), _mm_load_ps(b5 + i)), mscale2);
    __m128 a1m = _mm_mul_ps(_mm_add_ps(_mm_load_ps(b2 + i), _mm_load_ps(b4 + i)), mscale1);
    __m128 a0m = _mm_add_ps(_mm_add_ps(a2m, a1m), _mm_mul_ps(_mm_load_ps(b3 + i), mscale0));
    _mm_store_ps(ao + i, a0m);
  }
}

void am2dVolume3(float *a, float *b, int nx, int nz) {
  libfftv::ssetranspose(nz, nx, a, nz, b, nx);
  medium3_2d(b, a, nz, nx);
  average_2d(a, b, nz, nx);
  libfftv::ssetranspose(nx, nz, b, nx, a, nz);
  medium3_2d(a, b, nx, nz);
  average_2d(b, a, nx, nz);
  return;
}

void aa2dVolume3(float *a, float *b, int nx, int nz) {
  libfftv::ssetranspose(nz, nx, a, nz, b, nx);
  average_2d(b, a, nz, nx);
  average_2d(a, b, nz, nx);
  libfftv::ssetranspose(nx, nz, b, nx, a, nz);
  average_2d(a, b, nx, nz);
  average_2d(b, a, nx, nz);
  return;
}

void aa2dVolume2(float *a, float *b, int nx, int nz) {
  libfftv::ssetranspose(nz, nx, a, nz, b, nx);
  average_2d(b, a, nz, nx);
  libfftv::ssetranspose(nx, nz, a, nx, b, nz);
  average_2d(b, a, nx, nz);
  return;
}

void m33Volume(float *a, float *b, int nny, int nnz) { // only for 3 point
  int nThreads = omp_get_max_threads();
  size_t nxz = size_t(nnz);
  memcpy(b, a, nxz * sizeof(float));
  memcpy(b + (nny - 1) * nxz, a + (nny - 1) * nxz, nxz * sizeof(float));
#pragma omp parallel for num_threads(nThreads) schedule(static)
  for(int iy = 1; iy < nny - 1; iy++) {
    float *__restrict x0 = a + (iy - 1) * nxz;
    float *__restrict y0 = a + iy * nxz;
    float *__restrict z0 = a + (iy + 1) * nxz;
    float *__restrict ut = b + iy * nxz; // output
    //*
    for(size_t i = 0; i < nxz; i += SSEsize) {
      __m128 a00 = _mm_load_ps(x0 + i);
      __m128 a01 = _mm_load_ps(y0 + i);
      __m128 a02 = _mm_load_ps(z0 + i);
      __m128 aaa = _mm_add_ps(_mm_add_ps(a00, a01), a02);
      __m128 amn = _mm_min_ps(_mm_min_ps(a00, a01), a02);
      __m128 amx = _mm_max_ps(_mm_max_ps(a00, a01), a02);
      _mm_store_ps(ut + i, _mm_sub_ps(_mm_sub_ps(aaa, amn), amx));
    }
    //*/
    //for(size_t i=0; i<nxz;i++) ut[i]=x0[i]+y0[i]+z0[i]-max(max(x0[i],y0[i]),z0[i])-min(min(x0[i],y0[i]),z0[i]);
  }
  return;
}

void mdmVolume(float *a, float *b, int nny, int nnz, int sizeh) {
  int lens = 2 * sizeh + 1;
  size_t nxz = size_t(nnz);
  float *bufm = new float[nxz * lens + 128];
  int *indx = new int[nxz * (2 * lens + 1) + 128];
  int *indi = indx + nxz * lens;
  int *idtm = indi + nxz * lens;
  for(int j = 0; j < sizeh; j++)
    for(size_t i = 0; i < nxz; i++)
      bufm[i + j * nxz] = a[i];
  for(int j = 0; j < sizeh; j++)
    for(size_t i = 0; i < nxz; i++)
      indx[i + j * nxz] = j;
  for(int j = 0; j <= sizeh; j++)
    for(size_t i = 0; i < nxz; i++)
      bufm[i + (j + sizeh) * nxz] = a[i + j * nxz];
  for(int j = 0; j <= sizeh; j++)
    for(size_t i = 0; i < nxz; i++)
      indx[i + (j + sizeh) * nxz] = j + sizeh;
  memcpy(indi, indx, nxz * lens * sizeof(int));
  for(int j = 0; j < 2 * sizeh; j++)
    for(int k = 0; k < 2 * sizeh; k++) {
      for(size_t i = 0; i < nxz; i++) {
        if(bufm[i + k * nxz] > bufm[i + (k + 1) * nxz]) {
          float tmp = bufm[i + (k + 1) * nxz];
          bufm[i + (k + 1) * nxz] = bufm[i + k * nxz];
          bufm[i + k * nxz] = tmp;
          int tmi = indx[i + (k + 1) * nxz];
          indx[i + (k + 1) * nxz] = indx[i + k * nxz];
          indx[i + k * nxz] = tmi;
          tmi = indi[i + indx[i + (k + 1) * nxz] * nxz];
          indi[i + indx[i + (k + 1) * nxz] * nxz] = indi[i + indx[i + k * nxz] * nxz];
          indi[i + indx[i + k * nxz] * nxz] = tmi;
        }
      }
    }
  for(size_t i = 0; i < nxz; i++)
    b[i] = bufm[i + sizeh * nxz];
  for(int iy = 1; iy < nny; iy++) {
    int iyg = min((nny - 1), (iy + sizeh));
    int iyr = (iy - 1) % lens;
    for(size_t i = 0; i < nxz; i++)
      idtm[i] = 0;
    for(int j = 0; j < 2 * sizeh + 1; j++)
      for(size_t i = 0; i < nxz; i++)
        if(bufm[i + j * nxz] < a[i + iyg * nxz]) idtm[i]++;

    for(size_t i = 0; i < nxz; i++) {
      int idc = indi[i + iyr * nxz];
      if(idtm[i] == idc) {
        bufm[i + indi[i + iyr * nxz] * nxz] = a[i + iyg * nxz];
      } else if(idtm[i] > idc) {
        for(int k = idc; k < idtm[i] - 1; k++)
          bufm[i + k * nxz] = bufm[i + (k + 1) * nxz]; // move the data
        for(int k = idc; k < idtm[i] - 1; k++)
          indx[i + k * nxz] = indx[i + (k + 1) * nxz];
        for(int k = idc; k < idtm[i] - 1; k++)
          indi[i + indx[i + k * nxz] * nxz] = k;
        bufm[i + (idtm[i] - 1) * nxz] = a[i + iyg * nxz];
        indi[i + iyr * nxz] = (idtm[i] - 1);
        indx[i + (idtm[i] - 1) * nxz] = iyr;
      } else {
        for(int k = idc; k > idtm[i]; k--)
          bufm[i + k * nxz] = bufm[i + (k - 1) * nxz]; // move the data
        for(int k = idc; k > idtm[i]; k--)
          indx[i + k * nxz] = indx[i + (k - 1) * nxz];
        for(int k = idc; k > idtm[i]; k--)
          indi[i + indx[i + k * nxz] * nxz] = k;
        bufm[i + idtm[i] * nxz] = a[i + iyg * nxz];
        indx[i + idtm[i] * nxz] = iyr;
        indi[i + iyr * nxz] = idtm[i];
      }
    }
    memcpy(b + iy * nxz, bufm + sizeh * nxz, nxz * sizeof(float));
  }
  delete[] bufm;
  delete[] indx;
  return;
}

