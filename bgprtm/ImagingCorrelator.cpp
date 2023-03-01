/*
 * ImagingCorrelator.cpp
 *
 */

#include <math.h>
#include <string.h>
#include <xmmintrin.h>
#include <omp.h>

#include "ImagingCorrelator.h"
#include "libCommon/Assertion.h"

#include "MpiPrint.h"
using MpiPrint::print1m;

ImagingCorrelator::ImagingCorrelator(int nx, int ny, int nz, float dx, float dy, float dz, int nx2, int ny2, int nz2, float kmin,
    int nThreads) :
    nx(nx), ny(ny), nz(nz), dx(dx), dy(dy), dz(dz), nx2(nx2), ny2(ny2), nz2(nz2), kmin(kmin), nThreads(nThreads) {
  nxz = (size_t)nx * (size_t)nz;
  nyz = (size_t)ny * (size_t)nz;
  nxy = (size_t)nx * (size_t)ny;
  nxyz = nxz * (size_t)ny;

  planxz1 = 0;
  planxz2 = 0;

  plany1 = 0;
  plany2 = 0;

  nzp2 = nz + 2;

  nxyzp2 = nxy * nzp2;

  srcBuff = (float*)_mm_malloc(nxyzp2 * sizeof(float) + 128, 16);
  recBuff = (float*)_mm_malloc(nxyzp2 * sizeof(float) + 128, 16);

  srcBufz = (float*)_mm_malloc(nxyzp2 * sizeof(float) + 128, 16);
  recBufz = (float*)_mm_malloc(nxyzp2 * sizeof(float) + 128, 16);

  if(ny2 > ny) {
    srcBufx = (float*)_mm_malloc(nxyzp2 * sizeof(float) + 128, 16);
    recBufx = (float*)_mm_malloc(nxyzp2 * sizeof(float) + 128, 16);
  } else {
    srcBufx = srcBuff;
    recBufx = recBuff;
  }

  srcBufy = srcBuff;
  recBufy = recBuff;

  kx = 0;
  kz = 0;
  ky = 0;

  rawOption = false;

  int nDblz = nz2 / nz, nDblx = nx2 / nx, nDbly = ny2 / ny;
  // default: ivz = 1, ivx = 2, ivy = 3, ivzx = 4, ivzy = 5, ivxy = 6, ivzxy = 7;
  if(nDbly == 1) {
    if(nDblz == 1) ivx = 1;
    else ivz = 1, ivx = 2, ivzx = 3;
  } else if(nDblx == 1) {
    if(nDblz == 1) ivy = 1; // only interp-y needed
    else ivz = 1, ivy = 2, ivzy = 3;
  } else if(nDblz == 1) ivx = 1, ivy = 2, ivxy = 3;
  // lines below: not used, set to 0
  if(nDbly == 1) ivy = ivxy = ivzy = ivzxy = 0;
  if(nDblx == 1) ivx = ivxy = ivzx = ivzxy = 0;
  if(nDblz == 1) ivz = ivzx = ivzy = ivzxy = 0;

  assertion(nDblz >= nDblx && nDblz >= nDbly,
            "Current implementation requires no interp or at least z-interp! (nDblz=%d, nDblx=%d, nDbly=%d)", nDblz, nDblx, nDbly);
  nloop = ivzxy ? 8 : ivzx ? 4 : ivz ? 2 : 1;
  print1m("nDblz=%d, nDblx=%d, nDbly=%d\n", nDblz, nDblx, nDbly);
  print1m("nloop=%d,  ivz = %d, ivx = %d, ivy = %d, ivzx = %d, ivzy = %d, ivxy = %d, ivzxy = %d\n", nloop, ivz, ivx, ivy, ivzx, ivzy, ivxy,
         ivzxy);

  //kmin = 1.0f/1500.0f;
  create();
}

ImagingCorrelator::~ImagingCorrelator() {
  if(planxz1) {
    fftwf_destroy_plan(planxz1);
    planxz1 = 0;
  }
  if(planxz2) {
    fftwf_destroy_plan(planxz2);
    planxz2 = 0;
  }

  if(plany1) {
    fftwf_destroy_plan(plany1);
    plany1 = 0;
  }
  if(plany2) {
    fftwf_destroy_plan(plany2);
    plany2 = 0;
  }

  if(srcBuff) _mm_free(srcBuff);
  if(recBuff) _mm_free(recBuff);

  if(srcBufz) _mm_free(srcBufz);
  if(recBufz) _mm_free(recBufz);

  if(ny2 > ny) {
    _mm_free(srcBufx);
    _mm_free(recBufx);
  }

  if(kx) _mm_free(kx);
  if(kz) _mm_free(kz);
  if(ky) _mm_free(ky);
}

void ImagingCorrelator::create() {
  int maxlen;
  scaleX = 1.0f / float(nx);
  scaleY = 1.0f / float(ny);
  scaleZ = 1.0f / float(nz);
  scalef = sqrtf(scaleX * scaleY * scaleZ);

  dkx = 2.0f * M_PI / (dx * nx);
  dky = 2.0f * M_PI / (dy * ny);
  dkz = 2.0f * M_PI / (dz * nz);

  if(ny > 1) {
    planxz1 = fftwf_plan_dft_r2c_3d(ny, nx, nz, (float*)srcBuff, (fftwf_complex*)srcBuff, FFTW_MEASURE);
    planxz2 = fftwf_plan_dft_c2r_3d(ny, nx, nz, (fftwf_complex*)srcBuff, (float*)srcBuff, FFTW_MEASURE);
  } else {
    planxz1 = fftwf_plan_dft_r2c_2d(nx, nz, (float*)srcBuff, (fftwf_complex*)srcBuff, FFTW_MEASURE);
    planxz2 = fftwf_plan_dft_c2r_2d(nx, nz, (fftwf_complex*)srcBuff, (float*)srcBuff, FFTW_MEASURE);
  }

  kz = assignW(nz, dkz);
  kx = assignW(nx, dkx);
  ky = assignW(ny, dky);
}

float* ImagingCorrelator::assignW(int n, float dw) {
  float *w = (float*)_mm_malloc(n * sizeof(float) + 128, 16);

  for(int i = 0; i < n / 2 + 1; i++)
    w[i] = -i * dw;
  for(int i = n / 2 + 1; i < n; i++)
    w[i] = (n - i) * dw;

  return w;
}

void ImagingCorrelator::buffPad2(float *in, float *out, int option) {
  size_t nxzp2 = (size_t)nx * (size_t)nzp2;
  int nxy = nx * ny;
  if(option == 1) {
#pragma omp parallel for schedule(static)
    for(int ixy = 0; ixy < nxy; ixy++) {
      int ix = ixy % nx, iy = ixy / nx;
      float *pin = in + iy * nxz + ix * nz;
      float *pout = out + iy * nxzp2 + ix * nzp2;
      for(int iz = 0; iz < nz; iz++)
        pout[iz] = pin[iz] * scalef;
      for(int iz = nz; iz < nzp2; iz++)
        pout[iz] = 0.0f;
    }
  } else {
#pragma omp parallel for schedule(static)
    for(int ixy = 0; ixy < nxy; ixy++) {
      int ix = ixy % nx, iy = ixy / nx;
      float *pin = in + iy * nxzp2 + ix * nzp2;
      float *pout = out + iy * nxz + ix * nz;
      for(int iz = 0; iz < nz; iz++)
        pout[iz] = pin[iz] * scalef;
    }
  }
}

// isign is for image, the 2nd sign is always +1
void ImagingCorrelator::crossone(float *image, float *img2, float *srcWave, float *recWave, int mz, int isign) {
  int nxy = nx * ny;
#pragma omp parallel for num_threads(nThreads) schedule(static)
  for(int ixy = 0; ixy < nxy; ixy++) {
    int ix = ixy % nx, iy = ixy / nx;
    for(int iz = 0; iz < nz; iz++) {
      size_t id1 = ((size_t)iy * nx + ix) * (size_t)nz + iz;
      size_t id2 = ((size_t)iy * nx + ix) * (size_t)mz + iz;
      float dot = srcWave[id2] * recWave[id2];
      if(rawOption) dot = std::abs(dot);
      if(image) image[id1] += isign * dot;
      if(img2) img2[id1] += dot;
    }
  }
}
void ImagingCorrelator::lapx(float *in, float *out) {
  int nxy = nx * ny;
#pragma omp parallel for num_threads(nThreads) schedule(static)
  for(int ixy = 0; ixy < nxy; ixy++) {
    int ix = ixy % nx, iy = ixy / nx;
    for(int iz = 0; iz < nzp2 / 2; iz++) {
      float kk = sqrtf(kx[ix] * kx[ix] + kz[iz] * kz[iz] + ky[iy] * ky[iy]);
      size_t idd = ((size_t)iy * nx + ix) * (size_t)(nzp2) + 2 * iz;
      if(kk > 1.0E-25) {
        float kkx = kx[ix] / kk;
        float temp = kkx * in[idd];
        out[idd] = -kkx * in[idd + 1];
        out[idd + 1] = temp;
      } else {
        out[idd] = 0.0f;
        out[idd + 1] = 0.0f;
      }
    }
  }
  fftwf_execute_dft_c2r(planxz2, (fftwf_complex*)out, (float*)out);
}

void ImagingCorrelator::lapz(float *in, float *out) {
  int nxy = nx * ny;
#pragma omp parallel for num_threads(nThreads) schedule(static)
  for(int ixy = 0; ixy < nxy; ixy++) {
    int ix = ixy % nx, iy = ixy / nx;
    for(int iz = 0; iz < nzp2 / 2; iz++) {
      float kk = sqrtf(kx[ix] * kx[ix] + kz[iz] * kz[iz] + ky[iy] * ky[iy]);
      size_t idd = ((size_t)iy * nx + ix) * (size_t)(nzp2) + 2 * iz;
      if(kk > 1.0E-25) {
        float kkz = kz[iz] / kk;
        float temp = kkz * in[idd];
        out[idd] = -kkz * in[idd + 1];
        out[idd + 1] = temp;
      } else {
        out[idd] = 0.0f;
        out[idd + 1] = 0.0f;
      }
    }
  }
  fftwf_execute_dft_c2r(planxz2, (fftwf_complex*)out, (float*)out);
}

void ImagingCorrelator::lapy(float *in, float *out) {
  int nxy = nx * ny;
#pragma omp parallel for num_threads(nThreads) schedule(static)
  for(int ixy = 0; ixy < nxy; ixy++) {
    int ix = ixy % nx, iy = ixy / nx;
    for(int iz = 0; iz < nzp2 / 2; iz++) {
      float kk = sqrtf(kx[ix] * kx[ix] + kz[iz] * kz[iz] + ky[iy] * ky[iy]);
      size_t idd = ((size_t)iy * nx + ix) * (size_t)(nzp2) + 2 * iz;
      if(kk > 1.0E-25) {
        float kky = ky[iy] / kk;
        float temp = kky * in[idd];
        out[idd] = -kky * in[idd + 1];
        out[idd + 1] = temp;
      } else {
        out[idd] = 0.0f;
        out[idd + 1] = 0.0f;
      }
    }
  }
  fftwf_execute_dft_c2r(planxz2, (fftwf_complex*)out, (float*)out);
}

void ImagingCorrelator::derive(float *srcWave, float *recWave) {
  //pad the input buffer;
  buffPad2(srcWave, srcBuff, +1);
  buffPad2(recWave, recBuff, +1);

  fftwf_execute_dft_r2c(planxz1, srcBuff, (fftwf_complex*)srcBuff);
  fftwf_execute_dft_r2c(planxz1, recBuff, (fftwf_complex*)recBuff);

  //
  lapz(srcBuff, srcBufz);
  lapz(recBuff, recBufz);

  //
  lapx(srcBuff, srcBufx);
  lapx(recBuff, recBufx);

  if(ny > 1) {
    lapy(srcBuff, srcBufy);
    lapy(recBuff, recBufy);
  }
}

void ImagingCorrelator::kfilter(float *wave, float *work) {
  //pad the input buffer;
  buffPad2(wave, work, +1);

  fftwf_execute_dft_r2c(planxz1, work, (fftwf_complex*)work);

  int nxy = nx * ny;
#pragma omp parallel for num_threads(nThreads) schedule(static)
  for(int ixy = 0; ixy < nxy; ixy++) {
    int ix = ixy % nx, iy = ixy / nx;
    for(int iz = 0; iz < nzp2 / 2; iz++) {
      float kk = sqrtf(kx[ix] * kx[ix] + kz[iz] * kz[iz] + ky[iy] * ky[iy]);
      size_t idd = ((size_t)iy * nx + ix) * (size_t)(nzp2) + 2 * iz;
      if(kk < kmin) kk = kmin;
      float rkk = 1.0f / kk;
      work[idd] *= rkk;
      work[idd + 1] *= rkk;
    }
  }

  fftwf_execute_dft_c2r(planxz2, (fftwf_complex*)work, (float*)work);

  buffPad2(work, wave, -1);
}

void ImagingCorrelator::costerm(float *image) {

  float *work = srcBuff;
  for(int i = 0; i < nloop; i++) {

    buffPad2(image + i * nxyz, work, +1);
    fftwf_execute_dft_r2c(planxz1, work, (fftwf_complex*)work);

    int nxy = nx * ny;
#pragma omp parallel for num_threads(nThreads) schedule(static)
    for(int ixy = 0; ixy < nxy; ixy++) {
      int ix = ixy % nx, iy = ixy / nx;
      for(int iz = 0; iz < nzp2 / 2; iz++) {
        float kk = kx[ix] * kx[ix] + kz[iz] * kz[iz] + ky[iy] * ky[iy];
        size_t idd = ((size_t)iy * nx + ix) * (size_t)(nzp2) + 2 * iz;
        work[idd] *= kk;
        work[idd + 1] *= kk;
      }
    }
    fftwf_execute_dft_c2r(planxz2, (fftwf_complex*)work, (float*)work);
    buffPad2(work, image + i * nxyz, -1);
  }
}

void ImagingCorrelator::sinterm(float *imgfull, float *imgcos) {

  if(imgfull) {
    for(int i = 0; i < nloop; i++) {
      imgmath(imgfull + i * nxyz, imgcos + i * nxyz);
    }
  }
}

void ImagingCorrelator::imgmath(float *imgfull, float *imgimp) {
  int nxy = nx * ny;
#pragma omp parallel for num_threads(nThreads) schedule(static)
  for(int ixy = 0; ixy < nxy; ixy++) {
    int ix = ixy % nx, iy = ixy / nx;
    for(int iz = 0; iz < nz; iz++) {
      size_t idd = ((size_t)iy * nx + ix) * ((size_t)nz) + iz;
      imgfull[idd] = 2.0f * imgfull[idd] - 0.5f * imgimp[idd];
    }
  }
}

void ImagingCorrelator::run(libfftv::FFTVFilter *fftv, float *image, float *imgFWI, float *imgRAW, float **srcBuf, float **recBuf,
    int option) {

  if(option == IMAGE1) imaging1(fftv, image, srcBuf, recBuf);
  else if(option == INVERSION) inversion1(fftv, image, imgFWI, srcBuf, recBuf);
  else if(option == INVERSION2) inversion2(fftv, image, imgFWI, srcBuf, recBuf);
  else if(option == ALLIN3) imaging3(fftv, image, imgFWI, imgRAW, srcBuf, recBuf);
  else print1m("no defined imaging option \n");
}

void ImagingCorrelator::imaging1(libfftv::FFTVFilter *fftv, float *image, float **srcBuf, float **recBuf) {

  //wavefield   ----> in this case, the interp will be done outside to accelerate the delay time gather 
  //interp(fftv, srcBuf);
  //interp(fftv, recBuf);

  for(int i = 0; i < nloop; i++) {
    crossone(image + i * nxyz, NULL, srcBuf[i], recBuf[i], nz, +1);
  }
}

void ImagingCorrelator::inversion1(libfftv::FFTVFilter *fftv, float *image, float *imgFWI, float **srcBuf, float **recBuf) {

  derive(srcBuf[0], recBuf[0]);

  //wavefield
  interp(fftv, srcBuf);
  interp(fftv, recBuf);
  for(int i = 0; i < nloop; i++) {
    crossone(image ? image + i * nxyz : NULL, imgFWI ? imgFWI + i * nxyz : NULL, srcBuf[i], recBuf[i], nz, +1);
  }

  //pz
  buffPad2(srcBufz, srcBuf[0], -1);
  buffPad2(recBufz, recBuf[0], -1);
  interp(fftv, srcBuf);
  interp(fftv, recBuf);
  for(int i = 0; i < nloop; i++) {
    crossone(image ? image + i * nxyz : NULL, imgFWI ? imgFWI + i * nxyz : NULL, srcBuf[i], recBuf[i], nz, -1);
  }

  //px
  buffPad2(srcBufx, srcBuf[0], -1);
  buffPad2(recBufx, recBuf[0], -1);
  interp(fftv, srcBuf);
  interp(fftv, recBuf);
  for(int i = 0; i < nloop; i++) {
    crossone(image ? image + i * nxyz : NULL, imgFWI ? imgFWI + i * nxyz : NULL, srcBuf[i], recBuf[i], nz, -1);
  }

  //py
  if(ny > 1) {
    buffPad2(srcBufy, srcBuf[0], -1);
    buffPad2(recBufy, recBuf[0], -1);
    interp(fftv, srcBuf);
    interp(fftv, recBuf);
    for(int i = 0; i < nloop; i++) {
      crossone(image ? image + i * nxyz : NULL, imgFWI ? imgFWI + i * nxyz : NULL, srcBuf[i], recBuf[i], nz, -1);
    }
  }

}

void ImagingCorrelator::inversion2(libfftv::FFTVFilter *fftv, float *image, float *imgFWI, float **srcWave, float **recWave) {

  if(imgFWI) {
    imaging1(fftv, imgFWI, srcWave, recWave);
  }

  kfilter(srcWave[0], srcBuff);
  kfilter(recWave[0], srcBuff);

  interp(fftv, srcWave);
  interp(fftv, recWave);
  for(int i = 0; i < nloop; i++) {
    crossone(image + i * nxyz, NULL, srcWave[i], recWave[i], nz, +1);
  }
}

void ImagingCorrelator::imaging3(libfftv::FFTVFilter *fftv, float *image, float *imgFWI, float *imgRAW, float **srcBuf, float **recBuf) {
  rawOption = true;
  if(imgRAW) imaging1(fftv, imgRAW, srcBuf, recBuf);

  rawOption = false;
  if(image || imgFWI) inversion1(fftv, image, imgFWI, srcBuf, recBuf);
}

void ImagingCorrelator::interleave(float *image, float **srcBuf, float **recBuf) {
  if(!image || nloop == 1) return;

  if(ivzxy) interleave3d(image, srcBuf, recBuf);
  else interleave2d(image, srcBuf, recBuf);
}

void ImagingCorrelator::interleave2d(float *image, float **srcBuf, float **recBuf) {
  int nDblx = ivx ? 2 : 1;
#pragma omp parallel for num_threads(nThreads)
  for(int i = 0; i < nloop; i++)
    memcpy(srcBuf[i], image + i * nxz, nxz * sizeof(float));

#pragma omp parallel for num_threads(nThreads)
  for(int ix = 0; ix < nx; ix++) {
    int ix1 = ix * nDblx;
    int ix2 = ix * nDblx + nDblx - 1;
    for(int iz = 0; iz < nz; iz++) {
      size_t iz1 = iz * 2;
      size_t iz2 = iz * 2 + 1;
      // 1D: 0, 1 for 0,z; 2D: 0,1,2,3 for 0,z,x,zx;
      size_t idd = ix * ((size_t)nz) + iz;
      size_t id0 = ix1 * ((size_t)nz2) + iz1; //x1z1
      size_t idz = ix1 * ((size_t)nz2) + iz2; //x1z2
      image[id0] = srcBuf[0][idd];
      image[idz] = srcBuf[ivz][idd];
      if(ivzx) {
        size_t idx = ix2 * ((size_t)nz2) + iz1; //x2z1
        size_t idzx = ix2 * ((size_t)nz2) + iz2; //x2z2
        image[idx] = srcBuf[ivx][idd];
        image[idzx] = srcBuf[ivzx][idd];
      }
    }
  }
}

void ImagingCorrelator::interleave3d(float *image, float **srcBuf, float **recBuf) {
  int nxy = nx * ny;

#pragma omp parallel for num_threads(nThreads)
  for(int i = 0; i < nloop; i++)
    memcpy(srcBuf[i], image + i * nxyz, nxyz * sizeof(float));
#pragma omp parallel for num_threads(nThreads) schedule(static)
  for(int ixy = 0; ixy < nxy; ixy++) {
    int ix = ixy % nx, iy = ixy / nx;
    size_t iy1 = iy * 2;
    size_t iy2 = iy * 2 + 1;
    size_t ix1 = ix * 2;
    size_t ix2 = ix * 2 + 1;
    for(int iz = 0; iz < nz; iz++) {
      size_t iz1 = iz * 2;
      size_t iz2 = iz * 2 + 1;
      // 0,z,x,y,zx,zy,xy,zxy
      size_t idd = (iy * nx + ix) * ((size_t)nz) + iz;
      size_t id0 = (iy1 * nx2 + ix1) * ((size_t)nz2) + iz1; //y1x1z1
      size_t idy = (iy2 * nx2 + ix1) * ((size_t)nz2) + iz1; //y2x1z1
      size_t idx = (iy1 * nx2 + ix2) * ((size_t)nz2) + iz1; //y1x2z1
      size_t idz = (iy1 * nx2 + ix1) * ((size_t)nz2) + iz2; //y1x1z2
      size_t idzy = (iy2 * nx2 + ix1) * ((size_t)nz2) + iz2; //y2x1z2
      size_t idzx = (iy1 * nx2 + ix2) * ((size_t)nz2) + iz2; //y1x2z2
      size_t idxy = (iy2 * nx2 + ix2) * ((size_t)nz2) + iz1; //y2x2z1
      size_t idzxy = (iy2 * nx2 + ix2) * ((size_t)nz2) + iz2; //y2x2z2
      // ivz = 1, ivx = 2, ivy = 3, ivzx = 4, ivzy = 5, ivxy = 6, ivzxy = 7
      image[id0] = srcBuf[0][idd];
      image[idz] = srcBuf[ivz][idd];
      image[idx] = srcBuf[ivx][idd];
      image[idy] = srcBuf[ivy][idd];
      image[idzx] = srcBuf[ivzx][idd];
      image[idzy] = srcBuf[ivzy][idd];
      image[idxy] = srcBuf[ivxy][idd];
      image[idzxy] = srcBuf[ivzxy][idd];
    }
  }
}

void ImagingCorrelator::interp(libfftv::FFTVFilter *fftv, float **srcBuf) {

  fftv->SetFilterType(libfftv::SHIFTHALF);
  if(ivz) fftv->run(srcBuf[0], srcBuf[ivz], NULL, 1); //z
  if(ivx) fftv->run(srcBuf[0], srcBuf[ivx], NULL, 2);  //x
  if(ivy) fftv->run(srcBuf[0], srcBuf[ivy], NULL, 3);  //y
  if(ivzx) fftv->run(srcBuf[ivz], srcBuf[ivzx], NULL, 2);  //zx
  if(ivzy) fftv->run(srcBuf[ivz], srcBuf[ivzy], NULL, 3);  //zy
  if(ivxy) fftv->run(srcBuf[ivx], srcBuf[ivxy], NULL, 3); //x-y
  if(ivzxy) fftv->run(srcBuf[ivzx], srcBuf[ivzxy], NULL, 3); //zx-y
#if 0
  int nvol = 2 * (1 + (nx > 1)) * (1 + (ny > 1));
  for(int i = 0; i < nvol; i++) {
    std::string info = "ImagingCorrelator::interp: srcBuf[" + std::to_string(i) + "]";
    Util::print_mem_crc(srcBuf[i], nz, nx * ny, info.c_str());
  }
#endif
}

