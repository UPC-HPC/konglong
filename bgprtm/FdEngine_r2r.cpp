#include "FdEngine_r2r.h"
using namespace PMLBUF;

FdEngine_r2r::FdEngine_r2r(int innx, int inny, int innz, float indx, float indy, float indz, int inNThreads) //removed int RhoCN*, by wolf
  : FdEngine(innx, inny, innz, indx, indy, indz, inNThreads) { //removed RhoCN*, by wolf
  int maxlen;
  scaleX = 1.0f / float(nx);
  scaleY = 1.0f / float(ny);
  scaleZ = 1.0f / float(nz);

  dkx    =  M_PI / (dx * nx) * 0.5f * scaleX;
  dky    =  M_PI / (dy * ny) * 0.5f * scaleY;
  dkz    =  M_PI / (dz * nz) * 0.5f * scaleZ;

  maxlen = MAX(nx + 2, ny + 2);
  maxlen = MAX(maxlen, nz + 2);
  float *work1 = (float *) _mm_malloc((maxlen + 2) * sizeof(float) + 128, 16);
  float *work2 = (float *) _mm_malloc((maxlen + 2) * sizeof(float) + 128, 16);

  planyf1  = 0;
  planyb1  = 0;
  planzf1 = fftwf_plan_r2r_1d(nz, work1, work2, FFTW_REDFT10, FFTW_MEASURE);
  planzb1 = fftwf_plan_r2r_1d(nz, work1, work2, FFTW_RODFT01, FFTW_MEASURE);
  planxf1 = fftwf_plan_r2r_1d(nx, work1, work2, FFTW_REDFT10, FFTW_MEASURE);
  planxb1 = fftwf_plan_r2r_1d(nx, work1, work2, FFTW_RODFT01, FFTW_MEASURE);
  if(ny > 1) planyf1 = fftwf_plan_r2r_1d(ny, work1, work2, FFTW_REDFT10, FFTW_MEASURE);
  if(ny > 1) planyb1 = fftwf_plan_r2r_1d(ny, work1, work2, FFTW_RODFT01, FFTW_MEASURE);
  planyf2  = 0;
  planyb2  = 0;
  planzf2 = fftwf_plan_r2r_1d(nz, work1, work2, FFTW_RODFT10, FFTW_MEASURE);
  planzb2 = fftwf_plan_r2r_1d(nz, work1, work2, FFTW_REDFT01, FFTW_MEASURE);
  planxf2 = fftwf_plan_r2r_1d(nx, work1, work2, FFTW_RODFT10, FFTW_MEASURE);
  planxb2 = fftwf_plan_r2r_1d(nx, work1, work2, FFTW_REDFT01, FFTW_MEASURE);
  if(ny > 1) planyf2 = fftwf_plan_r2r_1d(ny, work1, work2, FFTW_RODFT10, FFTW_MEASURE);
  if(ny > 1) planyb2 = fftwf_plan_r2r_1d(ny, work1, work2, FFTW_REDFT01, FFTW_MEASURE);

  kz = assignW(nz, dkz);
  kx = assignW(nx, dkx);
  ky = 0;
  if(ny > 1) ky = assignW(ny, dky);

  _mm_free(work1);
  _mm_free(work2);
}

FdEngine_r2r::~FdEngine_r2r() {
  if(planxf1) {fftwf_destroy_plan(planxf1);  planxf1 = 0;}
  if(planxf2) {fftwf_destroy_plan(planxf2);  planxf2 = 0;}
  if(planxb1) {fftwf_destroy_plan(planxb1);  planxb1 = 0;}
  if(planxb2) {fftwf_destroy_plan(planxb2);  planxb2 = 0;}

  if(planzf1) {fftwf_destroy_plan(planzf1);  planzf1 = 0;}
  if(planzf2) {fftwf_destroy_plan(planzf2);  planzf2 = 0;}
  if(planzb1) {fftwf_destroy_plan(planzb1);  planzb1 = 0;}
  if(planzb2) {fftwf_destroy_plan(planzb2);  planzb2 = 0;}

  if(planyf1) {fftwf_destroy_plan(planyf1);  planyf1 = 0;}
  if(planyf2) {fftwf_destroy_plan(planyf2);  planyf2 = 0;}
  if(planyb1) {fftwf_destroy_plan(planyb1);  planyb1 = 0;}
  if(planyb2) {fftwf_destroy_plan(planyb2);  planyb2 = 0;}
if(kx) _mm_free(kx);
  if(kz) _mm_free(kz);
  if(ky) _mm_free(ky);
}

float *FdEngine_r2r::assignW(int n, float dw) {
  float *w = (float *) _mm_malloc(n * sizeof(float) + 128, 16);
  for(int i = 0; i < n; i++) w[i] = -(i + 1) * dw;
  return w;
}


void FdEngine_r2r::fftderiv1_r2r(fftwf_plan &planf, fftwf_plan &planb, PML *pml1, PML *pml2,
                                       float *kx, float *wrk1, float *wrk2, float *q1, float *q2,
                                       int nz, int nx, int nbnd1, int nbnd2, int isign) {
  int nxp2 = nx + 4;

  for(int iz = 0; iz < nz; iz++) {

    //timeRecorder.start(DERIVATIVE_TIME);
    fftwf_execute_r2r(planf, &wrk1[iz * nxp2], &wrk2[iz * nxp2]);

    if(isign == 1) {
      for(int ix = 0; ix < nx - 1; ix++) wrk2[iz * nxp2 + ix] = -kx[ix] * wrk2[iz * nxp2 + ix + 1];
      wrk2[iz * nxp2 + nx - 1] = 0.0f;
    } else {
      for(int ix = nx - 1; ix > 0; ix--) wrk2[iz * nxp2 + ix] = kx[ix - 1] * wrk2[iz * nxp2 + ix - 1];
      wrk2[iz * nxp2] = 0.0f;
    }

    fftwf_execute_r2r(planb, &wrk2[iz * nxp2], &wrk1[iz * nxp2]);
    //timeRecorder.end(DERIVATIVE_TIME);

    //timeRecorder.start(PML_TIME);
    if(absType == ABSPML) {
      if(pml1) pml1->apply(wrk1 + iz * nxp2, q1 + iz * nbnd1, 1, -1);
      if(pml2) pml2->apply(wrk1 + iz * nxp2, q2 + iz * nbnd2, 1, +1);
    }
    //timeRecorder.end(PML_TIME);
  }

}

void FdEngine_r2r::dx1_2D(float *pin, float *pout, float *wrk, int iy, int iacc, int isign) {
  dx1_2D(pin, pout, wrk, iy, iacc, isign, nullptr, nullptr);
}

void FdEngine_r2r::dx1_2D(float *pin, float *pout, float *wrk, int iy, int iacc, int isign, float *velSlice, float *rhoSlice) {
  int nxp2 = nx + 4;

  float *wrk1 = wrk;
  float *wrk2 = wrk + nxp2 * nz;
  float *q1 = pmlBuf[X][TOP][ROUND1][XX] + iy * nz * nxbnd1;
  float *q2 = pmlBuf[X][BOT][ROUND1][XX] + iy * nz * nxbnd2;
  float *q3 = pmlBuf[X][TOP][ROUND2][XX] + iy * nz * nxbnd1;
  float *q4 = pmlBuf[X][BOT][ROUND2][XX] + iy * nz * nxbnd2;

  Util::transposeAndPad(nz, nx, nz, nxp2, pin, wrk1);

  if(isign == -1)
    fftderiv1_r2r(planxf2, planxb2, pmlX1, pmlX2,
                  kx, wrk1, wrk2, q1, q2,
                  nz, nx, nxbnd1, nxbnd2, isign);
  else
    fftderiv1_r2r(planxf1, planxb1, pmlX1, pmlX2,
                  kx, wrk1, wrk2, q3, q4,
                  nz, nx, nxbnd1, nxbnd2, isign);

  if(iacc)
    Util::transposeAndAdd(nx, nz, nxp2, nz, wrk1, pout, velSlice, rhoSlice);
  else
    Util::transpose(nx, nz, nxp2, nz, wrk1, pout);
}

void FdEngine_r2r::dy1_2D(float *pin, float *pout, float *wrk, int ix, int iacc, int isign) {
  dy1_2D(pin, pout, wrk, ix, iacc, isign, nullptr, nullptr);
}

void FdEngine_r2r::dy1_2D(float *pin, float *pout, float *wrk, int ix, int iacc, int isign, float *velSlice, float *rhoSlice) {
  int nyp2 = ny + 4;

  float *wrk1 = wrk;
  float *wrk2 = wrk + nyp2 * nz;
  float *q1 = pmlBuf[Y][TOP][ROUND1][YY] + ix * nz * nybnd1;
  float *q2 = pmlBuf[Y][BOT][ROUND1][YY] + ix * nz * nybnd2;
  float *q3 = pmlBuf[Y][TOP][ROUND2][YY] + ix * nz * nybnd1;
  float *q4 = pmlBuf[Y][BOT][ROUND2][YY] + ix * nz * nybnd2;

  Util::transposeAndPad(nz, ny, nxz, nyp2, pin, wrk1);

  if(isign == -1)
    fftderiv1_r2r(planyf2, planyb2, pmlY1, pmlY2,
                  ky, wrk1, wrk2, q1, q2,
                  nz, ny, nybnd1, nybnd2, isign);
  else
    fftderiv1_r2r(planyf1, planyb1, pmlY1, pmlY2,
                  ky, wrk1, wrk2, q3, q4,
                  nz, ny, nybnd1, nybnd2, isign);

  if(iacc)
    Util::transposeAndAdd(ny, nz, nyp2, nxz, wrk1, pout, velSlice, rhoSlice);
  else
    Util::transpose(ny, nz, nyp2, nxz, wrk1, pout);
}

void FdEngine_r2r::dz1_2D(float *pin, float *pout, float *wrk, int iy, int iacc, int isign) {
  dz1_2D(pin, pout, wrk, iy, iacc, isign, nullptr, nullptr);
}

void FdEngine_r2r::dz1_2D(float *pin, float *pout, float *wrk, int iy, int iacc, int isign, float *velSlice, float *rhoSlice) {
  int nzp2 = nz + 4;

  float *wrk1 = wrk;
  float *wrk2 = wrk + nx * nzp2;

  float *q1 = pmlBuf[Z][TOP][ROUND1][ZZ] + iy * nx * nzbnd1;
  float *q2 = pmlBuf[Z][BOT][ROUND1][ZZ] + iy * nx * nzbnd2;
  float *q3 = pmlBuf[Z][TOP][ROUND2][ZZ] + iy * nx * nzbnd1;
  float *q4 = pmlBuf[Z][BOT][ROUND2][ZZ] + iy * nx * nzbnd2;

  for(int ix = 0; ix < nx; ix++) {
    memcpy(wrk1 + ix * nzp2, pin + ix * nz, sizeof(float)*nz);
    wrk1[ix * nzp2 + nz]   = 0.0f;
    wrk1[ix * nzp2 + nz + 1] = 0.0f;
    wrk1[ix * nzp2 + nz + 2] = 0.0f;
    wrk1[ix * nzp2 + nz + 3] = 0.0f;
  }
  if(isign == -1)
    fftderiv1_r2r(planzf2, planzb2, pmlZ1, pmlZ2,
                  kz, wrk1, wrk2, q1, q2,
                  nx, nz, nzbnd1, nzbnd2, isign);
  else
    fftderiv1_r2r(planzf1, planzb1, pmlZ1, pmlZ2,
                  kz, wrk1, wrk2, q3, q4,
                  nx, nz, nzbnd1, nzbnd2, isign);

  if(iacc) {
    if(jacobz) {
      if (velSlice && rhoSlice)
      {
        for(int ix = 0; ix < nx; ix++)
          for(int iz = 0; iz < nz; iz++)
            pout[ix * nz + iz] -= wrk1[ix * nzp2 + iz] * jacobz[iz] * velSlice[ix * nz + iz] * rhoSlice[ix * nz + iz];
      }
      else if (velSlice)
      {
        for(int ix = 0; ix < nx; ix++)
          for(int iz = 0; iz < nz; iz++)
            pout[ix * nz + iz] -= wrk1[ix * nzp2 + iz] * jacobz[iz] * velSlice[ix * nz + iz];
      }
      else
      {//it is impossible that velSlice is null where rhoSlice is not null! by wolf
        for(int ix = 0; ix < nx; ix++)
          for(int iz = 0; iz < nz; iz++)
            pout[ix * nz + iz] += wrk1[ix * nzp2 + iz] * jacobz[iz]; // Here it is += to keep it consistent with the original version, while other case is -=. by wolf
      }
    } else {
      if (velSlice && rhoSlice)
      {
        for(int ix = 0; ix < nx; ix++)
          for(int iz = 0; iz < nz; iz++)
            pout[ix * nz + iz] -= wrk1[ix * nzp2 + iz] * velSlice[ix * nz + iz] * rhoSlice[ix * nz + iz];
      }
      else if (velSlice)
      {
        for(int ix = 0; ix < nx; ix++)
          for(int iz = 0; iz < nz; iz++)
            pout[ix * nz + iz] -= wrk1[ix * nzp2 + iz] * velSlice[ix * nz + iz];
      }
      else
      {//it is impossible that velSlice is null where rhoSlice is not null! by wolf
        for(int ix = 0; ix < nx; ix++)
          for(int iz = 0; iz < nz; iz++)
            pout[ix * nz + iz] += wrk1[ix * nzp2 + iz]; // Here it is += to keep it consistent with the original version, while other case is -=. by wolf
      }
    }
  } else {
    if(jacobz) {
      for(int ix = 0; ix < nx; ix++)
        for(int iz = 0; iz < nz; iz++)
          pout[ix * nz + iz] = wrk1[ix * nzp2 + iz] * jacobz[iz];
    } else {
      for(int ix = 0; ix < nx; ix++)
        memcpy(pout + ix * nz, wrk1 + ix * nzp2, sizeof(float)*nz);
    }
  }
}
~                                                                                                                                                 
                                                  
