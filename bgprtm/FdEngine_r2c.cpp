#include "FdEngine_r2c.h"
using namespace PMLBUF;

FdEngine_r2c::FdEngine_r2c(int innx, int inny, int innz, float indx, float indy, float indz, int inNThreads) // removed  int inRhoCN*, by wolf
  : FdEngine(innx, inny, innz, indx, indy, indz, inNThreads) { // removed  inRhoCN* by wolf
  int maxlen;
  scaleX = 1.0f / float(nx);
  scaleY = 1.0f / float(ny);
  scaleZ = 1.0f / float(nz);

  dkx    =  2.0f * M_PI / (dx * nx) * scaleX;
  dky    =  2.0f * M_PI / (dy * ny) * scaleY;
  dkz    =  2.0f * M_PI / (dz * nz) * scaleZ;

  maxlen = MAX(nx + 2, ny + 2);
  maxlen = MAX(maxlen, nz + 2);
  float *work1 = (float *) _mm_malloc((maxlen + 2) * sizeof(float) + 128, 16);
  float *work2 = (float *) _mm_malloc((maxlen + 2) * sizeof(float) + 128, 16);

  planzf = fftwf_plan_dft_r2c_1d(nz, (float *) work1, (fftwf_complex *) work2, FFTW_MEASURE | FFTW_DESTROY_INPUT);
  planzb = fftwf_plan_dft_c2r_1d(nz, (fftwf_complex *) work1, (float *) work2, FFTW_MEASURE | FFTW_DESTROY_INPUT);
  planxf = fftwf_plan_dft_r2c_1d(nx, (float *) work1, (fftwf_complex *) work2, FFTW_MEASURE | FFTW_DESTROY_INPUT);
  planxb = fftwf_plan_dft_c2r_1d(nx, (fftwf_complex *) work1, (float *) work2, FFTW_MEASURE | FFTW_DESTROY_INPUT);

  planyf  = 0;
  planyb  = 0;
  if(ny > 1) planyf = fftwf_plan_dft_r2c_1d(ny, (float *) work1, (fftwf_complex *) work2, FFTW_MEASURE | FFTW_DESTROY_INPUT);
  if(ny > 1) planyb = fftwf_plan_dft_c2r_1d(ny, (fftwf_complex *) work1, (float *) work2, FFTW_MEASURE | FFTW_DESTROY_INPUT);

  kz = assignW(nz, dkz);
  kx = assignW(nx, dkx);
  ky = 0;
  if(ny > 1) ky = assignW(ny, dky);
  
  _mm_free(work1);
  _mm_free(work2);
} 

FdEngine_r2c::~FdEngine_r2c() {
  if(planxf) {fftwf_destroy_plan(planxf);  planxf = 0;}
  if(planxb) {fftwf_destroy_plan(planxb);  planxb = 0;}
  if(planzf) {fftwf_destroy_plan(planzf);  planzf = 0;}
  if(planzb) {fftwf_destroy_plan(planzb);  planzb = 0;}
  if(planyf) {fftwf_destroy_plan(planyf);  planyf = 0;}
  if(planyb) {fftwf_destroy_plan(planyb);  planyb = 0;}

  if(kx) _mm_free(kx);
  if(kz) _mm_free(kz);
  if(ky) _mm_free(ky);
}

float *FdEngine_r2c::assignW(int n, float dw) {
  float *w = (float *) _mm_malloc(n * sizeof(float) + 128, 16);

  for(int i = 0; i < n / 2 + 1; i++) w[i] = -i * dw;
  for(int i = n / 2 + 1; i < n; i++) w[i] = (n - i) * dw;

  return w;
}
void FdEngine_r2c::dx1_2D(float *pin, float *pout, float *wrk, int iy, int iacc, int isign)
{
  dx1_2D(pin, pout, wrk, iy, iacc, isign, nullptr, nullptr);
}

void FdEngine_r2c::dx1_2D(float *pin, float *pout, float *wrk, int iy, int iacc, int isign, float *velSlice, float *rhoSlice) {
  int nxp2 = nx + 4;

  float *wrk1 = wrk;
  float *wrk2 = wrk + nxp2 * nz;
  float *q1 = pmlBuf[X][TOP][ROUND1][XX] + iy * nz * nxbnd1;
  float *q2 = pmlBuf[X][BOT][ROUND1][XX] + iy * nz * nxbnd2;
  float *q3 = pmlBuf[X][TOP][ROUND2][XX] + iy * nz * nxbnd1;
  float *q4 = pmlBuf[X][BOT][ROUND2][XX] + iy * nz * nxbnd2;

  Util::transposeAndPad(nz, nx, nz, nxp2, pin, wrk1);

  if(isign == -1)
    fftderiv1_r2c(planxf, planxb, pmlX1, pmlX2,
                  kx, wrk1, wrk2, q1, q2,
                  nz, nx, nxbnd1, nxbnd2, isign);
  else
    fftderiv1_r2c(planxf, planxb, pmlX1, pmlX2,
                  kx, wrk1, wrk2, q3, q4,
                  nz, nx, nxbnd1, nxbnd2, isign);

  if(iacc)
    Util::transposeAndAdd(nx, nz, nxp2, nz, wrk1, pout, velSlice, rhoSlice);
  else
    Util::transpose(nx, nz, nxp2, nz, wrk1, pout);
}
void FdEngine_r2c::dy1_2D(float *pin, float *pout, float *wrk, int ix, int iacc, int isign)
{
  dy1_2D(pin, pout, wrk, ix, iacc, isign, nullptr, nullptr);
}

void FdEngine_r2c::dy1_2D(float *pin, float *pout, float *wrk, int ix, int iacc, int isign, float *velSlice, float *rhoSlice) {
  int nyp2 = ny + 4;

  float *wrk1 = wrk;
  float *wrk2 = wrk + nyp2 * nz;
  float *q1 = pmlBuf[Y][TOP][ROUND1][YY] + ix * nz * nybnd1;
  float *q2 = pmlBuf[Y][BOT][ROUND1][YY] + ix * nz * nybnd2;
  float *q3 = pmlBuf[Y][TOP][ROUND2][YY] + ix * nz * nybnd1;
  float *q4 = pmlBuf[Y][BOT][ROUND2][YY] + ix * nz * nybnd2;

  Util::transposeAndPad(nz, ny, nxz, nyp2, pin, wrk1);

  if(isign == -1)
    fftderiv1_r2c(planyf, planyb, pmlY1, pmlY2,
                  ky, wrk1, wrk2, q1, q2,
                  nz, ny, nybnd1, nybnd2, isign);
  else
    fftderiv1_r2c(planyf, planyb, pmlY1, pmlY2,
                  ky, wrk1, wrk2, q3, q4,
                  nz, ny, nybnd1, nybnd2, isign);

  if(iacc)
    Util::transposeAndAdd(ny, nz, nyp2, nxz, wrk1, pout, velSlice, rhoSlice);
  else
    Util::transpose(ny, nz, nyp2, nxz, wrk1, pout);

}
void FdEngine_r2c::dz1_2D(float *pin, float *pout, float *wrk, int iy, int iacc, int isign)
{
  dz1_2D(pin, pout, wrk, iy, iacc, isign, nullptr, nullptr);
}

void FdEngine_r2c::dz1_2D(float *pin, float *pout, float *wrk, int iy, int iacc, int isign, float *velSlice, float *rhoSlice) {
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

  if(isign == -1) {
    fftderiv1_r2c(planzf, planzb, pmlZ1, pmlZ2,
                  kz, wrk1, wrk2, q1, q2,
                  nx, nz, nzbnd1, nzbnd2, isign);
  } else {
    fftderiv1_r2c(planzf, planzb, pmlZ1, pmlZ2,
                  kz, wrk1, wrk2, q3, q4,
                  nx, nz, nzbnd1, nzbnd2, isign);
  }
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
}  else {
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

void FdEngine_r2c::fftderiv1_r2c(fftwf_plan &planf, fftwf_plan &planb, PML *pml1, PML *pml2,
                                       float *kx, float *wrk1, float *wrk2, float *q1, float *q2,
                                       int nz, int nx, int nbnd1, int nbnd2, int isign) {
  int nxp2 = nx + 4;

  for(int iz = 0; iz < nz; iz++) {

    //timeRecorder.start(DERIVATIVE_TIME);
    fftwf_execute_dft_r2c(planf, (float *) &wrk1[iz * nxp2], (fftwf_complex *) &wrk2[iz * nxp2]);

    for(int ix = 0; ix < nx / 2; ix++) {
      float temp = kx[ix] * wrk2[iz * nxp2 + 2 * ix];
      wrk2[iz * nxp2 + 2 * ix] = -kx[ix] * wrk2[iz * nxp2 + 2 * ix + 1];
      wrk2[iz * nxp2 + 2 * ix + 1] = temp;
}

    int ix = nx / 2;
    float temp = kx[ix] * wrk2[iz * nxp2 + 2 * ix] * isign;
    wrk2[iz * nxp2 + 2 * ix] = -kx[ix] * wrk2[iz * nxp2 + 2 * ix + 1] * isign;
    wrk2[iz * nxp2 + 2 * ix + 1] = temp;

    ix = nx / 2 + 1;
    wrk2[iz * nxp2 + 2 * ix] = 0.0f;
    wrk2[iz * nxp2 + 2 * ix + 1] = 0.0f;


    fftwf_execute_dft_c2r(planb, (fftwf_complex *) &wrk2[iz * nxp2], (float *) &wrk1[iz * nxp2]);
    //timeRecorder.end(DERIVATIVE_TIME);

    //timeRecorder.start(PML_TIME);
    if(absType == ABSPML) {
      if(pml1) pml1->apply(wrk1 + iz * nxp2, q1 + iz * nbnd1, 1, -1);
      if(pml2) pml2->apply(wrk1 + iz * nxp2, q2 + iz * nbnd2, 1, +1);
    }
    //timeRecorder.end(PML_TIME);

  }

}

