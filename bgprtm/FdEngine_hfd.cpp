#include "FdEngine_hfd.h"
using namespace PMLBUF;

FdEngine_hfd::FdEngine_hfd(int innx, int inny, int innz, float indx, float indy, float indz, int inNThreads) : // removed int RhoCN* by wolf,
    FdEngine(innx, inny, innz, indx, indy, indz, inNThreads) {// removed int RhoCN* by wolf,
  dispersion_factor = 0.8f;
}

FdEngine_hfd::~FdEngine_hfd() {
}

void FdEngine_hfd::deriv1_hfd(PML *pml1, PML *pml2, float *wrk1, float *wrk2, float *q1, float *q2, int nz, int nx, int nbnd1, int nbnd2,
    int isign) {
  int nxp2 = nx + 4;

  for(int iz = 0; iz < nz; iz++) {

    float time_fdt1, time_fdt2;
    //timeRecorder.start(DERIVATIVE_TIME);
    fd_dev1(&wrk1[iz * nxp2], &wrk2[iz * nxp2], nx, dx, time_fdt1, time_fdt2, isign);
    //timeRecorder.end(DERIVATIVE_TIME);

    memcpy(&wrk1[iz * nxp2], &wrk2[iz * nxp2], nx * sizeof(float));

    //timeRecorder.start(PML_TIME);
    if(absType == ABSPML) {
      if(pml1) pml1->apply(wrk1 + iz * nxp2, q1 + iz * nbnd1, 1, -1);
      if(pml2) pml2->apply(wrk1 + iz * nxp2, q2 + iz * nbnd2, 1, +1);
    }
    //timeRecorder.end(PML_TIME);
  }
}
void FdEngine_hfd::dx1_2D(float *pin, float *pout, float *wrk, int iy, int iacc, int isign) {
  int nxp2 = nx + 4;

  float *wrk1 = wrk;
  float *wrk2 = wrk + nxp2 * nz;
  float *q1 = pmlBuf[X][TOP][ROUND1][XX] + iy * nz * nxbnd1;
  float *q2 = pmlBuf[X][BOT][ROUND1][XX] + iy * nz * nxbnd2;
  float *q3 = pmlBuf[X][TOP][ROUND2][XX] + iy * nz * nxbnd1;
  float *q4 = pmlBuf[X][BOT][ROUND2][XX] + iy * nz * nxbnd2;

  Util::transposeAndPad(nz, nx, nz, nxp2, pin, wrk1);

  if(isign == -1) deriv1_hfd(pmlX1, pmlX2, wrk1, wrk2, q1, q2, nz, nx, nxbnd1, nxbnd2, isign);
  else deriv1_hfd(pmlX1, pmlX2, wrk1, wrk2, q3, q4, nz, nx, nxbnd1, nxbnd2, isign);

  if(iacc) Util::transposeAndAdd(nx, nz, nxp2, nz, wrk1, pout);
  else Util::transpose(nx, nz, nxp2, nz, wrk1, pout);
}
void FdEngine_hfd::dy1_2D(float *pin, float *pout, float *wrk, int ix, int iacc, int isign) {
  int nyp2 = ny + 4;

  float *wrk1 = wrk;
  float *wrk2 = wrk + nyp2 * nz;
  float *q1 = pmlBuf[Y][TOP][ROUND1][YY] + ix * nz * nybnd1;
  float *q2 = pmlBuf[Y][BOT][ROUND1][YY] + ix * nz * nybnd2;
  float *q3 = pmlBuf[Y][TOP][ROUND2][YY] + ix * nz * nybnd1;
  float *q4 = pmlBuf[Y][BOT][ROUND2][YY] + ix * nz * nybnd2;

  Util::transposeAndPad(nz, ny, nxz, nyp2, pin, wrk1);

  if(isign == -1) deriv1_hfd(pmlY1, pmlY2, wrk1, wrk2, q1, q2, nz, ny, nybnd1, nybnd2, isign);
  else deriv1_hfd(pmlY1, pmlY2, wrk1, wrk2, q3, q4, nz, ny, nybnd1, nybnd2, isign);

  if(iacc) Util::transposeAndAdd(ny, nz, nyp2, nxz, wrk1, pout);
  else Util::transpose(ny, nz, nyp2, nxz, wrk1, pout);
}
void FdEngine_hfd::dz1_2D(float *pin, float *pout, float *wrk, int iy, int iacc, int isign) {
  int nzp2 = nz + 4;

  float *wrk1 = wrk;
  float *wrk2 = wrk + nx * nzp2;

  float *q1 = pmlBuf[Z][TOP][ROUND1][ZZ] + iy * nx * nzbnd1;
  float *q2 = pmlBuf[Z][BOT][ROUND1][ZZ] + iy * nx * nzbnd2;
  float *q3 = pmlBuf[Z][TOP][ROUND2][ZZ] + iy * nx * nzbnd1;
  float *q4 = pmlBuf[Z][BOT][ROUND2][ZZ] + iy * nx * nzbnd2;

  for(int ix = 0; ix < nx; ix++) {
    memcpy(wrk1 + ix * nzp2, pin + ix * nz, sizeof(float) * nz);
    wrk1[ix * nzp2 + nz] = 0.0f;
    wrk1[ix * nzp2 + nz + 1] = 0.0f;
    wrk1[ix * nzp2 + nz + 2] = 0.0f;
    wrk1[ix * nzp2 + nz + 3] = 0.0f;
  }

  if(isign == -1) deriv1_hfd(pmlZ1, pmlZ2, wrk1, wrk2, q1, q2, nx, nz, nzbnd1, nzbnd2, isign);
  else deriv1_hfd(pmlZ1, pmlZ2, wrk1, wrk2, q3, q4, nx, nz, nzbnd1, nzbnd2, isign);

  if(iacc) {
    if(jacobz) {
      for(int ix = 0; ix < nx; ix++)
        for(int iz = 0; iz < nz; iz++)
          pout[ix * nz + iz] += wrk1[ix * nzp2 + iz] * jacobz[iz];
    } else {
      for(int ix = 0; ix < nx; ix++)
        for(int iz = 0; iz < nz; iz++)
          pout[ix * nz + iz] += wrk1[ix * nzp2 + iz];
    }
  } else {
    if(jacobz) {
      for(int ix = 0; ix < nx; ix++)
        for(int iz = 0; iz < nz; iz++)
          pout[ix * nz + iz] = wrk1[ix * nzp2 + iz] * jacobz[iz];
    } else {
      for(int ix = 0; ix < nx; ix++)
        memcpy(pout + ix * nz, wrk1 + ix * nzp2, sizeof(float) * nz);
    }
  }
}
                         
