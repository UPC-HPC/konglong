#include "boundarytaper.h"
#include "GetPar.h"
#include "Wavefield.h"
#include <omp.h>

void BndTaper::create(float coef) {
  nThreads = init_num_threads();

  xtaper = 0;
  ytaper = 0;
  ztaper = 0;
  if(xbound > 0) xtaper = new float[xbound];
  if(ybound > 0) ytaper = new float[ybound];
  if(zbound > 0) ztaper = new float[zbound];
  float diff = 1.0 - coef;

  for(int i = 0; i < xbound; i++)
    xtaper[i] = 1.0 - diff * (0.5 * (1.0 + cos(PI * i / float(xbound))));

  for(int i = 0; i < ybound; i++)
    ytaper[i] = 1.0 - diff * (0.5 * (1.0 + cos(PI * i / float(ybound))));

  for(int i = 0; i < zbound; i++)
    ztaper[i] = 1.0 - diff * (0.5 * (1.0 + cos(PI * i / float(zbound))));

}

BndTaper::~BndTaper() {
  if(xtaper) {delete[] xtaper; xtaper = NULL;}
  if(ytaper) {delete[] ytaper; ytaper = NULL;}
  if(ztaper) {delete[] ztaper; ztaper = NULL;}
}
void BndTaper::apply(Wavefield *myWavefield) {
  apply(myWavefield->w0);
  apply(myWavefield->w1);
}

void BndTaper::apply(float *vol) {
  #pragma omp parallel num_threads(nThreads)
  {
    #pragma omp for schedule(static)
    for(int iy = 0; iy < ybound; iy++) {
      size_t id = iy * myGrid->nx;
      id *= myGrid->nz;
      size_t ic = (myGrid->ny - iy - 1) * myGrid->nx;
      ic *= myGrid->nz;
      for(int ix = 0; ix < myGrid->nx; ix++) {
        for(int iz = 0; iz < myGrid->nz; iz++) {
          int idxz = ix * myGrid->nz + iz;
          vol[id + idxz] *= ytaper[iy];
        }
      }

      for(int ix = 0; ix < myGrid->nx; ix++) {
        for(int iz = 0; iz < myGrid->nz; iz++) {
          int idxz = ix * myGrid->nz + iz;
          vol[ic + idxz] *= ytaper[iy];
        }
      }
    }
#pragma omp for schedule(static)
    for(int iy = 0; iy < myGrid->ny; iy++) {
      for(int ix = 0; ix < xbound; ix++) {
        size_t id = iy * myGrid->nx + ix;
        id *= myGrid->nz;
        size_t ic = (iy + 1) * myGrid->nx - ix - 1;
        ic *= myGrid->nz;
        for(int iz = 0; iz < myGrid->nz; iz++)
          vol[id + iz] *= xtaper[ix];
      }
      for(int ix = 0; ix < xbound; ix++) {
        size_t id = iy * myGrid->nx + ix;
        id *= myGrid->nz;
        size_t ic = (iy + 1) * myGrid->nx - ix - 1;
        ic *= myGrid->nz;
        for(int iz = 0; iz < myGrid->nz; iz++)
          vol[ic + iz] *= xtaper[ix];
      }
    }

    #pragma omp for schedule(static)
    for(int iy = 0; iy < myGrid->ny; iy++) {
      for(int ix = 0; ix < myGrid->nx; ix++) {
        size_t id = iy * myGrid->nx + ix;
        id *= myGrid->nz;
        for(int iz = 0; iz < zbound; iz++) {
          int ic = myGrid->nz - iz - 1;
          vol[id + iz] *= ztaper[iz];
          vol[id + ic] *= ztaper[iz];
        }
      }
    }
  }
}

