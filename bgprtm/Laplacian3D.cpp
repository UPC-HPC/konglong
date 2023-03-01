/*
 * Laplacian3D.cpp
 *
 */

//
#include <math.h>
#include <string.h>
#include <omp.h>

#include "Laplacian3D.h"
#include "GetPar.h"
#include "libFFTV/fftvfilter.h"
#include "Grid.h"

void laplacian3D(float *image, Grid *grid) {
  int nThreads = init_num_threads();

  int nx = grid->nx;
  int ny = grid->ny;
  int nz = grid->nz;

  size_t gridSize = size_t(nx) * size_t(ny) * size_t(nz);

  float *work1 = new float[gridSize];
  float *work2 = new float[gridSize];

  libfftv::FFTVFilter *derive = new libfftv::FFTVFilter(nz, nx, ny, grid->dz, grid->dx, grid->dy, nThreads, 1);

  derive->SetFilterType(libfftv::DERIVATIVE2);

  derive->run(image, work1, NULL, 2);

  if(ny > 1) {
    derive->run(image, work2, NULL, 3);

#pragma omp for schedule(static)
    for(int iy = 0; iy < ny; iy++) {
      for(int ix = 0; ix < nx; ix++) {
        for(int iz = 0; iz < nz; iz++) {
          size_t id1 = size_t(iy * nx + ix) * size_t(nz) + iz;
          work1[id1] += work2[id1];
        }
      }
    }
  }

  derive->run(image, work2, NULL, 1);
  if(grid->mytype == RECTANGLE) {
#pragma omp for schedule(static)
    for(int iy = 0; iy < ny; iy++) {
      for(int ix = 0; ix < nx; ix++) {
        for(int iz = 0; iz < nz; iz++) {
          size_t id1 = size_t(iy * nx + ix) * size_t(nz) + iz;
          image[id1] = work1[id1] + work2[id1];
        }
      }
    }
  } else {
    float *jacobz2 = new float[nz];
    for(int iz = 0; iz < nz; iz++)
      jacobz2[iz] = grid->jacobz[iz] * grid->jacobz[iz];
#pragma omp for schedule(static)
    for(int iy = 0; iy < ny; iy++) {
      for(int ix = 0; ix < nx; ix++) {
        for(int iz = 0; iz < nz; iz++) {
          size_t id1 = size_t(iy * nx + ix) * size_t(nz) + iz;
          image[id1] = (work1[id1] + work2[id1]) * jacobz2[iz];
        }
      }
    }
    delete[] jacobz2;
  }

  delete derive;
  delete[] work1;
  delete[] work2;
}                                                
