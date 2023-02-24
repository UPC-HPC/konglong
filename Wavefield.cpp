#include "Wavefield.h"
#include <xmmintrin.h>
#include "string.h"
#include "stdlib.h"

#include "Grid.h"
#include "Model.h"
#include "Q.h"
#include "MpiPrint.h"
using MpiPrint::print1m;

Wavefield::Wavefield(Grid *myGrid, Model *model) : model(model) {
  nx_ = myGrid->nx;
  ny_ = myGrid->ny;
  nz_ = myGrid->nz;

  w0 = NULL;
  w1 = NULL;
  wb = NULL;
  wx = NULL;
  wy = NULL;
  wz = NULL;
  wr = NULL;

  appType = RTM;

  isInitialized = false;
  allocWx = true; // when prop engine requires 3D wx, wy, wz ...
  modelType_ = model->modeltype;


Wavefield::~Wavefield() {
  if(isInitialized) deallocateMemory();
}

/*
 * Allocate memory for TTI propagation (multithreaded)
 */
void Wavefield::allocateMemory(bool allocWx, int nthreads) {

  this->allocWx = allocWx;

  if(!isInitialized) {
    size_t nxyz = (size_t)nx_ * ny_ * nz_;
    string mem_free0 = libCommon::Utl::free_memory();
    w0 = allocMem3d(nx_, ny_, nz_, nthreads);
    w1 = allocMem3d(nx_, ny_, nz_, nthreads);

    if(allocWx) {
      wb = allocMem3d(nx_, ny_, nz_, nthreads); // moved from line 45 to here by wolf
      wx = allocMem3d(nx_, ny_, nz_, nthreads);
      wy = allocMem3d(nx_, ny_, nz_, nthreads);
      wz = allocMem3d(nx_, ny_, nz_, nthreads);
    }
    wr = allocMem3d(nx_, ny_, nz_, nthreads);

    if(model->useQ) {
      for(int iq = 0; iq < 2; iq++) {
        wq[iq].resize(Q::order);
        for(int i = 0; i < Q::order; i++)
          wq[iq][i].resize(nxyz), count_mem3d++;
      }
    }
    print1m("Allocated %7.2fGB for wavefields: nvol=%d [%dx%dx%d], MemFree: %s->%s\n",
            float(count_mem3d) * nxyz * sizeof(float) / 1024 / 1024 / 1024, count_mem3d, nz_, nx_, ny_, mem_free0.c_str(),
            libCommon::Utl::free_memory().c_str());
  }

  isInitialized = true;
}

/*
 * Free all memory allocated by Wavefield instance
 */
void Wavefield::deallocateMemory() {
  if(isInitialized) {
    _mm_free(w0);
    _mm_free(w1);

    if(allocWx) {
      _mm_free(wb); // move from line 74 to here by wolf, corresponding to the modification in line 47
      _mm_free(wx);
      _mm_free(wy);
      _mm_free(wz);
    }
    _mm_free(wr);

    for(int iq = 0; iq < 2; iq++)
      vector<vector<float>> { }.swap(wq[iq]);

    count_mem3d = 0;
  }
  isInitialized = false;
}

/*
 * Zero all memory allocated by Wavefield instance (multithreaded)
 */
void Wavefield::cleanMemory(int nThreads) {
  if(isInitialized) {
    size_t nxz = (size_t)nx_ * nz_;
#pragma omp parallel for num_threads(nThreads) schedule(static)
    for(int iy = 0; iy < ny_; iy++) {
      memset(&w0[iy * nxz], 0, nxz * sizeof(float));
      memset(&w1[iy * nxz], 0, nxz * sizeof(float));

      if(allocWx) {
        memset(&wb[iy * nxz], 0, nxz * sizeof(float)); // move from line 97 to here by wolf, corresponding to the modification in line 47
        memset(&wx[iy * nxz], 0, nxz * sizeof(float));
        memset(&wy[iy * nxz], 0, nxz * sizeof(float));
        memset(&wz[iy * nxz], 0, nxz * sizeof(float));
      }
      memset(&wr[iy * nxz], 0, nxz * sizeof(float));

      if(model->useQ) {
        for(int i = 0; i < Q::order; i++) {
          memset(&wq[0][i][iy * nxz], 0, nxz * sizeof(float));
          memset(&wq[1][i][iy * nxz], 0, nxz * sizeof(float));
        }
      }

    }

  }
}

/*
 * Quick and simple method to swap previous and next wavefield pointers
 */
void Wavefield::swapPointers() {
  std::swap(w0, w1);
  std::swap(iq0, iq1);
}

/*
 * Multithreaded allocation routine
 */
float* Wavefield::allocMem3d(int nx, int ny, int nz, int nThreads) {
  size_t gridsize = (size_t)nx * ny * nz;
  size_t nxz = (size_t)nx * nz;
  float *buffer = (float*)_mm_malloc(gridsize * sizeof(float) + 128, 16);
#pragma omp parallel for num_threads(nThreads) schedule(static)
  for(int iy = 0; iy < ny; iy++)
    memset(&buffer[iy * nxz], 0, nxz * sizeof(float));
  count_mem3d++;
  return buffer;
}

