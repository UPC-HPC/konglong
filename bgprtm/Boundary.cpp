/*
 * Boundary.cpp
 *
 */
#include <omp.h>

#include "Boundary.h"
#include "PML.h"
using namespace PMLBUF;
#include "Util.h"
#include "libFFTV/transpose.h"

#include "MpiPrint.h"
using MpiPrint::print1m;

Boundary::Boundary(int bndType, int nx, int ny, int nz, float dx, float dy, float dz, float dt, float vmax, int nThreads) :
    bndType(bndType), nx(nx), ny(ny), nz(nz), dx(dx), dy(dy), dz(dz), dt(dt), vmax(vmax), nThreads(nThreads) {
}

Boundary::~Boundary() {
  if(pmlX1) delete pmlX1;
  if(pmlX2) delete pmlX2;
  if(pmlY1) delete pmlY1;
  if(pmlY2) delete pmlY2;
  if(pmlZ1) delete pmlZ1;
  if(pmlZ2) delete pmlZ2;

  if(pmlX3) delete pmlX3;
  if(pmlX4) delete pmlX4;
  if(pmlY3) delete pmlY3;
  if(pmlY4) delete pmlY4;
  if(pmlZ3) delete pmlZ3;
  if(pmlZ4) delete pmlZ4;
    for(auto &tb_Round12_Pxyz : pmlBuf) {
    for(auto &Round12_Pxyz : tb_Round12_Pxyz) {
      for(auto &Pxyz : Round12_Pxyz) {
        for(auto &buf : Pxyz)
          delete[] buf, buf = nullptr;
      }
    }
  }

  if(work) delete[] work;
}

void Boundary::setBoundary(int nxbnd1, int nxbnd2, int nybnd1, int nybnd2, int nzbnd1, int nzbnd2) {
  print1m("nxbnd1, nxbnd2, nybnd1, nybnd2, nzbnd1, nzbnd2: %d %d %d %d %d %d \n", nxbnd1, nxbnd2, nybnd1, nybnd2, nzbnd1, nzbnd2);
  nxz = (size_t)nx * (size_t)nz;
  nyz = (size_t)ny * (size_t)nz;
  nxy = (size_t)nx * (size_t)ny;

  this->nxbnd1 = nxbnd1;
  this->nxbnd2 = nxbnd2;

  this->nybnd1 = nybnd1;
  this->nybnd2 = nybnd2;

  this->nzbnd1 = nzbnd1;
  this->nzbnd2 = nzbnd2;

  bool limit_slope = false; // no 2nd derivative in this case (yet)
  if(nxbnd1 > 1) {
    pmlX1 = new PML(nx, nxbnd1, nyz, dx, dt, vmax, limit_slope);
  }
  if(nxbnd2 > 1) {
    pmlX2 = new PML(nx, nxbnd2, nyz, dx, dt, vmax, limit_slope);
  }
  if(nybnd1 > 1) {
    pmlY1 = new PML(ny, nybnd1, nxz, dy, dt, vmax, limit_slope);
  }
  if(nybnd2 > 1) {
    pmlY2 = new PML(ny, nybnd2, nxz, dy, dt, vmax, limit_slope);
  }

  if(nzbnd1 > 1) {
    pmlZ1 = new PML(nz, nzbnd1, nxy, dz, dt, vmax, limit_slope);
  }
  if(nzbnd2 > 1) {
    pmlZ2 = new PML(nz, nzbnd2, nxy, dz, dt, vmax, limit_slope);
  }

  if(nxbnd1 > 1) {
    pmlX3 = new PML(nx, nxbnd1, nyz, dx, dt, vmax, limit_slope);
  }
  if(nxbnd2 > 1) {
    pmlX4 = new PML(nx, nxbnd2, nyz, dx, dt, vmax, limit_slope);
  }

  if(nybnd1 > 1) {
    pmlY3 = new PML(ny, nybnd1, nxz, dy, dt, vmax, limit_slope);
  }
  if(nybnd2 > 1) {
    pmlY4 = new PML(ny, nybnd2, nxz, dy, dt, vmax, limit_slope);
  }

  if(nzbnd1 > 1) {
    pmlZ3 = new PML(nz, nzbnd1, nxy, dz, dt, vmax, limit_slope);
  }
  if(nzbnd2 > 1) {
    pmlZ4 = new PML(nz, nzbnd2, nxy, dz, dt, vmax, limit_slope);
  }
allocMemory();
}

void Boundary::allocMemory() {
  int maxlen = max(nxz, nyz);
  work = new float[maxlen * nThreads];

  for(int round = 0; round < 2; round++) {
    for(int i = 0; i < dimPML; i++) {
      if(nzbnd1 > 0) pmlBuf[Z][TOP][round][i] = new float[nzbnd1 * nxy];
      if(nzbnd2 > 0) pmlBuf[Z][BOT][round][i] = new float[nzbnd2 * nxy];
      if(nxbnd1 > 0) pmlBuf[X][TOP][round][i] = new float[nxbnd1 * nyz];
      if(nxbnd2 > 0) pmlBuf[X][BOT][round][i] = new float[nxbnd2 * nyz];
      if(nybnd1 > 0) pmlBuf[Y][TOP][round][i] = new float[nybnd1 * nxz];
      if(nybnd2 > 0) pmlBuf[Y][BOT][round][i] = new float[nybnd2 * nxz];
    }
  }
  cleanMemory();
}

void Boundary::cleanMemory() {

  for(int round = 0; round < 2; round++) {
    for(int i = 0; i < dimPML; i++) {
#pragma omp parallel num_threads(nThreads)
      {
#pragma omp for schedule(static)
        for(int iy = 0; iy < ny; iy++) {
          if(nzbnd1 > 0) memset(pmlBuf[Z][TOP][round][i] + (size_t)iy * nzbnd1 * nx, 0, sizeof(float) * (nzbnd1 * nx));
          if(nzbnd2 > 0) memset(pmlBuf[Z][BOT][round][i] + (size_t)iy * nzbnd2 * nx, 0, sizeof(float) * (nzbnd2 * nx));
          if(nxbnd1 > 0) memset(pmlBuf[X][TOP][round][i] + (size_t)iy * nxbnd1 * nz, 0, sizeof(float) * (nxbnd1 * nz));
          if(nxbnd2 > 0) memset(pmlBuf[X][BOT][round][i] + (size_t)iy * nxbnd2 * nz, 0, sizeof(float) * (nxbnd2 * nz));
        }

#pragma omp for schedule(static)
        for(int ix = 0; ix < nx; ix++) {
          if(nybnd1 > 0) memset(pmlBuf[Y][TOP][round][i] + ix * nybnd1 * nz, 0, sizeof(float) * (nybnd1 * nz));
          if(nybnd2 > 0) memset(pmlBuf[Y][BOT][round][i] + ix * nybnd2 * nz, 0, sizeof(float) * (nybnd2 * nz));
        }
      }
    }
  }

  int maxlen = max(nxz, nyz);
#pragma omp parallel num_threads(nThreads)
  {
    int tid = omp_get_thread_num();
    memset(work + tid * maxlen, 0, sizeof(float) * maxlen);
  }
}

void Boundary::applyX(float *p, int indx) {

#pragma omp parallel num_threads(nThreads)
  {
    int tid = omp_get_thread_num();
#pragma omp for schedule(static)
    for(int iy = 0; iy < ny; iy++) {
      Util::transpose(nz, nx, nz, nx, p + iy * nxz, work + tid * nxz);

      if(indx == 1) {
        if(nxbnd1 > 1) pmlX1->apply(work + tid * nxz, pmlBuf[X][TOP][ROUND1][XX] + iy * nz * nxbnd1, nz, -1);
        if(nxbnd2 > 1) pmlX2->apply(work + tid * nxz, pmlBuf[X][BOT][ROUND1][XX] + iy * nz * nxbnd2, nz, +1);
      } else if(indx == 2) {
        if(nxbnd1 > 1) pmlX3->apply(work + tid * nxz, pmlBuf[X][TOP][ROUND2][XX] + iy * nz * nxbnd1, nz, -1);
        if(nxbnd2 > 1) pmlX4->apply(work + tid * nxz, pmlBuf[X][BOT][ROUND2][XX] + iy * nz * nxbnd2, nz, +1);
      }

      Util::transpose(nx, nz, nx, nz, work + tid * nxz, p + iy * nxz);
    }
  }
}

void Boundary::applyY(float *p, int indx) {

#pragma omp parallel num_threads(nThreads)
  {
    int tid = omp_get_thread_num();
#pragma omp for schedule(static)
    for(int ix = 0; ix < nx; ix++) {
      Util::transpose(nz, ny, nxz, ny, p + ix * nz, work + tid * nyz);

      if(indx == 1) {
        if(nybnd1 > 1) pmlY1->apply(work + tid * nyz, pmlBuf[Y][TOP][ROUND1][YY] + ix * nz * nybnd1, nz, -1);
        if(nybnd2 > 1) pmlY2->apply(work + tid * nyz, pmlBuf[Y][BOT][ROUND1][YY] + ix * nz * nybnd2, nz, +1);
      } else if(indx == 2) {
        if(nybnd1 > 1) pmlY3->apply(work + tid * nyz, pmlBuf[Y][TOP][ROUND2][YY] + ix * nz * nybnd1, nz, -1);
        if(nybnd2 > 1) pmlY4->apply(work + tid * nyz, pmlBuf[Y][BOT][ROUND2][YY] + ix * nz * nybnd2, nz, +1);
      }

      Util::transpose(ny, nz, ny, nxz, work + tid * nyz, p + ix * nz);
    }

  }
}
void Boundary::applyZ(float *p, int indx) {
#pragma omp parallel num_threads(nThreads)
  {
#pragma omp for schedule(static)
    for(int iy = 0; iy < ny; iy++) {
      if(indx == 1) {
        if(nzbnd1 > 1) pmlZ1->apply(p + iy * nxz, pmlBuf[Z][TOP][ROUND1][ZZ] + iy * nx * nzbnd1, nx, -1);
        if(nzbnd2 > 1) pmlZ2->apply(p + iy * nxz, pmlBuf[Z][BOT][ROUND1][ZZ] + iy * nx * nzbnd2, nx, +1);
      } else if(indx == 2) {
        if(nzbnd1 > 1) pmlZ3->apply(p + iy * nxz, pmlBuf[Z][TOP][ROUND2][ZZ] + iy * nx * nzbnd1, nx, -1);
        if(nzbnd2 > 1) pmlZ4->apply(p + iy * nxz, pmlBuf[Z][BOT][ROUND2][ZZ] + iy * nx * nzbnd2, nx, +1);
      }

    }
  }
// Util::print_mem_crc(qZ3, nx, nzbnd1 * ny, "qZ3");
}

