#include <math.h>
#include <string.h>
#include <xmmintrin.h>
#include <omp.h>
#include "FdEngine.h"

// do not include any special engines that link to extern libs
#include "FdEngine_fd.h"
#include "FdEngine_hfd.h"
#include "FdEngine_cfd.h"

#include "jseisIO/jseisUtil.h"
using jsIO::jseisUtil;

#include "fdWrapper.h"
#include "PML.h"
#include "fdm.hpp"
#include "GetPar.h"
#include "ModelLoader.h"

using namespace PMLBUF;
const vector<string> FD::EngineNames { "FFTV", "FFTWR2C", "FFTWR2R", "HFD", "CFD", "CFD_BATCH", "SINCOS", "FD" };

FdEngine::FdEngine(int innx, int inny, int innz, float indx, float indy, float indz, int inNThreads) { // revised by wolf to remove inRhoCN*
  update_nxyz(innx, inny, innz); // non-virtual

  dx = indx;
  dy = indy;
  dz = indz;
  dxi = 1 / dx;
  dyi = 1 / dy;
//  RhoCN* = inRhoCN*; // removed by wolf
  nThreads = inNThreads;

  absType = ABSPML;

  topPMLFlag = true;
  allPMLFlag = true;

  string sourceSurfaceType = global_pars["sourceSurfaceType"].as<string>("ABSORB");

  transform(sourceSurfaceType.begin(), sourceSurfaceType.end(), sourceSurfaceType.begin(), ::toupper);

  //if freesurface option, turn off pml at top boundary
  if(sourceSurfaceType.compare("FREESURFACE") == 0) topPMLFlag = false;

  topPMLFlag = getBool(global_pars["top_pml_flag"], topPMLFlag);
  allPMLFlag = getBool(global_pars["all_pml_flag"], allPMLFlag);
  bool is2nd = isDerive2nd();
  do_pml3d = getBool(global_pars["pml3d"], true);
  pml3d_scaler = global_pars["pml3d_scaler"].as<float>(0.3f);
  dimPML = (!do_pml3d || is2nd) ? 1 : (ny == 1) ? 2 : 3;
}
void FdEngine::update_nxyz(int nx0, int ny0, int nz0) {
  nx = nx0;
  ny = ny0;
  nz = nz0;
  nxz = (size_t)nx * nz;
  nyz = (size_t)ny * nz;
  nxy = (size_t)nx * ny;
  nxyz = nxy * nz;
}

FdEngine::~FdEngine() {
  if(pmlX1) delete pmlX1;
  if(pmlX2) delete pmlX2;
  if(pmlY1) delete pmlY1;
  if(pmlY2) delete pmlY2;
  if(pmlZ1) delete pmlZ1;
  if(pmlZ2) delete pmlZ2;

//  if(pmltX1) delete pmltX1;
//  if(pmltX2) delete pmltX2;
//  if(pmltY1) delete pmltY1;
//  if(pmltY2) delete pmltY2;
//  if(pmltZ1) delete pmltZ1;
//  if(pmltZ2) delete pmltZ2;

  for(auto &tb_Round12_Pxyz : pmlBuf) {
    for(auto &Round12_Pxyz : tb_Round12_Pxyz) {
      for(auto &Pxyz : Round12_Pxyz) {
        for(auto &buf : Pxyz)
          delete[] buf, buf = nullptr;
      }
    }
  }

//  for(auto &tb_Round12_Pxyz : pmlcnnBuf) {
//    for(auto &Round12_Pxyz : tb_Round12_Pxyz) {
//      for(auto &Pxyz : Round12_Pxyz) {
//        for(auto &buf : Pxyz)
//          delete[] buf, buf = nullptr;
//      }
//    }
//  } //removed by wolf
}

void FdEngine::setJacob(Grid *grid, float *jacobx, float *jacoby, float *jacobz) {
  if(!grid) return; // allow nullptr for code re-use
  assertion(nz == grid->nz, "nz=%d does not match grid.nz=%d!", nz, grid->nz);
  this->grid = grid;
  this->gridType = grid->mytype;
  this->jacobx = jacobx;
  this->jacoby = jacoby;
  this->jacobz = jacobz;
}

void FdEngine::setBoundary(int nxbnd1, int nxbnd2, int nybnd1, int nybnd2, int nzbnd1, int nzbnd2, float dt, float vmax) {
  this->nxbnd1 = nxbnd1;
  this->nxbnd2 = nxbnd2;
  this->nybnd1 = nybnd1;
  this->nybnd2 = nybnd2;
  this->nzbnd1 = nzbnd1;
  this->nzbnd2 = nzbnd2;
  this->dt = dt;
  this->vmax = vmax;

  bool limit_slope = false; // isDerive2nd();
  if(nxbnd1 > 1 && allPMLFlag) {
    pmlX1 = new PML(nx, nxbnd1, 1, dx, dt, vmax, limit_slope);
  }
if(nxbnd2 > 1 && allPMLFlag) {
    pmlX2 = new PML(nx, nxbnd2, 1, dx, dt, vmax, limit_slope);
  }

  if(nybnd1 > 1 && allPMLFlag) {
    pmlY1 = new PML(ny, nybnd1, 1, dy, dt, vmax, limit_slope);
  }
  if(nybnd2 > 1 && allPMLFlag) {
    pmlY2 = new PML(ny, nybnd2, 1, dy, dt, vmax, limit_slope);
  }

  if(nzbnd1 > 1 && allPMLFlag && topPMLFlag) {
    pmlZ1 = new PML(nz, nzbnd1, 1, dz, dt, vmax, limit_slope);
  }
  if(nzbnd2 > 1 && allPMLFlag) {
    pmlZ2 = new PML(nz, nzbnd2, 1, dz, dt, vmax, limit_slope);
  }

//  if(nxbnd1 > 1 && allPMLFlag) {
//    pmltX1 = new PML(nx, nxbnd1, 1, dx, dt, vmax, limit_slope);
//  }
//  if(nxbnd2 > 1 && allPMLFlag) {
//    pmltX2 = new PML(nx, nxbnd2, 1, dx, dt, vmax, limit_slope);
//  }
//
//  if(nybnd1 > 1 && allPMLFlag) {
//    pmltY1 = new PML(ny, nybnd1, 1, dy, dt, vmax, limit_slope);
//  }
//  if(nybnd2 > 1 && allPMLFlag) {
//    pmltY2 = new PML(ny, nybnd2, 1, dy, dt, vmax, limit_slope);
//  }
//
//  if(nzbnd1 > 1 && allPMLFlag && topPMLFlag) {
//    pmltZ1 = new PML(nz, nzbnd1, 1, dz, dt, vmax, limit_slope);
//  }
//  if(nzbnd2 > 1 && allPMLFlag) {
//    pmltZ2 = new PML(nz, nzbnd2, 1, dz, dt, vmax, limit_slope);
//  }//Talked with owl, and these six PMLs are initalized for rhoCNN and could be deleted. wolf On Nov 7 2022

  //allocate memory
  for(int round = 0; round < 2; round++)
    for(int i = 0; i < dimPML; i++) {
      if(nzbnd1 > 0) pmlBuf[Z][TOP][round][i] = new float[nzbnd1 * nxy];
      if(nzbnd2 > 0) pmlBuf[Z][BOT][round][i] = new float[nzbnd2 * nxy];
      if(nxbnd1 > 0) pmlBuf[X][TOP][round][i] = new float[nxbnd1 * nyz];
      if(nxbnd2 > 0) pmlBuf[X][BOT][round][i] = new float[nxbnd2 * nyz];
      if(nybnd1 > 0) pmlBuf[Y][TOP][round][i] = new float[nybnd1 * nxz];
      if(nybnd2 > 0) pmlBuf[Y][BOT][round][i] = new float[nybnd2 * nxz];

//      if(RhoCN*) {
//        if(nzbnd1 > 0) pmlcnnBuf[Z][TOP][round][i] = new float[nzbnd1 * nxy];
//        if(nzbnd2 > 0) pmlcnnBuf[Z][BOT][round][i] = new float[nzbnd2 * nxy];
//        if(nxbnd1 > 0) pmlcnnBuf[X][TOP][round][i] = new float[nxbnd1 * nyz];
//        if(nxbnd2 > 0) pmlcnnBuf[X][BOT][round][i] = new float[nxbnd2 * nyz];
//        if(nybnd1 > 0) pmlcnnBuf[Y][TOP][round][i] = new float[nybnd1 * nxz];
//        if(nybnd2 > 0) pmlcnnBuf[Y][BOT][round][i] = new float[nybnd2 * nxz];
//      }//removed by wolf
    }

  cleanPMLMemory();
}
void FdEngine::cleanPMLMemory() {
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
          
//          if(RhoCN*) {
//            if(nzbnd1 > 0) memset(pmlcnnBuf[Z][TOP][round][i] + (size_t)iy * nzbnd1 * nx, 0, sizeof(float) * (nzbnd1 * nx));
//            if(nzbnd2 > 0) memset(pmlcnnBuf[Z][BOT][round][i] + (size_t)iy * nzbnd2 * nx, 0, sizeof(float) * (nzbnd2 * nx));
//            if(nxbnd1 > 0) memset(pmlcnnBuf[X][TOP][round][i] + (size_t)iy * nxbnd1 * nz, 0, sizeof(float) * (nxbnd1 * nz));
//            if(nxbnd2 > 0) memset(pmlcnnBuf[X][BOT][round][i] + (size_t)iy * nxbnd2 * nz, 0, sizeof(float) * (nxbnd2 * nz));
//          }//removed by wolf
        }
#pragma omp for schedule(static)
        for(int ix = 0; ix < nx; ix++) {
          if(nybnd1 > 0) memset(pmlBuf[Y][TOP][round][i] + ix * nybnd1 * nz, 0, sizeof(float) * (nybnd1 * nz));
          if(nybnd2 > 0) memset(pmlBuf[Y][BOT][round][i] + ix * nybnd2 * nz, 0, sizeof(float) * (nybnd2 * nz));
//          if(RhoCN*) { 
//            if(nybnd1 > 0) memset(pmlcnnBuf[Y][TOP][round][i] + ix * nybnd1 * nz, 0, sizeof(float) * (nybnd1 * nz));
//            if(nybnd2 > 0) memset(pmlcnnBuf[Y][BOT][round][i] + ix * nybnd2 * nz, 0, sizeof(float) * (nybnd2 * nz));
//          }//removed by wolf
        }
        
      }
    } 
  } 
} 
ModelType FdEngine::determineModelType() {
  if(!global_pars[GLOBAL]) return ISO; // in case called without a good jobdeck

  if(global_pars[GLOBAL]["dip"] || global_pars[GLOBAL]["azimuth"] || global_pars[GLOBAL]["dipx"] || global_pars[GLOBAL]["dipy"]
      || global_pars[GLOBAL]["pjx"] || global_pars[GLOBAL]["pjy"]) {
    return TTI;
  }

  if(global_pars[LOCAL]["dip"] || global_pars[LOCAL]["azimuth"] || global_pars[LOCAL]["dipx"] || global_pars[LOCAL]["dipy"]
      || global_pars[LOCAL]["pjx"] || global_pars[LOCAL]["pjy"]) {
    return TTI;
  }

  if(global_pars[GLOBAL]["delta"] || global_pars[GLOBAL]["epsilon"]) return VTI;

  if(global_pars[LOCAL]["delta"] || global_pars[LOCAL]["epsilon"]) return VTI;

  return ISO;
}

string FdEngine::getEngineName() {
  string engineName = global_pars["Engine"].as<string>("FFTV");
  transform(engineName.begin(), engineName.end(), engineName.begin(), ::toupper);
  return engineName;
}
float FdEngine::getFdDispersion() {
  float dispersion = 1.0f;
  int force_tti = global_pars["force_tti"].as<int>(0);
  if(global_pars["fd_dispersion"]) {
    dispersion = global_pars["fd_dispersion"].as<float>();
    assertion(dispersion >= 0.1f && dispersion <= 1.0f, "Manual fd_dispersion must between 0.1-1.0 !");
    return dispersion;
  }

  ModelType modelType = determineModelType();
  string engineName = getEngineName();
  if(engineName.rfind("CFD", 0) == 0) dispersion = FdEngine_cfd::getDispersion();
  else if(engineName.rfind("FD", 0) == 0) dispersion = FdEngine_fd::getDispersion(modelType >= TTI || force_tti);

  return dispersion;
}

bool FdEngine::isDerive2nd() {
  ModelType modelType = determineModelType();
  int force_tti = global_pars["force_tti"].as<int>(0);
  if(modelType >= TTI || force_tti) return false;
  string engineName = getEngineName();
  return (engineName.rfind("FD", 0) == 0);
}
void FdEngine::dy_3D(int nth, float *in, float *py, vector<float*> &wrk2d, int iacc, int isign) {
  assertion(nth == 1, "Derive_nth=%d, 2nd derivative not implemented!", nth);

  if(ny > 1) {
#pragma omp parallel num_threads(nThreads) if(nThreads>1)
    {
      int tid = omp_get_thread_num();
#pragma omp for schedule(static)
      for(int ix = 0; ix < nx; ix++) {
        dy1_2D(in + ix * nz, py + ix * nz, wrk2d[tid], ix, iacc, isign);
      }
    }
  }
}

void FdEngine::dy_3D(int nth, float *in, float *py, vector<float*> &wrk2d, int iacc, int isign, float *velVolume, float *rhoVolume) {
  assertion(nth == 1, "Derive_nth=%d, 2nd derivative not implemented!", nth);

  if(ny > 1) {
#pragma omp parallel num_threads(nThreads) if(nThreads>1)
    {
      int tid = omp_get_thread_num();
#pragma omp for schedule(static)
      for(int ix = 0; ix < nx; ix++) {
        dy1_2D(in + ix * nz, py + ix * nz, wrk2d[tid], ix, iacc, isign, velVolume + ix* nz, rhoVolume? rhoVolume + ix * nz: nullptr);
      }
    }
  }
}
void FdEngine::dxz_2D(int nth, float *in, float *px, float *pz, float *wrk, int iy, int iacc, int isign) {
  assertion(nth == 1, "Derive_nth=%d, 2nd derivative not implemented!", nth);

  if(pz) dz1_2D(in, pz, wrk, iy, iacc, isign);
  if(px && nx > 1) dx1_2D(in, px, wrk, iy, iacc, isign);
}

void FdEngine::pml1_zbnd(float *pz, int ix, int iy, int isign) {
  pml1_zbnd(pz, ix, iy, isign, ZZ, 1.0f);
  if(do_pml3d) {
    pml1_xbnd(pz, ix, iy, isign, XZ, pml3d_scaler);
    pml1_ybnd(pz, ix, iy, isign, YZ, pml3d_scaler);
  }
}

void FdEngine::pml1_xbnd(float *px, int ix, int iy, int isign) {
  pml1_xbnd(px, ix, iy, isign, XX, 1.0f);
  if(do_pml3d) {
    pml1_zbnd(px, ix, iy, isign, ZX, pml3d_scaler);
    pml1_ybnd(px, ix, iy, isign, YX, pml3d_scaler);
  }
}

void FdEngine::pml1_ybnd(float *py, int ix, int iy, int isign) {
  pml1_ybnd(py, ix, iy, isign, YY, 1.0f);
  if(do_pml3d && ny > 1) {
    pml1_zbnd(py, ix, iy, isign, ZY, pml3d_scaler);
    pml1_xbnd(py, ix, iy, isign, XY, pml3d_scaler);
  }
}
void FdEngine::pml1_zbnd(float *pz, int ix, int iy, int isign, PMLBUF::Pxyz jzxy, float scaler) {
  size_t ixy = (size_t)iy * nx + ix;
  if(pmlZ1) pmlZ1->apply_single(pmlZ1->coef, scaler, pz, pmlBuf[Z][TOP][isign > 0 ? ROUND1 : ROUND2][jzxy] + ixy * nzbnd1);
  if(pmlZ2) pmlZ2->apply_single(pmlZ2->feoc, scaler, pz + nz - nzbnd2, pmlBuf[Z][BOT][isign > 0 ? ROUND1 : ROUND2][jzxy] + ixy * nzbnd2);
}

void FdEngine::pml1_xbnd(float *px, int ix, int iy, int isign, PMLBUF::Pxyz jzxy, float scaler) {
  if(ix < nxbnd1 && pmlX1) {
    size_t off = ((size_t)iy * nxbnd1 + ix) * nz;
    pmlX1->apply_single_trans(pmlX1->coef[ix] * scaler, px, pmlBuf[X][TOP][isign > 0 ? ROUND1 : ROUND2][jzxy] + off, nz);
  }
  if(ix >= nx - nxbnd2 && pmlX2) {
    size_t off = ((size_t)iy * nxbnd2 + ix - nx + nxbnd2) * nz;
    pmlX2->apply_single_trans(pmlX2->coef[nx - 1 - ix] * scaler, px, pmlBuf[X][BOT][isign > 0 ? ROUND1 : ROUND2][jzxy] + off, nz);
  }
}

void FdEngine::pml1_ybnd(float *py, int ix, int iy, int isign, PMLBUF::Pxyz jzxy, float scaler) {
  if(iy < nybnd1 && pmlY1) {
    size_t off = ((size_t)iy * nx + ix) * nz;
    pmlY1->apply_single_trans(pmlY1->coef[iy] * scaler, py, pmlBuf[Y][TOP][isign > 0 ? ROUND1 : ROUND2][jzxy] + off, nz);
  }
  if(iy >= ny - nybnd2 && pmlY2) {
    size_t off = ((size_t)(iy - ny + nybnd2) * nx + ix) * nz;
    pmlY2->apply_single_trans(pmlY2->coef[ny - 1 - iy] * scaler, py, pmlBuf[Y][BOT][isign > 0 ? ROUND1 : ROUND2][jzxy] + off, nz);
  }
}

void FdEngine::pml2_zbnd(float *trace, int ix, int iy) {
  size_t ixy = (size_t)iy * nx + ix;
  if(pmlZ1) pmlZ1->apply2_single(pmlZ1->coef, trace, pmlBuf[Z][TOP][ROUND1][ZZ] + ixy * nzbnd1, pmlBuf[Z][TOP][ROUND2][ZZ] + ixy * nzbnd1);
  if(pmlZ2) pmlZ2->apply2_single(pmlZ2->feoc, trace + nz - nzbnd2, pmlBuf[Z][BOT][ROUND1][ZZ] + ixy * nzbnd2,
                                 pmlBuf[Z][BOT][ROUND2][ZZ] + ixy * nzbnd2);
}

void FdEngine::pml2_xbnd(float *trace, int ix, int iy) {
  if(ix < nxbnd1 && pmlX1) {
    size_t off = ((size_t)iy * nxbnd1 + ix) * nz;
#if 1
    pmlX1->apply2_single_trans(pmlX1->coef[ix], trace, pmlBuf[X][TOP][ROUND1][XX] + off, pmlBuf[X][TOP][ROUND2][XX] + off, nz);
#else // less efficient
    pmlX1->apply_single_trans(pmlX1->coef[ix], trace, pmlBuf[X][TOP][ROUND1][XX] + off, nz);
    pmlX1->apply_single_trans(pmlX1->coef[ix], trace, pmlBuf[X][BOT][ROUND1][XX] + off, nz);
#endif
  }
  if(ix >= nx - nxbnd2 && pmlX2) {
    size_t off = ((size_t)iy * nxbnd2 + ix - nx + nxbnd2) * nz;
#if 1
    pmlX2->apply2_single_trans(pmlX2->coef[nx - 1 - ix], trace, pmlBuf[X][BOT][ROUND1][XX] + off, pmlBuf[X][BOT][ROUND2][XX] + off, nz);
#else // less efficient
    pmlX2->apply_single_trans(pmlX2->coef[nx - 1 - ix], trace, pmlBuf[X][BOT][ROUND1][XX] + off, nz);
    pmlX2->apply_single_trans(pmlX2->coef[nx - 1 - ix], trace, pmlBuf[X][BOT][ROUND2][XX] + off, nz);
#endif
  }
}

void FdEngine::pml2_ybnd(float *trace, int ix, int iy) {
  if(iy < nybnd1 && pmlY1) {
    size_t off = ((size_t)iy * nx + ix) * nz;
    pmlY1->apply2_single_trans(pmlY1->coef[iy], trace, pmlBuf[Y][TOP][ROUND1][YY] + off, pmlBuf[Y][TOP][ROUND2][YY] + off, nz);
  }
  if(iy >= ny - nybnd2 && pmlY2) {
    size_t off = ((size_t)(iy - ny + nybnd2) * nx + ix) * nz;
    pmlY2->apply2_single_trans(pmlY2->coef[ny - 1 - iy], trace, pmlBuf[Y][BOT][ROUND1][YY] + off, pmlBuf[Y][BOT][ROUND2][YY] + off, nz);
  }
}

void FdEngine::pml2_bnd(float *trace, int ix, int iy) {
  pml2_zbnd(trace, ix, iy);
  pml2_xbnd(trace, ix, iy);
  pml2_ybnd(trace, ix, iy);
}

size_t FdEngine::getSizeBuf2d() {
  size_t size2d = max((nx + 4) * (size_t)nz, (ny + 4) * (size_t)nz);
  size2d = max((nz + 4) * (size_t)nx, size2d);
  size2d *= 2;
  return size2d;
}

void FdEngine::init_sine(float *data, int nx, int ny, int nz, float nosc) {
  //return; // FIXME: turn it on if need it
  int nyquist = 0;
#pragma omp parallel for
  for(int iy = 0; iy < ny; iy++) {
    for(int ix = 0; ix < nx; ix++)
      for(int iz = 0; iz < nz; iz++) {
        size_t ixyz = iz + nz * (ix + (size_t)nx * iy);
        if(nyquist > 1) data[ixyz] = sinf(2 * M_PI * nosc * ix * ix / nx / nx) * cosf(M_PI * iz);
        else {
          data[ixyz] = sinf(2 * M_PI * nosc * ix * ix / nx / nx) * sinf(2 * M_PI * nosc * iz * iz / nz / nz); // 2D
          if(nyquist) data[ixyz] += cosf(M_PI * iz);
        }
        if(ny > 1) data[ixyz] *= sinf(2 * M_PI * nosc * iy * iy / ny / ny);
      }
  }

}
int FdEngine::benchmark(int argc, char **argv) {
#define LOOP_FACTOR 100.0
  int N1 = 512, N2 = 512, N3 = 512;
  if(argc > 1) N1 = atoi(argv[1]);
  if(argc > 2) N2 = atoi(argv[2]);
  if(argc > 3) N3 = atoi(argv[3]);

  int savejs = 0;
  int force_dx1 = 0;

  if(argc > 4) savejs = atoi(argv[4]);
  if(argc > 5) force_dx1 = atoi(argv[5]);
  int nloop = (int)ceil(LOOP_FACTOR * 200 / N1 * 200 / N2 * 200 / N3);
  if(savejs) nloop = 1;

  int nThreads = omp_get_max_threads();
  float dx = 50.0f, dy = 50.0f, dz = 50.0f;
  FdEngine_fd fd(N2, N3, N1, dx, dy, dz, nThreads);
  float dispersion = fd.dispersion_factor;
  fd.update_nxyz(N2 / dispersion, N3 / dispersion, N1 / dispersion);
  fd.update_buffers();
  fd.setJacob(nullptr, nullptr, nullptr, nullptr);

  int nz = fd.nz, nx = fd.nx, ny = fd.ny;
  size_t nxy = fd.nxy, nxz = fd.nxz, nyz = fd.nyz, nxyz = fd.nxyz;

  size_t size2d = max((nx + 4) * (size_t)nz, (ny + 4) * (size_t)nz);
  size2d = max((nz + 4) * (size_t)nx, size2d);
  size2d *= 2;

  printf("num_threads=%d, n1=%d, n2=%d, n3=%d (nz=%d, nx=%d, ny=%d)\n", nThreads, N1, N2, N3, nz, nx, ny), fflush(stdout);
  vector<float> vbuf(nxyz, 0), vx(nxyz, 0), vy(nxyz, 0), vz(nxyz, 0), vv(nxyz, 0);
  vector<vector<float>> wrk(nThreads, vector<float>(size2d));
FdEngine::init_sine(&vbuf[0], nx, ny, nz);
  if(savejs) jseisUtil::save_zxy("sine.js", &vbuf[0], nz, nx, ny, dz, dx, dy);

  double t0, t1, t2, t3, t4;

#if 1
  t0 = omp_get_wtime();
  for(int i = 0; i < nloop; i++) {
    if(!fd.do_laplace || force_dx1) {
#pragma omp parallel for num_threads(nThreads) if(ny>1)
      for(int iy = 0; iy < ny; iy++) {
        int tid = omp_get_thread_num();
        fd.dx1_2D(&vbuf[iy * nxz], &vx[iy * nxz], &wrk[tid][0], iy, 0, 1);
      }
    } else fd.laplacian(&vbuf[0], &vx[0], nullptr, nullptr, FD::DX2);
  }
  printf("   px time * %d = %.3f\n", nloop, t1 = omp_get_wtime() - t0);
  if(savejs) jseisUtil::save_zxy("sine_px.js", &vx[0], nz, nx, ny, dz, dx, dy);

  t0 = omp_get_wtime();
  for(int i = 0; i < nloop; i++) {
    if(!fd.do_laplace || force_dx1) {
#pragma omp parallel for num_threads(nThreads) if(ny>1)
      for(int iy = 0; iy < ny; iy++) {
        int tid = omp_get_thread_num();
        fd.dz1_2D(&vbuf[iy * nxz], &vz[iy * nxz], &wrk[tid][0], iy, 0, 1);
      }
    } else fd.laplacian(&vbuf[0], &vz[0], nullptr, nullptr, FD::DZ2);
  }
  printf("   pz time * %d = %.3f\n", nloop, t2 = omp_get_wtime() - t0);
  if(savejs) jseisUtil::save_zxy("sine_pz.js", &vz[0], nz, nx, ny, dz, dx, dy);

  t0 = omp_get_wtime();
for(int i = 0; i < nloop; i++) {
    if(!fd.do_laplace || force_dx1) {
      if(ny > 1) {
#pragma omp parallel for num_threads(nThreads)
        for(int ix = 0; ix < nx; ix++) {
          int tid = omp_get_thread_num();
          fd.dy1_2D(&vbuf[ix * nz], &vy[ix * nz], &wrk[tid][0], ix, 0, 1);
        }
      }
    } else fd.laplacian(&vbuf[0], &vy[0], nullptr, nullptr, FD::DY2);
  }
  printf("   py time * %d = %.3f\n", nloop, t3 = omp_get_wtime() - t0);
  if(savejs) jseisUtil::save_zxy("sine_py.js", &vy[0], nz, nx, ny, dz, dx, dy);
#endif

  if(fd.do_laplace) {
    t0 = omp_get_wtime();
    for(int i = 0; i < nloop; i++) {
      fd.laplacian(&vbuf[0], &vv[0]);
      //fd.laplacian(&vbuf[0], &vz[0], &vx[0], &vy[0]);
    }
    printf("   unrolled lap time * %d = %.3f\n", nloop, t4 = omp_get_wtime() - t0);
    if(savejs) jseisUtil::save_zxy("sine_lap.js", &vv[0], nz, nx, ny, dz, dx, dy);
  }

  return 0;
#undef LOOP_FACTOR

  return 0;
}
/*
 mpic++ -c -o obj/FdEngine.o FdEngine.cpp -DMAIN=1 -std=c++14 -fopenmp -fPIC -g -Wall -Wno-unused -O3 -ffast-math -DNDEBUG -fstack-protector-all  -march=native -I../libfftavx -I../model_builder -I -I -I/home/owl/local/seiswave/include -m64 -I/opt/intel/oneapi/mkl/latest/include -DNO_OIIO=1 -Wno-reorder && \
 $CXX obj/FdEngine.o -Wl,-no-undefined -fopenmp -Wl,-rpath=/home/owl/local/seiswave/lib -L/home/owl/local/seiswave/lib -Wl,-rpath=/opt/intel/oneapi/mkl/latest/lib/intel64,-rpath=/opt/intel/oneapi/mkl/latest/../compiler/lib/intel64,-rpath=/opt/intel/oneapi/mkl/latest/../../compiler/latest/linux/compiler/lib/intel64 -L/opt/intel/oneapi/mkl/latest/lib/intel64 -L/opt/intel/oneapi/mkl/latest/../compiler/lib/intel64 -L/opt/intel/oneapi/mkl/latest/../../compiler/latest/linux/compiler/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl -l:libWaveProp.a /home/owl/local/lib.CentOS/liblevmar.a  -lmodbuild -lswio -l:libjseisIO.a  -lCommon -l:libyaml-cpp.a  -lfftv -lFortranCode -lz -lm -Wl,-rpath=/opt/rh/devtoolset-9/libportable -L/opt/rh/devtoolset-9/libportable -lgfortran -o test
 */
Node global_pars;
int main(int argc, char **argv) {
  return FdEngine::benchmark(argc, argv);
}

#endif
            
