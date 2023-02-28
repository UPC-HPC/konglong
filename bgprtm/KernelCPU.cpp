#include "KernelCPU.h"

#include <xmmintrin.h>
#include <string.h>
#include <omp.h>
#include "FdEngine.h"
#include "FdEngine_r2c.h"
#include "FdEngine_r2r.h"
#include "FdEngine_hfd.h"
#include "FdEngine_cfd.h"
#include "FdEngine_fd.h"

#include "Source.h"
#include "Grid.h"
#include "timing.h"

#include "libCommon/CommonMath.h"
#include "jseisIO/jseisUtil.h"
using jsIO::jseisUtil;

#include "MpiPrint.h"
using MpiPrint::print1m;

KernelCPU::KernelCPU(int nx, int ny, int nz, float dx, float dy, float dz, int engine, int nThreads) ://removed RhoCN* by wolf
    nx(nx), ny(ny), nz(nz), dx(dx), dy(dy), dz(dz), engine(engine), nThreads(nThreads) {
  nxz = (size_t)nx * nz;
  nxy = (size_t)nx * ny;

  if(engine == FD::FFTWR2C) derive = new FdEngine_r2c(nx, ny, nz, dx, dy, dz, nThreads);//removed RhoCN* by wolf
  else if(engine == FD::FFTWR2R) derive = new FdEngine_r2r(nx, ny, nz, dx, dy, dz, nThreads);
  else if(engine == FD::HFD) derive = new FdEngine_hfd(nx, ny, nz, dx, dy, dz, nThreads);
  else if(engine == FD::CFD) derive = new FdEngine_cfd(nx, ny, nz, dx, dy, dz, nThreads);
  else if(engine == FD::FD) derive = new FdEngine_fd(nx, ny, nz, dx, dy, dz, nThreads);


  size2d = derive->getSizeBuf2d();

  force_tti = global_pars["force_tti"].as<int>(0);

  allocMemory();
}

KernelCPU::~KernelCPU() {
  for(int tid = 0; tid < nThreads; tid++) {
    _mm_free(wrk2d[tid]);
    _mm_free(px2d[tid]);
    _mm_free(pz2d[tid]);
  }
//  _mm_free(prho); // removed by wolf

  delete derive;
}

void KernelCPU::cleanPMLMemory() {
  derive->cleanPMLMemory();
}

void KernelCPU::allocMemory() {
//  prho = (float*)_mm_malloc(sizeof(float) * nxz * ny + 128, 16);//removed by wolf

  wrk2d.resize(nThreads);
  px2d.resize(nThreads);
  pz2d.resize(nThreads);
#pragma omp parallel num_threads(nThreads)
  {
    int tid = omp_get_thread_num();
    wrk2d[tid] = (float*)_mm_malloc(sizeof(float) * size2d * nThreads + 128, 16);
    px2d[tid] = (float*)_mm_malloc(sizeof(float) * nxz * nThreads + 128, 16);
    pz2d[tid] = (float*)_mm_malloc(sizeof(float) * nxz * nThreads + 128, 16);
    memset(wrk2d[tid], 0, sizeof(float) * size2d);
    memset(px2d[tid], 0, sizeof(float) * nxz);
    memset(pz2d[tid], 0, sizeof(float) * nxz);
  }
}

void KernelCPU::setModel(float *vel, float *rho, float *del, float *eps, float *pjx, float *pjy) {
  this->velVolume = vel;
  this->rhoVolume = rho;
  this->delVolume = del;
  this->epsVolume = eps;
  this->pjxVolume = pjx;
  this->pjyVolume = pjy;
}

void KernelCPU::setJacob(Grid *grid, float *jacobx, float *jacoby, float *jacobz) {
  derive->setJacob(grid, jacobx, jacoby, jacobz);
}

void KernelCPU::setBoundary(int nxbnd1, int nxbnd2, int nybnd1, int nybnd2, int nzbnd1, int nzbnd2, float dt, float vmax) {
  derive->setBoundary(nxbnd1, nxbnd2, nybnd1, nybnd2, nzbnd1, nzbnd2, dt, vmax);
}

void KernelCPU::TTI(float *p0, float *pb, float *pr, ModelType modelType) {

  if(ny > 1) derive->dy_3D(1, p0, pr, wrk2d, 0, 1);

#pragma omp parallel for num_threads(nThreads) if(nThreads>1 && ny>1)
  for(int iy = 0; iy < ny; iy++) {
    int tid = omp_get_thread_num();
//    print1m("iy=%03d, tid=%02d, ny=%d (TTI) \n", iy, tid, ny), fflush(stdout);
    derive->dxz_2D(1, p0 + iy * nxz, px2d[tid], pz2d[tid], wrk2d[tid], iy, 0, 1);

    if(rhoVolume) divideRho(px2d[tid], pr + iy * nxz, pz2d[tid], iy); // removed RhoCN* by wolf

    if(modelType == ::TTI) ScalarTTI(px2d[tid], pr + iy * nxz, pz2d[tid], iy);
    else if(modelType == ::VTI) ScalarVTI(px2d[tid], pr + iy * nxz, pz2d[tid], iy);

    derive->dx1_2D(px2d[tid], pb + iy * nxz, wrk2d[tid], iy, 1, -1, velVolume + iy * nxz, rhoVolume?rhoVolume+iy * nxz:nullptr);

    derive->dz1_2D(pz2d[tid], pb + iy * nxz, wrk2d[tid], iy, 1, -1, velVolume + iy * nxz, rhoVolume?rhoVolume+iy * nxz:nullptr);

  }

  if(ny > 1) derive->dy_3D(1, pr, pb, wrk2d, 1, -1, velVolume, rhoVolume);
  // if(libCommon::maxfabs(p0, nz, nx * ny) != 0) jseisUtil::save_zxy("/tmp/tti.js", pb, nz, nx, ny, dz, dx, dy), exit(0);

  //if(rhoVolume && RhoCN* == 0) multiplyRho(pb); // removed by wolf
  //if(rhoVolume && RhoCN* == 1) multiplycnnRho(pb, p0, prho); // only for ISO //TODO
}

void KernelCPU::VTI(float *p0, float *pb, float *pr) {
  TTI(p0, pb, pr, ::VTI);
}

void KernelCPU::ISO(float *p0, float *pb, float *pr, int bndType, int iz0, int tprlen) { // p0->pb

  timeRecorder.start(DERIVATIVE_TIME);
  int do_derive2nd = (derive->derive_cap & FD::order2) && !rhoVolume && !force_tti; //removed RhoCN* by wolf
  int do_laplacian = derive->do_laplace && do_derive2nd;
  static int printed;
  if(!printed) printed = 1, print1m("do_laplacian=%d\n", do_laplacian);
  if(do_laplacian) { // laplacian has it's own OMP loops
    derive->laplacian(p0, pr);
    applyVel_minus(pr, pb);

    // if(libCommon::maxfabs(p0, nz, nx * ny) != 0) jseisUtil::save_zxy("/tmp/iso.js", pb, nz, nx, ny, dz, dx, dy), exit(0);

    //if(rhoVolume && RhoCN* == 1) multiplycnnRho(pb, p0, prho); // RhoCN*==0 won't go to this branch // removed by wolf

#if 0
  } else if(do_derive2nd) {
#pragma omp parallel num_threads(nThreads) if(nThreads>1)
    {
      int tid = omp_get_thread_num();
      int ny_pll = (ny == 1) ? nThreads : ny;
#pragma omp for schedule(static)
      for(int izy = 0; izy < ny_pll; izy++) {
        int iy = (ny == 1) ? 0 : izy;
        int iz0 = (ny == 1) ? min(tid * ((nz - 1) / nThreads + 1), nz) : 0;
        int iz1 = (ny == 1) ? min((tid + 1) * ((nz - 1) / nThreads + 1), nz) : nz;
        derive->dx2(p0 + iy * nxz, pb + iy * nxz, wrk2d[tid], iy, iz0, iz1, 0);
      }

      { // no for-loop needed, manually spread nxy over nThreads
        size_t ixy0 = min(tid * ((nxy - 1) / nThreads + 1), nxy);
        size_t ixy1 = min((tid + 1) * ((nxy - 1) / nThreads + 1), nxy);
        derive->dz2(p0 + ixy0 * nz, pb + ixy0 * nz, wrk2d[tid], ixy0, ixy1, 1);
      }

      if(ny > 1) {
#pragma omp for schedule(static)
        for(int ix = 0; ix < nx; ix++) {
          derive->dy2(p0 + ix * nz, pb + ix * nz, wrk2d[tid], ix, 1);
        }
      }
    }
    //if(rhoVolume && RhoCN* == 1) multiplycnnRho(pb, p0, prho); // RhoCN*==0 won't go to this branch //removed by wolf
#endif
  } else { // TTI-alike version
    TTI(p0, pb, pr, ::ISO);
  }
  timeRecorder.end(DERIVATIVE_TIME);

}

void KernelCPU::ScalarTTI(float *wx, float *wy, float *wz, int iy) {
  if(ny == 1) {
    ScalarTTI2D(wx, wz, iy);
    return;
  }
  size_t offset = (size_t)iy * nxz;
  float *__restrict eps = epsVolume + offset;
  float *__restrict del = delVolume + offset;
  float *__restrict pjx = pjxVolume + offset;
  float *__restrict pjy = pjyVolume + offset;

  __m128 one = _mm_set1_ps(1.0f);
  __m128 two = _mm_set1_ps(2.0f);
  __m128 eight = _mm_set1_ps(8.0f);
  __m128 half = _mm_set1_ps(0.5f);
  __m128 small = _mm_set1_ps(1e-32);

  for(size_t i = 0; i < nxz; i += SSEsize) {
    __m128 myeps = _mm_load_ps(eps + i);
    __m128 mydel = _mm_sub_ps(myeps, _mm_load_ps(del + i));
    myeps = _mm_add_ps(one, _mm_mul_ps(two, myeps));
    __m128 axisx = _mm_load_ps(pjx + i);
    __m128 axisy = _mm_load_ps(pjy + i);
    __m128 axisz = _mm_sqrt_ps(_mm_sub_ps(one, _mm_add_ps(_mm_mul_ps(axisx, axisx), _mm_mul_ps(axisy, axisy))));
    __m128 vectx = _mm_load_ps(wx + i);
    __m128 vecty = _mm_load_ps(wy + i);
    __m128 vectz = _mm_load_ps(wz + i);

    __m128 vect2 = _mm_add_ps(_mm_add_ps(_mm_mul_ps(vectx, vectx), _mm_mul_ps(vecty, vecty)), _mm_mul_ps(vectz, vectz));
    __m128 veczz = _mm_add_ps(_mm_add_ps(_mm_mul_ps(vectx, axisx), _mm_mul_ps(vecty, axisy)), _mm_mul_ps(vectz, axisz));
    __m128 vecz2 = _mm_mul_ps(veczz, veczz);
    __m128 vech2 = _mm_sub_ps(vect2, vecz2);
    __m128 eclip = _mm_add_ps(_mm_mul_ps(myeps, vech2), vecz2);

    __m128 parts = _mm_sqrt_ps(
        _mm_sub_ps(
            one, _mm_div_ps(_mm_mul_ps(eight, _mm_mul_ps(mydel, _mm_mul_ps(vech2, vecz2))), _mm_add_ps(_mm_mul_ps(eclip, eclip), small))));
    __m128 scale = _mm_mul_ps(half, _mm_add_ps(one, parts));
    scale = _mm_min_ps(scale, one);
    axisx = _mm_mul_ps(veczz, axisx);
    axisy = _mm_mul_ps(veczz, axisy);
    axisz = _mm_mul_ps(veczz, axisz);
    _mm_store_ps(wx + i, _mm_mul_ps(_mm_add_ps(axisx, _mm_mul_ps(_mm_sub_ps(vectx, axisx), myeps)), scale));
    _mm_store_ps(wy + i, _mm_mul_ps(_mm_add_ps(axisy, _mm_mul_ps(_mm_sub_ps(vecty, axisy), myeps)), scale));
    _mm_store_ps(wz + i, _mm_mul_ps(_mm_add_ps(axisz, _mm_mul_ps(_mm_sub_ps(vectz, axisz), myeps)), scale));
  }
}

void KernelCPU::ScalarTTI2D(float *wx, float *wz, int iy) {
  float *__restrict eps = epsVolume;
  float *__restrict del = delVolume;
  float *__restrict pjx = pjxVolume;

  __m128 one = _mm_set1_ps(1.0f);
  __m128 two = _mm_set1_ps(2.0f);
  __m128 eight = _mm_set1_ps(8.0f);
  __m128 half = _mm_set1_ps(0.5f);
  __m128 small = _mm_set1_ps(1e-32);

  for(size_t i = 0; i < nxz; i += SSEsize) {
    __m128 myeps = _mm_load_ps(eps + i);
    __m128 mydel = _mm_sub_ps(myeps, _mm_load_ps(del + i));
    myeps = _mm_add_ps(one, _mm_mul_ps(two, myeps));
    __m128 axisx = _mm_load_ps(pjx + i);
    __m128 axisz = _mm_sqrt_ps(_mm_sub_ps(one, _mm_mul_ps(axisx, axisx)));
    __m128 vectx = _mm_load_ps(wx + i);
    __m128 vectz = _mm_load_ps(wz + i);

    __m128 vect2 = _mm_add_ps(_mm_mul_ps(vectx, vectx), _mm_mul_ps(vectz, vectz));
    __m128 veczz = _mm_add_ps(_mm_mul_ps(vectx, axisx), _mm_mul_ps(vectz, axisz));
    __m128 vecz2 = _mm_mul_ps(veczz, veczz);
    __m128 vech2 = _mm_sub_ps(vect2, vecz2);
    __m128 eclip = _mm_add_ps(_mm_mul_ps(myeps, vech2), vecz2);

    __m128 parts = _mm_sqrt_ps(
        _mm_sub_ps(
            one, _mm_div_ps(_mm_mul_ps(eight, _mm_mul_ps(mydel, _mm_mul_ps(vech2, vecz2))), _mm_add_ps(_mm_mul_ps(eclip, eclip), small))));
    __m128 scale = _mm_mul_ps(half, _mm_add_ps(one, parts));
    scale = _mm_min_ps(scale, one);
    axisx = _mm_mul_ps(veczz, axisx);
    axisz = _mm_mul_ps(veczz, axisz);
    _mm_store_ps(wx + i, _mm_mul_ps(_mm_add_ps(axisx, _mm_mul_ps(_mm_sub_ps(vectx, axisx), myeps)), scale));
    _mm_store_ps(wz + i, _mm_mul_ps(_mm_add_ps(axisz, _mm_mul_ps(_mm_sub_ps(vectz, axisz), myeps)), scale));
  }
}

void KernelCPU::ScalarVTI(float *wx, float *wy, float *wz, int iy) {
  if(ny == 1) {
    ScalarVTI2D(wx, wz, iy);
    return;
  }
  size_t offset = (size_t)iy * nxz;
  float *__restrict eps = epsVolume + offset;
  float *__restrict del = delVolume + offset;

  __m128 one = _mm_set1_ps(1.0f);
  __m128 two = _mm_set1_ps(2.0f);
  __m128 eight = _mm_set1_ps(8.0f);
  __m128 half = _mm_set1_ps(0.5f);
  __m128 small = _mm_set1_ps(1e-32);

  for(size_t i = 0; i < nxz; i += SSEsize) {
    __m128 myeps = _mm_load_ps(eps + i);
    __m128 mydel = _mm_sub_ps(myeps, _mm_load_ps(del + i));
    myeps = _mm_add_ps(one, _mm_mul_ps(two, myeps));
    __m128 vectx = _mm_load_ps(wx + i);
    __m128 vecty = _mm_load_ps(wy + i);
    __m128 vectz = _mm_load_ps(wz + i);
    __m128 vecx2 = _mm_add_ps(_mm_mul_ps(vectx, vectx), _mm_mul_ps(vecty, vecty));
    __m128 vecz2 = _mm_mul_ps(vectz, vectz);
    __m128 eclip = _mm_add_ps(_mm_mul_ps(myeps, vecx2), vecz2);
    __m128 parts = _mm_sqrt_ps(
        _mm_sub_ps(
            one, _mm_div_ps(_mm_mul_ps(eight, _mm_mul_ps(mydel, _mm_mul_ps(vecx2, vecz2))), _mm_add_ps(_mm_mul_ps(eclip, eclip), small))));
    __m128 scale = _mm_mul_ps(half, _mm_add_ps(one, parts));
    __m128 scale2 = _mm_mul_ps(myeps, _mm_min_ps(scale, one));

    _mm_store_ps(wx + i, _mm_mul_ps(vectx, scale2));
    _mm_store_ps(wy + i, _mm_mul_ps(vecty, scale2));
  }

}

void KernelCPU::ScalarVTI2D(float *wx, float *wz, int iy) {
  float *__restrict eps = epsVolume;
  float *__restrict del = delVolume;

  __m128 one = _mm_set1_ps(1.0f);
  __m128 two = _mm_set1_ps(2.0f);
  __m128 eight = _mm_set1_ps(8.0f);
  __m128 half = _mm_set1_ps(0.5f);
  __m128 small = _mm_set1_ps(1e-32);

  for(size_t i = 0; i < nxz; i += SSEsize) {
    __m128 myeps = _mm_load_ps(eps + i);
    __m128 mydel = _mm_sub_ps(myeps, _mm_load_ps(del + i));
    myeps = _mm_add_ps(one, _mm_mul_ps(two, myeps));
    __m128 vectx = _mm_load_ps(wx + i);
    __m128 vectz = _mm_load_ps(wz + i);
    __m128 vecx2 = _mm_mul_ps(vectx, vectx);
    __m128 vecz2 = _mm_mul_ps(vectz, vectz);
    __m128 eclip = _mm_add_ps(_mm_mul_ps(myeps, vecx2), vecz2);
    __m128 parts = _mm_sqrt_ps(
        _mm_sub_ps(
            one, _mm_div_ps(_mm_mul_ps(eight, _mm_mul_ps(mydel, _mm_mul_ps(vecx2, vecz2))), _mm_add_ps(_mm_mul_ps(eclip, eclip), small))));
    __m128 scale = _mm_mul_ps(half, _mm_add_ps(one, parts));
    __m128 scale2 = _mm_mul_ps(myeps, _mm_min_ps(scale, one));
    _mm_store_ps(wx + i, _mm_mul_ps(vectx, scale2));
  }

}

void KernelCPU::applyVel_minus(float *pr, float *pb) {
#pragma omp parallel for num_threads(nThreads) schedule(static)
  for(int iy = 0; iy < ny; iy++) {
    for(size_t ixz = 0; ixz < nxz; ixz += SSEsize) {
      size_t i = iy * nxz + ixz;
      __m128 v2 = _mm_load_ps(velVolume + i);
      __m128 vectx = _mm_load_ps(pr + i);
      _mm_store_ps(pb + i, _mm_sub_ps(_mm_load_ps(pb + i), _mm_mul_ps(vectx, v2)));
    }
  }
}


void KernelCPU::divideRho(float *px, float *py, float *pz, int iy) {
  float *__restrict rho = rhoVolume + (size_t)iy * nxz;

  for(size_t i = 0; i < nxz; i++) {
    if(nx > 1) px[i] /= rho[i];
    if(ny > 1) py[i] /= rho[i];
    pz[i] /= rho[i];
  }
}



void KernelCPU::apply_symmetry(float *pz, int sym, int iz0, int tprlen, int nn) {

  int iz2 = iz0 * 2; // symmetric point iz0 * 2
//    int ny = myGrid->ny, nx = myGrid->nx, nz = myGrid->nz;

  int taper_apply_symmetry = 1;
//    print1m("symmetry = %d, iz0=%d, tparlen=%d,  \n", sym, iz0);

  if(taper_apply_symmetry == 1) {
    vector<float> taper(tprlen);
    float a = 0.0;

    for(int iz = 0; iz < tprlen; iz++) {
      taper[iz] = 0.5f * (1 - cosf((float) M_PI * (iz + 0.5) / tprlen));
      //          print1m("iz=%d, taper=%f\n",iz, taper[iz]);

      /*
       float tmp = float ( iz) / float (tprlen );
       float a3 = 10*(1-a);
       float a4 = -15*(1-a);
       float a5 = 6*(1-a);
       taper[iz] = a + (a3 +a4*tmp+ a5*tmp*tmp)* tmp*tmp*tmp;
       */
    }
    for(int ix = 0; ix < nn; ix++) {
      float *__restrict p = pz + ((size_t)ix) * nz;

      for(int iz = tprlen; iz < iz0; iz++) {
        p[iz] = sym * p[iz2 - iz];
      }
      if(sym == -1) p[iz0] = 0;
      for(int iz = 0; iz < tprlen; iz++) {
        float f = taper[iz];
        p[iz] = sym
            * (p[iz2 - iz] * f
                + (1 - f) * (p[iz2 - iz - 2] + p[iz2 - iz + 2] + 4 * (p[iz2 - iz - 1] + p[iz2 - iz + 1]) + 6 * p[iz2 - iz]) / 16.) * f;
        //              p[iz] =  sym * p[iz2 - iz]*taper[iz];
      }
    }
  } else {
    for(int ix = 0; ix < nn; ix++) {
      float *__restrict p = pz + ((size_t)ix) * nz;
      for(int iz = 0; iz < iz0; iz++)
        p[iz] = sym * p[iz2 - iz];
      if(sym == -1) p[iz0] = 0;
      p[0] = 0.5 * (p[0] + p[1]);
    }
  }
}

