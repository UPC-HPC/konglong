/*
 * FdEngine_fd.cpp
 *
 *  Created on: Mar 1, 2022
 *      Author: owl
 */

#include "FdEngine_fd.h"
#include "libCommon/Taylor.h"
using libCommon::Taylor;
#include "jseisIO/jseisUtil.h"
using jsIO::jseisUtil;
#include "libCommon/CommonMath.h"
#include "libCommon/Options.h"

FdEngine_fd::FdEngine_fd(int nx0, int ny0, int nz0, float dx0, float dy0, float dz0, int nThreads0) : // removed int RhoCN*0, by wolf
    FdEngine(nx0, ny0, nz0, dx0, dy0, dz0, nThreads0) { //revised by wolf to remove RhoCN*0
  derive_cap = FD::Capabilities(FD::order1 | FD::order2);
  parallel_xz = true;
  assertion(nx > 2 && nz > 2, "The current code requires nx(%d), nz(%d) > 2!", nx, nz);

  // update order and ne here if needed
  order = getOrder();
  ne = order / 2;
  use_taylor = isTaylor();

  dispersion_factor = getDispersion(!isDerive2nd()); // 8th
  do_laplace = true;

  coef1.resize(ne + 1); // only ne needed, first one is left empty for convenience
  coef2.resize(ne + 1);
  float sum2max = 0;
  for(int nev = 1; nev <= ne; nev++) {
coef1[nev].resize(nev * 2 + 1, 0);
    coef1[nev][nev] = 0;
    for(int j = 1; j <= nev; j++) {
      float c = (use_taylor ? Taylor::coef1 : Taylor::optim1)[nev - 1][j - 1];
      coef1[nev][nev - j] = -c;
      coef1[nev][nev + j] = c;
    }

    coef2[nev].resize(nev * 2 + 1, 0);
    coef2[nev][nev] = (use_taylor ? Taylor::coef2 : Taylor::optim2)[nev - 1][0];
    float sum2 = -coef2[nev][nev];
    int s2 = 2;
    for(int j = 1; j <= nev; j++) {
      float c = (use_taylor ? Taylor::coef2 : Taylor::optim2)[nev - 1][j];
      coef2[nev][nev - j] = c;
      coef2[nev][nev + j] = c;
      sum2 += s2 * c;
      s2 = -s2;
    }
    sum2max = max(sum2max, sum2);
  }

#if 1 // for compatibility
  setJacob(nullptr, nullptr, nullptr, nullptr);
#endif

  k1max = sqrtf(sum2max);
  float k1max_static = get_k1max();
  assertion(k1max <= k1max_static, "k1max(%f) exceeds estimation get_k1max() (%f) for calcDt()!", k1max, k1max_static);
  update_buffers();
}

FdEngine_fd::~FdEngine_fd() {
}

float FdEngine_fd::get_k1max() {
  int order = getOrder();
  int ne = order / 2;
  bool use_taylor = isTaylor();
  float sum2 = -(use_taylor ? Taylor::coef2 : Taylor::optim2)[ne - 1][0];
  int s2 = 2;
  for(int j = 1; j <= ne; j++) {
    float c = (use_taylor ? Taylor::coef2 : Taylor::optim2)[ne - 1][j];
    sum2 += s2 * c;
    s2 = -s2;
  }
  return sqrtf(sum2) * 1.01f;
}

void FdEngine_fd::update_buffers() {
  vbufz.resize(nThreads);
#pragma omp parallel num_threads(nThreads)
  {
    int tid = omp_get_thread_num();
    vbufz[tid].resize(nz);
    vbufz[tid][0] = 0;
  }
}
void FdEngine_fd::setJacob(Grid *grid, float *jacobx, float *jacoby, float *jacobz) {
  FdEngine::setJacob(grid, jacobx, jacoby, nullptr);
  bool reg = !grid || gridType == RECTANGLE;

  vector<float> zp(nz), zpp(nz);
  if(!reg) {
    Taylor::derive1(&zp[0], &grid->zgrid[0], nz, 1.0, ne, const_ne);
    Taylor::derive2(&zpp[0], &grid->zgrid[0], nz, 1.0, ne, const_ne);
//    printf("zp (dz=%f): [%s]\n", dz, COptions::floats2str(zp).c_str());
//    printf("zpp (dz=%f): [%s]\n", dz, COptions::floats2str(zpp).c_str());
//    Taylor::derive1(&zpp[0], &zp[0], nz, 1.0, ne);
//    printf("zpp (dz=%f): [%s]\n", dz, COptions::floats2str(zpp).c_str());
  }

  vector<vector<float>>(ne * 2 + 1, vector<float>(nz, 0)).swap(coef1z); // only nz-2 needed for var-ne, leave the first one empty for convenience
  vector<vector<float>>(ne * 2 + 1, vector<float>(nz, 0)).swap(coef2z);
  int edge = const_ne ? 0 : 1;
  for(int iz = edge; iz < nz - edge; iz++) {
    float zpi = reg ? 1 / dz : 1 / zp[iz]; // dtau/dz
    float zpi2 = zpi * zpi; // (dtau/dz)^2
    float slope = reg ? 0 : -zpp[iz] * zpi2; // d2tau_dz2 / zpi

    int nev = const_ne ? ne : min(ne, min(iz, nz - 1 - iz));

    coef1z[ne][iz] = 0;
    coef2z[ne][iz] = zpi2 * (use_taylor ? Taylor::coef2 : Taylor::optim2)[nev - 1][0];
    for(int j = 1; j <= nev; j++) {
      float c = zpi * (use_taylor ? Taylor::coef1 : Taylor::optim1)[nev - 1][j - 1];
      coef1z[ne - j][iz] = -c;
      coef1z[ne + j][iz] = c;
      float a = zpi2 * (use_taylor ? Taylor::coef2 : Taylor::optim2)[nev - 1][j];
      float b = slope * c;
      coef2z[ne - j][iz] = a - b;
coef2z[ne + j][iz] = a + b;
    }
  }
  zpi0 = reg ? 1 / dz : 1 / zp[0];
  zpi1 = reg ? 1 / dz : 1 / zp[nz - 1];
  // printf("zpi0*dz=%f, zpi1*dz=%f\n", zpi0 * dz, zpi1 * dz);
}

bool FdEngine_fd::isTaylor() {
  return getBool("fd_use_taylor", false);
}

int FdEngine_fd::getOrder(string engineName) {
  string numpart = engineName.substr(2);
  if(engineName.rfind("FD", 0) != 0) return 0;

  int order = 0;
  try {
    order = std::stoi(numpart);
  } catch(std::invalid_argument const &ex) {
  }
  if(order < 4) order = 8;
  return order;
}

int FdEngine_fd::getOrder() {
  return getOrder(global_pars["Engine"].as<string>("FFTV"));
}
float FdEngine_fd::getDispersion(bool derive1st) {
  int order = getOrder();
  assertion(order >= 8 && order <= 24, "Dispersion values other than 8th-24th order to be added!");
  if(order == 24) return derive1st ? 0.775f : 0.87f;
  if(order == 22) return derive1st ? 0.771f : 0.857f;
  if(order == 20) return derive1st ? 0.759f : 0.86f;
  if(order == 18) return derive1st ? 0.734f : 0.831f;
  if(order == 16) return derive1st ? 0.715f : 0.785f;
  if(order == 14) return derive1st ? 0.678f : 0.744f;
  if(order == 12) return derive1st ? 0.62f : 0.739f;
  if(order == 10) return derive1st ? 0.558f : 0.68f; // 10th
  return derive1st ? 0.46f : 0.6f; // 8th
}

#define YUNROLL 16 // 32: 1.54s, 16: 1.55s, 8: 1.7s, 4: 2.2s, 2: 3.0s
void FdEngine_fd::laplacian(float *in, float *out, float *px, float *py, int op) {
  laplacian_irreg(in, out, px, py, op); // this is general purpose one, not slow either

  // KernelCPU call PML_standalone() after this
}
void FdEngine_fd::laplacian_reg(float *in, float *out, float *px, float *py, int op) { // for speed benchmark only, edge not handled
  assertion(order == 8 && gridType == RECTANGLE,
            "This is a hard-coded benchmark 8th order Laplacian w/o BC for const dz!\n" "(current order=%d, gridType=%d[%d is RECTANGLE])",
            order, gridType, RECTANGLE);
  assertion(!px && !py, "output px2 and py2 is not implemented!");

  const float dx2i = 1. / (dx * dx);
  const float dy2i = 1. / (dy * dy);
  const float dz2i = 1. / (dz * dz);
  const float xc[] = { coef2[4][4] * dx2i, coef2[4][5] * dx2i, coef2[4][6] * dx2i, coef2[4][7] * dx2i, coef2[4][8] * dx2i };
  const float yc[] = { coef2[4][4] * dy2i, coef2[4][5] * dy2i, coef2[4][6] * dy2i, coef2[4][7] * dy2i, coef2[4][8] * dy2i };
  const float zc[] = { coef2[4][4] * dz2i, coef2[4][5] * dz2i, coef2[4][6] * dz2i, coef2[4][7] * dz2i, coef2[4][8] * dz2i };

#pragma omp parallel for num_threads(nThreads) collapse(2)
  for(int iy0 = 0; iy0 < ny; iy0 += YUNROLL) { /* intermixed y and x looping */
    for(int ix = 0; ix < nx; ix++) {
      for(int iy = iy0; iy < min(ny, iy0 + YUNROLL); iy++) {
        float *__restrict u = in + iy * nxz + ix * nz;
        float *__restrict v = out + iy * nxz + ix * nz;
        const float xc0 = xc[0] + yc[0];

        for(int iz = 4; iz < nz - 4; iz++) // hi order middle z
          v[iz] = (xc0 + zc[0]) * u[iz] + zc[1] * (u[iz - 1] + u[iz + 1]) + zc[2] * (u[iz - 2] + u[iz + 2])
              + zc[3] * (u[iz - 3] + u[iz + 3]) + zc[4] * (u[iz - 4] + u[iz + 4]);
        if(ix >= 4 && ix < nx - 4 && iy >= 4 && iy < ny - 4) for(int m = 1; m <= 4; m++) // hi order middle xy
          for(int iz = 0; iz < nz; iz++)
            v[iz] += xc[m] * (u[iz - m * nz] + u[iz + m * nz]) + yc[m] * (u[iz - m * nxz] + u[iz + m * nxz]);
      }
    }
  }
}
void FdEngine_fd::laplacian_irreg(float *in, float *out, float *px, float *py, int op) {
  assertion(coef2z[0].size() == (size_t )nz, "coef2z size does not match with nz! (forgot to call setJacob(nullptr, ...)?");

  bool skip_x = !(op & FD::DX2), skip_y = !(ny > 2 && (op & FD::DY2)), skip_z = !(op & FD::DZ2);
  const float dx2i = 1. / (dx * dx);
  const float dy2i = 1. / (dy * dy);
  int edge = const_ne ? 0 : 1;

#pragma omp parallel for num_threads(nThreads) collapse(2)
  for(int iy0 = 0; iy0 < ny; iy0 += YUNROLL) { /* intermixed y and x looping */
    for(int ix = 0; ix < nx; ix++) {
      int tid = omp_get_thread_num();
      float *__restrict buf0 = &vbufz[tid][0];
      for(int iy = iy0; iy < min(ny, iy0 + YUNROLL); iy++) {
        float *__restrict u = in + iy * nxz + ix * nz;
        float *__restrict v = out + iy * nxz + ix * nz;
        float *__restrict vx = px + iy * nxz + ix * nz;
        float *__restrict vy = py + iy * nxz + ix * nz;

        if(!skip_z) {
          for(int iz = edge; iz < nz - edge; iz++) {
            v[iz] = coef2z[ne][iz] * u[iz];
            if(isnanf(v[iz])) printf("iz=%d, coeff=%f\n", iz, coef2z[ne][iz]), exit(1);
          }
          for(int m = -ne; m <= ne; m++) {
            if(m == 0) continue;
            float *__restrict zc = &coef2z[ne + m][0];
            int iz0 = max(edge, -m), iz1 = min(nz - edge, nz - m);
            for(int iz = iz0; iz < iz1; iz++) {
              v[iz] += zc[iz] * u[iz + m];
              if(isnanf(v[iz])) printf("m=%d, iz=%d, coeff=%f\n", m, iz, zc[iz]), exit(1);
            }
          }
if(!const_ne) v[0] = v[1], v[nz - 1] = v[nz - 2]; // 2nd order edge.
          if(!do_pml3d) pml2_zbnd(v, ix, iy);
        } else if(out) memset(v, 0, sizeof(float) * nz);

        // px2
        if(!skip_x) {
          float *__restrict buf = px ? vx : buf0;
#if 0 // variable order along edge, unstable for left side of sigsbee2b shot325 (total 1-531)
          if(ix == 0 || ix == nx - 1) {
            int sx = (ix == 0) ? nz : -nz;
            for(int iz = 0; iz < nz; iz++)
              buf[iz] = (u[iz] - 2 * u[iz + sx] + u[iz + 2 * sx]) * dx2i;
          } else {
            int nev = min(ne, min(ix, nx - 1 - ix));
            float c = coef2[nev][nev] * dx2i;
            for(int iz = 0; iz < nz; iz++) // m=0
              buf[iz] = c * u[iz];

            for(int m = 1; m <= nev; m++) {
              c = coef2[nev][nev + m] * dx2i;
              for(int iz = 0; iz < nz; iz++)
                buf[iz] += c * (u[iz - m * nz] + u[iz + m * nz]);
            }
          }
#else // constant order
          float c = coef2[ne][ne] * dx2i;
          for(int iz = 0; iz < nz; iz++) // m=0
            buf[iz] = c * u[iz];

          int nev0 = min(ne, ix);
          int nev1 = min(ne, nx - 1 - ix);
          for(int m = 1; m <= ne; m++) {
            c = coef2[ne][ne + m] * dx2i;
            if(m <= nev0) for(int iz = 0; iz < nz; iz++)
              buf[iz] += c * u[iz - m * nz];
            if(m <= nev1) for(int iz = 0; iz < nz; iz++)
              buf[iz] += c * u[iz + m * nz];
          }
#endif
          if(!do_pml3d) pml2_xbnd(buf, ix, iy);
          if(!px) for(int iz = 0; iz < nz; iz++)
            v[iz] += buf[iz];
        } // end of px2

        // py2
        if(!skip_y) {
          float *__restrict buf = py ? vy : buf0;
#if 0 // variable order along edge
          if(iy == 0 || iy == ny - 1) {
            int sy = (iy == 0) ? nxz : -nxz;
            for(int iz = 0; iz < nz; iz++)
              buf[iz] = (u[iz] - 2 * u[iz + sy] + u[iz + 2 * sy]) * dy2i;
          } else {
            int nev = min(ne, min(iy, ny - 1 - iy));
            float c = coef2[nev][nev] * dy2i;
            for(int iz = 0; iz < nz; iz++) // m=0
              buf[iz] = c * u[iz];

for(int m = 1; m <= nev; m++) {
              c = coef2[nev][nev + m] * dy2i;
              for(int iz = 0; iz < nz; iz++)
                buf[iz] += c * (u[iz - m * nxz] + u[iz + m * nxz]);
            }
          }
#else // constant order
          float c = coef2[ne][ne] * dy2i;
          for(int iz = 0; iz < nz; iz++) // m=0
            buf[iz] = c * u[iz];

          int nev0 = min(ne, iy);
          int nev1 = min(ne, ny - 1 - iy);
          for(int m = 1; m <= ne; m++) {
            c = coef2[ne][ne + m] * dy2i;
            if(m <= nev0) for(int iz = 0; iz < nz; iz++)
              buf[iz] += c * u[iz - m * nxz];
            if(m <= nev1) for(int iz = 0; iz < nz; iz++)
              buf[iz] += c * u[iz + m * nxz];
          }
#endif
          if(!do_pml3d) pml2_ybnd(buf, ix, iy);
          if(!py) for(int iz = 0; iz < nz; iz++)
            v[iz] += buf[iz];
        } // end of py2

        if(do_pml3d && out) pml2_bnd(v, ix, iy); // 3D
      }
    }
  }
}
void FdEngine_fd::dy_3D(int nth, float *in, float *py, vector<float*> &wrk2d, int iacc, int isign) {
  dy_3D(nth, in, py, wrk2d, iacc, isign, nullptr, nullptr);
}

void FdEngine_fd::dy_3D(int nth, float *in, float *py, vector<float*> &wrk2d, int iacc, int isign, float *velVolume, float *rhoVolume) {
  assertion(nth == 1 || iacc == 0, "Derive_nth=%d, iacc=%d: 2nd derivative with iacc=1 is not implemented!", nth);
  if(ny == 1) return;

  if(nth == 1) dxyz1(in, nullptr, py, nullptr, iacc, isign, velVolume, rhoVolume);
  else laplacian_irreg(in, nullptr, nullptr, py, FD::DY2);
}


void FdEngine_fd::dxyz1(float *in, float *px, float *py, float *pz, int iacc, int isign) {
  dxyz1(in, px, py, pz, iacc, isign, nullptr, nullptr);
}

void FdEngine_fd::dxyz1(float *in, float *px, float *py, float *pz, int iacc, int isign, float *velVolume, float *rhoVolume) {
  assertion(coef1z[0].size() == (size_t )nz, "coef1z size does not match with nz! (forgot to call setJacob(nullptr, ...)?");

#pragma omp parallel for num_threads(nThreads) collapse(2)
  for(int iy0 = 0; iy0 < ny; iy0 += YUNROLL) { /* intermixed y and x looping */
    for(int ix = 0; ix < nx; ix++) {
      for(int iy = iy0; iy < min(ny, iy0 + YUNROLL); iy++) {
        float *__restrict u = in + iy * nxz + ix * nz;
        float *__restrict vz = pz ? pz + iy * nxz + ix * nz : nullptr;
        float *__restrict vx = px ? px + iy * nxz + ix * nz : nullptr;
        float *__restrict vy = py ? py + iy * nxz + ix * nz : nullptr;
        float *__restrict velSlice = velVolume ? velVolume + iy * nxz + ix * nz : nullptr;
        float *__restrict rhoSlice = rhoVolume ? rhoVolume + iy * nxz + ix * nz : nullptr;

        dxyz1_nz(1, u, vx, vy, vz, ix, iy, iacc, isign, velSlice, rhoSlice);
}
    }
  }
}//revised by wolf on Sep. 21, 2022

void FdEngine_fd::dxyz1_nz_2Dhelper(int order, float *in, float *px, float *py, float *pz, int ix, int iy, int iacc, int isign, float *velSlice, float *rhoSlice) {
  float *__restrict u = in + ix * nz;
  float *__restrict vz = pz ? pz + ix * nz : nullptr;
  float *__restrict vy = py ? py + ix * nz : nullptr;
  float *__restrict vx = px ? px + ix * nz : nullptr;
  float *__restrict velSeg = velSlice ? velSlice + ix * nz : nullptr;
  float *__restrict rhoSeg = rhoSlice ? rhoSlice + ix * nz : nullptr;
  dxyz1_nz(1, u, vx, vy, vz, ix, iy, iacc, isign, velSeg, rhoSeg);
}//revised by wolf on Sep. 21, 2022

void FdEngine_fd::dxz_2D(int nth, float *in, float *px, float *pz, float *wrk, int iy, int iacc, int isign) {
  dxz_2D(nth, in, px, pz, wrk, iy, iacc, isign, nullptr, nullptr);
}
void FdEngine_fd::dxz_2D(int nth, float *in, float *px, float *pz, float *wrk, int iy, int iacc, int isign, float *velSlice, float *rhoSlice) {
  assertion(nth == 1, "Derive_nth=%d, 2nd derivative not implemented!", nth);
  if(nThreads > 1 && ny == 1) {
#pragma omp parallel for num_threads(nThreads)
    for(int ix = 0; ix < nx; ix++)
      dxyz1_nz_2Dhelper(1, in, px, nullptr, pz, ix, iy, iacc, isign, velSlice, rhoSlice);
  } else { // make sure omp_get_thread_num() will get outer parallel region number
    for(int ix = 0; ix < nx; ix++)
      dxyz1_nz_2Dhelper(1, in, px, nullptr, pz, ix, iy, iacc, isign, velSlice, rhoSlice);
  }
}

void FdEngine_fd::dxyz1_nz(int order, float *u, float *px, float *py, float *pz, int ix, int iy, int iacc, int isign, float *vel, float *rho) {
  int tid = omp_get_thread_num(); // in the case of nested omp, be careful!
  // if(ix == 0) printf("iy=%03d, tid=%02d, ny=%d (dxyz1_nz)\n", iy, tid, ny), fflush(stdout);
  float *__restrict buf0 = &vbufz[tid][0];
  int edge = const_ne ? 0 : 1;

 if(pz) {
    float *__restrict vz = iacc ? buf0 : pz;
    memset(vz, 0, sizeof(float) * nz);
    for(int m = -ne; m <= ne; m++) {
      if(m == 0) continue;
      float *__restrict zc = &coef1z[ne + m][0];
      int iz0 = max(edge, -m), iz1 = min(nz - edge, nz - m);
      for(int iz = iz0; iz < iz1; iz++)
        vz[iz] += zc[iz] * u[iz + m];
    }
    // if(!const_ne)  vz[0] = (u[1] - u[0]) * zpi0, vz[nz - 1] = (u[nz - 1] - u[nz - 2]) * zpi1; // 1st order edge.
    if(!const_ne) vz[0] = vz[1], vz[nz - 1] = vz[nz - 2];

    pml1_zbnd(vz, ix, iy, isign);
    if(iacc){
      if(rho) {
        for(int iz = 0; iz < nz; iz++)
          pz[iz] -= vz[iz] * vel[iz] * rho[iz];
      }
      else {
        for(int iz = 0; iz < nz; iz++)
          pz[iz] -= vz[iz] * vel[iz];
      }
    }
  }
// px1
  if(px) {
    float *__restrict vx = iacc ? buf0 : px;
    memset(vx, 0, sizeof(float) * nz);
    int nev0 = min(ne, ix);
    int nev1 = min(ne, nx - 1 - ix);
    for(int m = 1; m <= ne; m++) {
      float c = coef1[ne][ne + m] * dxi;
      if(m <= nev0) for(int iz = 0; iz < nz; iz++)
        vx[iz] -= c * u[iz - m * nz];
      if(m <= nev1) for(int iz = 0; iz < nz; iz++)
        vx[iz] += c * u[iz + m * nz];
    }

    pml1_xbnd(vx, ix, iy, isign);
    if(iacc) {
      if(rho) {
        for(int iz = 0; iz < nz; iz++)
          px[iz] -= vx[iz] * vel[iz] * rho[iz];
      }
      else {
        for(int iz = 0; iz < nz; iz++)
          px[iz] -= vx[iz] * vel[iz];
      }
    }
  } // end of px1
 // py1
  if(py) {
    float *__restrict vy = iacc ? buf0 : py;
    memset(vy, 0, sizeof(float) * nz);
    int nev0 = min(ne, iy);
    int nev1 = min(ne, ny - 1 - iy);
    for(int m = 1; m <= ne; m++) {
      float c = coef1[ne][ne + m] * dyi;
      if(m <= nev0) for(int iz = 0; iz < nz; iz++)
        vy[iz] -= c * u[iz - m * nxz];
      if(m <= nev1) for(int iz = 0; iz < nz; iz++)
        vy[iz] += c * u[iz + m * nxz];
    }

    pml1_ybnd(vy, ix, iy, isign);
    if(iacc) {
      if(rho) {
        for(int iz = 0; iz < nz; iz++)
          py[iz] -= vy[iz] * vel[iz] * rho[iz];
      } else {
        for(int iz = 0; iz < nz; iz++)
          py[iz] -= vy[iz] * vel[iz];
      }
    }
  } // end of py1
}

void FdEngine_fd::dz1_2D(float *in, float *out, float *wrk, int iy, int iacc, int isign) {
  dz1_2D(in, out, wrk, iy, iacc, isign, nullptr, nullptr);
}

void FdEngine_fd::dx1_2D(float *in, float *out, float *wrk, int iy, int iacc, int isign) {
  dx1_2D(in, out, wrk, iy, iacc, isign, nullptr, nullptr);
}
void FdEngine_fd::dz1_2D(float *in, float *out, float *wrk, int iy, int iacc, int isign, float *velSlice, float *rhoSlice){
  dxz_2D(1, in, nullptr, out, wrk, iy, iacc, isign, velSlice, rhoSlice);
}//added by wolf

void FdEngine_fd::dx1_2D(float *in, float *out, float *wrk, int iy, int iacc, int isign, float *velSlice, float *rhoSlice){
  dxz_2D(1, in, out, nullptr, wrk, iy, iacc, isign, velSlice, rhoSlice);
} //added by wolf

// this function should not have any OMP
void FdEngine_fd::dy1_2D(float *in, float *out, float *wrk, int ix, int iacc, int isign) {
  for(int iy = 0; iy < ny; iy++) {
    float *__restrict u = in + iy * nxz;
    float *__restrict vy = out + iy * nxz;
    dxyz1_nz(1, u, nullptr, vy, nullptr, ix, iy, iacc, isign);
  }
}


#undef YUNROLL

