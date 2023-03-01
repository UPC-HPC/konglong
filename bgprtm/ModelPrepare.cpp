#include "ModelPrepare.h"
#include "Grid.h"
#include "fdm.hpp"
#include "Model.h"
#include "ModelLoader.h"
#include "ModelVolumeID.h"
#include "FdEngine_fd.h"
#include "Q.h"
#include "volumefilter.h"
#include "GetPar.h"
#include <string.h>
#include <omp.h>
#include <xmmintrin.h>
#include "libCommon/CommonMath.h"
#include "libCommon/Utl.hpp"
#include "MpiPrint.h"
#include "stdio.h"
using MpiPrint::print1m;

ModelPrepare::ModelPrepare(Grid *myGrid, Model *myModel, ModelLoader *myModelLoader, int nThreads) : myGrid(myGrid), myModel(myModel), myModelLoader(
    myModelLoader), nThreads(nThreads) {
  volVel = 0;
  volEps = 0;
  volDel = 0;
  volPjx = 0;
  volPjy = 0;
  volRho = 0;
  volReflectivity = 0;
  volVel2 = 0;
}

ModelPrepare::~ModelPrepare() {
}

float* ModelPrepare::allocMem3d(int nx, int ny, int nz, int nThreads) const {
  size_t gridsize = (size_t)nx * ny * nz;
  size_t nxz = (size_t)nx * nz;
  float *waveField = (float*)_mm_malloc(gridsize * sizeof(float) + 128, 16);
#pragma omp parallel for num_threads(nThreads) schedule(static)
  for(int iy = 0; iy < ny; iy++)
    memset(&waveField[iy * nxz], 0, nxz * sizeof(float));
  return waveField;
}

void ModelPrepare::velPrepare(const char *velFile) {
  volVel = AniRegrid(myModel->fdms[VEL], -1);
  myModel->free(VEL);
  myGrid->saveModel(velFile, volVel);
}

void ModelPrepare::velPrepare_backup(const char *velFile) {

  Fdm *vellfdm = myModel->fdms[VEL];
  FdmHeader hdr = vellfdm->getHeader();
  float *vel = vellfdm->getdata();
#pragma omp parallel for num_threads(nThreads) schedule(static)
  for(int iy = 0; iy < hdr.ny; iy++) {
    for(int ix = 0; ix < hdr.nx; ix++) {
      for(int iz = 0; iz < hdr.nz; iz++) {
        size_t ido = (iy * size_t(hdr.nx) + ix) * hdr.nz + iz;
        vel[ido] = 1.0 / vel[ido];
      }
    }
  }

  //
  ModelType mtype = myModel->modeltype;
  print1m("The model Type is %d  %d  %d  %d\n", mtype, ISO, VTI, TTI);
  print1m("nz=%d\n", myGrid->nz);

  for(int iz = 0; iz < myGrid->nz; iz++) {
    print1m("iz=%d, %f, %f\n", iz, myGrid->getmyz(iz), myGrid->dzgrid[iz]);
  }
  // always have velocity
  volVel = allocMem3d(myGrid->nx, myGrid->ny, myGrid->nz, nThreads);
  float *halfz = (float*)malloc(myGrid->nz * sizeof(float));
  for(int i = 0; i < myGrid->nz; i++) {
    int id = MIN(i + 1, myGrid->nz - 1);
    float mydz = 0.25 * (myGrid->dzgrid[i] + myGrid->dzgrid[id]);   // linear interpolation of dz
    halfz[i] = myGrid->getmyz(i) + 0.25 * myGrid->dzgrid[i] + 0.5 * mydz;
  }

#pragma omp parallel for num_threads(nThreads) schedule(static)
  for(int iy = 0; iy < myGrid->ny; iy++) {
    float *halfv = (float*)malloc(2 * myGrid->nz * sizeof(float));
    float *halfv2 = halfv + myGrid->nz;
    for(int ix = 0; ix < myGrid->nx; ix++) {
      for(int iz = 0; iz < myGrid->nz; iz++) {
        vector3 xx = myGrid->getxloc(ix, iy, iz);
        vector3 xz;
        xz.setvec(myGrid->getmyx(ix, halfz[iz]), myGrid->getmyy(iy, halfz[iz]), halfz[iz]);
        halfv[iz] = myModel->getvel(xx);
        halfv2[iz] = myModel->getvel(xz);
      }
      float *vout = volVel + (iy * myGrid->nx + ix) * myGrid->nz;
      vout[0] = halfv[0];
      for(int iz = 1; iz < myGrid->nz; iz++) {
        vout[iz] = (halfv[iz] + halfv2[iz] + halfv2[iz - 1]) / 3.0f;
      }
    }
    free(halfv);
  }
  free(halfz);

  //free the memory
  myModel->free(VEL);

  //back to velocity
#pragma omp parallel for num_threads(nThreads) schedule(static)
  for(int iy = 0; iy < myGrid->ny; iy++) {
    for(int ix = 0; ix < myGrid->nx; ix++) {
      for(int iz = 0; iz < myGrid->nz; iz++) {
        size_t id = ((size_t)iy * myGrid->nx + ix) * myGrid->nz + iz;
        volVel[id] = 1.0f / volVel[id];  // back to velocity
      }
    }
  }

  myGrid->saveModel(velFile, volVel);
}


void ModelPrepare::vel2Prepare(const char *vel2File) {
  volVel2 = AniRegrid(myModel->fdms[VEL2], -1);
  myModel->free(VEL2);
  myGrid->saveModel(vel2File, volVel2);
}

void ModelPrepare::vel2Prepare_backup(const char *vel2File) {

  Fdm *vellfdm = myModel->fdms[VEL2];
  FdmHeader hdr = vellfdm->getHeader();
  float *vel = vellfdm->getdata();
#pragma omp parallel for num_threads(nThreads) schedule(static)
  for(int iy = 0; iy < hdr.ny; iy++) {
    for(int ix = 0; ix < hdr.nx; ix++) {
      for(int iz = 0; iz < hdr.nz; iz++) {
        size_t ido = (iy * size_t(hdr.nx) + ix) * hdr.nz + iz;
        vel[ido] = 1.0 / vel[ido];
      }
    }
  }

  //
  ModelType mtype = myModel->modeltype;
  print1m("The model Type is %d  %d  %d  %d\n", mtype, ISO, VTI, TTI);
  print1m("nz=%d\n", myGrid->nz);

  for(int iz = 0; iz < myGrid->nz; iz++) {
    print1m("iz=%d, %f, %f\n", iz, myGrid->getmyz(iz), myGrid->dzgrid[iz]);
  }
  // always have velocity
  volVel2 = allocMem3d(myGrid->nx, myGrid->ny, myGrid->nz, nThreads);
  float *halfz = (float*)malloc(myGrid->nz * sizeof(float));
  for(int i = 0; i < myGrid->nz; i++) {
    int id = MIN(i + 1, myGrid->nz - 1);
    float mydz = 0.25 * (myGrid->dzgrid[i] + myGrid->dzgrid[id]);   // linear interpolation of dz
    halfz[i] = myGrid->getmyz(i) + 0.25 * myGrid->dzgrid[i] + 0.5 * mydz;
  }

#pragma omp parallel for num_threads(nThreads) schedule(static)
  for(int iy = 0; iy < myGrid->ny; iy++) {
    float *halfv = (float*)malloc(2 * myGrid->nz * sizeof(float));
    float *halfv2 = halfv + myGrid->nz;
    for(int ix = 0; ix < myGrid->nx; ix++) {
      for(int iz = 0; iz < myGrid->nz; iz++) {
        vector3 xx = myGrid->getxloc(ix, iy, iz);
        vector3 xz;
        xz.setvec(myGrid->getmyx(ix, halfz[iz]), myGrid->getmyy(iy, halfz[iz]), halfz[iz]);
        halfv[iz] = myModel->getvel(xx);
        halfv2[iz] = myModel->getvel(xz);
      }
      float *vout = volVel2 + (iy * myGrid->nx + ix) * myGrid->nz;
      vout[0] = halfv[0];
      for(int iz = 1; iz < myGrid->nz; iz++) {
        vout[iz] = (halfv[iz] + halfv2[iz] + halfv2[iz - 1]) / 3.0f;
      }
    }
    free(halfv);
  }
  free(halfz);

  //free the memory
  myModel->free(VEL2);

  //back to velocity
#pragma omp parallel for num_threads(nThreads) schedule(static)
  for(int iy = 0; iy < myGrid->ny; iy++) {
    for(int ix = 0; ix < myGrid->nx; ix++) {
      for(int iz = 0; iz < myGrid->nz; iz++) {
        size_t id = ((size_t)iy * myGrid->nx + ix) * myGrid->nz + iz;
        volVel2[id] = 1.0f / volVel2[id];  // back to velocity
      }
    }
  }

  myGrid->saveModel(vel2File, volVel2);
}

void ModelPrepare::vtiPrepare(const char *epsFile, const char *delFile) {
  myModelLoader->loadModelVolume(EPS);
  volEps = AniRegrid(myModel->fdms[EPS], 0);
  myModel->free(EPS);
  myGrid->saveModel(epsFile, volEps);

  myModelLoader->loadModelVolume(DEL);
  volDel = AniRegrid(myModel->fdms[DEL], 0);
  myModel->free(DEL);
  myGrid->saveModel(delFile, volDel);
}

void ModelPrepare::vtiPrepare_backup(const char *epsFile, const char *delFile) {
  myModelLoader->loadModelVolume(EPS);
  Prepare4ModelRegrid(myModel->fdms[EPS]);
  volEps = convertToComputeVol(*myModel, *myGrid, EPS, nThreads);
  myModel->free(EPS);
  myGrid->saveModel(epsFile, volEps);

  myModelLoader->loadModelVolume(DEL);
  Prepare4ModelRegrid(myModel->fdms[DEL]);
  volDel = convertToComputeVol(*myModel, *myGrid, DEL, nThreads);
  myModel->free(DEL);
  myGrid->saveModel(delFile, volDel);
}

void ModelPrepare::ttiPrepare(const char *pjxFile, const char *pjyFile) {
  myModelLoader->loadModelVolume(PJX); // also loads PJY
  volPjx = AniRegrid(myModel->fdms[PJX], 0);
  myModel->free(PJX);
  myGrid->saveModel(pjxFile, volPjx);

  Prepare4ModelRegrid(myModel->fdms[PJY]);
  volPjy = AniRegrid(myModel->fdms[PJY], 0);
  myModel->free(PJY);
  myGrid->saveModel(pjyFile, volPjy);
}

void ModelPrepare::ttiPrepare_backup(const char *pjxFile, const char *pjyFile) {
  myModelLoader->loadModelVolume(PJX); // also loads PJY

  Prepare4ModelRegrid(myModel->fdms[PJX]);
  volPjx = convertToComputeVol(*myModel, *myGrid, PJX, nThreads);
  myModel->free(PJX);
  myGrid->saveModel(pjxFile, volPjx);

  Prepare4ModelRegrid(myModel->fdms[PJY]);
  volPjy = convertToComputeVol(*myModel, *myGrid, PJY, nThreads);
  myModel->free(PJY);
  myGrid->saveModel(pjyFile, volPjy);
}

float* ModelPrepare::AniRegrid(Fdm *myfdm, int isgn) {
  float *volume = allocMem3d(myGrid->nx, myGrid->ny, myGrid->nz, nThreads);

  FdmHeader hdr = myfdm->getHeader();
  float *vol = myfdm->getdata();

  // Specifically or velocity
  if(isgn == -1) {
#pragma omp parallel for num_threads(nThreads) schedule(static)
    for(int iy = 0; iy < hdr.ny; iy++) {
      for(int ix = 0; ix < hdr.nx; ix++) {
        for(int iz = 0; iz < hdr.nz; iz++) {
          size_t ido = (iy * size_t(hdr.nx) + ix) * hdr.nz + iz;
          vol[ido] = 1.0 / vol[ido];
        }
      }
    }
  }

  float *zz1 = (float*)malloc(myGrid->nz * sizeof(float));
  float *zz2 = (float*)malloc(myGrid->nz * sizeof(float));

  for(int iz = 0; iz < myGrid->nz; iz++) {
    if(iz == 0) {
      zz1[iz] = myGrid->getmyz(0);
      zz2[iz] = 0.5 * (myGrid->getmyz(0) + myGrid->getmyz(1));
    } else if(iz == myGrid->nz - 1) {
      zz1[iz] = 0.5 * (myGrid->getmyz(iz - 1) + myGrid->getmyz(iz));
      zz2[iz] = myGrid->getmyz(iz);
    } else {
      zz1[iz] = 0.5 * (myGrid->getmyz(iz - 1) + myGrid->getmyz(iz));
      zz2[iz] = 0.5 * (myGrid->getmyz(iz) + myGrid->getmyz(iz + 1));
    }
  }

#pragma omp parallel num_threads(nThreads)
  {
    float *fdmGridz = (float*)malloc(hdr.nz * sizeof(float));
    float *fdmValue = (float*)malloc(hdr.nz * sizeof(float));
    float *myValue = (float*)malloc(myGrid->nz * sizeof(float));
    for(int iz = 0; iz < hdr.nz; iz++) {
      fdmGridz[iz] = hdr.z0 + iz * hdr.dz;
    }

#pragma omp for
    for(int iy = 0; iy < myGrid->ny; iy++) {
      for(int ix = 0; ix < myGrid->nx; ix++) {
        float myx = myGrid->x0 + ix * myGrid->dx;
        float myy = myGrid->y0 + iy * myGrid->dy;

        if(myGrid->ny > 1) {
          for(int izz = 0; izz < hdr.nz; izz++) {
            vector3 xx;
            xx.x = myx;
            xx.y = myy;
            xx.z = fdmGridz[izz];
            fdmValue[izz] = myfdm->getvalue(xx);
          }
        } else {
          for(int izz = 0; izz < hdr.nz; izz++) {
            vector2 xx;
            xx.x = myx;
            xx.z = fdmGridz[izz];
            fdmValue[izz] = myfdm->getvalue(xx);
          }
        }

        for(int iz = 0; iz < myGrid->nz; iz++) {
          myValue[iz] = AveRegrid(zz1[iz], zz2[iz], fdmValue, fdmGridz, hdr.nz);
        }

        float *vout = volume + ((size_t)iy * myGrid->nx + ix) * myGrid->nz;
        for(int iz = 0; iz < myGrid->nz; iz++) {
          vout[iz] = myValue[iz];
        }

      }
    }
    free(myValue);
    free(fdmGridz);
    free(fdmValue);
  }

  free(zz1);
  free(zz2);

  // Specifically or velocity
  if(isgn == -1) {
#pragma omp parallel for num_threads(nThreads) schedule(static)
    for(int iy = 0; iy < myGrid->ny; iy++) {
      for(int ix = 0; ix < myGrid->nx; ix++) {
        for(int iz = 0; iz < myGrid->nz; iz++) {
          size_t id = ((size_t)iy * myGrid->nx + ix) * myGrid->nz + iz;
          volume[id] = 1.0f / volume[id];  // back to velocity
        }
      }
    }

  }

  return volume;
}

void ModelPrepare::Prepare4ModelRegrid(Fdm *myfdm) {
  FdmHeader hdr = myfdm->getHeader();
  float *vel = myfdm->getdata();

  int mygriddx = (int)nearbyintf(0.5 * myGrid->dx / hdr.dx);
  int mygriddy = (int)nearbyintf(0.5 * myGrid->dy / hdr.dy);
  int mygriddz = (int)nearbyintf(0.5 * myGrid->dz / hdr.dz);

  if(myGrid->ny > 1) avgVolume3D(vel, hdr.nx, hdr.ny, hdr.nz, mygriddx, mygriddy, mygriddz);
  else avgVolume2D(vel, hdr.nx, hdr.nz, mygriddx, mygriddz);
}

float* ModelPrepare::convertToComputeVol(const Model &model, const Grid &grid, ModelVolID id, int nThreads) const {
  float *volume = allocMem3d(grid.nx, grid.ny, grid.nz, nThreads);
  const Fdm &fdm = model.getFdm(id);
  if(grid.ny > 1) {
#pragma omp parallel for num_threads(nThreads) schedule(static)
    for(int iy = 0; iy < grid.ny; iy++) {
      for(int ix = 0; ix < grid.nx; ix++) {
        for(int iz = 0; iz < grid.nz; iz++) {
          vector3 xx = grid.getxloc(ix, iy, iz);
          size_t idx = ((size_t)iy * grid.nx + ix) * grid.nz + iz;
          volume[idx] = fdm.getvalue(xx);
        }
      }
    }
  } else {
    for(int ix = 0; ix < grid.nx; ix++) {
      for(int iz = 0; iz < grid.nz; iz++) {
        vector3 xx = grid.getxloc(ix, 0, iz);
        size_t idx = ((size_t)ix) * grid.nz + iz;
        volume[idx] = fdm.getvalue(vector2(xx.x, xx.z));
      }
    }

  }

  return volume;
}

void ModelPrepare::velPrepare(string velFile) {
  velPrepare(velFile.c_str());
}

void ModelPrepare::vel2Prepare(string vel2File) {
  vel2Prepare(vel2File.c_str());
}

void ModelPrepare::velPrepare_backup(string velFile) {
  velPrepare_backup(velFile.c_str());
}

void ModelPrepare::vel2Prepare_backup(string vel2File) {
  vel2Prepare_backup(vel2File.c_str());
}

void ModelPrepare::vtiPrepare(string epsFile, string delFile) {
  vtiPrepare(epsFile.c_str(), delFile.c_str());
}

void ModelPrepare::vtiPrepare_backup(string epsFile, string delFile) {
  vtiPrepare_backup(epsFile.c_str(), delFile.c_str());
}

void ModelPrepare::ttiPrepare(string pjxFile, string pjyFile) {
  ttiPrepare(pjxFile.c_str(), pjyFile.c_str());
}

void ModelPrepare::ttiPrepare_backup(string pjxFile, string pjyFile) {
  ttiPrepare_backup(pjxFile.c_str(), pjyFile.c_str());
}

void ModelPrepare::rhoPrepare_backup(string rhoFile) {
  myModelLoader->loadModelVolume(RHO);
  volRho = AniRegrid(myModel->fdms[RHO], 0);
  /*
   float* volRho0 = SmoothingVol(volRho);
   volRho = SmoothingVol(volRho0);
   myGrid->saveModel(rhoFile, volRho);
   myModel->free(RHO);
   _mm_free(volRho);
   _mm_free(volRho0);
   */

  float *sqrtRho = SqrtVol(volRho);
  float *cnnRho = CanonicalVol(sqrtRho, volRho);
  myGrid->saveModel(rhoFile, cnnRho);

  myModel->free(RHO);
  _mm_free(sqrtRho);
  _mm_free(cnnRho);

}
//March 2018
void ModelPrepare::rhoPrepare(string rhoFile) {
  myModelLoader->loadModelVolume(RHO);
  Prepare4ModelRegrid(myModel->fdms[RHO]);
  volRho = convertToComputeVol(*myModel, *myGrid, RHO, nThreads);
  myModel->free(RHO);
  myGrid->saveModel(rhoFile, volRho);
}

void ModelPrepare::reflectivityPrepare(string reflectivityFile) {
  myModelLoader->loadModelVolume(REFLECTIVITY);
  Prepare4ModelRegrid(myModel->fdms[REFLECTIVITY]);
  volReflectivity = convertToComputeVol(*myModel, *myGrid, REFLECTIVITY, nThreads);
  myModel->free(REFLECTIVITY);
  myGrid->saveModel(reflectivityFile, volReflectivity);
}

void ModelPrepare::qPrepare(string qFile) {
  myModelLoader->loadModelVolume(Q);
  Prepare4ModelRegrid(myModel->fdms[Q]);
  volQ = convertToComputeVol(*myModel, *myGrid, Q, nThreads);
  invQmax = libCommon::maxf(volQ, myGrid->nz, myGrid->nx * myGrid->ny, nThreads > 1);
  myModel->free(Q);
  myGrid->saveModel(qFile, volQ);
}

float* ModelPrepare::CanonicalVol(float *sqrtRho, float *Rho) {
  float *volout = allocMem3d(myGrid->nx, myGrid->ny, myGrid->nz, nThreads);

#pragma omp parallel for num_threads(nThreads) schedule(static)
  for(int iy = 0; iy < myGrid->ny; iy++) {
    for(int ix = 0; ix < myGrid->nx; ix++) {
      for(int iz = 0; iz < myGrid->nz; iz++) {
        int iz1 = iz - 1;
        if(iz1 < 0) iz1 = 1;

        int iz2 = iz + 1;
        if(iz2 > myGrid->nz - 1) iz2 = myGrid->nz - 2;

        float dz1, dz2;
        if(iz1 == 1) {
          dz1 = myGrid->dzgrid[0];
        } else {
          dz1 = myGrid->dzgrid[iz1];
        }
        if(iz2 == myGrid->nz - 2) {
          dz2 = myGrid->dzgrid[myGrid->nz - 2];
        } else {
          dz2 = myGrid->dzgrid[iz];
        }
        size_t id1 = ((size_t)iy * myGrid->nx + ix) * myGrid->nz + iz1;
        size_t id = ((size_t)iy * myGrid->nx + ix) * myGrid->nz + iz;
        size_t id2 = ((size_t)iy * myGrid->nx + ix) * myGrid->nz + iz2;
        volout[id] = 0.5 * sqrtRho[id]
            * ((1. / Rho[id1] + 1. / Rho[id]) * sqrtRho[id1] - (1. / Rho[id1] + 2. / Rho[id] + 1. / Rho[id2]) * sqrtRho[id]
                + (1. / Rho[id2] + 1. / Rho[id]) * sqrtRho[id2]) / dz1 / dz2;
      }
    }
  }

#pragma omp parallel for num_threads(nThreads) schedule(static)
  for(int iy = 0; iy < myGrid->ny; iy++) {
    for(int ix = 0; ix < myGrid->nx; ix++) {
      int ix1 = ix - 1;
      if(ix1 < 0) ix1 = 1;

      int ix2 = ix + 1;
      if(ix2 > myGrid->nx - 1) ix2 = myGrid->nx - 2;

      float dx = myGrid->dx;
      for(int iz = 0; iz < myGrid->nz; iz++) {
        size_t id1 = ((size_t)iy * myGrid->nx + ix1) * myGrid->nz + iz;
        size_t id = ((size_t)iy * myGrid->nx + ix) * myGrid->nz + iz;
        size_t id2 = ((size_t)iy * myGrid->nx + ix2) * myGrid->nz + iz;
        volout[id] = volout[id]
            + 0.5 * sqrtRho[id]
                * ((1. / Rho[id1] + 1. / Rho[id]) * sqrtRho[id1] - (1. / Rho[id1] + 2. / Rho[id] + 1. / Rho[id2]) * sqrtRho[id]
                    + (1. / Rho[id2] + 1. / Rho[id]) * sqrtRho[id2]) / dx / dx;
      }
    }
  }

  // Specifically or velocity
  if(myGrid->ny == 1) {
  } else {
    for(int iy = 0; iy < myGrid->ny; iy++) {
      int iy1 = iy - 1;
      if(iy1 < 0) iy1 = 1;

      int iy2 = iy + 1;
      if(iy2 > myGrid->ny - 1) iy2 = myGrid->ny - 2;

      float dy = myGrid->dy;
      for(int ix = 0; ix < myGrid->nx; ix++) {
        for(int iz = 0; iz < myGrid->nz; iz++) {
          size_t id1 = ((size_t)iy1 * myGrid->nx + ix) * myGrid->nz + iz;
          size_t id = ((size_t)iy * myGrid->nx + ix) * myGrid->nz + iz;
          size_t id2 = ((size_t)iy2 * myGrid->nx + ix) * myGrid->nz + iz;
          volout[id] = volout[id]
              + 0.5 * sqrtRho[id]
                  * ((1. / Rho[id1] + 1. / Rho[id]) * sqrtRho[id1] - (1. / Rho[id1] + 2. / Rho[id] + 1. / Rho[id2]) * sqrtRho[id]
                      + (1. / Rho[id2] + 1. / Rho[id]) * sqrtRho[id2]) / dy / dy;
        }
      }
    }
  }

  return volout;

}

float* ModelPrepare::SmoothingVol(float *volin) {
  float *volout = allocMem3d(myGrid->nx, myGrid->ny, myGrid->nz, nThreads);

#pragma omp parallel for num_threads(nThreads) schedule(static)
  for(int iy = 0; iy < myGrid->ny; iy++) {
    for(int ix = 0; ix < myGrid->nx; ix++) {
      for(int iz = 0; iz < myGrid->nz; iz++) {
        int iz1 = MAX(iz - 1, 0);
        int iz2 = MIN(iz + 1, myGrid->nz - 1);
        size_t id1 = ((size_t)iy * myGrid->nx + ix) * myGrid->nz + iz1;
        size_t id = ((size_t)iy * myGrid->nx + ix) * myGrid->nz + iz;
        size_t id2 = ((size_t)iy * myGrid->nx + ix) * myGrid->nz + iz2;
        volout[id] = 0.25 * volin[id1] + 0.5 * volin[id] + 0.25 * volin[id2];
      }
    }
  }

#pragma omp parallel for num_threads(nThreads) schedule(static)
  for(int iy = 0; iy < myGrid->ny; iy++) {
    for(int ix = 0; ix < myGrid->nx; ix++) {
      int ix1 = MAX(ix - 1, 0);
      int ix2 = MIN(ix + 1, myGrid->nx - 1);
      for(int iz = 0; iz < myGrid->nz; iz++) {
        size_t id1 = ((size_t)iy * myGrid->nx + ix1) * myGrid->nz + iz;
        size_t id = ((size_t)iy * myGrid->nx + ix) * myGrid->nz + iz;
        size_t id2 = ((size_t)iy * myGrid->nx + ix2) * myGrid->nz + iz;
        volin[id] = 0.25 * volout[id1] + 0.5 * volout[id] + 0.25 * volout[id2];
      }
    }
  }

  // Specifically or velocity
  if(myGrid->ny == 1) {
#pragma omp parallel for num_threads(nThreads) schedule(static)
    for(int iy = 0; iy < myGrid->ny; iy++) {
      for(int ix = 0; ix < myGrid->nx; ix++) {
        for(int iz = 0; iz < myGrid->nz; iz++) {
          size_t id = ((size_t)iy * myGrid->nx + ix) * myGrid->nz + iz;
          volout[id] = volin[id];
        }
      }
    }
  } else {
    for(int iy = 0; iy < myGrid->ny; iy++) {
      int iy1 = MAX(iy - 1, 0);
      int iy2 = MIN(iy + 1, myGrid->ny - 1);
      for(int ix = 0; ix < myGrid->nx; ix++) {
        for(int iz = 0; iz < myGrid->nz; iz++) {
          size_t id1 = ((size_t)iy1 * myGrid->nx + ix) * myGrid->nz + iz;
          size_t id = ((size_t)iy * myGrid->nx + ix) * myGrid->nz + iz;
          size_t id2 = ((size_t)iy2 * myGrid->nx + ix) * myGrid->nz + iz;
          volout[id] = 0.25 * volin[id1] + 0.5 * volin[id] + 0.25 * volin[id2];
        }
      }
    }
  }

  return volout;

}

float* ModelPrepare::SqrtVol(float *volin) {
  float *volout = allocMem3d(myGrid->nx, myGrid->ny, myGrid->nz, nThreads);

#pragma omp parallel for num_threads(nThreads) schedule(static)
  for(int iy = 0; iy < myGrid->ny; iy++) {
    for(int ix = 0; ix < myGrid->nx; ix++) {
      for(int iz = 0; iz < myGrid->nz; iz++) {
        size_t id = ((size_t)iy * myGrid->nx + ix) * myGrid->nz + iz;
        volout[id] = sqrt(volin[id]);
      }
    }
  }

  return volout;

}

int ModelPrepare::getDimension() {
  string dim_str = global_pars["dimension"].as<string>("3D");
  transform(dim_str.begin(), dim_str.end(), dim_str.begin(), ::toupper);
  int dim = ThreeD;
  if(dim_str.find("1D") == 0) dim = OneD;
  else if(dim_str.find("2D") == 0) dim = TwoD;
  if(dim_str.find("D3") != string::npos || dim_str.find("D5") != string::npos) dim |= Sim3D;
  return dim;
}

float ModelPrepare::calcDeltaTime() {
  int dim = getDimension();

  float k1max = PI;
  string engineName = global_pars["Engine"].as<string>("FFTV");
  transform(engineName.begin(), engineName.end(), engineName.begin(), ::toupper);
  if(engineName.rfind("FD", 0) == 0) k1max = FdEngine_fd::get_k1max();


  ModelType mtype = myModel->modeltype;
  float *myVel = volVel;
  if(getBool(global_pars["dual_flood"], false)){
    for(int iy = 0; iy < myGrid->ny; iy++) {
      for(int ix = 0; ix < myGrid->nx; ix++) {
        size_t Idxy = (size_t)iy * myGrid->nx + ix;
        Idxy *= myGrid->nz;
        for(int iz = 0; iz < myGrid->nz; iz++) {
          size_t IDxyz = Idxy + iz;
          myVel[IDxyz] = max(myVel[IDxyz], volVel2[IDxyz]);
        }
      }
    }
  }

  // calculate dt
  float mydt = 1000.0;
  if(mtype == ISO) {
    //March 2018

    if(myModel->useRho) { //remove RhoCN* by wolf
      float *__restrict rho = volRho;
      float rhomin = rho[0];
      for(int iy = 0; iy < myGrid->ny; iy++) {
        for(int ix = 0; ix < myGrid->nx; ix++) {
          size_t Idxy = (size_t)iy * myGrid->nx + ix;
          Idxy *= myGrid->nz;
          for(int iz = 0; iz < myGrid->nz; iz++) {
            size_t IDxyz = Idxy + iz;
            float rdx2 = (dim & OneD) ? 0 : 1. / (myGrid->dxgrid[iz] * myGrid->dxgrid[iz]);
            float rdy2 = (dim & ThreeD) ? 1. / (myGrid->dygrid[iz] * myGrid->dygrid[iz]) : 0;
            float rdz2 = 1. / (myGrid->dzgrid[iz] * myGrid->dzgrid[iz]);
            float lambda = myVel[IDxyz] * sqrt(rho[IDxyz]) * sqrt(rdx2 + rdy2 + rdz2);
            mydt = MIN(mydt, 2.0 / k1max / lambda);
            rhomin = MIN(rhomin, rho[IDxyz]);
            //                    float  dxyz  = MIN(MIN(myGrid->dxgrid[iz], myGrid->dygrid[iz]), myGrid->dzgrid[iz]);
            //                    mydt = MIN(mydt, dxyz/myVel[IDxyz]);
          }
        }
      }

      if(rhomin < 1E-7) libCommon::Utl::fatal("Input density has value close to zero");

      mydt = mydt * sqrt(rhomin);
      print1m("Rhomin=%f, mydt=%f \n", rhomin, mydt);
    } else {
      for(int iy = 0; iy < myGrid->ny; iy++) {
        for(int ix = 0; ix < myGrid->nx; ix++) {
          size_t Idxy = (size_t)iy * myGrid->nx + ix;
          Idxy *= myGrid->nz;
          for(int iz = 0; iz < myGrid->nz; iz++) {
            size_t IDxyz = Idxy + iz;
            float rdx2 = (dim & OneD) ? 0 : 1. / (myGrid->dxgrid[iz] * myGrid->dxgrid[iz]);
            float rdy2 = (dim & ThreeD) ? 1. / (myGrid->dygrid[iz] * myGrid->dygrid[iz]) : 0;
            float rdz2 = 1. / (myGrid->dzgrid[iz] * myGrid->dzgrid[iz]);
            float lambda = myVel[IDxyz] * sqrt(rdx2 + rdy2 + rdz2);
            mydt = MIN(mydt, 2.0 / k1max / lambda);
            //                    float  dxyz  = MIN(MIN(myGrid->dxgrid[iz], myGrid->dygrid[iz]), myGrid->dzgrid[iz]);
            //                    mydt = MIN(mydt, dxyz/myVel[IDxyz]);
          }
        }
      }
    }

  } else if(mtype == VTI) {
    float *myEps = volEps;
    float *myDel = volDel;
    if(myModel->useRho) { //remove RhoCN* by wolf
      float *__restrict rho = volRho;
      float rhomin = rho[0];
      for(int iy = 0; iy < myGrid->ny; iy++) {
        for(int ix = 0; ix < myGrid->nx; ix++) {
          size_t Idxy = (size_t)iy * myGrid->nx + ix;
          Idxy *= myGrid->nz;
          for(int iz = 0; iz < myGrid->nz; iz++) {
            size_t IDxyz = Idxy + iz;
            float rdx2 = (dim & OneD) ? 0 : 1. / (myGrid->dxgrid[iz] * myGrid->dxgrid[iz]);
            float rdy2 = (dim & ThreeD) ? 1. / (myGrid->dygrid[iz] * myGrid->dygrid[iz]) : 0;
            float rdz2 = 1. / (myGrid->dzgrid[iz] * myGrid->dzgrid[iz]);
            float tmp = (1.0 + 2. * myEps[IDxyz]) * (rdx2 + rdy2) + rdz2;
            float lambda = myVel[IDxyz] * sqrt(rho[IDxyz])
                * sqrt((tmp + sqrt(tmp * tmp - 8 * (myEps[IDxyz] - myDel[IDxyz]) * (rdx2 + rdy2) * rdz2)) / 2.0);
            mydt = MIN(mydt, 2.0 / k1max / lambda / sqrtf(1 + 2.0f * myEps[IDxyz]));
            //                    float  dxyz  = MIN(MIN(myGrid->dxgrid[iz], myGrid->dygrid[iz]), myGrid->dzgrid[iz]);
            //                    mydt = MIN(mydt, dxyz/(myVel[IDxyz]*sqrt(1.0+2.*myEps[IDxyz])));
          }
        }
      }
      if(rhomin < 1E-7) libCommon::Utl::fatal("Input density has value close to zero");

      mydt = mydt * sqrt(rhomin);
    } else {
      for(int iy = 0; iy < myGrid->ny; iy++) {
        for(int ix = 0; ix < myGrid->nx; ix++) {
          size_t Idxy = (size_t)iy * myGrid->nx + ix;
          Idxy *= myGrid->nz;
          for(int iz = 0; iz < myGrid->nz; iz++) {
            size_t IDxyz = Idxy + iz;
            float rdx2 = (dim & OneD) ? 0 : 1. / (myGrid->dxgrid[iz] * myGrid->dxgrid[iz]);
            float rdy2 = (dim & ThreeD) ? 1. / (myGrid->dygrid[iz] * myGrid->dygrid[iz]) : 0;
            float rdz2 = 1. / (myGrid->dzgrid[iz] * myGrid->dzgrid[iz]);
            float tmp = (1.0 + 2. * myEps[IDxyz]) * (rdx2 + rdy2) + rdz2;
            float lambda = myVel[IDxyz] * sqrt((tmp + sqrt(tmp * tmp - 8 * (myEps[IDxyz] - myDel[IDxyz]) * (rdx2 + rdy2) * rdz2)) / 2.0);
            mydt = MIN(mydt, 2.0 / k1max / lambda / sqrtf(1 + 2.0f * myEps[IDxyz]));
            //                    float  dxyz  = MIN(MIN(myGrid->dxgrid[iz], myGrid->dygrid[iz]), myGrid->dzgrid[iz]);
            //                    mydt = MIN(mydt, dxyz/(myVel[IDxyz]*sqrt(1.0+2.*myEps[IDxyz])));
          }
        }
      }
    }

  } else if(mtype == TTI) {
    float *myEps = volEps;
    float *myDel = volDel;
    float *myPjx = volPjx;
    float *myPjy = volPjy;

    int nc = 1;
    if(dim & TwoD) nc = 4;
    else if(dim & ThreeD) nc = 8;

    if(myModel->useRho) { //remove RhoCN* by wolf
      float *__restrict rho = volRho;
      float rhomin = rho[0];
      for(int iy = 0; iy < myGrid->ny; iy++) {
        for(int ix = 0; ix < myGrid->nx; ix++) {
          size_t Idxy = (size_t)iy * myGrid->nx + ix;
          Idxy *= myGrid->nz;
          for(int iz = 0; iz < myGrid->nz; iz++) {
            size_t IDxyz = Idxy + iz;
            float sintheta = sqrtf(myPjx[IDxyz] * myPjx[IDxyz] + myPjy[IDxyz] * myPjy[IDxyz]);
            float costheta = sqrtf(1.0f - sintheta * sintheta);
            float cosphi = 1.0f;
            float sinphi = 0.0;
            if(std::abs(sintheta) > 1.0E-15) {
              cosphi = -myPjx[IDxyz] / sintheta;
              sinphi = -myPjy[IDxyz] / sintheta;
            }

            float akx = 1.0f / myGrid->dxgrid[iz];
            float aky = 1.0f / myGrid->dygrid[iz];
            float akz = 1.0f / myGrid->dzgrid[iz];
            float rdx2 = (dim & OneD) ? 0 : (cosphi * costheta * akx + sinphi * costheta * aky - sintheta * akz);
            float rdy2 = (dim & ThreeD) ? (-sinphi * akx + cosphi * aky) : 0;
            float rdz2 = (cosphi * sintheta * akx + sinphi * sintheta * aky + costheta * akz);
            rdx2 *= rdx2;
            rdy2 *= rdy2;
            rdz2 *= rdz2;

            float lambda = 0.0f;

            for(int ic = 1; ic < nc; ic++) {
              int icz = ic & 1;
              int icx = (ic >> 1) & 1;
              int icy = (ic >> 2) & 1;

              float tmp1 = (1.0 + 2. * myEps[IDxyz]) * (rdx2 * icx + rdy2 * icy) + rdz2 * icz;
              float tmp2 = myVel[IDxyz]
                  * sqrt((tmp1 + sqrt(tmp1 * tmp1 - 8 * (myEps[IDxyz] - myDel[IDxyz]) * (rdx2 * icx + rdy2 * icy) * rdz2 * icz)) / 2.0);

              lambda = MAX(lambda, tmp2);
            }
            lambda *= sqrtf(rho[IDxyz]);
            mydt = MIN(mydt, 2.0 / k1max / lambda / sqrtf(1 + 2.0f * myEps[IDxyz]));
          }
        }
      }
      if(rhomin < 1E-7) libCommon::Utl::fatal("Input density has value close to zero");

      mydt = mydt * sqrtf(rhomin);
    } else {
      for(int iy = 0; iy < myGrid->ny; iy++) {
        for(int ix = 0; ix < myGrid->nx; ix++) {
          size_t Idxy = (size_t)iy * myGrid->nx + ix;
          Idxy *= myGrid->nz;
          for(int iz = 0; iz < myGrid->nz; iz++) {
            size_t IDxyz = Idxy + iz;
            float sintheta = sqrtf(myPjx[IDxyz] * myPjx[IDxyz] + myPjy[IDxyz] * myPjy[IDxyz]);
            float costheta = sqrtf(1.0f - sintheta * sintheta);
            float cosphi = 1.0f;
            float sinphi = 0.0;
            if(std::abs(sintheta) > 1.0E-15) {
              cosphi = -myPjx[IDxyz] / sintheta;
              sinphi = -myPjy[IDxyz] / sintheta;
            }

            float akx = 1.0f / myGrid->dxgrid[iz];
            float aky = 1.0f / myGrid->dygrid[iz];
            float akz = 1.0f / myGrid->dzgrid[iz];
            float rdx2 = (dim & OneD) ? 0 : (cosphi * costheta * akx + sinphi * costheta * aky - sintheta * akz);
            float rdy2 = (dim & ThreeD) ? (-sinphi * akx + cosphi * aky) : 0;
            float rdz2 = (cosphi * sintheta * akx + sinphi * sintheta * aky + costheta * akz);
            rdx2 *= rdx2;
            rdy2 *= rdy2;
            rdz2 *= rdz2;

            float lambda = 0.0f;

            for(int ic = 1; ic < nc; ic++) {
              int icz = ic & 1;
              int icx = (ic >> 1) & 1;
              int icy = (ic >> 2) & 1;

              float tmp1 = (1.0 + 2. * myEps[IDxyz]) * (rdx2 * icx + rdy2 * icy) + rdz2 * icz;
              float tmp2 = myVel[IDxyz]
                  * sqrt((tmp1 + sqrt(tmp1 * tmp1 - 8 * (myEps[IDxyz] - myDel[IDxyz]) * (rdx2 * icx + rdy2 * icy) * rdz2 * icz)) / 2.0);

              lambda = MAX(lambda, tmp2);
            }
            }

            mydt = MIN(mydt, 2.0 / k1max / lambda / sqrtf(1 + 2.0f * myEps[IDxyz]));

          }
        }
      }
    }
  }

  float maxfreq = global_pars["maxFreq"].as<float>();
  float qrefFreq = global_pars["qRefFreq"].as<float>(maxfreq / 3);
  if(myModel->useQ) {
    float rescale = 1 - invQmax * (3 + 2 / FLT_PI * logf(maxfreq / qrefFreq));
    print1m("dt_rescale due to Q: %f [max(1/Q)=%f, maxfreq=%f, ref_freq=%f]\n", rescale, invQmax, maxfreq, qrefFreq);
    mydt *= rescale;
  }

  //free the memory
  freeMemory();

  if(global_pars["tdsp_correction"].as<int>(2) == 0 && !global_pars["prop"]["nppc"]) global_pars["prop"]["nppc"] = 20;
  if(global_pars["prop"]["nppc"]) {
    float nppc = global_pars["prop"]["nppc"].as<float>();
    float mfreq = global_pars["maxFreq"].as<float>();
    float nppc_stab = 1 / (mfreq * mydt);
    if(nppc < nppc_stab) print1m("prop->nppc (%f) is smaller than nppc_stability (%f), dt_prop not changed!\n", nppc, nppc_stab);
    else {
      float dt = 1 / (mfreq * nppc);
      print1m("Forcing dt from %f to %f, (nppc from %f to %f)\n", mydt, dt, nppc_stab, nppc);
      mydt = dt;
    }
    }
  }

  print1m("calculated dt: %f \n", mydt);
  if(global_pars["prop"]["dt"] || global_pars["prop"]["dt_scale"]) {
    float dt_input = global_pars["prop"]["dt"].as<float>(mydt);
    dt_input *= global_pars["prop"]["dt_scale"].as<float>(1.0f);
    if(dt_input < mydt) {
      mydt = dt_input;
      print1m("dt overrided by user: %f \n", mydt);
    } else {
      print1m("user defined dt (%f) >= the stability calculation (%f), dt_prop not changed!\n", dt_input, mydt);
    }
  }
  if(myModel->useQ) Q::fitQCoeff(mydt, qrefFreq, maxfreq);

  return mydt;
}

void ModelPrepare::freeMemory() {
  if(volVel) _mm_free(volVel);
  if(volVel2) _mm_free(volVel2);
  if(volQ) _mm_free(volQ);
  if(volEps) _mm_free(volEps);
  if(volDel) _mm_free(volDel);
  if(volPjx) _mm_free(volPjx);
  if(volPjy) _mm_free(volPjy);
  if(volRho) _mm_free(volRho);
  if(volReflectivity) _mm_free(volReflectivity);
}


