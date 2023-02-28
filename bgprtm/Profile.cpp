#include <stdlib.h>
#include <float.h>
#include "Profile.h"
#include "ProfileWrapper.h"
#include "Util.h"
#include "fdm.hpp"
#include "GetPar.h"

void Profile::create() {
  FdmHeader myheader = myvel->getHeader();
  nz = myheader.nz;
  profile = new float[nz];
  dz = myheader.dz;
  z0 = myheader.z0;
  velmin = getMinValue();
}

Profile::~Profile() {
  if(profile) {
    delete[] profile;
    profile = NULL;
  }
}

void Profile::getVprofile(float x, float y) {
  FdmHeader myheader = myvel->getHeader();
  float dxi = 1.0f / myheader.dx;
  float dyi = 1.0f / myheader.dy;
  int ix = int((x - myheader.x0) * dxi);
  int iy = int((y - myheader.y0) * dyi);
  ix = MIN(ix, (myheader.nx - 1));
  iy = MIN(iy, (myheader.ny - 1));
  ix = MAX(ix, 0);
  iy = MAX(iy, 0);
  getVprofile(ix, iy);
}

void Profile::getVprofile(int ix, int iy) {
  FdmHeader myheader = myvel->getHeader();
  size_t offset = (iy * myheader.nx + ix) * nz;
  float *dataloc = myvel->getdata();
  dataloc += offset;
  memcpy(profile, dataloc, nz * sizeof(float));
}

float Profile::getMinValue() {
  FdmHeader myheader = myvel->getHeader();
  float *dataloc = myvel->getdata();
  float minvel = 999999999.0, minvel_glb = 999999999.0;
  for(int iz = 0; iz < myheader.nz; iz++)
    profile[iz] = 9.99e24;
  for(int iy = 0; iy < myheader.ny; iy++) {
    for(int ix = 0; ix < myheader.nx; ix++) {
      for(int iz = 0; iz < myheader.nz; iz++) {
        size_t id = (size_t)(iy * myheader.nx + ix) * (size_t)myheader.nz + (size_t)iz;
        profile[iz] = MIN(profile[iz], dataloc[id]);
      }
    }
  }

  float *zloc = (float *) malloc(myheader.nz * sizeof(float));
  float *vmin = (float *) malloc(myheader.nz * sizeof(float));
  float *fd = (float *) malloc(myheader.nz * sizeof(float));
  float *bb = (float *) malloc(myheader.nz * sizeof(float));
  float *am = (float *) malloc(7 * myheader.nz * sizeof(float));
  float *alm = (float *) malloc(3 * myheader.nz * sizeof(float));
  int *indx = (int *) malloc(3 * myheader.nz * sizeof(int));

  for(int iz = 0; iz < myheader.nz; iz++) {
    zloc[iz] = myheader.z0 + iz * myheader.dz;
    minvel_glb = MIN(profile[iz], minvel_glb);
  }

  int nsmth = int(minvel_glb / maxfreq / myheader.dz + 0.5);
  nsmth = MAX(nsmth, 5);

  minvel0(zloc, profile, vmin, myheader.nz, fd, bb, am, alm, indx, nsmth);

  for(int iz = 0; iz < myheader.nz; iz++) {
    //    printf("%f, %f, %f \n", zloc[iz], profile[iz], vmin[iz]);
    profile[iz] = vmin[iz];
    minvel = MIN(minvel, profile[iz]);
  }

  free(zloc);
  free(vmin);
  free(fd);
  free(bb);
  free(am);
  free(alm);
  free(indx);

  return minvel;
}

float Profile::getMinValue_backup() {
  FdmHeader myheader = myvel->getHeader();
  float *dataloc = myvel->getdata();
  float minvel = 999999999.0;
  for(int iz = 0; iz < nz; iz++)
    profile[iz] = 9.99e24;
  for(int iy = 0; iy < myheader.ny; iy++) {
    for(int ix = 0; ix < myheader.nx; ix++) {
      for(int iz = 0; iz < nz; iz++) {
        size_t id = (size_t)(iy * myheader.nx + ix) * (size_t)nz + (size_t)iz;
        profile[iz] = MIN(profile[iz], dataloc[id]);
      }
    }
  }

  for(int iz = 0; iz < nz; iz++)
    minvel = MIN(minvel, profile[iz]);
  for(int iz = 0; iz < nz; iz++)
    printf("%d, %f\n", iz, profile[iz]);
  return minvel;
}

void Profile::MinVelSmooth() {
  float *buff = new float[nz];
  memcpy(buff, profile, nz * sizeof(float));
  buff[0] = 0.5f * (buff[0] + buff[1]);
  buff[nz - 1] = 0.5f * (buff[nz - 1] + buff[nz - 2]);
  for(int iz = 1; iz < nz - 1; iz++)
    buff[iz] += 0.5f * (profile[iz - 1] + profile[iz + 1]); // Moving average
  for(int iz = 1; iz < nz - 1; iz++)
    buff[iz] *= 0.5f;

  for(int iz = 0; iz < nz; iz++)
    profile[iz] = MIN(profile[iz], buff[iz]);  // Take the minimum
  delete[] buff;
}

void Profile::MinVelForZGrid() {
  getMinValue();
  float gdz = velmin / (2.0f * maxfreq);
  float factor = MAX(1.0, (gdz / dz));
  for(int iloop = 0; iloop < 6 * factor; iloop++)
    MinVelSmooth(); // I love 6
}

float Profile::TimeGetDepth(float t) {
  float t0 = 0.0;
  float myz = z0;
  float myvel = profile[0];
  float dt0 = dz * 0.5 / myvel;
  float mydz = dz * 0.5;
  int iz = 0;

  while(t0 + dt0 < t) {
    myz = myz + mydz;
    t0 = t0 + dt0;
    iz++;
    if(iz == 0) {
      mydz = dz * 0.5;
      myvel = profile[0];
    } else if(iz >= nz - 1) {
      mydz = dz;
      myvel = profile[nz - 1];
    } else {
      mydz = dz;
      myvel = profile[iz];
    }
    dt0 = mydz / myvel;
  }

  dt0 = t - t0;
  mydz = dt0 * profile[MIN(iz, nz - 1)];
  myz = myz + mydz;
  return myz;
}

float Profile::getValue(float myz) {
  float zz = MAX(0.0f, (myz - z0));
  float dzvi = 1.0 / dz;
  float av1 = zz * dzvi;
  int iv1 = int(av1);
  float www = av1 - iv1;
  iv1 = MIN((nz - 1), iv1);
  int iv2 = MIN(nz - 1, iv1 + 1);
  float myvel = profile[iv1] * (1 - www) + profile[iv2] * www;
  return myvel;
}

float Profile::getVmin(float z00, float zmax0) {
  float dzvi = 1.0 / dz;
  int iz1 = int((z00 - z0) * dzvi);
  int iz2 = int((zmax0 - z0) * dzvi + 1);
#if 1 // seems other places cannot handle it yet
  assertion(iz1 >= 0 && iz2 <= nz,
            "ERROR: Propagation range iz1 or iz2 exceeds velocity range.\n  iz1 = %d, iz2 = %d, nv = %d, z0 = %f, zmax = %f\n",
            iz1, iz2, nz, z00, zmax0);
#else
  iz1 = std::max(0, iz1), iz2 = std::min(0, nz);
#endif

  float vmin = FLT_MAX;
  for(int iz = iz1; iz < iz2; iz++) {
    vmin = MIN(vmin, profile[iz]);
  }
  return vmin;
}

