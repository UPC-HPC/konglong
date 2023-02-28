/*
 * Lagrange.cpp
 *
 */


#include <math.h>
#include <string.h>
#include <xmmintrin.h>
#include <omp.h>

#include "Lagrange.h"
#include "Grid.h"
#include "Util.h"

#include "MpiPrint.h"
using MpiPrint::print1m;

Lagrange::Lagrange(Grid *gridIn, Grid *gridOut, int nPoints, int nThreads)
  : gridIn(gridIn), gridOut(gridOut), nPoints(nPoints), nThreads(nThreads) {

  coefz = coefx = coefy = 0;
  norderz = norderx = nordery = 0;
  nbeginz = nbeginx = nbeginy = 0;

  initCoefs();

  if(gridIn->nx > 1) workz.resize((size_t) gridOut->nz * gridIn->nx);

  if(gridIn->ny > 1) {
    workz.resize((size_t) gridOut->nz * gridIn->nx  * gridIn->ny);
    workx.resize((size_t) gridOut->nz * gridOut->nx * gridIn->ny);
  }

  print1m("Using Lagrange interpolation. \n");
  // gridIn->print();
  // gridOut->print();
}

Lagrange::~Lagrange() {

  if(coefz) delete[] coefz;
  if(coefx) delete[] coefx;
  if(coefy) delete[] coefy;

  if(norderz) delete[] norderz;
  if(norderx) delete[] norderx;
  if(nordery) delete[] nordery;

  if(nbeginz) delete[] nbeginz;
  if(nbeginx) delete[] nbeginx;
  if(nbeginy) delete[] nbeginy;
}


void Lagrange::initCoefs() {
  coefz = new float[gridOut->nz * nPoints];
  norderz = new int[gridOut->nz];
  nbeginz = new int[gridOut->nz];

  float *zgridOut = new float[gridOut->nz];
  for(int iz = 0; iz < gridOut->nz; iz++) zgridOut[iz] = gridOut->z0 + iz * gridOut->dz;

  if(gridIn->mytype == RECTANGLE) {
    float *zgridIn = new float[gridIn->nz];
    for(int iz = 0; iz < gridIn->nz; iz++) zgridIn[iz] = gridIn->z0 + iz * gridIn->dz;
    initCoef(zgridIn, gridIn->nz, zgridOut, gridOut->nz, nPoints, coefz, norderz, nbeginz);

    // printf( "nz = %d, nPoints = %d \n ", gridIn->nz , nPoints);
    // for (int iz = 0; iz < gridOut->nz; iz++)
    //   for (int iPoint = 0; iPoint < nPoints; iPoint++)
    //  printf( "iz = %d, iPoint = %d, nbeginz = %d, norderz = %d, coefz = %f \n ", iz, iPoint, nbeginz[iz], norderz[iz], coefz[iz+gridOut->nz*iPoint]);

    delete[] zgridIn;
  } else {
    initCoef(&gridIn->zgrid[0], gridIn->nz, zgridOut, gridOut->nz, nPoints, coefz, norderz, nbeginz);
  }
  delete[] zgridOut;

  if(gridIn->nx > 1) {
    coefx = new float[gridOut->nx * nPoints];

    norderx = new int[gridOut->nx];
    nbeginx = new int[gridOut->nx];

    float *xgridIn = new float[gridIn->nx];
    for(int ix = 0; ix < gridIn->nx; ix++) xgridIn[ix] = gridIn->x0 + ix * gridIn->dx;

    float *xgridOut = new float[gridOut->nx];
    for(int ix = 0; ix < gridOut->nx; ix++) xgridOut[ix] = gridOut->x0 + ix * gridOut->dx;

    initCoef(xgridIn, gridIn->nx, xgridOut, gridOut->nx, nPoints, coefx, norderx, nbeginx);

    delete[] xgridIn;
    delete[] xgridOut;
  }

  if(gridIn->ny > 1) {
    coefy = new float[gridOut->ny * nPoints];
    nordery = new int[gridOut->ny];
    nbeginy = new int[gridOut->ny];

    float *ygridIn = new float[gridIn->ny];
    for(int iy = 0; iy < gridIn->ny; iy++) ygridIn[iy] = gridIn->y0 + iy * gridIn->dy;

    float *ygridOut = new float[gridOut->ny];
    for(int iy = 0; iy < gridOut->ny; iy++) ygridOut[iy] = gridOut->y0 + iy * gridOut->dy;

    initCoef(ygridIn, gridIn->ny, ygridOut, gridOut->ny, nPoints, coefy, nordery, nbeginy);

    delete[] ygridIn;
    delete[] ygridOut;
  }
}


void Lagrange::initCoef(float *tblIn, int nin, float *tblOut, int nout, int nPoints, float *coef, int *norder, int *nbeg) {
  if(nPoints / 2 * 2 != nPoints) {
    printf("Error in gen_lagrange_coef: nPoints must be an enven number!\n");
    exit(-1);
  }
  int mintp = nPoints / 2;
  for(int iz = 0; iz < nout; ++iz) {
    int iz0 = checkGrid(tblOut[iz], tblIn, nin);
    size_t id = iz * nPoints;
    for(int i = 0; i < nPoints; ++i) coef[id + i] = 0.0;

    if(iz0 == -1) {
      norder[iz] = 1;
      nbeg[iz] = iz0;
    } else if(iz0 == nin) {
      norder[iz] = 1;
      nbeg[iz] = nin - 1;
    } else if(iz0 < mintp) {
      norder[iz] = (iz0 + 1) * 2;
      nbeg[iz] = 0;
    } else if(nin - 1 - iz0 < mintp) {
      norder[iz] = (nin - 1 - iz0) * 2;
      nbeg[iz] = nin  - norder[iz];
    } else {
      norder[iz] = nPoints;
      nbeg[iz] = iz0 - mintp;
    }
    getCoef(tblIn, nin, nbeg[iz], norder[iz], tblOut[iz], &coef[id]);

    //      printf("iz: %d, tblOut: %f, iz0: %d, tblIn: %f, norder: %d, ibeg: %d, coef[0]: %f \n", iz, tblOut[iz], iz0, tblIn[iz0], norder[iz], nbeg[iz], coef[id]);
  }
}

int Lagrange::checkGrid(float z, float *ztbl, int nztbl) {
  if(z < ztbl[0]) {
    return -1;
  } else if(z > ztbl[nztbl - 1]) {
    return nztbl;
  } else {
    int iz = 0;
    while(z > ztbl[iz + 1] && iz <= nztbl - 1) iz++;
    return iz;
  }
}
void Lagrange::getCoef(float *ztbl, int nztbl, int nzbeg, int norder, float z, float *intpcoef) {
  if(norder == 0) {
    intpcoef[0] = 1.0;
  } else {
    for(int i = 0; i < norder; ++i) {
      intpcoef[i] = 1.0;
      for(int j = 0; j < norder; ++j) {
        if(j != i) {
          intpcoef[i] = intpcoef[i] * (z - ztbl[j + nzbeg]) / (ztbl[i + nzbeg] - ztbl[j + nzbeg]);
        }
      }
      //      printf("i: %d, %f\n", i, intpcoef[i]);
    }
  }
}

void Lagrange::apply(float *dataIn, float *dataOut) {
  if(gridOut->ny > 1 || gridIn->ny > 1){
    interpz(dataIn, &workz[0], gridIn->nz, gridOut->nz, gridIn->nx, gridIn->ny, nPoints);
    interpx(&workz[0], &workx[0], gridIn->nx, gridOut->nx, gridIn->ny, gridOut->nz, nPoints);
    interpy(&workx[0], dataOut, gridIn->ny, gridOut->ny, gridOut->nx, gridOut->nz, nPoints);
  } else if(gridOut->nx > 1) {
    interpz(dataIn, &workz[0], gridIn->nz, gridOut->nz, gridIn->nx, gridIn->ny, nPoints);
    interpx(&workz[0], dataOut, gridIn->nx, gridOut->nx, gridIn->ny, gridOut->nz, nPoints);
  } else {
    interpz(dataIn, dataOut, gridIn->nz, gridOut->nz, gridIn->nx, gridIn->ny, nPoints);
  }
}

void Lagrange::interpz(float *dataIn, float *dataOut, int nzIn, int nzOut, int nx, int ny, int nintp) {

  int nxy = nx * ny;

  #pragma omp parallel for num_threads(nThreads) schedule(static)
  for(int ixy = 0; ixy < nxy; ixy++) {
    size_t id_out = (size_t) ixy * nzOut;
    size_t id_in = (size_t) ixy * nzIn;
    for(int iz = 0; iz < nzOut; ++iz) {
      dataOut[id_out + iz] = applyCoef(nbeginz[iz], norderz[iz], &coefz[iz * nintp], &dataIn[id_in]);
    }
    //    printf("ixy: %d, nxy: %d, last value: %f\n", ixy, nxy,  dataOut[id_out+nzOut]);
  }
}

void Lagrange::interpx(float *dataIn, float *dataOut, int nxIn, int nxOut, int ny, int nz, int nintp) {

  size_t nxzIn  = nxIn  * nz;
  size_t nxzOut = nxOut * nz;

  #pragma omp parallel num_threads(nThreads)
  {
    float *wrk1 = new float[nxzIn];
    float *wrk2 = new float[nxzOut];
    #pragma omp for schedule(static)
    for(int iy = 0; iy < ny; iy++) {

      Util::transpose(nz, nxIn, nz, nxIn, dataIn + iy * nxzIn, wrk1);

      for(int iz = 0; iz < nz; iz++) {
        for(int ix = 0; ix < nxOut; ix++) {
          wrk2[ix + iz * nxOut] = applyCoef(nbeginx[ix], norderx[ix], &coefx[ix * nintp], &wrk1[iz * nxIn]);
        }
      }

      Util::transpose(nxOut, nz, nxOut, nz, wrk2, dataOut + iy * nxzOut);
    }

    delete[] wrk1;
    delete[] wrk2;
  }
}

void Lagrange::interpy(float *dataIn, float *dataOut, int nyIn, int nyOut, int nx, int nz, int nintp) {

  size_t nxz = nx * nz;

  #pragma omp parallel num_threads(nThreads)
  {
    float *wrk1 = new float[nyIn * nz];
    float *wrk2 = new float[nyOut * nz];
    #pragma omp for schedule(static)
    for(int ix = 0; ix < nx; ix++) {

      Util::transpose(nz, nyIn, nxz, nyIn, dataIn + ix * nz, wrk1);

      for(int iz = 0; iz < nz; iz++) {
        for(int iy = 0; iy < nyOut; iy++) {
          wrk2[iy + iz * nyOut] = applyCoef(nbeginy[iy], nordery[iy], &coefy[iy * nintp], &wrk1[iz * nyIn]);
        }
      }

      Util::transpose(nyOut, nz, nyOut, nxz, wrk2, dataOut + ix * nz);
    }
    delete[] wrk1;
    delete[] wrk2;
  }
}

float Lagrange::applyCoef(int nbeg, int norder, float *coef, float *data) {
  float yval = 0.0;
  for(int i = 0; i < norder; ++i) {
    yval += data[nbeg + i] * coef[i];
    //    printf("i: %d, data: %f, coef: %f, nbeg: %d\n", i, data[nbeg+i], coef[i], nbeg);
  }
  return yval;
}

