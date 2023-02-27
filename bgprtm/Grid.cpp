#include <stdlib.h>
#include <omp.h>

#include "GetPar.h"
#include "Grid.h"
#include "Vector3.h"
#include "Profile.h"
#include "libFFTV/numbertype.h"
#include "libCommon/padfft.h"
#include "libCommon/Options.h"
#include "libCommon/Assertion.h"
#include "libCommon/Taylor.h"
using libCommon::Taylor;

Grid::Grid(int type, int nx0, int ny0, int nz0, float dx0, float dy0, float dz0, float z00, float z11, int nThreads0) {
  mytype = type;
  nx = nx0;
  ny = ny0;
  nzalloc = nnz = nz = nz0;
  dx = dx0;
  dy = dy0;
  dz = dz0;
  z0 = zmin = z00;
  iz0 = 0;
  zmax = z11;
  create();
  nThreads = nThreads0;
}

Grid::Grid(Grid *input, int nDblz, int nDblx, int nDbly) : nDblz(nDblz), nDblx(nDblx), nDbly(nDbly) {
  assertion(nDblz == 1 || nDblz == 2, "nDblz(%d) must be either 1 or 2", nDblz);
  assertion(nDblx == 1 || nDblx == 2, "nDblx(%d) must be either 1 or 2", nDblx);
  assertion(nDbly == 1 || nDbly == 2, "nDbly(%d) must be either 1 or 2", nDbly);

  *this = *input;  //copy the constructor

  if(nDblz == 2) {
    nnz *= 2;
    nz *= 2, dz /= 2.0;
    nzalloc = MAX(nz, nnz);
    if(nx > 1 && nDblx == 2) nx *= 2, dx /= 2.0;
    if(ny > 1 && nDbly == 2) ny *= 2, dy /= 2.0;
  }
  alloctables();
  if(nDblz == 2) {
    iz0 *= 2;
    nztop *= 2;
    if(mytype != RECTANGLE) {
      if(nx > 1 && nDblx == 2) slopex *= 0.5;
      if(ny > 1 && nDbly == 2) slopey *= 0.5;
      doublezsample(input);
      fillazmaps();
    }
    setupJacob();
  }
  mysize = (size_t)nx * (size_t)ny * (size_t)nz;
}

void Grid::doublezsample(Grid *input) {
  for(int i = 0; i < input->nz; i++) {
    zgrid[i * 2] = input->zgrid[i];
    dzgrid[i * 2] = input->dzgrid[i] * 0.5f;
    if(nDblx == 2) dxgrid[i * 2] = input->dxgrid[i] * 0.5f;
    if(nDbly == 2) dygrid[i * 2] = input->dygrid[i] * 0.5f;
  }
  for(int i = 0; i < input->nz; i++) {
    int id = MIN(i + 1, input->nz - 1);
    float mydz = 0.25 * (input->dzgrid[i] + input->dzgrid[id]);   // linear interpolation of dz
    float myz = input->zgrid[i] + 0.25 * input->dzgrid[i] + 0.5 * mydz;
    zgrid[i * 2 + 1] = myz;
    dzgrid[i * 2 + 1] = mydz;
    if(nDblx == 2) dxgrid[i * 2 + 1] = input->dxgrid[i] * 0.5f;
    if(nDbly == 2) dygrid[i * 2 + 1] = input->dygrid[i] * 0.5f;
  }
}

void Grid::create() {
  nztop = 0;
  slopex = 0.0f;
  slopey = 0.0f;
  mysize = nx * ny;
  mysize *= nz;                      // Total size of each volume
  alloctables();
  //printf("The nz is %d   %p  %p\n", nz, zgrid, dzgrid);
}

void Grid::alloctables() {
  dxgrid.clear();
  dxgrid.resize(nzalloc, dx);
  dygrid.clear();
  dygrid.resize(nzalloc, dy);
  dzgrid.clear();
  dzgrid.resize(nzalloc, dz);
  if(mytype == RECTANGLE) return;    // Rectangle grid is the basic grid

  zgrid.resize(nzalloc);
  azmaps.resize(nzalloc);
  jacobz.resize(nzalloc);
  if(mytype == IRREGULAR) return;

  jacobx.resize(nzalloc);
  jacoby.resize(nzalloc);
}

void Grid::setupGrid(Profile *prof_in, int zbnd, int nzuppad) {
  nztop = iz0 = zbnd + nzuppad;  // set up the boundary
  switch(mytype) {
  case RECTANGLE:
    setupRectangle();
    break;
  case IRREGULAR:
  case YPYRAMID:
  case XPYRAMID:
  case XYPYRAMID:
    int nz_tmp = boundedzgrid(prof_in, zbnd);
    nz = libCommon::padfft(nz_tmp);
    nzalloc = MAX(nz, nnz);
    create();
    nztop = zbnd + nzuppad;
    boundedzgrid(prof_in, zbnd);
    fillazmaps();
    break;
  }
  float mydx = getdx(z0);
  float mydy = getdy(z0);
  apertx = (nx - 1) * mydx;        // surface aperturex
  aperty = (ny - 1) * mydy;
  return;
}

void Grid::setupGrid(float *zgrid0, float *dzgrid0, int zbnd, int nzuppad) {
  nztop = iz0 = zbnd + nzuppad;  // set up the boundary
  switch(mytype) {
  case RECTANGLE:
    setupRectangle();
    break;
  case IRREGULAR:
    memcpy(&zgrid[0], zgrid0, sizeof(float) * nz);
    memcpy(&dzgrid[0], dzgrid0, sizeof(float) * nz);
    //    printf("Copying zgrid ...\n");
    //    printZGrid();
    //    exit(0);
    fillazmaps();
    z0 = zgrid[0];
    break;
  }
  float mydx = getdx(z0);
  float mydy = getdy(z0);
  apertx = (nx - 1) * mydx;        // surface aperturex
  aperty = (ny - 1) * mydy;
}

float Grid::getdy(float myz) const {
  if(myz < z0 || mytype == RECTANGLE || mytype == IRREGULAR || mytype == XPYRAMID) return dy;
  float zz = myz - z0;
  float a = dy + zz * slopey;
  return a;
}

float Grid::getdx(float myz) const {
  if(myz < z0 || mytype == RECTANGLE || mytype == IRREGULAR || mytype == YPYRAMID) return dx;
  float zz = myz - z0;
  float a = dx + zz * slopex;
  return a;
}

void Grid::simplezgrid(Profile *prof_in) {
  float dzmin = dz;
  float vmin = prof_in->getVmin(z0, zmax);
  float vmini = 1.0 / vmin;
  //printf("The zgrid %f  \n", vmin);
  fflush(stdout);
  float myz = z0;
  int idz = 0;
  float myvel = prof_in->getValue(myz);
  float mydz = myvel * dzmin * vmini;
  zgrid[idz] = myz;
  dzgrid[idz] = mydz;                                  // First grid is done
  //iz0          = 0;
  while(myz < zmax) {
    float myz1 = myz + mydz;
    float myvel1 = prof_in->getValue(myz1);
    float mydz1 = myvel1 * dzmin * vmini;
    myz += 0.5 * (mydz + mydz1);
    mydz = mydz1;
    idz++;
    zgrid[idz] = myz;
    dzgrid[idz] = mydz;
    // printf("The zgrid %d  %d   %f  %f\n", nz, idz, zgrid[idz], dzgrid[idz]);
  }

  nz = idz + 1;
  return;
}

float Grid::irregzloc(float zz, float range, float dz1) const {
  float dzi1 = 1.0 / dz1;
  float rangei = 1.0 / range;
  float dzi2 = 1.0 / (range - dz1);
  float aa = dzi2 * (rangei - 0.5 * dzi1);
  float bb = rangei - aa * range;
  return aa * zz * zz + bb * zz;
}

// this one ignore z0, compare with getIDzf()
float Grid::getmyZloc(float zz) const {
  if(mytype == RECTANGLE) return zz / dz + iz0;
  float az = zgrid[0];
  if(zz <= az) return (zz - az) / (zgrid[1] - zgrid[0]);

  int jz = 0;
  while(zz >= az && jz < (nz - 1))
    az = zgrid[++jz];
  az = zgrid[--jz];
  float diff = zz - az;
  float zloc = jz + diff / (zgrid[jz + 1] - zgrid[jz]);
#if 0
  printZGrid();
  printf("%f, %f\n", zz, zloc);
  exit(1);
#endif
  return zloc;
}

float Grid::getmyZloc_backup(float zz) const {
  if(mytype == RECTANGLE) return zz / dz + iz0;
  float az = zgrid[0];
  int jz = 0;
  while(zz >= az && jz < (nz - 1))
    az = zgrid[++jz];
  az = zgrid[--jz];
  float diff = zz - az;
  float range = zgrid[jz + 1] - az;
  int kz = MIN(jz + 1, nz - 1);
  float hdz = 0.375 * dzgrid[jz] + 0.125 * dzgrid[kz];
  return jz + irregzloc(diff, range, hdz);
}

float Grid::getmyXloc(float xx, float zz) const {
  float dxu = getdx(zz);
  float posi = (xx - x0) / dxu;
  return posi;
}

float Grid::getmyYloc(float yy, float zz) const {
  float dyu = getdy(zz);
  float posi = (yy - y0) / dyu;
  return posi;
}

void Grid::fillazmaps() {
  for(int iz = 0; iz < nnz; iz++) {
    float zz = zmin + iz * dz;               // regular grid z sampling, start with zmin
    azmaps[iz] = getmyZloc(zz);
    //printf("The irregular int  %d  %d  %f  %f  %f  %f\n", iz, jz, azmaps[iz], range, diff, dzgrid[jz] );
  }
}


void Grid::setupSlope(float dymax, float dxmax) {
  setupJacob();
  if(mytype == RECTANGLE || mytype == IRREGULAR) return;
  float zrange = zmax - z0;
  float zrangei = 1.0 / zrange;
  float ddy = dymax - dy;
  slopey = ddy * zrangei;
  if(mytype == YPYRAMID) return;
  slopex = (dxmax - dx) * zrangei;
  setupJacob();
  for(int iz = 0; iz < nz; iz++) {
    float thismyz = getmyz(iz);
    dxgrid[iz] = getdx(thismyz);
    dygrid[iz] = getdy(thismyz);
  }
  return;
}

int Grid::getIDx(float x, float z) const {
  return (int)nearbyintf(getIDxf(x, z));
}

float Grid::getIDxf(float x, float z) const {
  float dxu = getdx(z);
  float idx = (x - x0) / dxu;
  return idx;
}

int Grid::getIDy(float y, float z) const {
  return (int)nearbyintf(getIDyf(y, z));
}

float Grid::getIDyf(float y, float z) const {
  float dyu = getdy(z);
  float idx = (y - y0) / dyu;
  return idx;
}

int Grid::getIDz(float z) const {

  return (int)nearbyintf(getIDzf(z));

}

/*
 float Grid::getIDzf1(float z) const {
 if (mytype == RECTANGLE) return (z - z0) / dz;
 int iz = 0;
 while (zgrid[iz] < z && iz < nz - 1)
 iz++;
 if (iz == 0 || (iz == nz - 1 && z >= zgrid[iz])) return (float) iz;
 return (iz - 1) + (z - zgrid[iz - 1]) / (zgrid[iz] - zgrid[iz - 1]);
 }


 int Grid::getIDz(float z) const {
 return (int) nearbyintf(getIDzf(z));
 }

 */

float Grid::getIDzf(float z) const {
  if(mytype == RECTANGLE) return (z - z0) / dz;
  int iz = 0;
  while(zgrid[iz] < z && iz < nz - 1)
    iz++;
  if(iz == 0 || (iz == nz - 1 && z >= zgrid[iz])) return (float)iz;
  return (iz - 1) + (z - zgrid[iz - 1]) / (zgrid[iz] - zgrid[iz - 1]);
}

vector3 Grid::getxloc(int ix, int iy, int iz) const {
  vector3 xloc;
  xloc.x = getmyx(ix, iz);
  xloc.y = getmyy(iy, iz);
  xloc.z = getmyz(iz);
  return xloc;
}

int zbetween(float zz0, float *zz, int nz) {
  int iz;

  if(zz0 < zz[0]) {
    iz = -1;
  } else if(zz0 < 0.5 * (zz[0] + zz[1])) {
    iz = 0;
  } else {
    iz = 0;
    while(iz <= nz - 2 && zz0 >= 0.5 * (zz[iz] + zz[iz + 1])) {
      iz++;
    }
    if(iz == nz - 1) {
      if(zz0 < zz[nz - 1]) {
        iz = nz - 1;
      } else {
        iz = nz;
      }
    }
  }
  return iz;
}

float AveRegrid(float zz1, float zz2, float *Az, float *zz, int nz) {
  float ave;
  float length, sum;
  float dlength, dsum;

  int iz1, iz2;

  // zbtween guarantees returning in the range of [-1,nz]
  iz1 = zbetween(zz1, zz, nz);
  iz2 = zbetween(zz2, zz, nz);

  if(iz1 == -1 && iz2 == -1) {
    ave = Az[0];
  } else if(iz1 == nz && iz2 == nz) {
    ave = Az[nz - 1];
  } else if(iz1 == iz2) {
    ave = Az[iz1];
  } else {
    if(iz1 == -1) {
      length = zz[0] - zz1;
      sum = length * Az[0];
    } else if(iz1 == 0) {
      length = 0.5 * (zz[0] + zz[1]) - zz1;
      sum = length * Az[0];
    } else if(iz1 == nz - 1) {
      length = zz[nz - 1] - zz1;
      sum = length * Az[nz - 1];
    } else {
      length = 0.5 * (zz[iz1] + zz[iz1 + 1]) - zz1;
      sum = length * Az[iz1];
    }
    for(int iz = iz1 + 1; iz < iz2; iz++) {
      if(iz == 0) {
        dlength = 0.5 * (zz[1] - zz[0]);
        dsum = dlength * Az[0];
      } else if(iz == nz - 1) {
        dlength = 0.5 * (zz[nz - 1] - zz[nz - 2]);
        dsum = dlength * Az[nz - 1];
      } else {
        dlength = 0.5 * (zz[iz + 1] - zz[iz - 1]);
        dsum = dlength * Az[iz];
      }
      length = length + dlength;
      sum = sum + dsum;
    }

    if(iz2 >= nz || iz2 <= 0) {
      int iz = (iz2 >= nz) ? nz - 1 : 0;
      dlength = zz2 - zz[iz];
      dsum = dlength * Az[iz];
    } else {
      dlength = zz2 - 0.5 * (zz[iz2] + zz[iz2 - 1]);
      dsum = dlength * Az[iz2];
    }
    length = length + dlength;
    sum = sum + dsum;
    ave = sum / length;
  }
  return ave;

}

int ZnumGrid(Profile *prof_in, float z0, float zmax, float dzmin) {
  float vmin = prof_in->getVmin(z0, zmax);
  float vmini = 1.0 / vmin;
  //printf("The zgrid %f  \n", vmin);
  fflush(stdout);
  float myz = z0;
  int idz = 0;
  float delt = dzmin * vmini;

  while(myz < zmax) {
    float tau = idz * delt;
    myz = prof_in->TimeGetDepth(tau);
    idz++;
  }
  int nz = idz;
  return nz;
}

int ZnumGrid_backup(Profile *prof_in, float z0, float zmax, float dzmin) {
  float vmin = prof_in->getVmin(z0, zmax);
  float vmini = 1.0 / vmin;
  //printf("The zgrid %f  \n", vmin);
  fflush(stdout);
  float myz = z0;
  int idz = 0;
  float myvel = prof_in->getValue(myz);
  float mydz = myvel * dzmin * vmini;
  while(myz < zmax) {
    float myz1 = myz + mydz;
    float myvel1 = prof_in->getValue(myz1);
    float mydz1 = myvel1 * dzmin * vmini;
    myz += 0.5 * (mydz + mydz1);
    mydz = mydz1;
    idz++;
  }
  int nz = idz + 1;
  return nz;
}

int Grid::boundedzgrid(Profile *prof_in, int nzbnd) {
  z0 = zmin;
  float vmin = prof_in->getVmin(z0, zmax);
  float vmini = 1.0 / vmin;
  float myz = z0;
  int zbnd = nztop;
  int idz = zbnd;
  float dzmin = dz;
  float delt = dzmin * vmini;

  zgrid[idz] = z0;
  idz++;

  while(myz < zmax) {
    float tau = (idz - zbnd) * delt;
    myz = prof_in->TimeGetDepth(tau);
    zgrid[idz] = myz;
    idz++;
  }

  float mydz0 = zgrid[zbnd + 1] - zgrid[zbnd];

  for(int iz = zbnd - 1; iz >= 0; iz--) {    // upper boundary
    zgrid[iz] = zgrid[iz + 1] - mydz0;
  }

  for(int iz = 0; iz < nzbnd; iz++) {       // lower boundary
    float tau = (idz - zbnd) * delt;
    myz = prof_in->TimeGetDepth(tau);
    zgrid[idz] = myz;
    idz++;
  }

  while(idz < nz) {
    float tau = (idz - zbnd) * delt;
    myz = prof_in->TimeGetDepth(tau);
    zgrid[idz] = myz;
    idz++;
  }
  z0 = zgrid[0];                   //

  for(idz = 0; idz < nz - 1; idz++) {
    dzgrid[idz] = zgrid[idz + 1] - zgrid[idz];
  }
  dzgrid[nz - 1] = dzgrid[nz - 2];
  idz++;

  return idz;
}

int Grid::boundedzgrid_backup(Profile *prof_in, int nzbnd) {
  int niter = 5;
  float myz1, myvel1, mydz1;
  int zbnd = nztop;
  z0 = zmin;
  float dzmin = dz;
  float vmin = prof_in->getVmin(z0, zmax);
  float vmini = 1.0 / vmin;
  //printf("The zgrid %f  \n", vmin);
  fflush(stdout);
  float myz = z0;
  int idz = zbnd;
  float myvel = prof_in->getValue(myz);
  float mydz = myvel * dzmin * vmini;
  zgrid[idz] = z0;
  dzgrid[idz] = mydz;
  //iz0          = zbnd;
  for(int iz = zbnd - 1; iz >= 0; iz--) {    // upper boundary
    myz1 = myz - mydz;
    myvel1 = prof_in->getValue(myz1);
    mydz1 = myvel1 * dzmin * vmini;
    myz -= 0.5 * (mydz + mydz1);
    mydz = mydz1;
    zgrid[iz] = myz;
    dzgrid[iz] = mydz;
  }

  myz = z0;                        // set back iz to z0
  while(myz < zmax) {
    mydz1 = mydz;
    for(int i = 0; i < niter; i++) {
      myz1 = myz + 0.5 * (mydz + mydz1);
      myvel1 = prof_in->getValue(myz1);
      mydz1 = myvel1 * dzmin * vmini;
    }
    myz += 0.5 * (mydz + mydz1);
    mydz = mydz1;
    idz++;
    zgrid[idz] = myz;
    dzgrid[idz] = mydz;

  }

  for(int iz = 0; iz < nzbnd; iz++) {       // lower boundary
    mydz1 = mydz;
    for(int i = 0; i < niter; i++) {
      myz1 = myz + 0.5 * (mydz + mydz1);
      myvel1 = prof_in->getValue(myz1);
      mydz1 = myvel1 * dzmin * vmini;
    }
    myz += 0.5 * (mydz + mydz1);
    mydz = mydz1;
    idz++;
    zgrid[idz] = myz;
    dzgrid[idz] = mydz;
  }
  z0 = zgrid[0];                   //
  if(idz + 1 < nz) {
    for(int iz = 0; iz < nz - idz - 1; iz++) {       // lower boundary
      mydz1 = mydz;
      for(int i = 0; i < niter; i++) {
        myz1 = myz + 0.5 * (mydz + mydz1);
        myvel1 = prof_in->getValue(myz1);
        mydz1 = myvel1 * dzmin * vmini;
      }
      myz += 0.5 * (mydz + mydz1);
      mydz = mydz1;
      idz++;
      zgrid[idz] = myz;
      dzgrid[idz] = mydz;
    }
  }
  z0 = zgrid[0];                   //
  return idz + 1;
}

int ZnumGridbnd(Profile *prof_in, float z0, float zmax, float dzmin, int zbnd) {
  float vmin = prof_in->getVmin(z0, zmax);
  float vmini = 1.0 / vmin;
  //printf("The zgrid %f  \n", vmin);
  fflush(stdout);
  float myz = z0;
  int idz = zbnd;
  float myvel = prof_in->getValue(myz);
  float mydz = myvel * dzmin * vmini;
  //zgrid [idz] = z0;
  //dzgrid[idz] = mydz;
  for(int iz = zbnd - 1; iz >= 0; iz--) {    // upper boundary
    float myz1 = myz - mydz;
    float myvel1 = prof_in->getValue(myz1);
    float mydz1 = myvel1 * dzmin * vmini;
    myz -= 0.5 * (mydz + mydz1);
    mydz = mydz1;
    //zgrid [iz] = myz;
    //dzgrid[iz] = mydz;
  }

  myz = z0;                       // set back iz to z0
  while(myz < zmax) {
    float myz1 = myz + mydz;
    float myvel1 = prof_in->getValue(myz1);
    float mydz1 = myvel1 * dzmin * vmini;
    myz += 0.5 * (mydz + mydz1);
    mydz = mydz1;
    //zgrid [idz] = myz;
    //dzgrid[idz] = mydz;
    idz++;
  }

  for(int iz = 0; iz < zbnd; iz++) {    // lower boundary
    float myz1 = myz + mydz;
    float myvel1 = prof_in->getValue(myz1);
    float mydz1 = myvel1 * dzmin * vmini;
    myz += 0.5 * (mydz + mydz1);
    mydz = mydz1;
    //zgrid [idz] = myz;
    //dzgrid[idz] = mydz;
    idz++;
  }

  int nz = idz + 1;
  return nz;
}

void Grid::setMidpoint(float midx0, float midy0) {
  centerx = midx0;
  centery = midy0;
  x0 = midx0 - 0.5 * apertx;  // irregular grid
  y0 = midy0 - 0.5 * aperty;
}

void Grid::setOrigin(float x0, float y0) {
  this->x0 = x0;
  this->y0 = y0;
  centerx = x0 + 0.5f * apertx; // regular grid
  centery = y0 + 0.5f * aperty;
}
void Grid::setIncXY(int incx, int incy) {
  this->incx = incx;
  this->incy = incy;
}

void Grid::setupJacob() {
  for(int iz = 0; iz < nz; iz++) {
    if(jacobx.size()) jacobx[iz] = dx / getdx(zgrid[iz]);
    if(jacoby.size()) jacoby[iz] = dy / getdy(zgrid[iz]);
  }
  int order = 8;
  if(jacobz.size()) {
    Taylor::derive1(&jacobz[0], &zgrid[0], nz, dz, order / 2);
    for(int i = 0; i < nz; i++)
      jacobz[i] = 1 / jacobz[i];
//    cout << "jacobz: [" << COptions::floats2str(jacobz) << "]\n";
  }
}

void Grid::savefdm(const char *filename, float *vol, float gx0, float gy0, float gz0) {
  // printf("adde %p \n  %d  %d  %d\n", vol, nx, ny, nz);
  saveFdmCube(vol, filename, x0, y0, z0, nx, ny, nz, dx, dy, dz, gx0, gy0, gz0);
}

void Grid::saveModel(const char *filename, float *vol) {
  //printf("adde %p \n  %d  %d  %d\n", vol, nx, ny, nz);
  saveGlobalModel(vol, x0, y0, z0, nx, ny, nz, dx, dy, dz, filename);
}

void Grid::saveModel(string filename, float *vol) {
  saveModel(filename.c_str(), vol);
}

void Grid::FillVolume(Grid *gridin, float *volumein, float *volumeut) {
  if(gridin->nx < 2 || gridin->ny < 2) {
    //printf("grid: FillVolume2D\n");
    FillVolume2D(gridin, volumein, volumeut);
  } else {
    //printf("grid: FillVolume3D\n");
    FillVolume3D(gridin, volumein, volumeut);
  }
}

void Grid::FillVolume2D(Grid *gridin, float *volumein, float *volumeut) {
  float *weightz = new float[nz];
  //printf("gridin: iz0=%d, dz=%f\n", gridin->iz0, gridin->dz);
  for(int iz = 0; iz < nz; iz++) {
    float myz = getmyz(iz);
    weightz[iz] = gridin->getIDzf(myz); // need to use z0
    //printf("iz=%d, myz=%f, weight=%f\n", iz, myz, weightz[iz]);
  }

  Interpolant *myin = new Interpolant(LANCZOS, 9, nThreads);

  size_t size1D = MAX(nx, gridin->nx);
  size_t size2D = size1D * MAX(nz, gridin->nz);
  float *buff1Din = new float[size1D * nThreads];
  float *buff1Dut = new float[size1D * nThreads];
  float *buff2D = new float[size2D];

  //#pragma omp parallel num_threads(nThreads)
  {
    int tid = omp_get_thread_num();
    float *work1Din = buff1Din + tid * size1D;
    float *work1Dut = buff1Dut + tid * size1D;
    memset(work1Din, 0, sizeof(size1D));
    memset(work1Dut, 0, sizeof(size1D));

    //#pragma omp for schedule(static)
    for(int iz = 0; iz < gridin->nz; iz++) {

      for(int ix = 0; ix < gridin->nx; ix++) {
        size_t idin = ix * gridin->nz + iz;
        work1Din[ix] = volumein[idin];
      }

      float xin0 = gridin->getmyx(0, iz);
      float myz = gridin->getmyz(iz);
      float dxin = gridin->getdx(myz);

      int izut = getIDz(myz);
      float xut0 = getmyx(0, izut);
      float dxut = getdx(myz);

      myin->interpolate1D(work1Din, gridin->nx, xin0, dxin, work1Dut, nx, xut0, dxut);

      for(int ix = 0; ix < nx; ix++) {
        size_t idin = ix * gridin->nz + iz;
        buff2D[idin] = work1Dut[ix];
      }
    }

    //#pragma omp for schedule(static)
    //printf("required volumeout size: nx=%d,nz=%d,%ld\n", nx, nz, (size_t) nx * nz), fflush(stdout);
    for(int ix = 0; ix < nx; ix++) {
      size_t idin = ix * gridin->nz;
      size_t idut = ix * nz;
      float *ddin = buff2D + idin;
      float *ddut = volumeut + idut;
      myin->interpolate1D(ddin, ddut, weightz, gridin->nz, nz);
    }
  }
  delete[] buff1Dut;
  delete[] buff1Din;
  delete[] weightz;
  delete[] buff2D;
  delete myin;
  return;
}

void Grid::FillVolume3D(Grid *gridin, float *volumein, float *volumeut) {

  size_t size2D = MAX(nx, gridin->nx) * MAX(ny, gridin->ny);
  size_t sizeMx = size2D * MAX(nz, gridin->nz);   // max size of the volume
  vector<float> buff3D(sizeMx, 0.0f);
  vector<float> buff2Din(size2D * nThreads);
  vector<float> buff2Dut(size2D * nThreads);

  vector<float> weightz(nz);
  for(int iz = 0; iz < nz; iz++) {
    float myz = getmyz(iz);
    weightz[iz] = gridin->getIDzf(myz); // need to use z0
    // printf("iz, weightz: %d %f \n", iz, weightz[iz]);
  }

  Interpolant myin(LANCZOS, 9, nThreads);
  //myin.printTable();

#pragma omp parallel num_threads(nThreads)
  {
    int tid = omp_get_thread_num();
#pragma omp for schedule(static)
    for(int iz = 0; iz < gridin->nz; iz++) {
      for(int iy = 0; iy < gridin->ny; iy++) {
        for(int ix = 0; ix < gridin->nx; ix++) {
          size_t id2d = iy * gridin->nx + ix;
          size_t idin = id2d * gridin->nz + iz;
          buff2Din[id2d + tid * size2D] = volumein[idin];
        }
      }
      float xin0 = gridin->getmyx(0, iz);
      float yin0 = gridin->getmyy(0, iz);
      float myz = gridin->getmyz(iz);
      float dxin = gridin->getdx(myz);
      float dyin = gridin->getdy(myz);

      int izut = getIDz(myz);
      float xut0 = getmyx(0, izut);
      float yut0 = getmyy(0, izut);
      float dxut = getdx(myz);
      float dyut = getdy(myz);
      myin.interpolate2D(&buff2Din[tid * size2D], gridin->nx, gridin->ny, xin0, yin0, dxin, dyin, &buff2Dut[tid * size2D], nx, ny, xut0,
                         yut0, dxut, dyut);
      //printf("xin0, yin0, dxin, dyin, xut0, yut0, dxut, dyut: %f %f %f %f : %f %f %f %f\n",xin0, yin0, dxin, dyin, xut0, yut0, dxut, dyut);
      for(int iy = 0; iy < ny; iy++) {
        for(int ix = 0; ix < nx; ix++) {
          size_t id2d = iy * nx + ix;
          size_t idin = id2d * gridin->nz + iz;
          buff3D[idin] = buff2Dut[id2d + tid * size2D];
        }
      }
    }

#pragma omp for schedule(static)
    for(int iy = 0; iy < ny; iy++) {
      for(int ix = 0; ix < nx; ix++) {
        size_t id00 = iy * nx + ix;
        size_t idin = id00 * gridin->nz;
        size_t idut = id00 * nz;
        float *ddin = &buff3D[idin];
        float *ddut = volumeut + idut;
        myin.interpolate1D(ddin, ddut, &weightz[0], gridin->nz, nz);
      }
    }
  }

  return;
}

float Grid::getmyz(int iz) const {
  if(mytype == RECTANGLE) return z0 + iz * dz;
  iz = MIN(nz - 1, iz);
  iz = MAX(0, iz);
  return zgrid[iz];
}
void Grid::printZGrid() const {
  for(int i = 0; i < nz; i++) {
    printf("%d, %f \n", i, zgrid[i]);
  }
}

void Grid::print(int verbose) const {
  printf(" -------- Grid Info ----------\n");
  printf("nx %d, ny %d, nz %d \n", nx, ny, nz);
  printf("nnz %d, nzalloc %d, iz0 %d, nztop %d \n", nnz, nzalloc, iz0, nztop);
  printf("dx %f, dy %f, dz %f \n", dx, dy, dz);
  printf("x0 %f, y0 %f, z0 %f \n", x0, y0, z0);
  printf("centerx %f, centery %f\n", centerx, centery);
  printf("apertx %f, aperty %f\n", apertx, aperty);
  printf("zmin %f, zmax %f \n", zmin, zmax);
  printf("slopex %f, slopey %f \n", slopex, slopey);
  printf("size %ld, nThreads %d\n", mysize, nThreads);
  if(!verbose) return;
  this->printFloatVector(zgrid, "--- zGrid info ---");
  this->printFloatVector(dxgrid, "--- dxGrid info ---");
  this->printFloatVector(dygrid, "--- dyGrid info ---");
  this->printFloatVector(dzgrid, "--- dzGrid info ---");
  this->printFloatVector(azmaps, "--- azmaps info ---");
  this->printFloatVector(jacobx, "--- jacobx info ---");
  this->printFloatVector(jacoby, "--- jacoby info ---");
  this->printFloatVector(jacobz, "--- jacobz info ---");
}
void Grid::printFloatVector(vector<float> v, string info) const {
  printf("%s \n", info.c_str());
  int size = (int)v.size();
  printf("vector size %d\n", size);
    for(int i = 0; i < size; i++)
        printf("  index %d  val %f\n", i, v[i]);
}

void Grid::saveGrid(string file){
    ofstream ofs(file.c_str());
    if(!ofs.is_open())libCommon::Utl::fatal(string("can't open file ")+file);
    ofs<<"type: "<<mytype<<endl;
    ofs<<"nx: "<<nx<<endl;
    ofs<<"ny: "<<ny<<endl;
    ofs<<"nz: "<<nz<<endl;
    ofs<<"nnz: "<<nnz<<endl;
    ofs<<"nzalloc: "<<nzalloc<<endl;
    ofs<<"iz0: "<<iz0<<endl;
    ofs<<"nztop: "<<nztop<<endl;
    ofs<<"incx: "<<incx<<endl;
    ofs<<"incy: "<<incy<<endl;
    ofs<<"dx: "<<dx<<endl;
    ofs<<"dy: "<<dy<<endl;
    ofs<<"dz: "<<dz<<endl;
    ofs<<"x0: "<<x0<<endl;
    ofs<<"y0: "<<y0<<endl;
    ofs<<"z0: "<<z0<<endl;
    ofs<<"centerx: "<<centerx<<endl;
    ofs<<"centery: "<<centery<<endl;
    ofs<<"apertx: "<<apertx<<endl;
    ofs<<"aperty: "<<aperty<<endl;
    ofs<<"zmin: "<<zmin<<endl;
    ofs<<"zmax: "<<zmax<<endl;
    ofs<<"slopex: "<<slopex<<endl;
    ofs<<"slopey: "<<slopey<<endl;
    ofs<<"mysize: "<<mysize<<endl;
    outVec(zgrid, "zgrid", ofs);
    outVec(dxgrid, "dxgrid", ofs);
    outVec(dygrid, "dygrid", ofs);
    outVec(dzgrid, "dzgrid", ofs);
    ofs.close();
}

void Grid::readGrid(string file){
    ifstream ifs(file.c_str());
    if(!ifs.is_open())libCommon::Utl::fatal(string("can't open file ")+file);
    string tmp;
    ifs>>tmp>>mytype;
    ifs>>tmp>>nx;
    ifs>>tmp>>ny;
    ifs>>tmp>>nz;
    ifs>>tmp>>nnz;
    ifs>>tmp>>nzalloc;
    ifs>>tmp>>iz0;
    ifs>>tmp>>nztop;
    ifs>>tmp>>incx;
    ifs>>tmp>>incy;
    ifs>>tmp>>dx;
    ifs>>tmp>>dy;
    ifs>>tmp>>dz;
    ifs>>tmp>>x0;
    ifs>>tmp>>y0;
    ifs>>tmp>>z0;
    ifs>>tmp>>centerx;
    ifs>>tmp>>centery;
    ifs>>tmp>>apertx;
    ifs>>tmp>>aperty;
    ifs>>tmp>>zmin;
    ifs>>tmp>>zmax;
    ifs>>tmp>>slopex;
    ifs>>tmp>>slopey;
    ifs>>tmp>>mysize;
    inVec(zgrid,  ifs);
    inVec(dxgrid, ifs);
    inVec(dygrid, ifs);
    inVec(dzgrid, ifs);
    ifs.close();
}


void Grid::outVec(vector<float>& vec, string str, ofstream& ofs){
    ofs<<str<<":  "<<vec.size()<<endl;
    for(size_t i=0; i<vec.size(); i++)
        ofs<<"   "<<i+1<<"  "<<vec[i]<<endl;
    ofs<<endl;
}

void Grid::inVec(vector<float>& vec, ifstream& ifs){
    string tmp;
    size_t n, m;
    ifs>>tmp>>n;
    vec.resize(n);
    for(size_t i=0; i<n; i++)
        ifs>>m>>vec[i];
}


size_t Grid::bufferSize(){
    return 10*sizeof(int) + 14*sizeof(float) + sizeof(size_t) + 4*sizeof(int) + (zgrid.size()+dxgrid.size()+dygrid.size()+dzgrid.size()) * sizeof(float);
}
char* Grid::toBuffer(char* buf){
    int* iptr = (int*)buf;
    *iptr++ = mytype;
    *iptr++ = nx;
    *iptr++ = ny;
    *iptr++ = nz;
    *iptr++ = nnz;
    *iptr++ = nzalloc;
    *iptr++ = iz0;
    *iptr++ = nztop;
    *iptr++ = incx;
    *iptr++ = incy;
    float* fptr = (float*)iptr;
    *fptr++ = dx;
    *fptr++ = dy;
    *fptr++ = dz;
    *fptr++ = x0;           // X position of first sample
    *fptr++ = y0;
    *fptr++ = z0;           // depth of first sample. maybe negative because of boundary
    *fptr++ = centerx;           // for Pyramid use, in index
    *fptr++ = centery;
    *fptr++ = apertx;           // surface aperture of the grid
    *fptr++ = aperty;
    *fptr++ = zmin;           // depth of min z excluding boundary (c.f. z0)
    *fptr++ = zmax;
    *fptr++ = slopex;
    *fptr++ = slopey;
    size_t* sptr = (size_t*)fptr;
    *sptr++ = mysize;
    buf = (char*)sptr;
    buf = vecToBuf(zgrid, buf);
    buf = vecToBuf(dxgrid, buf);
    buf = vecToBuf(dygrid, buf);
    buf = vecToBuf(dzgrid, buf);
    return buf;
}
char* Grid::fromBuffer(char* buf){
    int* iptr = (int*)buf;
    mytype = *iptr++;
    nx = *iptr++;
    ny = *iptr++;
    nz = *iptr++;
    nnz = *iptr++;
    nzalloc = *iptr++;
    iz0 = *iptr++;
    nztop = *iptr++;
    incx = *iptr++;
    incy = *iptr++;
    float* fptr = (float*)iptr;
    dx = *fptr++;
    dy = *fptr++;
    dz = *fptr++;
    x0 = *fptr++;           // X position of first sample
    y0 = *fptr++;
    z0 = *fptr++;           // depth of first sample. maybe negative because of boundary
    centerx = *fptr++;           // for Pyramid use, in index
    centery = *fptr++;
    apertx = *fptr++;           // surface aperture of the grid
    aperty = *fptr++;
    zmin = *fptr++;           // depth of min z excluding boundary (c.f. z0)
    zmax = *fptr++;
    slopex = *fptr++;
    slopey = *fptr++;
    size_t* sptr = (size_t*)fptr;
    mysize = *sptr++;
    buf = (char*)sptr;
    buf = vecFromBuf(zgrid, buf);
    buf = vecFromBuf(dxgrid, buf);
    buf = vecFromBuf(dygrid, buf);
    buf = vecFromBuf(dzgrid, buf);
    return buf;
}
char* Grid::vecToBuf(vector<float>& vec, char* buf){
    int* iptr = (int*)buf;
    int n = vec.size();
    *iptr++ = n;
    float* fptr = (float*)iptr;
    for(int i=0; i<n; i++)
        *fptr++ = vec[i];
    buf = (char*)fptr;
    return buf;
}
char* Grid::vecFromBuf(vector<float>& vec, char* buf){
    int* iptr = (int*)buf;
    int n = *iptr++;
    vec.clear();
    if(n>0){
        vec.resize(n);
        float* fptr = (float*)iptr;
        for(int i=0; i<n; i++)
            vec[i] = *fptr++;
        buf = (char*)fptr;
    }else
        buf = (char*)iptr;
    return buf;
}
