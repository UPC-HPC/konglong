#include "Interpolant.h"

#include <omp.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "libCommon/Assertion.h"
#include "libCommon/CommonMath.h"

Interpolant::Interpolant(InType mytype, int mysize, int nThreads, bool varsize, bool detect_zero, int resamp) :
    mytype(mytype), mysize(mysize), nThreads(nThreads), varsize(varsize), detect_zero(detect_zero), resamp(resamp), tbsize(0) {
  assertion(mysize >= 1, "can not create interpolant due to small mysize %d", mysize);
  assertion(!detect_zero || varsize, "for detect_zero=true, varsize need to be true to be meaningful!");

  buildTable();
}

void Interpolant::buildTable() {
  tbsize = mysize * resamp + 2;           // half is enough due to the symmetry
  tables.resize(tbsize, 0.0);
  float dx = 1.0 / float(resamp);
  for(int i = 0; i < tbsize; i++) {
    float x = i * dx;
    switch(mytype) {
    case LINEAR:
      tables[i] = 1.0 - x;
      break;
    case LAGLANGE:
      tables[i] = lagrange(x, mysize);
      break;
    case LANCZOS:
      tables[i] = lanczos(x, mysize);
    }
  }
}

void Interpolant::printTable() {
  printf("The paramener is  %d    %d   %d\n", mysize, resamp, tbsize);
  float dx = 1.0 / float(resamp);
  for(int i = 0; i < tbsize; i += 20) {
    printf("Interpolant Table  %3d  %3d   %f   %f\n", tbsize, i, i * dx, tables[i]);
  }
}

Interpolant::~Interpolant() {
}

float Interpolant::getcoeff(float x, int n) {
  if(n == 0) n = mysize;

  float xx = ABS(x);
  if(xx > mysize) return 0.0;
  if(tables.size() > 0 && n == mysize) return valueFromTable(xx);
  switch(mytype) {
  case LINEAR:
    return 1.0 - x;
  case LAGLANGE:
    return lagrange(x, n);
  case LANCZOS:
    return lanczos(x, n);
  }
  return 0.0f;
}

float Interpolant::lagrange(float x, int n) {
  if(!x) return 1.0;
  if(int(x) == x) return 0.0;
  float upper = 1.0;
  float down = 1.0;
  for(int i = 1; i <= n; i++) {
    upper *= (x - i);
    down *= (0.0 - i);
  }
  return upper / down;
}

float Interpolant::lanczos(float x, int n) {
  return libCommon::sincf(x) * libCommon::sincf(x / n);
}

float Interpolant::valueFromTable(float xx) {
  float dd = xx * resamp;
  int id = int(dd);
  if(id >= tbsize - 1) return tables[tbsize - 1];

  float ww = dd - id;
  return (tables[id] * (1.0 - ww) + tables[id + 1] * ww);
}

void Interpolant::interpolate1D(float *input, int nin, float xoi, float dxi, float *output, int nou, float xoo, float dxo) {
  int i0 = 0, i1 = nin - 1; // i0, i1 inclusive
  if(detect_zero) i0 = firstNonZero(input, nin), i1 = lastNonZero(input, nin);

  if(i1 - i0 <= 0) {  // avoid crash later
    for(int i = 0; i < nou; i++)
      output[i] = input[0];
    return;
  }
  // now i1 > i0
  float dix = 1.0 / dxi;
  memset(output, 0, nou * sizeof(float));
  for(int i = 0; i < nou; i++) {
    float myx = xoo + i * dxo;
    float axx = (myx - xoi) * dix;
    if(axx < i0 || axx > i1) {
      output[i] = 0;
      continue;
    }

    if(!varsize) {
      int isnap = (int)floorf(axx);
      int id1 = isnap - mysize + 1;
      int id2 = isnap + mysize;
      for(int j = max(i0, id1); j <= min(id2, i1); j++) {
        float dist = axx - j;
        output[i] += input[j] * getcoeff(dist);
      }
    } else {
      int isnap = min(i1 - 1, (int)floorf(axx));
      int n_edge = min(isnap, i1 - 1 - isnap);
      int nsize = min(mysize, n_edge + 1);
      int id1 = isnap - nsize + 1;
      int id2 = isnap + nsize;
      for(int j = id1; j <= id2; j++) {
        float dist = axx - j;
        output[i] += input[j] * getcoeff(dist, nsize);
      }
    }
  }
}

void Interpolant::interpolate1D(float *input, float *output, float *weight, int nin, int nou) {
  int i0 = 0, i1 = nin - 1; // i0, i1 inclusive
  if(detect_zero) i0 = firstNonZero(input, nin), i1 = lastNonZero(input, nin);

  if(i1 - i0 <= 0) {  // avoid crash later
    for(int i = 0; i < nou; i++)
      output[i] = input[0];
    return;
  }

  // now i1 > i0
  memset(output, 0, nou * sizeof(float));
  for(int i = 0; i < nou; i++) {
    float axx = weight[i];
    if(axx < i0 || axx > i1) {
      output[i] = 0;
      continue;
    }

    if(!varsize) {
      int isnap = (int)floorf(axx);
      int id1 = isnap - mysize + 1;
      int id2 = isnap + mysize;
      for(int j = max(i0, id1); j <= min(id2, i1); j++) {
        float dist = axx - j;
        output[i] += input[j] * getcoeff(dist);
      }
    } else {
      int isnap = min(i1 - 1, (int)floorf(axx));
      int n_edge = min(isnap, i1 - 1 - isnap);
      int nsize = min(mysize, n_edge + 1);
      int id1 = isnap - nsize + 1;
      int id2 = isnap + nsize;
      for(int j = id1; j <= id2; j++) {
        float dist = axx - j;
        output[i] += input[j] * getcoeff(dist, nsize);
      }
    }
  }
}

void Interpolant::VolumeShiftHalfX(float *input, float *output, int nx, int ny, int nz) {
  memset(output, 0, sizeof(float) * nx * ny * nz);
#pragma omp parallel for num_threads(nThreads) schedule(static)
  for(int iy = 0; iy < ny; iy++) {
    for(int ix = 0; ix < nx; ix++) {
      float *duu = output + (iy * nx + ix) * nz;
      float mxx = ix + 0.5;
      int id1 = int(mxx) - mysize + 1;
      int id2 = int(mxx) + mysize;
      for(int j = max(0, id1); j <= min(id2, (nx - 1)); j++) {
        float dist = mxx - j;
        float *din = input + (iy * nx + j) * nz;
        float ww = getcoeff(dist);
        for(int k = 0; k < nz; k++)
          duu[k] += din[k] * ww;
      }
    }
  }
}

void Interpolant::VolumeShiftHalfY(float *input, float *output, int nx, int ny, int nz) {
  memset(output, 0, sizeof(float) * nx * ny * nz);
  int nxz = nx * nz;
#pragma omp parallel for num_threads(nThreads) schedule(static)
  for(int iy = 0; iy < ny; iy++) {
    float *duu = output + iy * nxz;
    float myy = iy + 0.5;
    int id1 = int(myy) - mysize + 1;
    int id2 = int(myy) + mysize;
    for(int j = max(0, id1); j <= min(id2, (ny - 1)); j++) {
      float dist = myy - j;
      float *din = input + j * nxz;
      float ww = getcoeff(dist);
      for(int k = 0; k < nxz; k++)
        duu[k] += din[k] * ww;
    }
  }
}

void Interpolant::VolumeShiftHalfZ(float *input, float *output, int nx, int ny, int nz) {
  memset(output, 0, sizeof(float) * nx * ny * nz);
  int nxz = nx * nz;

#pragma omp parallel for num_threads(nThreads) schedule(static)
  for(int iy = 0; iy < ny; iy++) {
    float *bufin = new float[nxz];
    float *bufut = new float[nxz];
    float *dii = input + iy * nxz;
    float *duu = output + iy * nxz;
    memcpy(bufin, dii, nxz * sizeof(float));
    transpose(bufin, nx, nz);
    memset(bufut, 0, nxz * sizeof(float));
    for(int iz = 0; iz < nz; iz++) {
      float *duu = bufut + iz * nx;
      float mxx = iz + 0.5;
      int id1 = int(mxx) - mysize + 1;
      int id2 = int(mxx) + mysize;
      for(int j = max(0, id1); j <= min(id2, (nz - 1)); j++) {
        float dist = mxx - j;
        float *din = bufin + iz * nx;
        float ww = getcoeff(dist);
        for(int k = 0; k < nx; k++)
          duu[k] += din[k] * ww;
      }
    }
    transpose(bufut, nz, nx);
    memcpy(duu, bufut, nxz * sizeof(float));
    delete[] bufin;
    delete[] bufut;
  }

}

void Interpolant::interpolate2D(float *input, int ninx, int niny, float xoi, float yoi, float dxi, float dyi, float *output, int noux,
    int nouy, float xoo, float yoo, float dxo, float dyo) {
  float diy = 1.0 / dyi;
  memset(output, 0, sizeof(float) * noux * nouy);

  if(niny == 1) {
    interpolate1D(input, ninx, xoi, dxi, output, noux, xoo, dxo);
    for(int i = 1; i < nouy; i++)
      memcpy(output + i * noux, output, sizeof(float) * noux);
    return;
  }

  float *dut = new float[ninx];
  for(int i = 0; i < nouy; i++) {
    memset(dut, 0, ninx * sizeof(float));
    float *duu = output + i * noux;
    float myy = yoo + i * dyo;
    float ayy = (myy - yoi) * diy;
    int id1 = int(ayy) - mysize + 1;
    int id2 = int(ayy) + mysize;
    for(int j = max(0, id1); j <= min(id2, (niny - 1)); j++) {
      float dist = ayy - j;
      float *din = input + j * ninx;
      //float* dut  = output  +   i*ninx;
      float ww = getcoeff(dist);
      for(int k = 0; k < ninx; k++)
        dut[k] += din[k] * ww;
    }
    interpolate1D(dut, ninx, xoi, dxi, duu, noux, xoo, dxo);
  }

  delete[] dut;
}

int Interpolant::firstNonZero(float *d, int n, float eps) {
  int i = 0;
  for(; i < n; i++)
    if(fabsf(d[i]) > eps) break;
  return i;
}

int Interpolant::lastNonZero(float *d, int n, float eps) {
  int i = n - 1;
  for(; i >= 0; i--)
    if(fabsf(d[i]) > eps) break;
  return i;
}

void Interpolant::transpose1(float *m, int w, int h) {

  printf("The transpose Dim  %d  %d\n", w, h);

  float *tmp = new float[w * h];

  for(int i = 0; i < w; i++) {
    for(int j = 0; j < h; j++)
      tmp[i * h + j] = m[j * w + i];
  }
  memcpy(m, tmp, w * h * sizeof(float));
  delete[] tmp;
}

void Interpolant::transpose(float *m, int w, int h) {
  int start, next, i;
  float tmp;

  for(start = 0; start <= w * h - 1; start++) {
    next = start;
    i = 0;
    do {
      i++;
      next = (next % h) * w + next / h;
    } while(next > start);
    if(next < start || i == 1) continue;

    tmp = m[next = start];
    do {
      i = (next % h) * w + next / h;
      m[next] = (i == start) ? tmp : m[i];
      next = i;
    } while(next > start);
  }
}

