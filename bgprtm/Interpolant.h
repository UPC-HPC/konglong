#ifndef INTERPOLANT_H
#define INTERPOLANT_H

#include "Util.h"

enum InType {
  LANCZOS = 1, LINEAR, LAGLANGE
};
// type of interpolation algorithm

class Interpolant {
public:
  InType mytype;
  int mysize;           // interplant half length,
  int nThreads;
  int resamp;           // coefficients is interpolated from pre-calculated coeffs, efficient even for single trace (high order)
  vector<float> tables;
  int tbsize;
  bool varsize, detect_zero;
  Interpolant(InType t1, int size0, int nThreads, bool varsize = true, bool detect_zero = false, int resampl = 1024);
  float getcoeff(float x, int n = 0);
  void buildTable();
  void printTable();
  void interpolate1D(float *input, float *output, float *weight, int nin, int nou);
  void interpolate1D(float *input, int nin, float xoi, float dxi, float *output, int nou, float xoo, float dxo);
  void interpolate2D(float *input, int ninx, int niny, float xoi, float yoi, float dxi, float dyi, float *output,
                     int noux, int nouy, float xoo, float yoo, float dxo, float dyo);
  void VolumeShiftHalfX(float *input, float *output, int nx, int ny, int nz);
  void VolumeShiftHalfY(float *input, float *output, int nx, int ny, int nz);
  void VolumeShiftHalfZ(float *input, float *output, int nx, int ny, int nz);

  static int firstNonZero(float *d, int n, float eps = 0);
  static int lastNonZero(float *d, int n, float eps = 0);
  ~Interpolant();

private:
  float valueFromTable(float xx);
  float lagrange(float x, int n);
  float lanczos(float x, int n);
  void transpose(float *m, int w, int h);
  void transpose1(float *m, int w, int h);
};

#endif

