/*
 * WaveletWrapper.cpp
 */

#include <stdio.h>
#include <stdlib.h>

#include "WaveletWrapper.h"

int ricker_design_n(float maxfreq, float fpeak, float dt) {

  int n = 0;
  ricker_design_(&maxfreq, &fpeak, &dt, &n);
  return n;
}

void ricker_assign_n(float fpeak, float *wavelet, float dt, int n) {
  ricker_assign_(&fpeak, wavelet, &dt, &n);
}

int spiky_design_n(float fmax, float fhigh, float frdb, float dt) {
  int n = 0;
  spiky_design_(&fmax, &fhigh, &frdb, &dt, &n);
  return n;
}

void spiky_assign_n(float fmax, float fhigh, float frdb, float *wavelet, float dt, int n, float *wrk) {
  spiky_assign_(&fmax, &fhigh, &frdb, wavelet, &dt, &n, wrk);
}

int ormsby_design_n(float f1, float f2, float f3, float f4, float frdb, float dt) {
  int n = 0;
  ormsby_design_(&f1, &f2, &f3, &f4, &frdb, &dt, &n);
  return n;
}

void ormsby_assign_n(float f1, float f2, float f3, float f4, float frdb, float *wavelet, float dt, int n, float *wrk1,
                     float *wrk2) {
  ormsby_assign_(&f1, &f2, &f3, &f4, &frdb, wavelet, &dt, &n, wrk1, wrk2);
}

