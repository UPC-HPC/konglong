/*
 * Derivative_fftv.cpp
 *
 */

#include "Derivative_fftv.h"

Derivative_fftv::Derivative_fftv(Grid *inGrid,  Boundary *inBnd, int modelType, int inNThreads)
  : Derivative(inGrid, inBnd, modelType, inNThreads) {
  fftv = new libfftv::FFTVFilter(nz, nx, ny, dz, dx, dy, nThreads, 1);
}

Derivative_fftv::~Derivative_fftv() {
  if(fftv) {
    delete fftv;
    fftv = NULL;
  }
}

// to be overloaded
void Derivative_fftv::dx1(float *in, float *out, int isign) {
  fftv->SetFilterType(libfftv::DERIVATIVE);  // First order of derivative
  fftv->run(in, out, NULL, 2);
}

void Derivative_fftv::dy1(float *in, float *out, int isign) {
  fftv->SetFilterType(libfftv::DERIVATIVE);  // First order of derivative
  fftv->run(in, out, NULL, 3);
}
void Derivative_fftv::dz1(float *in, float *out, int isign) {
  fftv->SetFilterType(libfftv::DERIVATIVE);  // First order of derivative
  fftv->run(in, out, NULL, 1);
}
void Derivative_fftv::dx2(float *in, float *out) {
  fftv->SetFilterType(libfftv::DERIVATIVE2);
  fftv->run(in, out, NULL, 2);
}
void Derivative_fftv::dy2(float *in, float *out) {
  fftv->SetFilterType(libfftv::DERIVATIVE2);  // First order of derivative
  fftv->run(in, out, NULL, 3);
}
void Derivative_fftv::dz2(float *in, float *out) {
  fftv->SetFilterType(libfftv::DERIVATIVE2);  // First order of derivative
  fftv->run(in, out, NULL, 1);
}

