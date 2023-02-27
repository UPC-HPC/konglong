/*
 * Derivative_sincos.cpp
 *
 */

#include "Derivative_sincos.h"

Derivative_sincos::Derivative_sincos(Grid *inGrid,  Boundary *inBnd, int modelType, int inNThreads)
  : Derivative(inGrid, inBnd, modelType, inNThreads), myDerivate_sincos(nz, nx, ny, dz, dx, dy, nThreads) {};

// to be overloaded
void Derivative_sincos::dx1(float *in, float *out, int isign) {
  if(isign == 1) { // first time
    myDerivate_sincos.runCosSin(in, out, 2);
  }
  //myDerivate_sincos.runSinCos(in, out, 2);
  else
    myDerivate_sincos.runSinCos(in, out, 2);
  //myDerivate_sincos.runCosSin(in, out, 2);
}

void Derivative_sincos::dy1(float *in, float *out, int isign) {
  if(isign == 1)// first time
    myDerivate_sincos.runCosSin(in, out, 3);
  //myDerivate_sincos.runSinCos(in, out, 3);
  else
    myDerivate_sincos.runSinCos(in, out, 3);
  //myDerivate_sincos.runCosSin(in, out, 3);
}
void Derivative_sincos::dz1(float *in, float *out, int isign) {
  if(isign == 1)// first time
    myDerivate_sincos.runCosSin(in, out, 1);
  //myDerivate_sincos.runSinCos(in, out, 1);
  else
    myDerivate_sincos.runSinCos(in, out, 1);
  //myDerivate_sincos.runCosSin(in, out, 1);
}

