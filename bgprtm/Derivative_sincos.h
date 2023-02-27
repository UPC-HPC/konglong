
#ifndef DERIVATIVE_SINCOS_H_
#define DERIVATIVE_SINCOS_H_

#include "Derivative.h"
#include "derivative.h"

class Derivative_sincos : public Derivative {
public:
  /*
   * construct
   */
  Derivative_sincos(Grid *grid,  Boundary *bnd, int modelType, int nThreads);

  /*
   * Destruct
   */
  virtual ~Derivative_sincos() {};

  // to be overloaded
  virtual void dx1(float *in, float *out, int isign);  // isign : 1: first time to derivative 2: second time to derivative
  virtual void dy1(float *in, float *out, int isign);
  virtual void dz1(float *in, float *out, int isign);
  virtual void dx2(float *in, float *out) {printf("Error: Derivative_sincos::dx2 not implement yet!\n"); exit(1);}
  virtual void dy2(float *in, float *out) {printf("Error: Derivative_sincos::dy2 not implement yet!\n"); exit(1);}
  virtual void dz2(float *in, float *out) {printf("Error: Derivative_sincos::dz2 not implement yet!\n"); exit(1);}

private:
  libfftavx::Derivative1 myDerivate_sincos;
};


#endif /* DERIVATIVE_FFTV_H_ */

