/*
 * Derivative_fftv.h
 *
 */

#ifndef DERIVATIVE_FFTV_H_
#define DERIVATIVE_FFTV_H_

#include "libFFTV/fftvfilter.h"
#include "Derivative.h"

class Derivative_fftv : public Derivative {
public:
  /*
   * construct
   */
  Derivative_fftv(Grid *grid,  Boundary *bnd, int modelType, int nThreads);

  /*
   * Destruct
   */
  virtual ~Derivative_fftv();

  // to be overloaded
  virtual void dx1(float *in, float *out, int isign);  // isign : 1: first time to derivative 2: second time to derivative
  virtual void dy1(float *in, float *out, int isign);
  virtual void dz1(float *in, float *out, int isign);
  virtual void dx2(float *in, float *out);
  virtual void dy2(float *in, float *out);
  virtual void dz2(float *in, float *out);
public:
  libfftv::FFTVFilter *fftv;
};


#endif /* DERIVATIVE_FFTV_H_ */

