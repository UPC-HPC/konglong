#ifndef FDENGINE_HFD_H_
#define FDENGINE_HFD_H_

#include "FdEngine.h"

class FdEngine_hfd : public FdEngine {
public:
  /*
   * ctor
   */
  FdEngine_hfd(int nx, int ny, int nz, float dx, float dy, float dz, int nThreads); // removed int RhoCN* by wolf,

  /*
   * dtor
   */
  virtual ~FdEngine_hfd();

  void dx1_2D(float *in, float *out, float *wrk, int iy, int iacc, int isign);
  void dy1_2D(float *in, float *out, float *wrk, int ix, int iacc, int isign);
  void dz1_2D(float *in, float *out, float *wrk, int iy, int iacc, int isign);

private:

  void deriv1_hfd(PML *pml1, PML *pml2, float *wrk1, float *wrk2, float *q1, float *q2,
                  int nz, int nx, int nbnd1, int nbnd2, int isign);
};


#endif /* FDENGINE_H_ */

