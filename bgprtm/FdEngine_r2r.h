#ifndef FDENGINE_R2R_H_
#define FDENGINE_R2R_H_

#include "FdEngine.h"

class FdEngine_r2r: public FdEngine {
public:
  /*
   * ctor
   */
  FdEngine_r2r(int nx, int ny, int nz, float dx, float dy, float dz, int nThreads);//removed int RhoCN*, by wolf

  /*
   * dtor
   */
  virtual ~FdEngine_r2r();

  void dx1_2D(float *in, float *out, float *wrk, int iy, int iacc, int isign) override;
  void dy1_2D(float *in, float *out, float *wrk, int ix, int iacc, int isign) override;
  void dz1_2D(float *in, float *out, float *wrk, int iy, int iacc, int isign) override;
  void dx1_2D(float *in, float *out, float *wrk, int iy, int iacc, int isign, float *velSlice, float *rhoSlice) override;
  void dy1_2D(float *in, float *out, float *wrk, int ix, int iacc, int isign, float *velSlice, float *rhoSlice) override;
  void dz1_2D(float *in, float *out, float *wrk, int iy, int iacc, int isign, float *velSlice, float *rhoSlice) override;


private:
  float *assignW(int n, float dw);

  void fftderiv1_r2r(fftwf_plan &planf, fftwf_plan &planb, PML *pml1, PML *pml2,
                     float *kx, float *wrk1, float *wrk2, float *q1, float *q2,
                     int nz, int nx, int nbnd1, int nbnd2, int isign);
private:
  fftwf_plan planxf1;
  fftwf_plan planxf2;
  fftwf_plan planxb1;
  fftwf_plan planxb2;
  fftwf_plan planyf1;
  fftwf_plan planyf2;
  fftwf_plan planyb1;
  fftwf_plan planyb2;
  fftwf_plan planzf1;
  fftwf_plan planzf2;
  fftwf_plan planzb1;
  fftwf_plan planzb2;

  float dkx;
  float dky;
  float dkz;

  float *kx;
  float *ky;
  float *kz;
};


#endif /* FDENGINE_H_ */

