#ifndef FDENGINE_R2C_H_
#define FDENGINE_R2C_H_

#include "FdEngine.h"

class FdEngine_r2c : public FdEngine {
public:
  /*
   * ctor
   */
  FdEngine_r2c(int innx, int inny, int innz, float indx, float indy, float indz, int inNThreads); // removed  int inRhoCN*, by wolf

  /*
   * dtor
   */
  virtual ~FdEngine_r2c();

  void dx1_2D(float *in, float *out, float *wrk, int iy, int iacc, int isign);
  void dy1_2D(float *in, float *out, float *wrk, int ix, int iacc, int isign);
  void dz1_2D(float *in, float *out, float *wrk, int iy, int iacc, int isign);
  void dx1_2D(float *in, float *out, float *wrk, int iy, int iacc, int isign, float *velSlice, float *rhoSlice) override;
  void dy1_2D(float *in, float *out, float *wrk, int ix, int iacc, int isign, float *velSlice, float *rhoSlice) override;
  void dz1_2D(float *in, float *out, float *wrk, int iy, int iacc, int isign, float *velSlice, float *rhoSlice) override;


private:

  float *assignW(int n, float dw);

  void fftderiv1_r2c(fftwf_plan &planf, fftwf_plan &planb, PML *pml1, PML *pml2,
                     float *kx, float *wrk1, float *wrk2, float *q1, float *q2,
                     int nz, int nx, int nbnd1, int nbnd2, int isign);
private:
  float dkx;
  float dky;
  float dkz;

  float *kx;
  float *ky;
  float *kz;

  fftwf_plan planxf;
  fftwf_plan planxb;
  fftwf_plan planyf;
  fftwf_plan planyb;
  fftwf_plan planzf;
  fftwf_plan planzb;

};


#endif /* FDENGINE_H_ */
                                   
