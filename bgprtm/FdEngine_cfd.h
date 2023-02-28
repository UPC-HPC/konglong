#ifndef FDENGINE_CFD_H_
#define FDENGINE_CFD_H_

#include "FdEngine.h"
#include "Cfd_plan.h"
class FdEngine_cfd : public FdEngine {
public:
  /*
   * ctor
   */
  FdEngine_cfd(int nx, int ny, int nz, float dx, float dy, float dz, int nThreads); // removed int RhoCN* by wolf,

  /*
   * dtor
   */
  virtual ~FdEngine_cfd();

  void dx1_2D(float *in, float *out, float *wrk, int iy, int iacc, int isign) override;
  void dy1_2D(float *in, float *out, float *wrk, int ix, int iacc, int isign) override;
  void dz1_2D(float *in, float *out, float *wrk, int iy, int iacc, int isign) override;
  void dx1_2D(float *in, float *out, float *wrk, int iy, int iacc, int isign, float *velSlice, float *rhoSlice) override;
  void dy1_2D(float *in, float *out, float *wrk, int ix, int iacc, int isign, float *velSlice, float *rhoSlice) override;
  void dz1_2D(float *in, float *out, float *wrk, int iy, int iacc, int isign, float *velSlice, float *rhoSlice) override;

  static float getDispersion() {return 0.9f;}
//private:
//
//  void deriv1_cfd(float *am, float *bm, float *bl, PML *pml1, PML *pml2, float *wrk1, float *wrk2, float *q1, float *q2,
//                  int nz, int nx, int nbnd1, int nbnd2, int isign);
////
//
//private:
//  float *bmz1, *blz1, *amz1, *bmz2, *blz2, *amz2;
//  float *bmx1, *blx1, *amx1, *bmx2, *blx2, *amx2;
//  float *bmy1, *bly1, *amy1, *bmy2, *bly2, *amy2;

private:

  void cfd_dev_z_c(float *am, float *bm, float *bl, int  n, float *win, float *wout);
  void banmul_dev_z_c(float *a, int n, float *x, float *b);
  void banbks_dev_z_c(float *a, int n, float *al, float *b);
  void cfd_dev_z_batch(float *am, float *bm, float *bl, int  n, int nbatch, float *win, float *wout);
  void banmul_dev_z_batch(float *am, int  n, int nbatch, float *win, float *wout);
  void banbks_dev_z_batch(float *a, int n, int nbatch, float *al, float *b);
  void banbks_dev_z_c_batch(float *a, int n, float *al, float *b);

  void cfd_dev_xy_batch(float *am, float *bm, float *bl, int  n, int nbatch, int ldim, float *win, float *wout);
  void banmul_dev_xy_batch(float *a, int n, int nbatch, int ldim, float *x, float *b);
  void banbks_dev_xy_batch(float *a, int n, int nbatch, int ldim, float *al, float *b);

private:
  CfdPlan myCfd_x;
  CfdPlan myCfd_y;
  CfdPlan myCfd_z;
};


#endif /* FDENGINE_H_ */

