/*
 * Derivative_cfd.h
 *
 */

#ifndef DERIVATIVE_CFD_H_
#define DERIVATIVE_CFD_H_

#include "Derivative.h"
#include "Cfd_plan.h"


class Derivative_cfd : public Derivative {
public:

  Derivative_cfd(Grid *inGrid,  Boundary *inBnd, int modelType, int inNThreads);

  virtual ~Derivative_cfd();

  void dx1(float *in, float *out, int isign);
  void dy1(float *in, float *out, int isign);
  void dz1(float *in, float *out, int isign);

  void dx2(float *in, float *out) {this->dx1(in, mywk, 1); this->dx1(mywk, out, 2);}
  void dy2(float *in, float *out) {this->dy1(in, mywk, 1); this->dy1(mywk, out, 2);}
  void dz2(float *in, float *out) {this->dz1(in, mywk, 1); this->dz1(mywk, out, 2);}

  static float getDispersion() {return 0.9f;}

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
  float *mywk;
};

#endif /* DERIVATIVE_H_ */

