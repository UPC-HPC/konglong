#ifndef CFD_PLAN_H_
#define CFD_PLAN_H_

#include <iostream>
#include <math.h>


struct CfdPlan {
public:
  CfdPlan(int inN, float dd);
  ~CfdPlan();

private:
  void init_cfd(int nz, float dz, float *bmz1, float *blz1, float *amz1, float *bmz2, float *blz2, float *amz2);
  void reducematrix(float *a, float *b, int n1, int n2a, int n2b);
  void cfd_transpose(float *a, int n1, int n2);
  void bandec(float *a, int n, int m1, float *al);
  void init_cfddev1(float *b, float *bl, float *a, int n, float dx, int isgn);

public:
  int n;
  float *bm1, *bm2;
  float *bl1, *bl2;
  float *am1, *am2;
};

#endif /* CFD_PLAN_H_ */

