/*
 * fdWrapper.cpp
 *
 *  Created on: Sep 28, 2018
 *      Author: tiger
 */

#include "fdWrapper.h"


void fd_dev1(float *din, float *dout, int n, float dx, float &time_fdt1, float &time_fdt2, int isgn) {

  fd_dev1_(din, dout, &n, &dx, &time_fdt1, &time_fdt2, &isgn);

}


void init_cfd(int  n, float  delta, float *bm1, float *bl1, float *am1, float *bm2, float *bl2, float *am2) {
  init_cfd_(&n, &delta, bm1, bl1, am1, bm2, bl2, am2);
}


void cfd_dev1(float *am, float *bm, float *bl, int  n, float *win, float *wout) {
  cfd_dev1_(am, bm, bl, &n, win, wout);
}

