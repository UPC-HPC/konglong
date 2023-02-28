
/*
 * fdWrapper.h
 *
 *  Created on: Sep 28, 2018
 *      Author: tiger
 */

#ifndef FDWRAPPER_H_
#define FDWRAPPER_H_

#ifdef  __cplusplus
extern "C" {
#endif

void fd_dev1(float *din, float *dout, int  n, float  dx, float &time_fdt1, float &time_fdt2, int  isgn);
void fd_dev1_(float *din, float *dout, int *n, float *dx, float *time_fdt1, float *time_fdt2, int *isgn);

void init_cfd(int  n, float  delta, float *bm1, float *bl1, float *am1, float *bm2, float *bl2, float *am2);
void init_cfd_(int *n, float *delta, float *bm1, float *bl1, float *am1, float *bm2, float *bl2, float *am2);

void cfd_dev1(float *am, float *bm, float *bl, int  n, float *win, float *wout);
void cfd_dev1_(float *am, float *bm, float *bl, int *n, float *win, float *wout);

#ifdef  __cplusplus
}
#endif

#endif /* LIBWAVEPROP_FDWRAPPER_H_ */

