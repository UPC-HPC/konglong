
/*
 * WaveletWrapper.h
 *
 *  Created on: Aug 25, 2015
 *      Author: tiger
 */

#ifndef WAVELETWRAPPER_H_
#define WAVELETWRAPPER_H_

#ifdef  __cplusplus
extern "C" {
#endif

void ricker_design_(float *maxfreq, float *fpeak, float *dt, int *n);
void ricker_assign_(float *fpeak, float *wavelet, float *dt, int *n);

void spiky_design_(float *fmax, float *fhigh, float *frdb, float *dt, int *n);
void spiky_assign_(float *fmax, float *fhigh, float *frdb, float *wavelet, float *dt, int *n, float *wrk);

void ormsby_design_(float *f1, float *f2, float *f3, float *f4, float *frdb, float *dt, int *n);
void ormsby_assign_(float *f1, float *f2, float *f3, float *f4, float *frdb, float *wavelet, float *dt, int *n, float *wrk1, float *wrk2);


int  ricker_design_n(float maxfreq, float fpeak, float dt);
void ricker_assign_n(float fpeak, float *wavelet, float dt, int n);

int  spiky_design_n(float fmax, float fhigh, float frdb, float dt);
void spiky_assign_n(float fmax, float fhigh, float frdb, float *wavelet, float dt, int n, float *wrk);

int  ormsby_design_n(float f1, float f2, float f3, float f4, float frdb, float dt);
void ormsby_assign_n(float f1, float f2, float f3, float f4, float frdb, float *wavelet, float dt, int n, float *wrk1, float *wrk2);

#ifdef  __cplusplus
}
#endif


#endif /* WAVELETWRAPPER_H_ */

