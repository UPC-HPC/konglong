/*
 * ProfileWrapper.h
 *
 *  Created on: Aug 25, 2015
 *      Author: tiger
 */

#ifndef PROFILEWRAPPER_H_
#define PROFILEWRAPPER_H_

#ifdef  __cplusplus
extern "C" {
#endif

void minvel0(float *z, float *v, float *vmin, int n, float *fd, float *b, float *a, float *al, int *indx, int nsmth);
void minvel0_(float *z, float *v, float *vmin, int *n, float *fd, float *b, float *a, float *al, int *indx, int *nsmth);

#ifdef  __cplusplus
}
#endif


#endif /* PROFILEWRAPPER_H_ */

