/*
 * ProfileWrapper.cpp
 *
 *  Created on: Aug 25, 2015
 *      Author: tiger
 */

#include "ProfileWrapper.h"

void minvel0(float *z, float *v, float *vmin, int n, float *fd, float *b, float *a, float *al, int *indx, int nsmth) {
  minvel0_(z, v, vmin, &n, fd, b, a, al, indx, &nsmth);
}

