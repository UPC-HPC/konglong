#ifndef COMMON_UTIL_H
#define COMMON_UTIL_H_

#include "gpurtm.h"

void set_data_3d(float *data,
                 const int nx, const int ny, const int nz,
                 const int pattern_type);

int compare_data_3d(float *h_data, float*d_data,
                 const int nx, const int ny, const int nz,
                 const float threshold=1.E-6, const int nerr_threshold=3);
#endif
