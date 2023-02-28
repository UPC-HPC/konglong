/*
 * FreqUtil.h
 *
 *  Created on: Mar 1, 2020
 *      Author: owl
 */

#ifndef LIBWAVEPROP_FREQUTIL_H_
#define LIBWAVEPROP_FREQUTIL_H_

#include <cmath>
#include <complex>

#ifdef NO_MKL
#include <fftw3.h>
#else
#include <fftw/fftw3.h>
#endif

#include <assert.h>
#include <cstdio>
#include <string>
#include <vector>
#include <string.h>
using std::vector;
using std::complex;

#include "libCommon/CommonMath.h"
#include "libCommon/padfft.h"
using libCommon::padfft;
using libCommon::padfft_table;
using libCommon::fft_padding;
class FreqUtil {
public:
  FreqUtil(int nt, int pad = 0);
  virtual ~FreqUtil();

  void timeshift_in(float *in);
  void timeshift_out(float *out, float t_over_dt);
  void timeshift(float *in, float *out, float t_over_dt) {
    timeshift_in(in), timeshift_out(out, t_over_dt);
  }

  int nt, ntfft, ntbuf, nw;
  vector<float> buf;
  vector<complex<float> > bufw, bufw2;
private:
  fftwf_plan planf, planb;
};

#endif /* LIBWAVEPROP_FREQUTIL_H_ */

