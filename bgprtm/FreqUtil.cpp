/*
 * FreqUtil.cpp
 *
 *  Created on: Mar 1, 2020
 *      Author: owl
 */

#include "FreqUtil.h"

FreqUtil::FreqUtil(int nt, int pad) : nt(nt) {
  ntfft = padfft(nt + pad);
  ntbuf = r2c_size(ntfft);
  nw = ntbuf / 2;
  buf.resize(ntfft), bufw.resize(nw), bufw2.resize(nw);
  planf = fftwf_plan_dft_r2c_1d(ntfft, (float*)&buf[0], (fftwf_complex*)&bufw[0], FFTW_ESTIMATE);
  planb = fftwf_plan_dft_c2r_1d(ntfft, (fftwf_complex*)&bufw[0], (float*)&buf[0], FFTW_ESTIMATE);
}

FreqUtil::~FreqUtil() {
  fftwf_destroy_plan(planf);
  fftwf_destroy_plan(planb);
}

void FreqUtil::timeshift_in(float *in) {
  memcpy(&buf[0], in, sizeof(float) * nt);
  fft_padding(&buf[0], nt, ntfft, 0);
  fftwf_execute_dft_r2c(planf, &buf[0], (fftwf_complex*)&bufw[0]);
}
void FreqUtil::timeshift_out(float *out, float t_over_dt) {
  float scaler = 1.0f / ntfft;
  float dwdt = 2.0f * M_PI / ntfft;
  t_over_dt *= dwdt;
  for(int iw = 0; iw < nw; iw++)
    bufw2[iw] = bufw[iw] * exp(complex<float>(0, -iw * t_over_dt));

  fftwf_execute_dft_c2r(planb, (fftwf_complex*)&bufw2[0], &buf[0]);
  for(int i = 0; i < nt; i++)
    out[i] = buf[i] * scaler;
}

