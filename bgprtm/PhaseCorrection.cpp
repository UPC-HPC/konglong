/*
 * PhaseCorrection.cpp
 *
 */
#include "GetPar.h"
#include "PhaseCorrection.h"
#include "libCommon/CommonMath.h"
#include "libCommon/padfft.h"
#include "libFFTV/fft1dfunc.h"

#include "MpiPrint.h"
using MpiPrint::print1m;

PhaseCorrection::PhaseCorrection(int nt, float dt, float Dt, float t0, float fmax, DIRECTION dir, float w_pow, float phase_deg, int do_tdsp,
    int ntO, float dtO, float t0O) :
    nt(nt), dt(dt), Dt(Dt), t0(t0), fmax(fmax), direction(dir), w_pow(w_pow), phase(phase_deg * FLT_PI / 180), ntOut(ntO), dtOut(dtO), t0Out(
        t0O) {

  c_phase = std::exp(Complex(0.0f, -phase)); // exp(-iwt)

  float ratio = global_pars["tpad_ratio"].as<float>(0.5f);

  ntfft = libfftv::padfftv(nearbyintf(nt * (1 + ratio)));
  nw = ntfft / 2 + 1;

  skip_tdsp_correction = global_pars["skip_tdsp_correction"].as<int>(0) & dir;
  tdsp_correction = global_pars["tdsp_correction"].as<int>(2);
  if(!do_tdsp) tdsp_correction = 0;  // in case of src with different options ...

  resample = (dtOut > 0 && ntOut > 0);

  if(resample) nt_dft = ntOut, dt_dft = dtOut, t_shift = t0Out - t0;
  else nt_dft = nt, dt_dft = dt, t_shift = 0, t0Out = t0 + t_shift; // for t_shift != 0, need to give identical dtOut as dt, then can specify t0Out
ut

  it_shift = int(t_shift / dt_dft);
  t_shift_frac = t_shift - it_shift * dt_dft;

  init();
}

PhaseCorrection::~PhaseCorrection() {
}

static inline double roundDownAngle(double angle) { // @suppress("Unused static function")
  int n = (int)nearbyint(angle / (2 * M_PI));
  return angle - n * 2 * M_PI;
}

static inline float roundDownAngle(float angle) {
  int n = (int)nearbyint(angle / (2 * FLT_PI));
  return angle - n * 2 * FLT_PI;
}

// NOTE from Neptune:  Fwd: sin w/ Jacobian cos(x) : dft->fft,  Bwd: sin: fft->dft

// arccos(1-x*x/2) = x + x^3/24 + ...
static inline double dsp_acos(double x) {
  return acos(1 - 0.5 * (double)x * x) - x;
}

//sin(x) = x-x^3/6 +...
static inline double dsp_sin(double x) {
  return x - 2. * sin(x / 2.0);
}
void PhaseCorrection::init() {
  float tdsp_acos_ratio = 0.0f; // direction == FWD ? 0.0f : -1.0f; // FWD: RTM, BWD: Modeling
  tdsp_acos_ratio = global_pars["tdsp_acos_ratio"].as<float>(tdsp_acos_ratio);
  if(getBool("tdsp_force_sin", false)) tdsp_acos_ratio = 0;
  else if(getBool("tdsp_force_acos", false)) tdsp_acos_ratio = 1.0f;

  do_fft_dft = direction == BWD;
  bool do_sin_jacobian = (direction == FWD) && !skip_tdsp_correction && !getBool("tdsp_skip_jacobian", false);
  if(getBool("tdsp_force_fftdft", false)) do_fft_dft = true;
  else if(getBool("tdsp_force_dftfft", false)) do_fft_dft = false;

  float it0 = (do_fft_dft ? t0Out : t0) / dt_dft;

  double dw = 2.0 * M_PI / (ntfft * dt);
  double dwdt = dw * dt_dft;
  nw1 = (int)nearbyintf(2.0f * FLT_PI * fmax / dw) + 1;
  print1m(
      "PhaseCorrection: nt=%d, ntfft=%d, nw1=%d, nw=%d, fmax=%f, dt=%f, dt_dft=%f, t0=%f, Dt=%f, tdsp_correction=%d, w_pow=%f, phase_deg=%f \n",
      nt, ntfft, nw1, nw, fmax, dt, dt_dft, t0, Dt, tdsp_correction, w_pow, phase * 180 / FLT_PI);
  nw1 = min(nw, nw1);

  if(tdsp_correction) table.resize((size_t)nw1 * nt_dft);
  Complex *ctable = &table[0];

  for(int iw = 0; iw < nw1; iw++) {
    double x = (double)iw * dw * Dt;

    // things in the it-loop better to be float for efficiency ...
    float factor = do_sin_jacobian ? 2.0f * cosf(x / 2) : 2.0f; // Jacobian
    float dsp = (Dt == 0 || skip_tdsp_correction) ? 0 : (tdsp_acos_ratio * dsp_acos(x) + (1 - tdsp_acos_ratio) * dsp_sin(x)) * dt_dft / Dt;
    // double phs = (it - it0) * (dsp(x) - 1) * iw * dw * dt_dft - (t0 - t_shift) * iw * dw; // for comparison with below
    float dsp = (Dt == 0 || skip_tdsp_correction) ? 0 : (tdsp_acos_ratio * dsp_acos(x) + (1 - tdsp_acos_ratio) * dsp_sin(x)) * dt_dft / Dt;
    // double phs = (it - it0) * (dsp(x) - 1) * iw * dw * dt_dft - (t0 - t_shift) * iw * dw; // for comparison with below
    if(dt == dt_dft) {
      for(int it = 0; it < nt_dft; it++) {
        float phs = (it - it0) * dsp - ((long(it - it_shift) * iw) % ntfft) * dw * dt_dft + t_shift_frac * iw * dw; // only for dt == dt_dft
        ctable[(size_t)iw * nt_dft + it] = std::exp(Complex(0.0f, phs)) * factor;
      }
    } else {
      for(int it = 0; it < nt_dft; it++) {
        float phs = (it - it0) * dsp - (float)dw * dt_dft * it * iw + (float)dw * t_shift * iw;
        phs = roundDownAngle(phs);
        ctable[(size_t)iw * nt_dft + it] = std::exp(Complex(0.0f, phs)) * factor;
      }
    }
  }

}

void PhaseCorrection::applyForward(float *trace, int ntr) {
  assertion(direction == FWD, "");
  if(!tdsp_correction && w_pow == 0 && phase == 0) return;

  if(!do_fft_dft) dft_fft(trace, ntr);  // Fwd: sin w/ jacobian, dft_fft
  else {
    if(tdsp_correction == 1) fft_dft(trace, ntr, trace);
    else dft_fft(trace, ntr);
  }
}

void PhaseCorrection::applyBackward(float *trace, int ntr, float *trcOut, float scaler) {
  if(trcOut == NULL) trcOut = trace;
  if(!tdsp_correction && w_pow == 0 && phase == 0) {
    assertion(trcOut == trace, "NO-OP requires trcOut==trace");
    return;
  }

  if(do_fft_dft) fft_dft(trace, ntr, trcOut);  // Bwd: sin, fft_dft
  else {
    if(tdsp_correction == 1) dft_fft(trace, ntr);
    else fft_dft(trace, ntr, trcOut);
  }
  if(scaler != 1.0f) for(int it = 0; it < ntOut; it++)
    trcOut[it] *= scaler;
}

void PhaseCorrection::fft_dft(float *trace, int ntr, float *trcOut, float tshift) {
  if(trcOut == NULL) trcOut = trace;

  float *fftbuf = new float[ntfft * 2]; //TODO: move this to init, but need to alloc enough for multi-threading
  Complex *fftbuc = (Complex*)fftbuf;
  Complex *ctable = &table[0];

  float dw = 2.0f * M_PI / (ntfft * dt); // fake scaler for w
  float scaler = 1.0f / ntfft;

  for(int ir = 0; ir < ntr; ir++) {
    float *ptrace = trace + (size_t)ir * nt;
    float *ptrcOut = trcOut + (size_t)ir * nt_dft;
    {
      // fft-fwd
      memset(fftbuf, 0, sizeof(float) * ntfft * 2);
      memcpy(fftbuf, ptrace, sizeof(float) * nt);
      libfftv::fft_r2c_fd(fftbuc, ntfft);
    }

    fftbuc[0] = fftbuc[0].real(); // clear the nyquist part for simplicity
    for(int iw = 0; iw < nw1; iw++) {
      float factor = (w_pow == 0) ? 1 : (iw == 0) ? 0 : powf(iw * dw, w_pow);
      fftbuc[iw] *= c_phase * factor * scaler;
      if(tshift != 0) fftbuc[iw] *= std::exp(Complex(0.0f, iw * dw * tshift));
    }
    fftbuc[0] = fftbuc[0].real(); // clear the nyquist part for simplicity

    if(tdsp_correction) {
      memset(ptrcOut, 0, sizeof(float) * nt_dft);

      fftbuf[0] *= 0.5f;
      fftbuf[1] = 0.0;
      for(int iw = 0; iw < nw1; iw++) {
        Complex *ptable = ctable + (size_t)iw * nt_dft;
        for(int it = 0; it < nt_dft; it++) {
          ptrcOut[it] += fftbuc[iw].real() * ptable[it].real() - fftbuc[iw].imag() * ptable[it].imag();
        }
      }
    } else {
      libfftv::fft_c2r_bd(fftbuc, ntfft);
      memcpy(ptrcOut, fftbuf, sizeof(float) * min(ntfft, nt_dft));
    }
  }

  delete[] fftbuf;
}

void PhaseCorrection::dft_fft(float *trace, int ntr) {
  assertion(!resample, "Resample option cannot call dft_fft()");

  float *fftbuf = new float[ntfft * 2]; //TODO: move this to init, but need to alloc enough for multi-threading
  Complex *fftbuc = (Complex*)fftbuf;

  Complex *ctable = &table[0];

  float dw = 2.0f * M_PI / (ntfft * dt); // fake scaler for w
  float scaler = 1.0f / ntfft;

  float *ptrace = trace;
  for(int itr = 0; itr < ntr; itr++) {
    if(tdsp_correction) {
      memset((void*)fftbuc, 0, sizeof(Complex) * ntfft);

      for(int iw = 0; iw < nw1; iw++) {
        Complex *ptable = ctable + (size_t)iw * nt_dft;
        for(int it = 0; it < nt; it++)
          fftbuc[iw] += ptrace[it] * conj(ptable[it]);
      }
    } else {
      memset(fftbuf, 0, sizeof(float) * ntfft * 2);
      memcpy(fftbuf, ptrace, sizeof(float) * nt);
      libfftv::fft_r2c_fd(fftbuc, ntfft);
    }

    fftbuc[0] = fftbuc[0].real(); // clear the nyquist part for simplicity
    for(int iw = 0; iw < nw1; iw++) {
      float factor = (w_pow == 0) ? 1 : (iw == 0) ? 0 : powf(iw * dw, w_pow);
      fftbuc[iw] *= c_phase * factor * scaler;
    }
    fftbuc[0] = fftbuc[0].real(); // clear the nyquist part for simplicity

    {
      // fft-bwd
      libfftv::fft_c2r_bd(fftbuc, ntfft);
      memcpy(ptrace, fftbuf, sizeof(float) * nt);
    }

    ptrace += nt;
  }

  delete[] fftbuf;
}

