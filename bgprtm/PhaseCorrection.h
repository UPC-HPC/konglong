/*
 * PhaseCorrection.h
 */

#ifndef PHASECORRECTION_H_
#define PHASECORRECTION_H_

#include <complex>
typedef std::complex<float> Complex;

class PhaseCorrection {
public:
  enum DIRECTION {
    NONE = 0, FWD = 1, BWD = 2
  };
public:
  PhaseCorrection(int nt, float dt, float Dt, float t0, float fmax, DIRECTION dir, float w_pow = 0, float phase_deg = 0,
                  int do_tdsp = 1, int ntOut = -1, float dtOut = -1.0f, float t0Out = 0);
  ~PhaseCorrection();

  void applyForward(float *trace, int ntr);
  void applyBackward(float *trace, int ntr, float *trcOut = NULL, float scaler = 1);
  void fft_dft(float *trace, int ntr, float *trcOut = NULL, float tshift = 0);
  void dft_fft(float *trace, int ntr);

private:
  //
  void init();

private:
  DIRECTION direction;
  int tdsp_correction;
  bool do_fft_dft = true;
  int skip_tdsp_correction; // if set to 3 (bit 1 + 2), then skip both FWD and BWD
  int resample;
  int nt, ntOut, nt_dft;
  int ntfft;
  float dt, Dt, dtOut, dt_dft, t0, t0Out, t_shift; // Dt: propagation dt, required to make correction, dtOut: dtOut for resample
  int it_shift; // the int part
  float t_shift_frac;
  int nw1, nw;
  float fmax;
  float w_pow;
  float phase;
  Complex c_phase;
  vector<Complex> table;
};

#endif /* PHASECORRECTION_H_ */

