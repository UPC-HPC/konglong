
#ifndef WAVELET_H
#define WAVELET_H

enum {
  RICKER = 1, SPIKY, ORMSBY, DELTA, USERWAVELET
};

enum {
  ZERO = 0, MINIMUM = 1
};

class Wavelet {
public:
  Wavelet(int waveletType, float fmax, int nt, float dt, float extra_srt_over_it0 = 0.0f, float t_delay = 0.0f, int sinc_avg = 0,
          int phase_type = 1);

  Wavelet(int waveletType, float f1, float f2, float f3, float fmax, int nt0, float dt, float extra_srt_over_it0, float t_delay, int sinc_avg,
          int phase_type = 1);

  ~Wavelet();

  int waveletType;
  int phase_type = 1; // default as original
  float fmax = -1;
  float f1 = -1, f2 = -1, f3 = -1, f4 = -1;
  int nt;
  int m;
  float dt;
  int sinc_avg;
  float extra_srt_over_it0, t_delay;
  int it0;          // the index of zero time
  float t0;
  float *mysource;

private:
  void init();
  void scan();

  void ricker();

  void spiky();

  void ormsby();

  void delta();

  void userwavelet();

};

#endif

