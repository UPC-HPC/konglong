/*
 * ImagingCorrelator.h
 *
 */

#ifndef IMAGINGCORRELATOR_H_
#define IMAGINGCORRELATOR_H_

#ifdef NO_MKL
#include <fftw3.h>
#else
#include <fftw/fftw3.h>
#endif
#include "Util.h"
#include "libFFTV/fftvfilter.h"

enum IMAGE_OPTION {
  IMAGE1 = 0, INVERSION = 1, INVERSION2 = 2, ALLIN3 = 3
};

class ImagingCorrelator {
public:
  /*
   * ctor
   */
  ImagingCorrelator(int nx, int ny, int nz, float dx, float dy, float dz, int nx2, int ny2, int nz2, float kmin, int nThreads);

  /*
   * dtor
   */
  virtual ~ImagingCorrelator();

  void run(libfftv::FFTVFilter *fftv, float *image, float *imgFWI, float *imgRAW, float **srcWave, float **recWave, int option);

  void interleave(float *image, float **srcBuf, float **recBuf);

  void costerm(float *image);

  void sinterm(float *imgfull, float *imgcos);

private:
  //allocate resource
  void create();

  float* assignW(int n, float dw);

  void buffPad2(float *in, float *out, int option);

  void lapx(float *in, float *out);
  void lapz(float *in, float *out);
  void lapy(float *in, float *out);

  void crossone(float *image, float *img2, float *srcWave, float *recWave, int mz, int isign);

  //  void onestep(float* image, float* srcWave, float* recWave);

  void imaging1(libfftv::FFTVFilter *fftv, float *image, float **srcWave, float **recWave);

  void inversion1(libfftv::FFTVFilter *fftv, float *image, float *imgFWI, float **srcWave, float **recWave);

  void inversion2(libfftv::FFTVFilter *fftv, float *image, float *imgFWI, float **srcWave, float **recWave);

  void imaging3(libfftv::FFTVFilter *fftv, float *image, float *imgFWI, float *imgRAW, float **srcBuf, float **recBuf);

  void interleave2d(float *image, float **srcBuf, float **recBuf);

  void interleave3d(float *image, float **srcBuf, float **recBuf);

  void interp(libfftv::FFTVFilter *fftv, float **srcBuf);

  void derive(float *srcWave, float *recWave);

  void kfilter(float *wave, float *work);

  void imgmath(float *imgfull, float *imgimp);

public:
  fftwf_plan planxz1;
  fftwf_plan planxz2;

  fftwf_plan plany1;
  fftwf_plan plany2;

private:
  int nx;
  int ny;
  int nz;
  int nx2;
  int ny2;
  int nz2;
  int nzp2;
  int nloop;
  int ivz = 1, ivx = 2, ivy = 3, ivzx = 4, ivzy = 5, ivxy = 6, ivzxy = 7;

  size_t nxz;
  size_t nyz;
  size_t nxy;
  size_t nxyz;
  size_t nxyzp2;

  float dx;
  float dy;
  float dz;
  float vmax;
  float kmin;

  float scalef;
  float scaleX;
  float scaleY;
  float scaleZ;

  float dkx;
  float dky;
  float dkz;

  int nThreads;
  bool rawOption;

  float *kx;
  float *ky;
  float *kz;

  float *srcBuff;
  float *recBuff;

public:
  float *srcBufz;
  float *recBufz;
  float *srcBufx;
  float *recBufx;
  size_t nxyzp2;

  float dx;
  float dy;
  float dz;
  float vmax;
  float kmin;

  float scalef;
  float scaleX;
  float scaleY;
  float scaleZ;

  float dkx;
  float dky;
  float dkz;

  int nThreads;
  bool rawOption;

  float *kx;
  float *ky;
  float *kz;

  float *srcBuff;
  float *recBuff;

public:
  float *srcBufz;
  float *recBufz;
  float *srcBufx;
  float *recBufx;
  size_t nxyzp2;

  float dx;
  float dy;
  float dz;
  float vmax;
  float kmin;

  float scalef;
  float scaleX;
  float scaleY;
  float scaleZ;

  float dkx;
  float dky;
  float dkz;

  int nThreads;
  bool rawOption;

  float *kx;
  float *ky;
  float *kz;

  float *srcBuff;
  float *recBuff;

public:
  float *srcBufz;
  float *recBufz;
  float *srcBufx;
  float *recBufx;
  size_t nxyzp2;

  float dx;
  float dy;
  float dz;
  float vmax;
  float kmin;

  float scalef;
  float scaleX;
  float scaleY;
  float scaleZ;

  float dkx;
  float dky;
  float dkz;

  int nThreads;
  bool rawOption;

  float *kx;
  float *ky;
  float *kz;

  float *srcBuff;
  float *recBuff;

public:
  float *srcBufz;
  float *recBufz;
  float *srcBufx;
  float *recBufx;
  size_t nxyzp2;

  float dx;
  float dy;
  float dz;
  float vmax;
  float kmin;

  float scalef;
  float scaleX;
  float scaleY;
  float scaleZ;

  float dkx;
  float dky;
  float dkz;

  int nThreads;
  bool rawOption;

  float *kx;
  float *ky;
  float *kz;

  float *srcBuff;
  float *recBuff;

public:
  float *srcBufz;
  float *recBufz;
  float *srcBufx;
  float *recBufx;
  float *srcBufy;
  float *recBufy;

};

#endif /* MAGINGCORRELATOR_H_ */

