#ifndef WAVEFIELD_H
#define WAVEFIELD_H

class Wavefield {
public:
  float *w0;
  float *w1;
  float *wb;
  float *wx;
  float *wy;
  float *wz;
  float *wr;

  Wavefield(int nx, int ny, int nz);
  ~Wavefield();
  void allocateMemory();
  void cleanMemory();
  void deallocateMemory();

private:
  bool isInitialized;
  int nx_, ny_, nz_;
};

#endif

