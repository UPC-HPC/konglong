#ifndef WAVEFIELD_H
#define WAVEFIELD_H

#include "Model.h"

class Grid;

enum AppType {
  RTM = 1, MOD
};

class Wavefield {
public:
  float *w0;
  float *w1;
  float *wb;
  float *wx;
  float *wy;
  float *wz;
  float *wr;
  vector<vector<float>> wq[2];
  int iq0 = 0, iq1 = 1; // rotate wq[0] and wq[1]
  int count_mem3d { };

  Wavefield(Grid *myGrid, Model *model);
  ~Wavefield();
  void allocateMemory(bool allocWx, int nthreads);
  void cleanMemory(int nthreads);
  void deallocateMemory();
  void swapPointers();
  void swapPointers_demig();

private:
  bool isInitialized;
  bool allocWx;
  int nx_, ny_, nz_;
  ModelType modelType_;
  Model *model;
  int appType;

  float* allocMem3d(int nx, int ny, int nz, int nThreads);
};

#endif

