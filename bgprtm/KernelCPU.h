#ifndef KERNELCPU_H_
#define KERNELCPU_H_

#include <stddef.h>
#include <vector>
using std::vector;
#include "Model.h"

class Grid;
class FdEngine;
class FdEngine_r2r;
class FdEngine_r2c;
class FdEngine_hfd;
class FdEngine_cfd;

class KernelCPU {
public:

  KernelCPU(int nx, int ny, int nz, float dx, float dy, float dz, int engine, int nThreds);

  virtual ~KernelCPU();

  void setModel(float *vel, float *rho, float *del, float *eps, float *pjx, float *pjy);

  void setJacob(Grid *grid, float *jacobx, float *jacoby, float *jacobz);

  void setBoundary(int nxbnd1, int nxbnd2, int nybnd1, int nybnd2, int nzbnd1, int nzbnd2, float dt, float vmax);

  void TTI(float *p0, float *pb, float *pr, ModelType modelType = ::TTI);

  void VTI(float *p0, float *pb, float *pr);

  void ISO(float *p0, float *pb, float *pr, int bndType, int iz0, int spread);

  void cleanPMLMemory();

private:
  void ScalarTTI(float *px, float *py, float *pz, int iy);
  void ScalarTTI2D(float *px, float *pz, int iy);

  void ScalarVTI(float *px, float *py, float *pz, int iy);
  void ScalarVTI2D(float *px, float *pz, int iy);

  void applyVel_minus(float *pr, float *pb);//added by wolf to revise ISO according to the modification of Propagator::kernel()

  void divideRho(float *px, float *py, float *pz, int iy);
//  void multiplyRho(float *pb);//removde by wolf
  void apply_symmetry(float *pz, int sym, int iz0, int tprlen, int n);
//  void multiplycnnRho(float *pb, float *p0, float *prho); // removed by wolf
//  void multiplycnnRho_backup(float *pb, float *p0); // removed by wolf

  void allocMemory();

  int nx, ny, nz;
  float dx, dy, dz;
  int engine;
  int force_tti;
  int nThreads;
  size_t nxz, nxy, size2d;
  float *velVolume { }, *rhoVolume { }, *delVolume { }, *epsVolume { }, *pjxVolume { }, *pjyVolume { };
  vector<float*> wrk2d, px2d, pz2d;
//  float *prho { };
  FdEngine *derive { };
};

#endif /* KERNELCPU_H_ */

