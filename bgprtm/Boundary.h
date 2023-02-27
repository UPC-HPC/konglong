/*
 * Boundary.h
 *
 */

#ifndef BOUNDARY_H_
#define BOUNDARY_H_

#include <stddef.h>

class PML;

class Boundary {
public:
  /*
   * constructor
   */
  Boundary(int bndType, int nx, int ny, int nz, float dx, float dy, float dz, float dt, float vmax, int nThreads);

  /*
   * distructor
   */
  virtual ~Boundary();

  void setBoundary(int nxbnd1, int nxbnd2, int nybnd1, int nybnd2, int nzbnd1, int nzbnd2);

  //  void applyX(float* p, int nbatch, int indx);

  //  void applyY(float* p, int nbatch, int indx);

  //  void applyZ(float* p, int nbatch, int indx);

  void applyX(float *p, int indx);
  void applyY(float *p, int indx);

  void applyZ(float *p, int indx);

  void allocMemory();

  void cleanMemory();

protected:
  int bndType { };
  int nx { }, ny { }, nz { };
  float dx { }, dy { }, dz { }, dt { };
  float vmax { };
  int nThreads { };
  int nxbnd1 { }, nxbnd2 { };
  int nybnd1 { }, nybnd2 { };
  int nzbnd1 { }, nzbnd2 { };
  size_t nxz { }, nyz { }, nxy { };
  float *work { };
  PML *pmlX1 { }, *pmlX2 { }, *pmlY1 { }, *pmlY2 { }, *pmlZ1 { }, *pmlZ2 { }, *pmlX3 { }, *pmlX4 { }, *pmlY3 { }, *pmlY4 { }, *pmlZ3 { },
      *pmlZ4 { };
  float *pmlBuf[3][2][2][3] { };
  int dimPML = 1;
};

#endif /* BOUNDARY_H_ */

