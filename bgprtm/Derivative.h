/*
 * Derivative.h
 *
 */

#ifndef DERIVATIVE_H_
#define DERIVATIVE_H_

#include <math.h>
#include <xmmintrin.h>
#include <omp.h>
#include "Grid.h"
#include "Wavefield.h"
#include "Boundary.h"

class Derivative {
public:
  Derivative(Grid *grid, Boundary *bnd, int modelType, int nThreads);

  /*
   * Destruct
   */
  virtual ~Derivative() {};

  // to be overloaded
  virtual void dx1(float *in, float *out, int isign) = 0;  // isign : 1: first time to derivative 2: second time to derivative
  virtual void dy1(float *in, float *out, int isign) = 0;
  virtual void dz1(float *in, float *out, int isign) = 0;
  virtual void dx2(float *in, float *out) = 0;
  virtual void dy2(float *in, float *out) = 0;
  virtual void dz2(float *in, float *out) = 0;


  void getDiverge(Wavefield *myWavefield);
  void getGradient(Wavefield *myWavefield);

  void cleanMemory() {bnd->cleanMemory();}

  //setup grid
  void setupGrid(int gridType, float slopex, float slopey, float *jacobx, float *jacoby, float *jacobz);

  void getDivergePXY(Wavefield *myWavefield);
  void getLaplacian(Wavefield *myWavefield);

private:
  void    getDiverge0(Wavefield *myWavefield);

  void    destretchz(float *volapply);
   void    destretchz2(float *volapply);

  void    rescaley(float *volapply);

  void    rescalex(float *volapply);

  void    dePyramidy(float *outy, float *outz);

  void    dePyramidx(float *outx, float *outz);

  void    addVolume(float *a, float *b);


protected:
  Boundary *bnd;
  int  iswitch;
  int  nThreads;

  int   nx;
  int   ny;
  int   nz;

  float dx;
  float dy;
  float dz;

  float slopex;
  float slopey;

  int   gridType;

  size_t nxz;
  size_t nyz;
  float *jacobz;
  float *jacobx;
  float *jacoby;

};


#endif /* DERIVATIVE_H_ */

