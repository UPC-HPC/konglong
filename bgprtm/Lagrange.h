/*
 * Lagrange.h
 *
 */

#ifndef SWPRO_LIBWAVEPROP_LAGRANGE_H_
#define SWPRO_LIBWAVEPROP_LAGRANGE_H_

#include <vector>
#include <memory>
using std::vector;
using std::unique_ptr;
class Grid;

class Lagrange {

public:
  /*
   * ctor
   */
  Lagrange(Grid *gridIn, Grid *gridOut, int nPoints, int nThreads);

  /*
   * dtor
   */
  virtual ~Lagrange();

  void apply(float *dataIn, float *dataOut);

private:
  void initCoefs();

  void initCoef(float *tblIn, int nin, float *tblOut, int nout, int nPoints, float *coefz, int *norderz, int *nbeginz);

  int checkGrid(float z, float *ztbl, int nztbl);

  void getCoef(float *ztbl, int nztbl, int nzbeg, int norder, float z, float *intpcoef);

  float applyCoef(int nbeg, int norder, float *coef, float *data);

  void interpz(float *dataIn, float *dataOut, int nzIn, int nzOut, int nx, int ny, int nintp);

  void interpx(float *dataIn, float *dataOut, int nxIn, int nxOut, int ny, int nz, int nintp);

  void interpy(float *dataIn, float *dataOut, int nyIn, int nyOut, int nx, int nz, int nintp);

protected:
  Grid *gridIn;
  Grid *gridOut;

  float *coefz, *coefx, *coefy;
  int *norderz, *norderx, *nordery;
  int *nbeginz, *nbeginx, *nbeginy;

  vector<float> workz;
  vector<float> workx;

  int nPoints, nThreads;

};

#endif /* SWPRO_LIBWAVEPROP_LAGRANGE_H_ */

