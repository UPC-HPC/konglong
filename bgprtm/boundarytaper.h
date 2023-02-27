#ifndef BOUNDARY_TAPER_H
#define BOUNDARY_TAPER_H

#include "Grid.h"

class Wavefield;

class BndTaper {    // Wave Propagation
public:
  Grid      *myGrid;
  int        xbound;
  int        ybound;
  int        zbound;
  float     *xtaper;
  float     *ytaper;
  float     *ztaper;
  int      nThreads;
  // Wave fields
public:
  BndTaper(Grid *mg, int xb, int yb, int zb, float coef) {myGrid = mg; xbound = xb; ybound = yb; zbound = zb; create(coef);};
  ~BndTaper();
  void apply(Wavefield *myWavefield); // Overload so that original function can become Private
private:
  void apply(float *vol);
  void create(float coef);

};



#endif
~          
