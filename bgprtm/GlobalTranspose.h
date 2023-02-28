/*
 * GlobalTranspose.h
 */

#ifndef GLOBALTRANSPOSE_H_
#define GLOBALTRANSPOSE_H_

//#include "Vec2.hpp"
#include "libCommon/Trace.hpp"
#include "libCommon/Vec2.hpp"

enum { NOTRANSPOSE = 0, WORLD2LOCAL, LOCAL2WORLD};

class GlobalTranspose {

public:
  GlobalTranspose();

  virtual ~GlobalTranspose();

  void init();

  void worldToLocal(double &x, double &y);

  void localToWorld(double &x, double &y);

  void worldToLocal(float &x, float &y);

  void localToWorld(float &x, float &y);

  void worldToLocal(libCommon::Trace &trace);

  void localToWorld(libCommon::Trace &trace);
    libCommon::Vec2i xyToLine_local(libCommon::Vec2d p);
    libCommon::Vec2i xyToLine_world(libCommon::Vec2d p);
    void xyToLine_local(double x, double y, int& ix, int& iy);
    void xyToLine_world(double x, double y, int& ix, int& iy);
    void lineToXY_local(int ix, int iy, double& x, double& y);
    void lineToXY_world(int ix, int iy, double& x, double& y);

private:

  libCommon::Vec2d p0,  p1,  p2;
  libCommon::Vec2i lp0, lp1, lp2;

  libCommon::Vec2d dir1, dir2;
  libCommon::Vec2d  l0;
  double x0, y0;

    int xf, yf, xl, yl;
    double cdpdx, cdpdy;
};


#endif /* GLOBALTRANSPOSE_H_ */

