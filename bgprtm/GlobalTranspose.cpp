/*
 * GlobalTranspose.cpp
 *
 */
#include <math.h>
#include "GlobalTranspose.h"
#include "GetPar.h"

#ifndef MIN
#define MIN(a,b) ((a) > (b) ? (b) : (a))
#endif

#ifndef MAX
#define MAX(a,b) ((a) < (b) ? (b) : (a))
#endif

GlobalTranspose::GlobalTranspose() {

  init();

  xf = MIN(MIN(lp0.x, lp1.x), lp2.x);
  xl = MAX(MAX(lp0.x, lp1.x), lp2.x);
  yf = MIN(MIN(lp0.y, lp1.y), lp2.y);
  yl = MAX(MAX(lp0.y, lp1.y), lp2.y);

  cout << "xf " << xf << " xl " << xl << endl;
  cout << "yf " << yf << " yl " << yl << endl;

  if(lp0.x != xf || lp0.y != yf)
    perror("first point need have smallest inline and smallest crossline");

  if(lp1.x != xl || lp1.y != yf)
    perror("second point need have smallest inline and largest crossline");

  if(lp2.x != xf || lp2.y != yl)
    perror("third point need have largest crossline and smallest crossline");


  cdpdx = (p1 - p0).length() / (xl - xf);
  cdpdy = (p2 - p0).length() / (yl - yf);

  cout << "dx " << cdpdx << " dy " << cdpdy << endl;

  x0 = cdpdx * (xf - 1);
  y0 = cdpdy * (yf - 1);
  l0 = libCommon::Vec2d(x0, y0);
  cout << "x0 " << x0 << " y0 " << y0 << endl;

  dir1 = p1 - p0;
  dir1.normalize();

  dir2 = p2 - p0;
  dir2.normalize();

  //dir1.print("dir1_p01");
  //dir2.print("dir2_p02");

  //<<"dir1 dot dir2 "<<dir1*dir2<<endl;
}

GlobalTranspose::~GlobalTranspose() {

}

void GlobalTranspose::init() {
  p0.x = global_pars["geometry"]["p0_x"].as<float>();
  p0.y = global_pars["geometry"]["p0_y"].as<float>();
  p1.x = global_pars["geometry"]["p1_x"].as<float>();
  p1.y = global_pars["geometry"]["p1_y"].as<float>();
  p2.x = global_pars["geometry"]["p2_x"].as<float>();
  p2.y = global_pars["geometry"]["p2_y"].as<float>();

  lp0.x = global_pars["geometry"]["p0_xl"].as<int>();
  lp0.y = global_pars["geometry"]["p0_il"].as<int>();
  lp1.x = global_pars["geometry"]["p1_xl"].as<int>();
  lp1.y = global_pars["geometry"]["p1_il"].as<int>();
  lp2.x = global_pars["geometry"]["p2_xl"].as<int>();
  lp2.y = global_pars["geometry"]["p2_il"].as<int>();
}

void GlobalTranspose::worldToLocal(double &x, double &y) {
  libCommon::Vec2d p(x, y);

  x = (p - p0) * dir1 + x0;
  y = (p - p0) * dir2 + y0;
}

void GlobalTranspose::localToWorld(double &x, double &y) {
  libCommon::Vec2d p(x, y);

  x = (p - l0) * dir1 + p0.x;
  y = (p - l0) * dir2 + p0.y;
}

void GlobalTranspose::worldToLocal(float &x, float &y) {
  libCommon::Vec2d p(x, y);

  x = (p - p0) * dir1 + x0;
  y = (p - p0) * dir2 + y0;
}

void GlobalTranspose::localToWorld(float &x, float &y) {
  libCommon::Vec2d p(x, y);

  x = (p - l0) * dir1 + p0.x;
  y = (p - l0) * dir2 + p0.y;
}

void GlobalTranspose::worldToLocal(libCommon::Trace &trace) {
  libCommon::Vec2d s(trace.getShotLoc().x, trace.getShotLoc().y);
  libCommon::Vec2d r(trace.getRecvLoc().x, trace.getRecvLoc().y);
    libCommon::Vec2d m(trace.getCdpLoc().x, trace.getCdpLoc().y);
  trace.setShotLoc(libCommon::Point((s - p0)*dir1 + x0, (s - p0)*dir2 + y0, trace.getShotLoc().z));
  trace.setRecvLoc(libCommon::Point((r - p0)*dir1 + x0, (r - p0)*dir2 + y0, trace.getRecvLoc().z));
    trace.setCdpLoc(libCommon::Point((m - p0)*dir1 + x0, (m - p0)*dir2 + y0, trace.getCdpLoc().z));
}

void GlobalTranspose::localToWorld(libCommon::Trace &trace) {
  libCommon::Vec2d s(trace.getShotLoc().x, trace.getShotLoc().y);
  libCommon::Vec2d r(trace.getRecvLoc().x, trace.getRecvLoc().y);
    libCommon::Vec2d m(trace.getCdpLoc().x, trace.getCdpLoc().y);
  trace.setShotLoc(libCommon::Point((s - l0)*dir1 + p0.x, (s - l0)*dir2 + p0.y, trace.getShotLoc().z));
  trace.setRecvLoc(libCommon::Point((r - l0)*dir1 + p0.x, (r - l0)*dir2 + p0.y, trace.getRecvLoc().z));
    trace.setCdpLoc(libCommon::Point((m - l0)*dir1 + p0.x, (m - l0)*dir2 + p0.y, trace.getCdpLoc().z));
}

libCommon::Vec2i GlobalTranspose::xyToLine_local(libCommon::Vec2d p){
    return libCommon::Vec2i( lround((p-l0) * dir1 / cdpdx) + xf, lround((p-l0) * dir2 / cdpdy) + yf);
}
libCommon::Vec2i GlobalTranspose::xyToLine_world(libCommon::Vec2d p){
    return libCommon::Vec2i( lround((p-p0) * dir1 / cdpdx) + xf, lround((p-p0) * dir2 / cdpdy) + yf);
}
void GlobalTranspose::xyToLine_local(double x, double y, int& ix, int& iy){
    libCommon::Vec2d p(x, y);
    ix = lround((p-l0) * dir1 / cdpdx) + xf;
    iy = lround((p-l0) * dir2 / cdpdy) + yf;
}
void GlobalTranspose::xyToLine_world(double x, double y, int& ix, int& iy){
  libCommon::Vec2d p(x, y);
  ix = lround((p-p0) * dir1 / cdpdx) + xf;
  iy = lround((p-p0) * dir2 / cdpdy) + yf;
}
void GlobalTranspose::lineToXY_local(int ix, int iy, double& x, double& y){
    x = x0 + (ix-xf)*cdpdx;
    y = y0 + (iy-yf)*cdpdx;
}
void GlobalTranspose::lineToXY_world(int ix, int iy, double& x, double& y){
  libCommon::Vec2d p = dir1*((ix-xf)*cdpdx) + dir2*((iy-yf)*cdpdy) + p0 ;
  x = p.x;
  y = p.y;
}

