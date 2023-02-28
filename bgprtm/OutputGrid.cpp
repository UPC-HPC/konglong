/*
 * OutputGrid.cpp
 */

#include "OutputGrid.h"

#include "GetPar.h"
#include <stdio.h>
#include <stdlib.h>
#include "fdm.hpp"
#include <jseisIO/jseisUtil.h>
#include "libSWIO/RecordIO.hpp"
#include "GlobalTranspose.h"

#include "MpiPrint.h"
using MpiPrint::print1m;

#include <boost/algorithm/string/predicate.hpp>
using boost::algorithm::iends_with;

OutputGrid::OutputGrid() {
  // TODO Auto-generated constructor stub
  x0 = y0 = z0 = 0.0;
  dx = dy = dz = 0.0;
  nx = ny = nz = 1;
  ix0 = iy0 = incx = incy = 1;

  initGrid();
}

OutputGrid::~OutputGrid() {
  // TODO Auto-generated destructor stub
}
void OutputGrid::initGrid() {
  if(global_pars["firstXline"] && global_pars["lastXline"]) // user defined output range
  read_grid_parameter();
  else // identical to velocity model
  read_grid_velfile();
}

void OutputGrid::setGrid() {
  if(!global_pars["firstXline"] || !global_pars["lastXline"]) {
    global_pars["CDPdx"] = dx / incx;
    global_pars["CDPdy"] = dy / incy;
    global_pars["RTMdz"] = dz;
  }
}

void OutputGrid::read_grid_parameter() {

  ix0 = global_pars["firstXline"].as<int>();
  int ilx = global_pars["lastXline"].as<int>();

  iy0 = global_pars["firstInline"].as<int>(1);
  int ily = global_pars["lastInline"].as<int>(1);

  incy = global_pars["incInline"].as<int>(1);
  incx = global_pars["incXline"].as<int>(1);

  assertion(global_pars["CDPdy"] && global_pars["CDPdy"], "'CDPdy' and 'CDPdy' are required for RTM!");

  double cdpdx = global_pars["CDPdx"].as<double>();
  double cdpdy = global_pars["CDPdy"].as<double>();
  dx = cdpdx * incx;
  dy = cdpdy * incy;

  assertion(global_pars["RTMdz"] && global_pars["zMax"], "'RTMdz' and 'zMax' are required for RTM!");
  dz = global_pars["RTMdz"].as<double>();

  float zMax = global_pars["zMax"].as<float>();

  double outputImageGridXAnchor = global_pars["outputImageGridXAnchor"].as<double>(0.0);
  double outputImageGridYAnchor = global_pars["outputImageGridYAnchor"].as<double>(0.0);
  double outputImageGridZAnchor = global_pars["outputImageGridZAnchor"].as<double>(0.0);

  float zMin = global_pars["zMin"].as<float>(0.0f);

  // local origin
  x0 = (ix0 - 1) * cdpdx + outputImageGridXAnchor;
  y0 = (iy0 - 1) * cdpdy + outputImageGridYAnchor;
  z0 = zMin + outputImageGridZAnchor;
  z0 = global_pars["RTMz0"].as<float>(z0);

  nx = (ilx - ix0) / incx + 1;
  ny = (ily - iy0) / incy + 1;
  nz = nearbyintf((zMax - z0) / dz) + 1;
}

void OutputGrid::read_grid_velfile() {
  Node trunk = global_pars["global"];
  Node branch = global_pars["global"]["vel"];
  string velFile = expEnvVars(branch["file"].as<string>(""));
  if(iends_with(velFile, ".fdm")) { // read grid info from FDM
    Fdm vel;
    vel.readheader(velFile.c_str());
    FdmHeader head = vel.getHeader();
    nz = head.nz, nx = head.nx, ny = head.ny;
    dz = head.dz, dx = head.dx * head.xi, dy = head.dy * head.yi;
    z0 = head.z0, x0 = head.x0, y0 = head.y0;
    incx = head.xi, incy = head.yi;
  } else if(iends_with(velFile, ".js")) {
    jsVel = velFile;
    libSeismicFileIO::JSDataReader reader(velFile);
    nz = reader.getAxisLen(0);
    z0 = reader.getAxisPhysicalOrigin(0);
    dz = reader.getAxisPhysicalDelta(0);
    nx = reader.getAxisLen(1);
    ix0 = reader.getAxisLogicalOrigin(1);
    incx = reader.getAxisLogicalDelta(1);
    x0 = reader.getAxisPhysicalOrigin(1);
    dx = reader.getAxisPhysicalDelta(1);
    ny = reader.getAxisLen(2);
    iy0 = reader.getAxisLogicalOrigin(2);
    incy = reader.getAxisLogicalDelta(2);
    y0 = reader.getAxisPhysicalOrigin(2);
    dy = reader.getAxisPhysicalDelta(2);
  } else {
    dz = branch["dz"].as<float>(trunk["dz"].as<float>(0.0f));
    dx = branch["dx"].as<float>(trunk["dx"].as<float>(0.0f));
    if(dz == 0 || dx == 0) {
      fprintf(stderr, "YAML keys 'dz' and 'dx' are required for model!\n\n");
      exit(-1);
    }
    dy = branch["dy"].as<float>(trunk["dy"].as<float>(50.0f));

    z0 = branch["z0"].as<float>(trunk["z0"].as<float>(0.0f));
    x0 = branch["x0"].as<float>(trunk["x0"].as<float>(0.0f));
    y0 = branch["y0"].as<float>(trunk["y0"].as<float>(0.0f));

    ny = 1;
    if(trunk["x1"]) nx = nearbyintf(trunk["x1"].as<float>() - x0) / dx + 1;
    if(branch["x1"]) nx = nearbyintf(branch["x1"].as<float>() - x0) / dx + 1;
    if(trunk["y1"]) ny = nearbyintf(trunk["y1"].as<float>() - y0) / dy + 1;
    if(branch["y1"]) ny = nearbyintf(branch["y1"].as<float>() - y0) / dy + 1;
    if(trunk["z1"]) nz = nearbyintf(trunk["z1"].as<float>() - z0) / dz + 1;
    if(branch["z1"]) nz = nearbyintf(branch["z1"].as<float>() - z0) / dz + 1;
    nz = branch["nz"].as<int>(trunk["nz"].as<int>(nz));
    nx = branch["nx"].as<int>(trunk["nx"].as<int>(nx));
    ny = branch["ny"].as<int>(trunk["ny"].as<int>(ny));

    ix0 = branch["xline_first"].as<int>(trunk["xline_first"].as<int>(1));
    iy0 = branch["iline_first"].as<int>(trunk["iline_first"].as<int>(1));
    incx = branch["xline_inc"].as<int>(trunk["xline_inc"].as<int>(1));
    incy = branch["iline_inc"].as<int>(trunk["iline_inc"].as<int>(1));
  }
  assertion(nx > 0 && ny > 0 && nz > 0, "nx=%d, ny=%d, nz=%d from velocity grid must >0!", nx, ny, nz);

  if(global_pars["geometry"]) {
    GlobalTranspose gtrans;
    gtrans.worldToLocal(x0, y0);
  }

  double outputImageGridXAnchor = global_pars["outputImageGridXAnchor"].as<double>(0.0);
  double outputImageGridYAnchor = global_pars["outputImageGridYAnchor"].as<double>(0.0);
  double outputImageGridZAnchor = global_pars["outputImageGridZAnchor"].as<double>(0.0);

  int ix00 = nearbyintf((x0 - outputImageGridXAnchor) / dx * incx) + 1; // incx inside () for incx=2, ix00 is even
  int iy00 = nearbyintf((y0 - outputImageGridYAnchor) / dy * incy) + 1;

  string str_follow = global_pars["coord_follow"].as<string>("NONE");
  transform(str_follow.begin(), str_follow.end(), str_follow.begin(), ::toupper);
  vector<string> list = COORD_FOLLOW;
  int follow = std::find(list.begin(), list.end(), str_follow) - list.begin();
  print1m("follow=%d (LOGICAL=%d, PHYSICAL=%d)\n", follow, COORD_LOGICAL, COORD_PHYSICAL);
  if(follow == COORD_LOGICAL) {
    x0 = (ix0 - 1) * dx / incx + outputImageGridXAnchor;
    y0 = (iy0 - 1) * dy / incy + outputImageGridYAnchor;
  } else if(follow == COORD_PHYSICAL) {
    ix0 = ix00, iy0 = iy00;
  } else assertion(ix00 == ix0 && iy00 == iy0, "Physical origin (%d,%d) and Logical coordinates (%d,%d) mismatch!\n"
                   "  Set 'coord_follow' to either %s or %s to force one!",
                   ix00, iy00, ix0, iy0, list[1].c_str(), list[2].c_str());

  if(global_pars["zMax"]) {
    float zMax = global_pars["zMax"].as<float>();
    float zMin = global_pars["zMin"].as<float>(0.0f);
    if(global_pars["RTMz0"]) z0 = global_pars["RTMz0"].as<float>();
    else if(zMin != z0) print1m("WARNING: velocity z0(%f) != parameter zMin(%f), using zMin ...!\n", z0, zMin), z0 = zMin;
    nz = nearbyintf((zMax - z0) / dz) + 1;
  }
}

