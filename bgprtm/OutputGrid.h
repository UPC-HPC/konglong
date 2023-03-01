/*
 * OutputGrid.h
 *
 */

#ifndef SWPRO_MPIRTM_OUTPUTGRID_H_
#define SWPRO_MPIRTM_OUTPUTGRID_H_

#include <string>
using std::string;

enum CoordFollow {
  COORD_NONE = 0, COORD_LOGICAL = 1, COORD_PHYSICAL = 2
};
#define COORD_FOLLOW {"NONE", "LOGICAL", "PHYSICAL"}

class OutputGrid {
public:
  OutputGrid();

  virtual ~OutputGrid();

  void initGrid();

  void setGrid();

  void read_grid_velfile();
  void read_grid_parameter();

public:
  string jsVel;
  float x0 { }, y0 { }, z0 { };
  double dx { }, dy { }, dz { };
  int nx { }, ny { }, nz { };
  int ix0 { }, iy0 { }, incx { }, incy { };

};

#endif /* SWPRO_MPIRTM_OUTPUTGRID_H_ */

