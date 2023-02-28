#ifndef GEOMETRY_H_
#define GEOMETRY_H_

#include <string>
#include <vector>
using std::string;
using std::vector;

struct GeomHeader {
  float x0, y0, z0;          // origin
  int nx, ny, nz;          // size
  float dx, dy, dz;          // step
  int nzbnd, nzuppad;
  int nxbnd, nybnd;
  int gridType;
};

class Geometry {
public:
  Geometry();

  ~Geometry();

  void read(const char *path);
  void read(string &path);

  void write(const char *path);
  void write(string &path);

  GeomHeader* getHeader() {
    return header;
  }

  float* getZGrid() {
    return zgrid.empty() ? nullptr : &zgrid[0];
  }

  float* getDzGrid() {
    return dzgrid.empty() ? nullptr : &dzgrid[0];
  }

  void setHeader(float x0, float y0, float z0, int nx, int ny, int nz, float dx, float dy, float dz, int nzbnd, int nzuppad, int nxbnd,
      int nybnd, int gridType);

  void setZGrid(float *zgrid);
  void setZGrid(vector<float> &zgrid);

  void setDzGrid(float *dzgrid);
  void setDzGrid(vector<float> &dzgrid);

  void printHeader();

  void printTable();

public:

  GeomHeader *header;

  vector<float> zgrid;
  vector<float> dzgrid;
};

#endif /* GEOMETRY_H_ */

