#ifndef GRID_H
#define GRID_H

#include <fstream>
#include "Model.h"
#include "Interpolant.h"

// Grid Type for Irregular XYZ
enum GRIDTYPE {
  RECTANGLE = 0, IRREGULAR, XPYRAMID, YPYRAMID, XYPYRAMID
};

// Forward declarations
class vector3;
class Profile;

#ifdef  __cplusplus
extern "C" {
#endif

int ZnumGrid(Profile *prof_in, float z0, float zmax, float dzmin);
int ZnumGrid_backup(Profile *prof_in, float z0, float zmax, float dzmin);
float AveRegrid(float zz1, float zz2, float *Vz, float *zz, int nz);
int zbetween(float z0, float *zz, int nz);
int ZnumGridbnd(Profile *prof_in, float z0, float zmax, float dzmin, int zbnd);  // with boundary

#ifdef  __cplusplus
}
#endif

class Grid {
public:
  int nx;           //
  int ny;
  int nz;           // real Nz of volume
  int nnz;           // regular output zone nz
  int nzalloc;           // MAX(nz, nnz)
  int iz0;           // the iz of the position zmin (i.e., the index of the first non-boundary)
  int nztop;
  int incx = 1, incy = 1; // for ILINE XLINE increment concept, mainly for imaging output only
  float dx;
  float dy;
  float dz;
  float x0;           // X position of first sample
  float y0;
  float z0;           // depth of first sample. maybe negative because of boundary
  float centerx;           // for Pyramid use, in index
  float centery;
  float apertx;           // surface aperture of the grid
  float aperty;
  float zmin;           // depth of min z excluding boundary (c.f. z0)
  float zmax;
  float slopex;
  float slopey;
  size_t mysize;
  vector<float> zgrid, dxgrid, dygrid, dzgrid, azmaps;
  vector<float> jacobx, jacoby, jacobz;
  int nDblx = 1, nDbly = 1, nDblz = 1;
  int nThreads;

public:
  int mytype;
  Grid(int type, int nx0, int ny0, int nz0, float dx0, float dy0, float dz0, float z00, float z11, int nThread0);
  Grid(Grid *input, int nDblz, int nDblx, int nDbly);
  ~Grid() {
  }
  void setupRectangle() {
    z0 = zmin - nztop * dz;
    apertx = (nx - 1) * dx;
    aperty = (ny - 1) * dy;
  }
  void setZtop(int zz) {
    nztop = zz;
  }
  void setMidpoint(float midx0, float midy0);
  void setOrigin(float x0, float y0);
  void setIncXY(int incx, int incy);
  void setupGrid(Profile *prof_in, int zbnd, int nzuppad);
  void setupGrid(float *zgrid, float *dzgrid, int zbnd, int nzuppad);
  void setupSlope(float dymax, float dxmax);
  void create();
  float getmyZloc(float myz) const;
  float getmyZloc_backup(float myz) const;
  float getmyXloc(float myx, float myz) const;
  float getmyYloc(float myy, float myz) const;
  void fillazmaps();
  int getIDx(float x, float z = 0.0f) const;
  float getIDxf(float x, float z = 0.0f) const;
  int getIDy(float y, float z = 0.0f) const;
  float getIDyf(float y, float z = 0.0f) const;
  int getIDz(float z) const;
  float getIDzf(float z) const;
  float getmyx(int ix, float myz) const {
    return x0 + ix * getdx(myz);
  }

  float getmyy(int iy, float myz) const {
    return y0 + iy * getdy(myz);
  }

  float getmyx(int ix, int iz) const {
    return x0 + ix * getdx(getmyz(iz));
  }

  float getmyy(int iy, int iz) const {
    return y0 + iy * getdy(getmyz(iz));
  }

  float getmyz(int iz) const;
  vector3 getxloc(int ix, int iy, int iz) const;

  // private:
  void simplezgrid(Profile *prof_in);
  int boundedzgrid(Profile *prof_in, int nzbnd);
  int boundedzgrid_backup(Profile *prof_in, int nzbnd);
  void alloctables();
  float getdy(float myz) const;
  float getdx(float myz) const;
  float irregzloc(float zz, float range, float dz1) const;
  void doublezsample(Grid *input);
  void setupJacob_backup() {
    for(int iz = 0; iz < nz; iz++) {
      if(jacobx.size()) jacobx[iz] = dx / getdx(zgrid[iz]);
      if(jacoby.size()) jacoby[iz] = dy / getdy(zgrid[iz]);
      if(jacobz.size()) jacobz[iz] = dz / dzgrid[iz];
    }
  }

  void setupJacob();
  void savefdm(const char *filename, float *vol, float gx0 = 0.0f, float gy0 = 0.0f, float gz0 = 0.0f);
  void saveModel(const char *filename, float *vol);
  void saveModel(string filename, float *vol);
  void FillVolume(Grid *gridin, float *volumein, float *volumeut);
  void FillVolume2D(Grid *gridin, float *volumein, float *volumeut);
  void FillVolume3D(Grid *gridin, float *volumein, float *volumeut);
    void printZGrid() const;
    void print(int verbose = 0) const;
    void printFloatVector(vector<float> v, string info) const;

    void saveGrid(string file);
    void readGrid(string file);
    void outVec(vector<float>& vec, string str, std::ofstream& ofs);
    void inVec(vector<float>& vec, std::ifstream& ifs);

    size_t bufferSize();
    char* toBuffer(char* buf);
    char* fromBuffer(char* buf);
    char* vecToBuf(vector<float>& vec, char* buf);
    char* vecFromBuf(vector<float>& vec, char* buf);
};

#endif

