#ifndef FDENGINE_H_
#define FDENGINE_H_

#ifdef NO_MKL
#include <fftw3.h>
#else
#include <fftw/fftw3.h>
#endif

#include "Util.h"
#include <math.h>
#include <string.h>
#include <xmmintrin.h>
#include <omp.h>

#include "fdWrapper.h"
#include "Grid.h"
#include "PML.h"
#include "fdm.hpp"
#include "GetPar.h"

#define ABSPML  0
#define ABSTAP  1
namespace FD {
enum Engine {
  FFTV = 0, FFTWR2C, FFTWR2R, HFD, CFD, CFD_BATCH, SINCOS, FD
};
extern const vector<string> EngineNames; // need to match the "enum Engine" index!!!
enum Capabilities { // bits for capabilities
  order0 = 0x01, order1 = 0x02, order2 = 0x04
};
enum OPBITS { // bits for operations
  DZ2 = 0x01, DX2 = 0x02, DY2 = 0x04
};
}

class PML;

class FdEngine {
public:
  /*
   * Note that nz,nx,ny is from fastest to slowest in memory
   */
  FdEngine(int innx, int inny, int innz, float indx, float indy, float indz, int inNThreads); //revised by wolf to removed int inRhoCN*

  /*
   * dtor
   */
  virtual ~FdEngine();

  void update_nxyz(int nx0, int ny0, int nz0);
  virtual void update_buffers() {
  }
  //setup jacob
  virtual void setJacob(Grid *grid, float *jacobx, float *jacoby, float *jacobz);
  void setBoundary(int nxbnd1, int nxbnd2, int nybnd1, int nybnd2, int nzbnd1, int nzbnd2, float dt, float vmax);

  size_t getSizeBuf2d();

  static ModelType determineModelType();
  static float getFdDispersion();
  static string getEngineName();
  static bool isDerive2nd(); // if ModelType is VTI/ISO and the engine uses pure 2nd derivative

  virtual void dy_3D(int nth_derive, float *in, float *py, vector<float*> &wrk2d, int iacc, int isign);
  virtual void dy_3D(int nth_derive, float *in, float *py, vector<float*> &wrk2d, int iacc, int isign, float *velVolume, float *rhoVolume);//added by wolf

  virtual void dxz_2D(int nth_derive, float *in, float *px, float *pz, float *wrk, int iy, int iacc, int isign);

  virtual void dz1_2D(float *in, float *out, float *wrk, int iy, int iacc, int isign) {
    assertion(false, "Not implemented!");
  }
  virtual void dx1_2D(float *in, float *out, float *wrk, int iy, int iacc, int isign) {
    assertion(false, "Not implemented!");
  }

  virtual void dy1_2D(float *in, float *out, float *wrk, int ix, int iacc, int isign) {  // this function should not have any OMP
    assertion(false, "Not implemented!");
  }

  virtual void dz1_2D(float *in, float *out, float *wrk, int iy, int iacc, int isign, float *vel, float *rho) {
    assertion(false, "Not implemented!");
  }//added by wolf, Sep. 21, 2022
  virtual void dx1_2D(float *in, float *out, float *wrk, int iy, int iacc, int isign, float *vel, float *rho) {
    assertion(false, "Not implemented!");
  }//added by wolf, Sep. 21, 2022
 virtual void dy1_2D(float *in, float *out, float *wrk, int ix, int iacc, int isign, float *vel, float *rho) {  // this function should not have any OMP
    assertion(false, "Not implemented!");
  }

  // ixy0/ixy1 is for PML mem positions, ixy1-ixy0 is for in/out batch size (already offset by ixy0)
  virtual void dz2(float *in, float *out, float *wrk, int ixy0, int ixy1, int iacc) {
    assertion(false, "dz2() not implemented! derive_cap=%s", bitset<8>(derive_cap).to_string().c_str());
  }
  // iy is slowest, iz is fastest. iy/iz0/iz1 is for PML mem positions, iz0/iz1 is for in/out positions (already offset by iy)
  virtual void dx2(float *in, float *out, float *wrk, int iy, int iz0, int iz1, int iacc) {
    assertion(false, "dx2() not implemented! derive_cap=%s", bitset<8>(derive_cap).to_string().c_str());
  }
  // ix slice. ix is for PML mem position. in/out is already offset partially by ix*nz
  virtual void dy2(float *in, float *out, float *wrk, int ix, int iacc) {
    assertion(false, "dy2() not implemented! derive_cap=%s", bitset<8>(derive_cap).to_string().c_str());
  }

  virtual void dxyz1(float *in, float *px, float *py, float *pz, int iacc, int isign) {
    assertion(false, "dxyz1() not implemented! do_dxyz1=false!");
  }
  // out: when px, py are nullptr, it's pz2+px2+py2, otherwise it's pz2
  virtual void laplacian(float *in, float *out, float *px = nullptr, float *py = nullptr, int op = FD::DX2 | FD::DY2 | FD::DZ2) {
    assertion(false, "laplacian() not implemented! do_laplace=false!");
  }

  void pml1_zbnd(float *pz, int ix, int iy, int isign);
  void pml1_xbnd(float *px, int ix, int iy, int isign);
  void pml1_ybnd(float *py, int ix, int iy, int isign);
  void pml2_zbnd(float *trace, int ix, int iy);
  void pml2_xbnd(float *trace, int ix, int iy);
  void pml2_ybnd(float *trace, int ix, int iy);
  void pml2_bnd(float *trace, int ix, int iy);
void cleanPMLMemory();

  static void init_sine(float *data, int nx, int ny, int nz, float nosc = 10.0);
  static int benchmark(int argc, char **argv);
public:
  float dispersion_factor = 1.0f;
  bool parallel_xz = false;
  bool do_laplace = false, do_dxyz1 = false;
  FD::Capabilities derive_cap = FD::order1; // default only supports 1st order
  float k1max = M_PI; // for stability

private:
  void pml1_zbnd(float *pz, int ix, int iy, int isign, PMLBUF::Pxyz jzxy, float scaler);
  void pml1_xbnd(float *px, int ix, int iy, int isign, PMLBUF::Pxyz jzxy, float scaler);
  void pml1_ybnd(float *py, int ix, int iy, int isign, PMLBUF::Pxyz jzxy, float scaler);

  virtual void dz1_2D_part(float *in, float *out, float *wrk, int ixy0, int ixy1, int iacc, int isign) {
    assertion(false, "Partial dz1(ixy0,ixy1) not implemented! parallel_xz = false!");
  }
  virtual void dx1_2D_part(float *in, float *out, float *wrk, int iy, int iz0, int iz1, int iacc, int isign) {
    assertion(false, "Partial dx1(iz0,iz1) not implemented! parallel_xz = false!");
  }

//  void RhoCN*_bnd(PML *pml1, PML *pml2, float *wrk1, float *q1, float *q2, float *q3, float *q4, int nz, int nx, int nbnd1, int nbnd2); // removed by wolf

public:
  int nx { }, ny { }, nz { };
  size_t nxz { }, nyz { }, nxy { }, nxyz { };
  float dx { }, dy { }, dz { };
  float dxi { }, dyi { };

protected:
  float dt { };
 float vmax { };

  float scaleX { };
  float scaleY { };
  float scaleZ { };

  float dkx { };
  float dky { };
  float dkz { };

  int nThreads { };
//  int RhoCN* { }; removed by wolf
  Grid *grid { };
  int gridType { };
  int absType { };
  bool topPMLFlag { };
  bool allPMLFlag { };

  float *jacobz { };
  float *jacobx { };
  float *jacoby { };

  float *kx { };
  float *ky { };
  float *kz { };

  float *pmlBuf[3][2][2][3] { }; // pmlBuf[XYZ][TOPBOT][Round][Pxyz],
//  float *pmlcnnBuf[3][2][2][3] { }; //removed by wolf
  // dim=2 or 3. [Dim][2][2][DIM][]= for doPML3D/2D, [Dim][2][2][1][] for doPML1D
  // replace qZ1  by pmlBuf[Z][TOP][ROUND1][ZZ], qZ2  by pmlBuf[Z][BOT][ROUND1][ZZ]
  // replace qZ1b by pmlBuf[Z][TOP][ROUND2][ZZ], qZ2b by pmlBuf[Z][BOT][ROUND2][ZZ]
  // replace qX1  by pmlBuf[X][TOP][ROUND1][XX], qX2  by pmlBuf[X][BOT][ROUND1][XX]
  // replace qX1b by pmlBuf[X][TOP][ROUND2][XX], qX2b by pmlBuf[X][BOT][ROUND2][XX]
  // replace qY1  by pmlBuf[Y][TOP][ROUND1][YY], qY2  by pmlBuf[Y][BOT][ROUND1][YY]
  // // float *qX1 { }, *qX2 { }, *qY1 { }, *qY2 { }, *qZ1 { }, *qZ2 { }, *qX1b { }, *qX2b { }, *qY1b { }, *qY2b { }, *qZ1b { }, *qZ2b { };
//  float *qtX1 { }, *qtX2 { }, *qtY1 { }, *qtY2 { }, *qtZ1 { }, *qtZ2 { }, *qtX1b { }, *qtX2b { }, *qtY1b { }, *qtY2b { }, *qtZ1b { },
//      *qtZ2b { };

  int nxbnd1 { }, nxbnd2 { };
  int nybnd1 { }, nybnd2 { };
  int nzbnd1 { }, nzbnd2 { };

  PML *pmlX1 { }, *pmlX2 { }, *pmlY1 { }, *pmlY2 { }, *pmlZ1 { }, *pmlZ2 { };
  PML *pmltX1 { }, *pmltX2 { }, *pmltY1 { }, *pmltY2 { }, *pmltZ1 { }, *pmltZ2 { };
  bool do_pml3d {};
  int dimPML = 1;
  float pml3d_scaler = 1.0f;
};

#endif /* FDENGINE_H_ */

