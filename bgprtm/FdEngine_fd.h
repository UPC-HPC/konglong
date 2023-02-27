/*
 * FdEngine_fd.h
 *
 *  Created on: Mar 1, 2022
 *      Author: owl
 */

#ifndef SWPRO_LIBWAVEPROP_FDENGINE_FD_H_
#define SWPRO_LIBWAVEPROP_FDENGINE_FD_H_

#include "FdEngine.h"

class FdEngine_fd: public FdEngine {
public:
  // Note that nz,nx,ny is from fastest to slowest in memory
  FdEngine_fd(int nx0, int ny0, int nz0, float dx0, float dy0, float dz0, int nThreads0); // removed int RhoCN*0 by wolf
  virtual ~FdEngine_fd();
  void setJacob(Grid *grid, float *jacobx, float *jacoby, float *jacobz) override;

  void laplacian(float *in, float *out, float *px = nullptr, float *py = nullptr, int op = FD::DX2 | FD::DY2 | FD::DZ2) override;
  void laplacian_reg(float *in, float *out, float *px, float *py, int op);
  void laplacian_irreg(float *in, float *out, float *px, float *py, int op);

  void dy_3D(int nth_derive, float *in, float *py, vector<float*> &wrk2d, int iacc, int isign) override;
  void dy_3D(int nth_derive, float *in, float *py, vector<float*> &wrk2d, int iacc, int isign, float *velVolume, float *rhoVolume) override;
  void dxyz1(float *in, float *px, float *py, float *pz, int iacc, int isign) override;
  void dxz_2D(int order, float *in, float *px, float *pz, float *wrk, int iy, int iacc, int isign) override;
  void dz1_2D(float *in, float *out, float *wrk, int iy, int iacc, int isign) override;
  void dx1_2D(float *in, float *out, float *wrk, int iy, int iacc, int isign) override;
  void dz1_2D(float *in, float *out, float *wrk, int iy, int iacc, int isign, float *velSlice, float *rhoSlice) override;//added by wolf, Sep. 21, 2022
  void dx1_2D(float *in, float *out, float *wrk, int iy, int iacc, int isign, float *velSlice, float *rhoSlice) override;//added by wolf, Sep. 21, 2022
  void dy1_2D(float *in, float *out, float *wrk, int ix, int iacc, int isign) override; // this function should not have any OMP
//void dy1_2D(float *in, float *out, float *wrk, int ix, int iacc, int isign, float *velSlice, float *rhoSlice) override; // this function should not have any OMP

  void dxyz1_nz(int order, float *u, float *vx, float *vy, float *vz, int ix, int iy, int iacc, int isign, float *vel=nullptr, float *rho=nullptr);
  //the last two arguments added by wolf on Sep. 21, 2022
  void update_buffers() override;

  static int getOrder(string engineName);
  static int getOrder();
  static bool isTaylor();
  static float getDispersion(bool derive1st);
  static float get_k1max();

private:
  void dxyz1_nz_2Dhelper(int order, float *in, float *px, float *py, float *pz, int ix, int iy, int iacc, int isign, float *velSlice, float *rhoSlice); // in/px/py/pz are iy_slice 2D mem
  //the last two arguments added by wolf on Sep. 21, 2022
  void dxz_2D(int nth, float *in, float *px, float *pz, float *wrk, int iy, int iacc, int isign, float *velSlice, float *rhoSlice);//added by wolf, Sep. 21, 2022
  void dxyz1(float *in, float *px, float *py, float *pz, int iacc, int isign, float *velVolume, float *rhoVolume);

public:
  int order = 8, ne = 4;
  bool const_ne = true;

protected:
  bool use_taylor = false;
  vector<vector<float>> coef1z, coef2z;
  vector<vector<float>> coef1, coef2;
  vector<vector<float>> vbufz;
 float zpi0 = 0, zpi1 = 0;
};

#endif /* SWPRO_LIBWAVEPROP_FDENGINE_FD_H_ */

