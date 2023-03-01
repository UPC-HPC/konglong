/*
 * PML.h
 *
 */

#ifndef PML_H_
#define PML_H_

#include <vector>
using std::vector;

namespace PMLBUF { // these names does not prevent mistakes but makes each index's meaning clear
enum ZXY {
  Z = 0, X, Y
};
enum TOPBOT {
  TOP = 0, BOT = 1
};
enum ROUND {
  ROUND1 = 0, ROUND2 = 1
};
enum Pxyz {
  ZZ = 0, ZX = 1, ZY = 2, XX = 0, XZ = 1, XY = 2, YY = 0, YZ = 1, YX = 2
};

}

class PML {
public:
  /*
   * constructor
   */
  PML(int n, int npml, int nbatch, float dx, float dt, float vmax, bool limit_slope);

  /*
   * destructor
   */
  virtual ~PML();

  void apply_single(vector<float> &coef, float scaler, float *p, float *q, int symmetry = 0);
  void apply2_single(vector<float> &coef, float *p, float *q, float *q2, int symmetry = 0);
  void apply_single_trans(float g, float *p, float *q, int nz);
  void apply2_single_trans(float g, float *p, float *q, float *q2, int nz);

  void apply(float *pwav, float *qpml, int nbatch, int topbot);
  void apply_trans(float *pwav, float *qpml, int nbatch, int topbot);

  // apply2 is same as applying with qpml then q2pml, but doing together is more efficient
  void apply2(float *pwav, float *qpml, float *q2pml, int nbatch, int topbot);
  void apply2_trans(float *pwav, float *qpml, float *q2pml, int nbatch, int topbot);

private:
  //init
  void init();

public:
  int n, npml, nbatch;
  float dx, dt, vmax;
  bool limit_slope;
  vector<float> coef, feoc;
};

#endif /* PML_H_ */

