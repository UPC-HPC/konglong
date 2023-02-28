/*
 * Q.h
 *
 *  Created on: Sep 19, 2021
 *      Author: owl
 */

#ifndef SWPRO_LIBWAVEPROP_Q_H_
#define SWPRO_LIBWAVEPROP_Q_H_

#include "Model.h"

class Q {
public:
  Q();
  virtual ~Q();

  static float fitQCoeff(float dt0, float fdom0, float fmax0, bool forWaveEquation = true, float weight0 = -1.0f, int n_try = 10,
                         int n_iter_per_order = 5000);
  static void populateCoeff();
  static void updateCq();

  static void preFitAdjust(); // to implement increasing wl constraint
  static void postFitAdjust(); // convert back to real wl

  static int order;
  static vector<float> coeffs; // coeffs for Q equations
  static float *cq, *wq;
  static float cq0, cqsum;
  static float dt, fmax, fdom, fmin;

  static void mpibcast(int rank);
};
#endif /* SWPRO_LIBWAVEPROP_Q_H_ */
