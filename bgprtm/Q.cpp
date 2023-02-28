/*
 * Q.cpp
 *
 *  Created on: Sep 19, 2021
 *      Author: owl
 */

#include <mpi.h>
#include <iostream>
#include <bits/stdc++.h>
#include "Q.h"
#include "GetPar.h"
#include "libCommon/levmar.h"
#include "libCommon/CommonMath.h"
#include "libCommon/Options.h"
#include "libCommon/Assertion.h"
#include "MpiPrint.h"

int Q::order = 0;
float Q::fmax = 60;
float Q::fdom = 20;
float Q::fmin = 6;
float Q::dt = 0;
float *Q::cq = nullptr;
float *Q::wq = nullptr;
float Q::cq0 = 0;
float Q::cqsum = 0;
vector<float> Q::coeffs;

Q::Q() {
}

Q::~Q() {
}

// sinh(x*wdt)/wdt
template<typename T>
inline T hyper_sinc(T x, float wdt) {
  if(wdt > FLT_EPSILON) return std::sinh(x * wdt) / wdt;
  T xdt2 = x * wdt;
  xdt2 = xdt2 * xdt2;
  return x * (T(1.0) + xdt2 / T(6.0) + xdt2 * xdt2 / T(120.0));
}
// sin(x*wdt)/wdt
template<typename T>
inline T sinc(T x, float wdt) {
  if(wdt > FLT_EPSILON) return std::sin(x * wdt) / wdt;
  T xdt2 = x * wdt;
  xdt2 = xdt2 * xdt2;
  return x * (T(1.0) - xdt2 / T(6.0) + xdt2 * xdt2 / T(120.0));
}

#define QPARAM_OFFSET 10

// error vector, (fit - true) for non-weighted version
void misfit(float *p, float *hx, int np, int nw2, void *args) {
  float *adata = (float*)args;
  int order = (int)nearbyintf(adata[0]);
  int net_value = (int)nearbyintf(adata[1]); // a hack to evaluate net value not error
  float w0dt = adata[2];
  float x0 = adata[3]; // wmin / w0
  float x1 = adata[4]; // wmax / w0
  float weight = adata[5]; // weight on imaginary
  float q_amp = adata[6];
  float q_phase = adata[7];
  int nw = nw2 / 2;
  float dx = (x1 - x0) / (nw - 1);
  float *fl = (np == order + 1) ? adata + QPARAM_OFFSET : p + order + 1;

  float *realx = hx, *imagx = hx + nw;
  for(int iw = 0; iw < nw; iw++) {
    float x = x0 + iw * dx; // w/w0
    float true_r = net_value || q_phase == 0 ? 0 : q_phase * (logf(x) * 2 / FLT_PI);
    float true_i = net_value ? 0 : q_amp;
    realx[iw] = -true_r + p[0]; // real part
    imagx[iw] = -true_i; // imaginary part
    float b = 0, cl = 0;
    for(int ip = 0; ip < order; ip++) {
      cl = p[ip + 1];
      float b0 = fl[ip];
      b += b0;

      complex<float> denom = complex<float>(b, x);
      denom = hyper_sinc(denom, w0dt);
      complex<float> cval = cl / denom;

      realx[iw] += cval.real();
      imagx[iw] += cval.imag();
    }
    if(!net_value) imagx[iw] *= weight;
    // if (net_value > 1) fprintf(stderr, "x=%f, real=%f, imag=%f\n", x, realx[iw], imagx[iw]);
  }
}

void jacob(float *p, float *J, int np, int nw2, void *args) {
  float *adata = (float*)args;
  int order = (int)nearbyintf(adata[0]);
  float w0dt = adata[2];
  float x0 = adata[3]; // wmin / w0
  float x1 = adata[4]; // wmax / w0
  float weight = adata[5]; // weight on imaginary
  float q_amp = adata[6];
  float q_phase = adata[7];
  int nw = nw2 / 2;
  float dx = (x1 - x0) / (nw - 1);
  float *fl = (np == order + 1) ? adata + QPARAM_OFFSET : p + order + 1;

  float *realJ = J + 1, *imagJ = J + nw * np + 1;
  for(int iw = 0, j = 0; iw < nw; iw++, j += np) {
    float x = x0 + iw * dx; // w/w0
    // real part
    J[j] = 1; // p[0]
    float b = 0, cl = 0;
    for(int ip = 0; ip < order; ip++) {
      cl = p[ip + 1];
      float b0 = fl[ip];
      b += b0;

      complex<float> carg = complex<float>(b, x);
      complex<float> cval = 1.0f / hyper_sinc(carg, w0dt);

      // jacob for c_l
      realJ[j + ip] = cval.real();
      imagJ[j + ip] = cval.imag() * weight; // same j since it's technically for iw=[nw-nw2)
      if(np > order + 1) { // jacob for beta_l
        cval = -cval * cval * cl * cosh(carg * w0dt);

#if 1 // b[ip] = sum(fl[i=0 to ip])
        if(ip == 0) {
          realJ[j + order + ip] = cval.real();
          imagJ[j + order + ip] = cval.imag() * weight;
        } else {
          realJ[j + order + ip] = realJ[j + order + ip - 1] + cval.real();
          imagJ[j + order + ip] = imagJ[j + order + ip - 1] + cval.imag() * weight;
        }
#else
        realJ[j + order + ip] = cval.real();
        imagJ[j + order + ip] = cval.imag() * weight;
#endif
      }
    }
  }
}

float costfun(float x[], float lb[], float ub[], int n, vector<float> &args) {
  int nw = 100;
  vector<float> errs(nw * 2);
  misfit(x, &errs[0], n, nw * 2, &args[0]);

  float sum = 0;
  for(int iw = 0; iw < nw; iw++) {
    sum += errs[iw] * errs[iw];
    sum += errs[nw + iw] * errs[nw + iw];
  }

  return sum;
}

#define Q_DEL_WL 0.001f
float Q::fitQCoeff(float dt0, float fdom0, float fmax0, bool forWaveEquation, float weight0, int n_try, int n_iter_per_order) {
  dt = dt0, fdom = fdom0, fmax = fmax0;
  fmin = std::min(fdom / 3, fmax / 10) / 10;

  int order0 = global_pars["q_order"].as<int>(0);
  float q_amp = global_pars["q_amp"].as<float>(1.0);
  float q_phase = global_pars["q_phase"].as<float>(1.0);
  if(order0 > 0) order = order0;
  else if(q_amp < 1) order = 4;
  float weight = 1;
  if(weight0 >= 0) weight = weight0;
  else if(q_amp < 1) weight = 0.1; // weight on imaginary, i.e. AMP

  if(!(MpiPrint::verbose < 2 && (MpiPrint::rank != 0 || MpiPrint::printed1))) fprintf(stderr, "fitQ: order=%d, weight_amp=%f\n", order,
                                                                                      weight), fflush(stderr);

  if(order == 4 && q_amp == 1) coeffs = vector<float> { 2.0832741, -0.040106162, -0.21464862, -1.0239593, -9.7581205, 0.024538578,
      0.18895587, 1.0043846, 6.2038145 };
  else if(order == 4 && q_amp != 1) coeffs = vector<float> { 0.19323404, -0.072588377, -0.28686559, -2.6335149, 6.3146358, 0.06119559,
      0.29910451, 1.7500319, 6.0417738 };
  else if(order == 5) coeffs = vector<float> { -1.8545429, -0.078886382, -0.36275145, -7.9591517, 3.7560601, 27.21925, 0.06355074,
      0.35049143, 2.7232373, 5.004149, 6.9939642 };
  else coeffs = vector<float> { -2.8199008, -0.25147924, -0.11749831, 3.5049336, -11.690312, 9.9806776, 29.893116, 0.16760345, 0.39558345,
      1.9931216, 2.3870831, 6.8417306, 6.9619336 };

  int np = order * 2 + 1;
  assertion((int )coeffs.size() == np, "coeffs.size=%ld does not match order=%d", coeffs.size(), np);

  preFitAdjust();

  float w0_dt = FLT_2_PI * fdom * dt;
  float cq0_fix = 0;
  for(int i = 0; i < order; i++) {
    cq0_fix += coeffs[1 + i] * (1 / hyper_sinc(coeffs[order + 1 + i], w0_dt) - 1 / coeffs[order + 1 + i]);
  }
  coeffs[0] -= cq0_fix;

  vector<float> lb(order * 2 + 1, 1.0f), ub(order * 2 + 1, 1.0f);

  /* the elements in coeffs vector are: cq0, cq1, eq1, cq2, eq2, cq3, eq3, cq4, eq4 */
  for(int i = 0; i < order; i++) {
    ub[i + 1] = 30;
    lb[i + 1] = -30; // cl/w0 values
    ub[order + i + 1] = 12; // wl/w0
    lb[order + i + 1] = Q_DEL_WL;
  }
  ub[0] = 2.5f - cq0_fix;
  lb[0] = -4 - cq0_fix;

  float opts[LM_OPTS_SZ], info[LM_INFO_SZ];
  opts[0] = LM_INIT_MU;
  opts[1] = 1E-15;
  opts[2] = 1E-15;
  opts[3] = 1E-20;
  opts[4] = -LM_DIFF_DELTA;

  // int np <- args[0], float w0_dt = args[2], fmin = args[3], fmax_fit = args[4]
  vector<float> args;
  args.push_back(order);
  args.push_back(0);
  args.push_back(w0_dt);
  args.push_back(fmin / fdom); //       // wmin_fit / w0
  float fit_range = 1.0;
  args.push_back(fit_range * fmax / fdom); // // wmax_fit / w0
  args.push_back(weight);
  args.push_back(q_amp);
  args.push_back(q_phase);
  args.resize(order + QPARAM_OFFSET, 0);
  memcpy(&args[QPARAM_OFFSET], &coeffs[order + 1], sizeof(float) * order); // copy wl parameters to args

  vector<float> coeffs_bak(coeffs), coeffs_min;
  int nw = 100;
  int term_reason = 0;
  vector<float> errs(nw * 2);

  int i_try = 0;
  srand(time(NULL));
  bool do_rand = getBool("qfit_rand", false);
  float e2_min = FLT_MAX;

  int np_linear = order + 1;
  int worksz = LM_BC_DER_WORKSZ(np, nw * 2); //2*n+4*m + n*m + m*m;
  vector<float> work(worksz, 0);
  srand(time(NULL));
  int n_iter = order * n_iter_per_order;
  do {
    if(i_try > 0) {
      for(int i = 0; i < order; i++)
        coeffs[i + order + 1] = Q_DEL_WL + 12.0f * powf(rand() / RAND_MAX, 2);
      preFitAdjust();
    }
    // if(i_try > 0 && term_reason > 3) coeffs = coeffs_bak, args[1] = dt * (1 - 0.5f * random() / RAND_MAX);

    memset(&work[0], 0, sizeof(float) * worksz);
    slevmar_bc_der(misfit, jacob, &coeffs[0], &errs[0], np_linear, nw * 2, &lb[0], &ub[0], NULL, n_iter, opts, info, &work[0], NULL,
                   &args[0]);

    {
      memset(&work[0], 0, sizeof(float) * worksz);
      slevmar_bc_der(misfit, jacob, &coeffs[0], &errs[0], np, nw * 2, &lb[0], &ub[0], NULL, n_iter, opts, info, &work[0], NULL, &args[0]);
    }

    if(info[1] < e2_min) e2_min = info[1], coeffs_min = coeffs;
    // if (do_rand) std::cerr << COptions::floats2str(coeffs, -order) << std::endl;

    if(!(MpiPrint::verbose < 2 && (MpiPrint::rank != 0 || MpiPrint::printed1))) {
      fprintf(stderr, "LEVMAR: e^2=%g->%g, iterations=%d, term reason: %d, lineqns: %d\n", info[0], info[1], (int)nearbyint(info[5]),
              (term_reason = (int)nearbyint(info[6])), (int)nearbyint(info[9]));
      std::cerr << "LEVMAR: [" << COptions::floats2str(coeffs) << "]" << std::endl << std::flush;
    }
  } while(++i_try < n_try && (term_reason >= 3 || do_rand));
  coeffs = coeffs_min;
  assertion(term_reason <= 3, "Check the Q-fitting coefficients!\n  dt=%.8e (2f0*dt=%f, 2fmax*dt=%f), fmin=%f, fdom=%f, fmax=%f\n", dt,
            2 * fdom * dt, 2 * fmax * dt, fmin, fdom, fmax);

  if(forWaveEquation) {
    postFitAdjust();

    cq = &coeffs[1];
    wq = &coeffs[order + 1];
    for(int i = 0; i < order; i++) {
      float wq_norm = wq[i] * w0_dt;
      cq[i] *= expf(-wq_norm) * 2 * w0_dt;
      wq[i] = expf(-2 * wq_norm);
    }

    if(!(MpiPrint::verbose < 2 && (MpiPrint::rank != 0 || MpiPrint::printed1))) std::cout << "Q::coeffs = [" << COptions::floats2str(coeffs)
        << "]" << std::endl;
    global_pars["_q_coeffs"] = COptions::floats2str(coeffs); // yaml's float precision is not enough
    updateCq();
  }

  return e2_min;
}

void Q::populateCoeff() {
  if(order == 0) return;

  if(global_pars["_q_coeffs"]) {
    string str_coeffs = global_pars["_q_coeffs"].as<string>();
    coeffs = COptions::str2floats(str_coeffs);
    updateCq();
  } else {
    float maxfreq = global_pars["maxFreq"].as<float>();
    float qrefFreq = global_pars["qRefFreq"].as<float>(maxfreq / 3);
    float dt = global_pars["_dt_prop"].as<float>();
    fitQCoeff(dt, qrefFreq, maxfreq);
  }
}

void Q::updateCq() {
  cq0 = coeffs[0];
  cq = &coeffs[1];
  wq = &coeffs[order + 1];
  cqsum = 0;
  for(int i = 0; i < order; i++) {
    cqsum += cq[i];
  }
}

void Q::preFitAdjust() {
  std::sort(&coeffs[order + 1], &coeffs[order * 2 + 1]);
  for(int i = order - 1; i > 0; i--)  // make sure parameters are increasing
    coeffs[order + i + 1] = max(Q_DEL_WL, coeffs[order + i + 1] - coeffs[order + i]);
}

void Q::postFitAdjust() {
  for(int i = 1; i < order; i++)
    coeffs[order + i + 1] += coeffs[order + i];
}

void Q::mpibcast(int rank) {
  MPI_Bcast(&Q::order, 1, MPI_INT, 0, MPI_COMM_WORLD);
  int np = order * 2 + 1;
#if 0
  MPI_Bcast(&Q::fmax, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&Q::fdom, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&Q::dt, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
#endif

  if(rank > 0) coeffs.resize(np);
  MPI_Bcast(&Q::coeffs[0], np, MPI_FLOAT, 0, MPI_COMM_WORLD);

  if(rank > 0) {
    global_pars["_q_coeffs"] = COptions::floats2str(coeffs);
    updateCq();
  }
}

#if (TEST > 0)
// mpic++ -DTEST=1 -g -O3 -std=c++11 Q.cpp ${HOME}/local/lib.Ubuntu/liblevmar.a -L bin -lWaveProp -lyaml-cpp -llapack -lblas -lCommon -fopenmp -o bin/a.out
Node global_pars;
int main(int argc, char **argv) {
  if(argc <= 3) {
    fprintf(stderr, "%s q_amp order weight\n", argv[0]);
    exit(-1);
  }

  float q_amp = atof(argv[1]);
  int q_order = atoi(argv[2]);
  float weight = atof(argv[3]);
  float q_phase = 1;
  global_pars["q_amp"] = q_amp;
  global_pars["q_order"] = q_order;
  global_pars["qfit_rand"] = true;

  int n_try = 400, n_iter_per_order = 5000;
  float dt = 0; // 2e-3
  float q_fref = 13, q_fmax = 40;
  float e2_min = Q::fitQCoeff(dt, q_fref, q_fmax, false, weight, n_try, n_iter_per_order);

  int nw = 201;
  vector<float> vals(nw * 2), vals0(nw * 2);
  vector<float> args;
  args.push_back(Q::order);
  args.push_back(1); // evaluate than err
  args.push_back(FLT_2_PI * Q::fdom * Q::dt);
  args.push_back(Q::fmin / Q::fdom); //       // wmin_fit / w0
  args.push_back(Q::fmax / Q::fdom); // // wmax_fit / w0
  args.push_back(1.0); // weight on imaginary
  args.push_back(q_amp);
  args.push_back(q_phase);
  args.resize(Q::order + QPARAM_OFFSET, 0);
  misfit(&Q::coeffs[0], &vals[0], Q::order * 2 + 1, nw * 2, &args[0]);

  Q::postFitAdjust();
  std::cerr << "Final: " << e2_min << " [" << COptions::floats2str(Q::coeffs) << "]" << std::endl;

  if(Q::dt > 0) {
    Q::fitQCoeff(0, q_fref, q_fmax, false, weight, n_try, n_iter_per_order);
    misfit(&Q::coeffs[0], &vals0[0], Q::order * 2 + 1, nw * 2, &args[0]); // mistfit of static best coeff (i.e. dt=0) vs non-zero dt
  } else vals0 = vals;

  float df = (Q::fmax - Q::fmin) / (nw - 1); // need to be consistent with args[2], args[3]
  printf("# freq Real Real_fit Real_fit0 Imag Imag_fit Imag_fit0\n");
  for(int i = 0; i < nw; i++) {
    float f = Q::fmin + i * df;
    float vr = q_phase * (logf(f / Q::fdom) * 2 / FLT_PI), vi = q_amp;
    printf("%f %g %g %g %g %g %g\n", f, vr, vals[i], vals0[i], vi, vals[nw + i], vals0[nw + i]);
  }

  return 0;
}
#endif // TEST

