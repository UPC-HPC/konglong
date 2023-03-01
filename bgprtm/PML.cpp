/*
 * PML.cpp
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "GetPar.h"
#include "PML.h"
#include "MpiPrint.h"
using MpiPrint::print1m;

PML::PML(int n, int npml, int nbatch, float dx, float dt, float vmax, bool limit_slope) :
    n(n), npml(npml), nbatch(nbatch), dx(dx), dt(dt), vmax(vmax), limit_slope(limit_slope) {
  init();
}

PML::~PML() {
}

void PML::init() {
  float pml_eps = global_pars["pml_eps"].as<float>(1e-3);
  int verbose_pml = global_pars["verbose_pml"].as<int>(0);
  assertion(npml > 1, "npml(%d) must > 1");

  coef.resize(npml), feoc.resize(npml);
  float beta = dx / (dt * vmax);
  float dg1 = logf(1 / pml_eps) / (npml * beta);
  if(!limit_slope) {
    for(int i = 0; i < npml; i++) {
      float temp = float(npml - i) / npml;
      coef[i] = 1.5f / 2 * temp * temp * dg1; // gamma*dt/2
      feoc[npml - 1 - i] = coef[i];
    }
  } else {
    float step = 1.002;
    float eps = pml_eps;
    float damping = 1.0f;
    do {
      float ghalf_old = 0;
      float sum_g = 0;
      for(int i = 0; i < npml; i++) {
        float exponent = min(85.0f, 2 * beta * sum_g); // ln(1e37) is around 85
        damping = exp(-exponent);
        feoc[i] = ghalf_old + 0.5f * min(eps / damping, dg1);
        sum_g += 2 * feoc[i];
        coef[npml - 1 - i] = feoc[i];
        eps *= step;
      }
    } while(damping > eps / step);
    print1m("PML: npml=%d, eps=%g, damping=%g\n", npml, eps / step, damping);
  }
  if(verbose_pml) {
    printf("### pml coef \n");
    for(int i = 0; i < npml; i++)
      printf("  coef[%d]=%g \n", i, coef[i]);
  }
}

void PML::apply_single(vector<float> &coef, float scaler, float *p, float *q, int symmetry) {
  for(int i = 0; i < npml; i++) {
    float g = coef[i] * scaler;
    float qi = (1.0f - g) / (1.0f + g) * q[i] - g * 2.0f / (1.0f - g * g) * p[i];
    q[i] = qi;
    p[i] = qi + p[i] / (1.0f - g);
  }
  if(symmetry != 0) p[npml] = p[npml - 1] * symmetry;
}

void PML::apply2_single(vector<float> &coef, float *p, float *q, float *q2, int symmetry) {
  for(int i = 0; i < npml; i++) {
    float g = coef[i];
    float pi = p[i];
    float qi = (1.0f - g) / (1.0f + g) * q[i] - g * 2.0f / (1.0f - g * g) * pi;
    q[i] = qi;
    pi = qi + pi / (1.0f - g);

    qi = (1.0f - g) / (1.0f + g) * q2[i] - g * 2.0f / (1.0f - g * g) * pi; // 2nd round, q2i here
    q2[i] = qi;
    p[i] = qi + pi / (1.0f - g);
  }
  if(symmetry != 0) p[npml] = p[npml - 1] * symmetry;
}

// topbot also means left_right, front_back ..., -1: top, 1: bot, 2: bot symmetric, 3: bot anti-symmetric
void PML::apply(float *pwav, float *qpml, int nbatch, int topbot) {
  float *p = pwav;
  float *q = qpml;

  int symmetry = (topbot == 2) ? 1 : (topbot == 3) ? -1 : 0;
  if(topbot == -1) {
    for(int j = 0; j < nbatch; j++, p += n, q += npml)
      apply_single(coef, 1.0f, p, q, symmetry);
  } else if(topbot > 0) {
    p += n - npml; // flag==1, thus p[npml-1] is p_old[n-1]
    if(topbot == 2 || topbot == 3) p--; // p[npml-1] is p_old[n-2]
    for(int j = 0; j < nbatch; j++, p += n, q += npml)
      apply_single(feoc, 1.0f, p, q, symmetry);
  }
}

void PML::apply2(float *pwav, float *qpml, float *q2pml, int nbatch, int topbot) {
  float *p = pwav;
  float *q = qpml;
  float *q2 = q2pml;

  int symmetry = (topbot == 2) ? 1 : (topbot == 3) ? -1 : 0;
  if(topbot == -1) {
    for(int j = 0; j < nbatch; j++, p += n, q += npml, q2 += npml)
      apply2_single(coef, p, q, q2, symmetry);
  } else if(topbot > 0) {
    p += n - npml; // flag==1, thus p[npml-1] is p_old[n-1]
    if(topbot == 2 || topbot == 3) p--; // p[npml-1] is p_old[n-2]
    for(int j = 0; j < nbatch; j++, p += n, q += npml, q2 += npml)
      apply2_single(feoc, p, q, q2, symmetry);
  }
}

void PML::apply_single_trans(float g, float *p, float *q, int nz) {
  for(int j = 0; j < nz; j++) {
    float pj = p[j];
    float qj = (1.0f - g) / (1.0f + g) * q[j] - g * 2.0f / (1.0f - g * g) * pj;
    q[j] = qj;
    p[j] = qj + pj / (1.0f - g);
  }
}

void PML::apply2_single_trans(float g, float *p, float *q, float *q2, int nz) {
  for(int j = 0; j < nz; j++) {
    float pj = p[j];
    float qj = (1.0f - g) / (1.0f + g) * q[j] - g * 2.0f / (1.0f - g * g) * pj;
    q[j] = qj;
    pj = qj + pj / (1.0f - g);

    qj = (1.0f - g) / (1.0f + g) * q2[j] - g * 2.0f / (1.0f - g * g) * pj; // 2nd round, q2j here
    q2[j] = qj;
    p[j] = qj + pj / (1.0f - g);
  }
}

// -1: top, 1: bot, 2: bot symmetric, 3: bot anti-symmetric
// fastest dim is nz, e.g., nz is real nz, while n/npml is along nx
void PML::apply_trans(float *pwav, float *qpml, int nz, int topbot) {
  float *p = pwav;
  float *q = qpml;
  int symmetry = (topbot == 2) ? 1 : (topbot == 3) ? -1 : 0;

  if(topbot > 0) p += nz * (n - npml);
  if(topbot == 2 || topbot == 3) p -= nz;
  for(int i = 0; i < npml; i++, p += n, q += npml) {
    float g = (topbot == -1) ? coef[i] : feoc[i];
    apply_single_trans(g, p, q, nz);
  }
  if(symmetry != 0) {
    float *pp = p - nz;
    for(int j = 0; j < nz; j++)
      p[j] = pp[j] * symmetry;
  }
}

