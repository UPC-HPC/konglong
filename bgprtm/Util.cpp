/*
 * Util.cpp
 *
 *  Created on: Jul 25, 2015
 *      Author: tiger
 */

#include <xmmintrin.h>

#include "Util.h"
#include <zlib.h>
#include "libCommon/Assertion.h"
#include "libCommon/CommonMath.h"
#include <jseisIO/jseisUtil.h>
using jsIO::jseisUtil;

Util::Util() {

}

Util::~Util() {

}

void Util::transpose(int nx, int ny, int dimIn, int dimOut, float *in, float *out) {
  int nx4, ny4;
  int ix, iy;
  size_t offin, offout;
  float *pin, *pout;
  __m128 v1, v2, v3, v4;

  nx4 = (nx / SSEsize) * SSEsize;
  ny4 = (ny / SSEsize) * SSEsize;

  offout = 0;
  // SSEsize x SSEsize blocks
  for(ix = 0; ix < nx4; ix += SSEsize) {
    pin = in + ix;
    pout = out + offout;
    offin = 0;

    for(iy = 0; iy < ny4; iy += SSEsize) {
      v1 = _mm_loadu_ps(pin + offin);
      offin += dimIn;
      v2 = _mm_loadu_ps(pin + offin);
      offin += dimIn;
      v3 = _mm_loadu_ps(pin + offin);
      offin += dimIn;
      v4 = _mm_loadu_ps(pin + offin);
      offin += dimIn;
      _MM_TRANSPOSE4_PS(v1, v2, v3, v4);
      _mm_storeu_ps(pout + iy, v1);
      _mm_storeu_ps(pout + dimOut + iy, v2);
      _mm_storeu_ps(pout + dimOut + dimOut + dimOut + iy, v4);
    }

    for(iy = ny4; iy < ny; iy++) {
      pout[iy] = pin[offin];
      pout[iy + dimOut] = pin[offin + 1];
      pout[iy + 2 * dimOut] = pin[offin + 2];
      pout[iy + 3 * dimOut] = pin[offin + 3];
      offin += dimIn;
    }
    offout += SSEsize * dimOut;
  }

  for(ix = nx4; ix < nx; ix++) {
    pin = in + ix;
    pout = out + (size_t)ix * (size_t)dimOut;
    offin = 0;
    for(iy = 0; iy < ny; iy++) {
      pout[iy] = pin[offin];
      offin += dimIn;
    }
  }
}

void Util::transposeAndPad(int nx, int ny, int dimIn, int dimOut, float *in, float *out) {
  int nx4, ny4;
  int ix, iy, offin, offout;
  float *sp, *dp;
  __m128 v1, v2, v3, v4;

  nx4 = (nx / 4) * 4;
  ny4 = (ny / 4) * 4;

  offout = 0;

  for(ix = 0; ix < nx4; ix += 4) {
    sp = in + ix;
    dp = out + offout;
    offin = 0;

    for(iy = 0; iy < ny4; iy += 4) {

      v1 = _mm_loadu_ps(sp + offin);
      offin += dimIn;
      v2 = _mm_loadu_ps(sp + offin);
      offin += dimIn;
      v3 = _mm_loadu_ps(sp + offin);
      offin += dimIn;
      v4 = _mm_loadu_ps(sp + offin);
      offin += dimIn;
      _MM_TRANSPOSE4_PS(v1, v2, v3, v4);
      _mm_storeu_ps(dp + iy, v1);
      _mm_storeu_ps(dp + dimOut + iy, v2);
      _mm_storeu_ps(dp + dimOut + dimOut + iy, v3);
      _mm_storeu_ps(dp + dimOut + dimOut + dimOut + iy, v4);
    }

    for(iy = ny4; iy < ny; iy++) {
      dp[iy] = sp[offin];
      dp[iy + dimOut] = sp[offin + 1];
      dp[iy + 2 * dimOut] = sp[offin + 2];
      dp[iy + 3 * dimOut] = sp[offin + 3];
      offin += dimIn;
    }

    for(iy = ny; iy < dimOut; iy++) {
      dp[iy] = 0.0f;
      dp[iy + dimOut] = 0.0f;
      dp[iy + 2 * dimOut] = 0.0f;
      dp[iy + 3 * dimOut] = 0.0f;
    }
    offout += 4 * dimOut;
  }

  for(ix = nx4; ix < nx; ix++) {
    sp = in + ix;
    dp = out + (size_t)ix * (size_t)dimOut;
    offin = 0;
    for(iy = 0; iy < ny; iy++) {
      dp[iy] = sp[offin];
      offin += dimIn;
    }
    for(iy = ny; iy < dimOut; iy++)
      dp[iy] = 0.0f;
  }
}

void Util::transposeAndAdd(int nx, int ny, int dimIn, int dimOut, float *in, float *out) {
  transposeAndAdd(nx, ny, dimIn, dimOut, in, out, nullptr, nullptr);
}

void Util::transposeAndAdd(int nx, int ny, int dimIn, int dimOut, float *in, float *out, float *velSlice, float *rhoSlice) {
  int nx4, ny4;
  int ix, iy, offin, offout;
  float *sp, *dp, *velSeg, *rhoSeg;
  __m128 v1, v2, v3, v4;

  nx4 = (nx / 4) * 4;
  ny4 = (ny / 4) * 4;

  offout = 0;
  if(velSlice && rhoSlice) {
    for(ix = 0; ix < nx4; ix += 4) {
      sp = in + ix;
      dp = out + offout;
      velSeg = velSlice + offout;
      rhoSeg = rhoSlice + offout;
      offin = 0;

      for(iy = 0; iy < ny4; iy += 4) {

        v1 = _mm_loadu_ps(sp + offin);
        offin += dimIn;
        v2 = _mm_loadu_ps(sp + offin);
        offin += dimIn;
        v3 = _mm_loadu_ps(sp + offin);
        offin += dimIn;
        v4 = _mm_loadu_ps(sp + offin);
        offin += dimIn;
        _MM_TRANSPOSE4_PS(v1, v2, v3, v4);
        _mm_storeu_ps(dp + iy, _mm_sub_ps(_mm_loadu_ps(dp + iy), _mm_mul_ps(_mm_loadu_ps(velSeg + iy), _mm_mul_ps(_mm_loadu_ps(rhoSeg + iy), v1))));
        _mm_storeu_ps(dp + dimOut + iy, _mm_sub_ps(_mm_loadu_ps(dp + dimOut + iy), _mm_mul_ps(_mm_loadu_ps(velSeg + dimOut + iy), _mm_mul_ps(_mm_loadu_ps(rhoSeg + dimOut + iy), v2))));
        _mm_storeu_ps(dp + dimOut + dimOut + iy, _mm_sub_ps(_mm_loadu_ps(dp + dimOut + dimOut + iy), _mm_mul_ps(_mm_loadu_ps(velSeg + dimOut + dimOut + iy), _mm_mul_ps(_mm_loadu_ps(rhoSeg + dimOut + dimOut + iy), v3))));
        _mm_storeu_ps(dp + dimOut + dimOut + dimOut + iy, _mm_sub_ps(_mm_loadu_ps(dp + dimOut + dimOut + dimOut + iy), _mm_mul_ps(_mm_loadu_ps(velSeg + dimOut + dimOut + dimOut + iy), _mm_mul_ps(_mm_loadu_ps(rhoSeg + dimOut + dimOut + dimOut + iy), v4))));
      }

      for(iy = ny4; iy < ny; iy++) {
        dp[iy] -= sp[offin] * velSeg[iy] * rhoSeg[iy];
        dp[iy + dimOut] -= sp[offin + 1] * velSeg[iy + dimOut] * rhoSeg[iy + dimOut];
        dp[iy + 2 * dimOut] -= sp[offin + 2] * velSeg[iy + 2 * dimOut] * rhoSeg[iy + 2 * dimOut];
        dp[iy + 3 * dimOut] -= sp[offin + 3] * velSeg[iy + 3 * dimOut] * rhoSeg[iy + 3 * dimOut];
        offin += dimIn;
      }
      offout += 4 * dimOut;
    }

    for(ix = nx4; ix < nx; ix++) {
      sp = in + ix;
      dp = out + (size_t)ix * (size_t)dimOut;
      velSeg = velSlice + (size_t)ix * (size_t)dimOut;
      rhoSeg = rhoSlice + (size_t)ix * (size_t)dimOut;
      offin = 0;
      for(iy = 0; iy < ny; iy++) {
        dp[iy] -= sp[offin]* velSeg[iy] * rhoSeg[iy];
        offin += dimIn;
      }
    }
  }
  else if(velSlice) {
    for(ix = 0; ix < nx4; ix += 4) {
      sp = in + ix;
      dp = out + offout;
      velSeg = velSlice + offout;
      offin = 0;

      for(iy = 0; iy < ny4; iy += 4) {

        v1 = _mm_loadu_ps(sp + offin);
        offin += dimIn;
        v2 = _mm_loadu_ps(sp + offin);
        offin += dimIn;
        v3 = _mm_loadu_ps(sp + offin);
        offin += dimIn;
        v4 = _mm_loadu_ps(sp + offin);
        offin += dimIn;
        _MM_TRANSPOSE4_PS(v1, v2, v3, v4);
        _mm_storeu_ps(dp + iy, _mm_sub_ps(_mm_loadu_ps(dp + iy), _mm_mul_ps(_mm_loadu_ps(velSeg + iy), v1)));
        _mm_storeu_ps(dp + dimOut + iy, _mm_sub_ps(_mm_loadu_ps(dp + dimOut + iy), _mm_mul_ps(_mm_loadu_ps(velSeg + dimOut + iy), v2)));
        _mm_storeu_ps(dp + dimOut + dimOut + iy, _mm_sub_ps(_mm_loadu_ps(dp + dimOut + dimOut + iy), _mm_mul_ps(_mm_loadu_ps(velSeg + dimOut + dimOut + iy), v3)));
        _mm_storeu_ps(dp + dimOut + dimOut + dimOut + iy, _mm_sub_ps(_mm_loadu_ps(dp + dimOut + dimOut + dimOut + iy), _mm_mul_ps(_mm_loadu_ps(velSeg + dimOut + dimOut + dimOut + iy), v4)));
      }

      for(iy = ny4; iy < ny; iy++) {
        dp[iy] -= sp[offin] * velSeg[iy];
        dp[iy + dimOut] -= sp[offin + 1] * velSeg[iy + dimOut];
        dp[iy + 2 * dimOut] -= sp[offin + 2] * velSeg[iy + 2 * dimOut];
        dp[iy + 3 * dimOut] -= sp[offin + 3] * velSeg[iy + 3 * dimOut];
        offin += dimIn;
      }
      offout += 4 * dimOut;
    }

    for(ix = nx4; ix < nx; ix++) {
      sp = in + ix;
      dp = out + (size_t)ix * (size_t)dimOut;
      velSeg = velSlice + (size_t)ix * (size_t)dimOut;
      offin = 0;
      for(iy = 0; iy < ny; iy++) {
        dp[iy] -= sp[offin]* velSeg[iy];
        offin += dimIn;
      }
    }
  }
  else {//it is impossible that velSlice is null while rhoSlice is not null
    for(ix = 0; ix < nx4; ix += 4) {
      sp = in + ix;
      dp = out + offout;
      offin = 0;

      for(iy = 0; iy < ny4; iy += 4) {

        v1 = _mm_loadu_ps(sp + offin);
        offin += dimIn;
        v2 = _mm_loadu_ps(sp + offin);
        offin += dimIn;
        v3 = _mm_loadu_ps(sp + offin);
        offin += dimIn;
        v4 = _mm_loadu_ps(sp + offin);
        offin += dimIn;
        _MM_TRANSPOSE4_PS(v1, v2, v3, v4);
        _mm_storeu_ps(dp + iy, _mm_add_ps(_mm_loadu_ps(dp + iy), v1));
        //IMPORTANT: HERE it is add to keep its original function when the last two arguments are nullptr.
        //In the other cases it is subtract! by wolf
        _mm_storeu_ps(dp + dimOut + iy, _mm_add_ps(_mm_loadu_ps(dp + dimOut + iy), v2));
        _mm_storeu_ps(dp + dimOut + dimOut + iy, _mm_add_ps(_mm_loadu_ps(dp + dimOut + dimOut + iy), v3));
        _mm_storeu_ps(dp + dimOut + dimOut + dimOut + iy, _mm_add_ps(_mm_loadu_ps(dp + dimOut + dimOut + dimOut + iy), v4));
      }

      for(iy = ny4; iy < ny; iy++) {
        dp[iy] += sp[offin];
        dp[iy + dimOut] += sp[offin + 1];
        dp[iy + 2 * dimOut] += sp[offin + 2];
        dp[iy + 3 * dimOut] += sp[offin + 3];
        offin += dimIn;
      }
      offout += 4 * dimOut;
    }

    for(ix = nx4; ix < nx; ix++) {
      sp = in + ix;
      dp = out + (size_t)ix * (size_t)dimOut;
      offin = 0;
      for(iy = 0; iy < ny; iy++) {
        dp[iy] += sp[offin];
        offin += dimIn;
      }
    }
  }
}

float Util::asumf(float *data, int n1, int nr) {
  float sum = 0.0f;
#pragma omp parallel for reduction(+:sum)
  for(int ir = 0; ir < nr; ir++) {
    for(int i1 = 0; i1 < n1; i1++) {
      sum += fabsf(data[(size_t)ir * n1 + i1]);
#if CHECK_NAN
      if(isnanf(data[(sire_t)ir * n1 + i1])) printf("   NaN detected: ir=%d, i1=%d\n", ir, i1);
#endif
    }
  }
  return sum;
}

float Util::print_mem_crc(float *data, int n1, int nr, const char *sid, int skipzero) {
  float ret = asumf(data, n1, nr);
  if(!skipzero || ret != 0) printf("crc(%s)=%0lX, sumabs=%g\n", sid,
                                   crc32(crc32(0L, Z_NULL, 0), (const unsigned char*)data, sizeof(float) * nr * n1), ret), fflush(stdout);
    return ret;
}

// when nlimit != NULL, coeff[*nlimit] is assigned, with ix0 returned and *nlimit updated.
// when nlimit == NULL, coeff[n] is assigned, with ix0 returned, but the non-zero elements count 2*m is not returned
int Util::CompactOrmsbySpreadCoeff(int n, float *coeff0, float x, float k1, float k2, int *nlimit, bool do_dipole, bool debug) {
  assertion(k1 <= k2 && k1 > 0. && k2 <= 0.5, "k1,k2 need to be in [0,0.5]. Also k1(%f) <= k2(%f) required!", k1, k2);
  if(n == 1 || (nlimit != NULL && *nlimit < 2)) {
    assertion(!do_dipole, "Length=1 does not support dipole SpreadCoeff!");
    coeff0[0] = 1.0f;
    if(nlimit != NULL) *nlimit = 1;
    if(debug) {
      std::cout << " n " << n << " nlimit " << nlimit << " x " << x << std::endl;
    }
    int offset = int(nearbyintf(x));
    if(n == 1) assertion(offset == 0, "x=%f should be 0 for n=1(1D/2D)! Try to correct the coordinates of source/receiver!");
    return offset;
  }

  bool use_sinc = (k1 == k2) || do_dipole;
  float a1 = k1 * FLT_PI;
  float a2 = k2 * FLT_PI;
  if(use_sinc) a1 *= 2;
  float sinc1, sinc2;

  int isgn = 1;
  float amp = 0.0;

  int ix = int(floorf(x));
  int m = std::min(ix, n - 1 - ix);
  if(nlimit != NULL) {
    m = std::min(m, *nlimit / 2);
    *nlimit = 2 * m; // update nlimit
  } else {
    memset(coeff0, 0, sizeof(float) * n);
  }

  int ix0 = ix - m + 1;
  if(debug) {
    std::cout << "ix " << ix << " m " << m << std::endl;
    std::cout << "x " << x << " n " << n << " nlimit " << nlimit << std::endl;
    std::cout << "ix0 " << ix0 << std::endl;
  }
  float *__restrict coeff = (nlimit != NULL) ? coeff0 - ix0 : coeff0; // shift coeff for nlimit case

  int ntaper = m / 2;
  int i0 = ix - m + 1, i1 = ix + m; // i1 inclusive
  float halfwin = max(i1 - x, x - i0);
  for(int i = i0; i < i1 + 1; i++) {
    float shift = i - x;
    float b1 = shift * a1, b2;
    sinc1 = do_dipole ? libCommon::dipole_unnorm(b1) : libCommon::sincf_unnorm(b1);
    if(!use_sinc) {
      b2 = shift * a2;
      sinc2 = libCommon::sincf_unnorm(b2);
      float val = (a2 * a2 * sinc2 * sinc2 - a1 * a1 * sinc1 * sinc1) / FLT_PI / (a2 - a1);
      int i_edge = min(i - i0, i1 - i);
      if(i_edge < ntaper && m > 1) val *= 0.5f * (1 - cosf(FLT_PI * i_edge / ntaper));
      coeff[i] = val;
    } else {
      float val = sinc1 * 2 * k1;
      if(m > 1) val *= libCommon::blackman(FLT_PI * (halfwin - x + i) / halfwin);
      coeff[i] = val;
    }
#if 0
    amp = amp + coeff[i] * isgn;
    isgn = -isgn;
#endif
  }
#if 0
  jseisUtil::save_zxy("/tmp/qc_kspread.js", coeff+i0, i1-i0+1, 1, 1);
  exit(0);
#endif

#if 0 // prefer to have dc residue than edge non-zero
  amp = amp / (2 * m);
  isgn = 1;
  for(int i = i0; i < i1 + 1; i++) {
    coeff[i] = coeff[i] - amp * isgn;
    isgn = -isgn;
    //printf("coef=%f , %d, %f, %f, %d, %d  \n",coeff[ii],ii,x, amp, i0, n);
  }
#endif

  return ix0;
}

/*
 void Util::CompactOrmsbySpreadCoeff2(int i0, int n, float* coeff, float x, float k1, float k2) {
 if (n == 1) {
 coeff[0] = 1.0f;
 return;
 }

 float pi = 3.14159265358987932;
 float a1 = k1 * pi;
 float a2 = k2 * pi;
 float sinc1, sinc2;

 int isgn = 1;
 float amp = 0.0;

 if (k1 <= 0. || k1 > 0.5 || k2 <= 0. || k2 > 0.5 || k1 >= k2) {
 printf("Error in Util::CompactOrmsbySpreadCoeff!");
 exit(1);
 }
 int m=MIN(int(x-i0), int(i0+n-x));
 for (int i = i0; i < n; i++) {
 coeff[i] = 0.0;
 }

 int ix = int(x);

 for (int i = ix-m; i < ix+m; i++) {
 int ii = i - ix + m;
 float shift = float(i) - x;
 float b1 = shift * a1;
 float b2 = shift * a2;
 if (fabs(shift) < 0.001) {
 sinc1 = 1. - b1 * b1 / 6.;
 sinc2 = 1. - b2 * b2 / 6.;
 } else {
 sinc1 = sin(b1) / b1;
 sinc2 = sin(b2) / b2;
 }
 coeff[ii] = (a2 * a2 * sinc2 * sinc2 - a1 * a1 * sinc1 * sinc1) / pi / (a2 - a1);
 amp = amp + coeff[ii] * isgn;
 isgn = -isgn;
 }

 amp = amp / (2*m);
 isgn = 1;
 for (int i = ix-m; i < ix+m; i++) {
 int ii = i - ix + m;
 coeff[ii] = coeff[ii] - amp * isgn;
 isgn = -isgn;
 //printf("coef=%f , %d, %f, %f  \n",coeff[ii],ii,x, amp);
 }
 //exit(0);
 }
 */

void Util::CompactOrmsbySpreadCoeff_backup(int i0, int n, float *coeff, float x, float k1, float k2) {
  if(n == 1) {
    coeff[0] = 1.0f;
    return;
  }

  float pi = 3.14159265358987932;
  float a1 = k1 * pi;
  float a2 = k2 * pi;
  float sinc1, sinc2;

  int isgn = 1;
  float amp = 0.0;

  if(k1 <= 0. || k1 > 0.5 || k2 <= 0. || k2 > 0.5 || k1 >= k2) {
    printf("Error in Util::CompactOrmsbySpreadCoeff!");
    exit(-1);
  }

  for(int i = i0; i < i0 + n; i++) {
    int ii = i - i0;
    float shift = float(i) - x;
    float b1 = shift * a1;
    float b2 = shift * a2;
    if(fabs(shift) < 0.001) {
      sinc1 = 1. - b1 * b1 / 6.;
      sinc2 = 1. - b2 * b2 / 6.;
    } else {
      sinc1 = sin(b1) / b1;
      sinc2 = sin(b2) / b2;
    }
    coeff[ii] = (a2 * a2 * sinc2 * sinc2 - a1 * a1 * sinc1 * sinc1) / pi / (a2 - a1);
    amp = amp + coeff[ii] * isgn;
    isgn = -isgn;
  }

  amp = amp / n;
  isgn = 1;
  for(int i = 0; i < n; i++) {
    coeff[i] = coeff[i] - amp * isgn;
    isgn = -isgn;
    //printf("coef=%f , %d, %f, %f  \n",coeff[i],i,x, amp);
  }
  exit(-1);
}

void Util::print_vector(vector<float> v, int n) {
  assert((size_t )n <= v.size());
  printf("[ ");
  for(int i = 0; i < n; i++)
    printf("%f ", v[i]);
  printf("]\n");
}

// pad vector with the value of the last element
void Util::pad_vector(vector<float> &v, int n) {
  size_t l = v.size();
  if(l > 0 && (size_t)n > l) v.resize(n, v[l - 1]);
}

void Util::pad_vector(vector<float> &v, int n, float val) {
  size_t l = v.size();
  if((size_t)n > l) v.resize(n, val);
}

