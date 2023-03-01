/*
 * Util.h
 *
 *  Created on: Jul 25, 2015
 *      Author: tiger
 */

#ifndef UTIL_H_
#define UTIL_H_

#include <math.h>
#include <complex>
#include <vector>

#include "libFFTV/baseinc.h"
#include "libCommon/Utl.hpp"

#ifndef SSEsize
#define SSEsize 4
#endif

#ifndef PI
#define PI 3.1415926535897932
#endif

#ifndef PI_2_degree
#define PI_2_degree 57.29577951308232157827
#endif

#ifndef degree_2_PI
#define degree_2_PI 0.01745329251994329555
#endif

#ifndef MIN
#define MIN(a,b) ((a) > (b) ? (b) : (a))
#endif

#ifndef MAX
#define MAX(a,b) ((a) < (b) ? (b) : (a))
#endif

#ifndef ABS
#define ABS(a) ((a) > 0 ? (a) : (-(a)))
#endif

#ifndef SIGN
#define SIGN(a) ((a) > 0 ? 1 : -1)
#endif

#ifndef Sign
#define Sign(a) ((a)? SIGN(a) : 0)
#endif

typedef std::complex<float> Complex;
using std::vector;

class Util {
public:
  /*
   * constructor
   */
  Util();

  /*
   * distructor
   */
  virtual ~Util();

  //interface for calling transpose
  static void transpose(int nx, int ny, int dimIn, int dimOut, float *in, float *out);

  static void transposeAndPad(int nx, int ny, int dimIn, int dimOut, float *in, float *out);

  static void transposeAndAdd(int nx, int ny, int dimIn, int dimOut, float *in, float *out);

  static void transposeAndAdd(int nx, int ny, int dimIn, int dimOut, float *in, float *out, float *velSlice, float *rhoSlice);
  //added last two arguments by wolf
  static float asumf(float *data, int n1, int nr);

  static float print_mem_crc(float *data, int n1, int nr, const char *sid = "", int skipzero = 0);

  // returns the offset
  static int CompactOrmsbySpreadCoeff(int n, float *mycoeff, float x, float k1, float k2, int *nlimit = NULL, bool do_dipole = false,
      bool debug = false);
  static int CompactOrmsbySpreadCoeff(int n, vector<float> &mycoeff, float x, float k1, float k2, int *nlimit = NULL,
      bool do_dipole = false, bool debug = false) {
    return CompactOrmsbySpreadCoeff(n, &mycoeff[0], x, k1, k2, nlimit, do_dipole, debug);
  }

  static void CompactOrmsbySpreadCoeff_backup(int i0, int n, float *mycoeff, float x, float k1, float k2);

  static void print_vector(vector<float> v, int n);

  static void pad_vector(vector<float> &v, int n);
  static void pad_vector(vector<float> &v, int n, float val);
};

#endif /* UTIL_H_ */

