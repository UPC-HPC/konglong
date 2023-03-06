#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#include "WaveFieldCompress.h"
#include "GetPar.h"
#include "libCommon/CommonMath.h"

int WaveFieldCompress::compression = -1;

WaveFieldCompress::WaveFieldCompress(int nThreads) :
    nThreads(nThreads) {
  table = NULL;
  wtaper = wztaper = NULL;
  npoint = 65536;
  ntaper = 20;
  nztaper = 10;
  init_compression();
  init();
}

void WaveFieldCompress::init_compression() {
  if(compression < 0) compression = global_pars["wf_compress"].as<int>(COMPRESS_16);
}

// return num of ushort
size_t WaveFieldCompress::nshort_pack(size_t nfloats) {
  init_compression();

  switch(compression) {
  case COMPRESS_NONE:
  case COMPRESS_16:
    return nfloats + 2;
  case COMPRESS_16P:
    return (nfloats + 1) / 2 * 2 + sizeof(float) / sizeof(ushort) * 2;
  case COMPRESS_8P:
    return (nfloats + 1) / 2 + sizeof(float) / sizeof(ushort) * 2;
  default:
    return 2 * nfloats;
  }
}

WaveFieldCompress::~WaveFieldCompress() {
  if(table) delete[] table, table = NULL;
  if(wtaper) delete[] wtaper, wtaper = NULL;
  if(wztaper) delete[] wztaper, wztaper = NULL;
}

void WaveFieldCompress::init() {
  if(compression == COMPRESS_16) {
    int i, j, *ipt;
    float f;
    float octave[1024];

    table = new float[npoint]();
    // special values
    table[0] = 0.f;
    table[1] = 1.f;
    ipt = (int*)&f;
    *ipt = 0x7fffffff;
    table[32768] = f;

    f = 1.f / ((float)(1 << 30));
    for(i = 2; i < 1024; i++)
      table[i] = f * i;

    f = 1.f / ((float)(1 << 14));
    for(i = 1024; i < 16384; i++)
      table[i] = f * i;

    f = 1.f / 1024.f;
    for(i = 0; i < 1024; i++)
      octave[i] = 1.f + i * f;

    for(j = 0; j < 16; j++) {
      f = 1.f / ((float)(1 << (20 - j)));
      for(i = 0; i < 1024; i++)
        table[16384 + j * 1024 + i] = f * octave[i];
    }

    for(i = 1; i < 32768; i++)
      table[i + 32768] = -table[i];
  }

  //taper
  wtaper = new float[ntaper]();
  for(int i = 0; i < ntaper; i++) {
    float temp = (float) M_PI * float(i) / float(ntaper);
    wtaper[i] = 0.5f * (1.0f - cosf(temp));
  }
  wztaper = new float[nztaper]();
  for(int i = 0; i < nztaper; i++) {
    float temp = (float) M_PI * float(i) / float(nztaper);
    wztaper[i] = 0.5f * (1.0f - cosf(temp));
  }
}

size_t WaveFieldCompress::nshort_volume(int nz, int nx, int ny) {
  init_compression();
  if(compression == COMPRESS_16) return nshort_pack((size_t)nz * nx) * ny;
  if(compression == COMPRESS_16P) return nshort_pack(nz) * nx * ny;
  return sizeof(float) / sizeof(ushort) * nz * nx * ny;
}
void WaveFieldCompress::shift(int *in32, ushort *out16, size_t n) {
  int frc, exp, idat;
  ushort out, sgn;

  for(size_t i = 0; i < n; i++) {

    idat = in32[i];
    sgn = (idat & 0x80000000) >> 16;
    exp = ((idat & 0x7f800000) >> 23) - 127;

    if(exp > 0) out = 0x8000;
    else if(exp == 0) out = sgn | 0x1;
    else if(exp > -5) {
      frc = (idat & 0x7fffff) | 0x800000;
      out = sgn | (frc >> (9 - exp));
    } else if(exp > -21) {
      exp += 20;
      frc = (idat & 0x7fffff);
      out = sgn | 0x4000 | (exp << 10) | (frc >> 13);
    } else if(exp > -30) {
      frc = (idat & 0x7fffff) | 0x800000;
      out = sgn | (frc >> (-7 - exp));
    } else out = 0;

    out16[i] = out;
  }
}

void WaveFieldCompress::compress16(float *in, ushort *out, int nx, int ny, int nz) {
  size_t nxz = (size_t)nx * (size_t)nz;
  float *work = new float[nxz * nThreads];
#pragma omp parallel num_threads(nThreads)
  {
    int tid = omp_get_thread_num();
    float *pwork = work + nxz * tid;
#pragma omp for schedule(static)
    for(int iy = 0; iy < ny; iy++) {
      float val, vmax, scale;

      float *pin = in + (size_t)iy * nxz;
      ushort *pout = out + (size_t)iy * (nxz + 2);

      vmax = fabsf(pin[0]);
      for(unsigned int i = 1; i < nxz; i++) {
        val = fabsf(pin[i]);
        if(val > vmax) vmax = val;
      }

      if(vmax > eps) {
        scale = 1.0f / vmax;
        for(size_t i = 0; i < nxz; i++)
          pwork[i] = pin[i] * scale;
        shift((int*)pwork, (ushort*)pout + 2, nxz);
        ((float*)pout)[0] = vmax;
      } else {
        memset(pout, 0, (nxz + 2) * sizeof(ushort));
      }
    }
  }
  delete[] work;
  work = NULL;
}

// uncompress and taper ...
void WaveFieldCompress::uncompress16(ushort *in, float *out, int nx, int ny, int nz) {
  size_t nxz = (size_t)nx * (size_t)nz;
#pragma omp parallel for num_threads(nThreads) schedule(static)
  for(int iy = 0; iy < ny; iy++) {
    float scale = 1;

    ushort *pin = in + (size_t)iy * (nxz + 2);
    float *pout = out + (size_t)iy * nxz;

    scale = ((float*)pin)[0];
    // printf("uncompress: scale=%g\n", scale), fflush(stdout);

    if(scale > eps) {

      for(int ix = 0; ix < nx; ix++) {
        for(int iz = 0; iz < nz; iz++) {
          size_t id = ix * nz + iz;
          pout[id] = scale * table[pin[id + 2]];
          if(ny > ntaper * 2) {
            if(iy < ntaper) pout[id] *= wtaper[iy];
            if(iy > ny - ntaper - 1) pout[id] *= wtaper[ny - iy - 1];
          }
          if(nx > ntaper * 2) {
            if(ix < ntaper) pout[id] *= wtaper[ix];
            if(ix > nx - ntaper - 1) pout[id] *= wtaper[nx - ix - 1];
          }
          if(nz > nztaper * 2) {
            if(iz < nztaper) pout[id] *= wztaper[iz];
            if(iz > nz - nztaper - 1) pout[id] *= wztaper[nz - iz - 1];
          }
        }
      }
    } else {
      memset(pout, 0, nxz * sizeof(float));
    }
  }
}

void WaveFieldCompress::compress(float *in, ushort *out, int nx, int ny, int nz) {
  size_t nshort = nshort_pack(nz);
  size_t nxy = (size_t)nx * ny;

  switch(compression) {
  case COMPRESS_16:
    compress16(in, out, nx, ny, nz);
    break;

  case COMPRESS_16P:
#pragma omp parallel for num_threads(nThreads) schedule(static)
    for(size_t ixy = 0; ixy < nxy; ixy++)
      compress16p(in + ixy * nz, out + ixy * nshort, nz);
    break;

  case COMPRESS_8P:
#pragma omp parallel for num_threads(nThreads) schedule(static)
    for(size_t ixy = 0; ixy < nxy; ixy++)
      compress8p(in + ixy * nz, out + ixy * nshort, nz);
    break;

  case COMPRESS_NONE:
  default:
#pragma omp parallel for num_threads(nThreads) schedule(static)
    for(size_t ixy = 0; ixy < nxy; ixy++)
      memcpy(out + ixy * nshort, in + ixy * nz, sizeof(float) * nz);
  }
}

void WaveFieldCompress::uncompress(ushort *in, float *out, int nx, int ny, int nz) {
  switch(compression) {
  case COMPRESS_16:
    uncompress16(in, out, nx, ny, nz);
    break;

  case COMPRESS_8P:
  case COMPRESS_16P:
  case COMPRESS_NONE:
  default:
    size_t nshort = nshort_pack(nz);
    size_t nxy = (size_t)nx * ny;
#pragma omp parallel for num_threads(nThreads) schedule(static)
    for(size_t ixy = 0; ixy < nxy; ixy++) {
      int iy = ixy / nx;
      int ix = ixy % nx;

      ushort *pin = in + ixy * nshort;
      float *pout = out + ixy * nz;

      if(compression == COMPRESS_16P) uncompress16p(pin, pout, nz);
      else if(compression == COMPRESS_8P) uncompress8p(pin, pout, nz);
      else memcpy(pout, pin, sizeof(float) * nz);

      // tapering wavefield edges
      for(int iz = 0; iz < nz; iz++) {
        if(ny > ntaper * 2) {
          if(iy < ntaper) pout[iz] *= wtaper[iy];
          if(iy > ny - ntaper - 1) pout[iz] *= wtaper[ny - iy - 1];
        }
        if(nx > ntaper * 2) {
          if(ix < ntaper) pout[iz] *= wtaper[ix];
          if(ix > nx - ntaper - 1) pout[iz] *= wtaper[nx - ix - 1];
        }
        if(nz > nztaper * 2) {
          if(iz < nztaper) pout[iz] *= wztaper[iz];
          if(iz > nz - nztaper - 1) pout[iz] *= wztaper[nz - iz - 1];
        }
      }
    }
  }
}

int WaveFieldCompress::compress16p(float *in, ushort *out, size_t n) {
  ushort *__restrict c = (ushort*)out;
  size_t i;
  unsigned int ui1, ui2;
  float v1, v2, v1max, v1min, v2max, v2min;
  float scale1, bias1, scale2, bias2;

  /* Read trace data and get absmax */
  if(n == 1) v1min = v1max = in[0], v2min = v2max = 0;
  else v1min = v1max = 0.5f * (in[0] + in[1]), v2min = v2max = 0.5f * (in[0] - in[1]);
  for(i = 0; i < n - 1; i += 2) {
    v1 = 0.5f * (in[i] + in[i + 1]);
    v2 = 0.5f * (in[i] - in[i + 1]);
    if(v1 > v1max) v1max = v1;
    if(v1 < v1min) v1min = v1;
    if(v2 > v2max) v2max = v2;
    if(v2 < v2min) v2min = v2;
  }
  if(i < n) { // still 1 number left
    v1 = in[i];
    if(v1 > v1max) v1max = v1;
    if(v1 < v1min) v1min = v1;
  }
  float range1 = 2 * max(fabsf(v1max), fabsf(v1min));
  float range2 = 2 * max(fabsf(v2max), fabsf(v2min));

  int bit1 = autobit16(range1, range2);

  int bit2 = 32 - bit1, bit3 = abs(bit1 - 16);
  int bit_range1 = (1 << bit1) - 1, bit_range2 = (1 << bit2) - 1, bit_range3 = (1 << bit3) - 1;
  range1 = max(range1, FLT_MIN * bit_range1);
  bias1 = -0.5f * range1;
  range2 = max(range2, FLT_MIN * bit_range2);
  bias2 = -0.5f * range2;

  //  printf("#range1=%f, bias1=%f, range2=%f, bias2=%f, v1min=%f, v1max=%f\n", range1, bias1, range2, bias2,
  //      v1min, v1max);

  /* If max is zero, then put scale and unscale to zero too */
  scale1 = range1 > 0 ? bit_range1 / range1 : 0.0;
  scale2 = range2 > 0 ? bit_range2 / range2 : 0.0;

  /* Apply the scale and load in char data */
  for(i = 0; i < n - 1; i += 2, in += 2) {
    float x1 = 0.5f * (in[0] + in[1]) - bias1, x2 = 0.5f * (in[0] - in[1]) - bias2;
    x1 = scale1 * x1;
    x2 = scale2 * x2;
    ui1 = (unsigned int)(nearbyintf(bit1 < 16 ? x2 : x1)); // x1 range 0 to 0xFFF for 12bit
    ui2 = (unsigned int)(nearbyintf(bit1 < 16 ? x1 : x2)); // x2 range 0 to 0xF for 4bit
    *c++ = ui1 >> bit3; // first 16 bits for ui1 which is bit1
    *c++ = ((ui1 & bit_range3) << (bit1 < 16 ? bit1 : bit2)) + (ui2); // last bit3 bits of ui1, and ui2 (bit2)
  }
  if(i < n) { // still 1 number left
    float x1 = in[0] - bias1;
    x1 = scale1 * x1;
    ui1 = (unsigned int)(nearbyintf(x1)); // x1 range 0 to 0xFFF for 12bit
    *c++ = ui1 >> bit3; // remove the least bit3 off
    *c++ = ui1 & bit_range3; // last bit3 bits of ui1
  }

  ((float*)c)[0] = range1;
  ((float*)c)[1] = range2;

  return bit1;
}

int WaveFieldCompress::uncompress16p(ushort *in, float *out, size_t n) {
  size_t i;
  unsigned int ui1, ui2;
  float x1, x2;
  ushort *__restrict c = (ushort*)in;
  float unscale1, unscale2, bias1 = 0, bias2 = 0;

  /* extract the unscale factor */
  float *c2 = (float*)(c + (n + 1) / 2 * 2);
  float range1 = c2[0], range2 = c2[1];
  int bit1 = autobit16(range1, range2);

  int bit2 = 32 - bit1, bit3 = abs(bit1 - 16);
  int bit_range1 = (1 << bit1) - 1, bit_range2 = (1 << bit2) - 1, bit_range3 = (1 << bit3) - 1;

  unscale1 = bit_range1 ? range1 / bit_range1 : 0;
  unscale2 = bit_range2 ? range2 / bit_range2 : 0;
  bias1 = -0.5f * unscale1 * bit_range1;
  bias2 = -0.5f * unscale2 * bit_range2;

  for(i = 0; i < n - 1; i += 2, c += 2) {
    ui1 = (((unsigned int)c[0]) << bit3) + (c[1] >> (bit1 < 16 ? bit1 : bit2));
    ui2 = (unsigned int)(c[1] & (bit1 < 16 ? bit_range1 : bit_range2));
    x1 = (float)(bit1 < 16 ? ui2 : ui1);
    x2 = (float)(bit1 < 16 ? ui1 : ui2);
    x1 = x1 * unscale1 + bias1;
    x2 = x2 * unscale2 + bias2;
    *out++ = x1 + x2;
    *out++ = x1 - x2;
  }

  if(i < n) { // still 1 number left
    ui1 = (((unsigned int)c[0]) << bit3) + (c[1]);
    x1 = (float)ui1;
    *out++ = x1 * unscale1 + bias1;
  }

  return bit1;
}

int WaveFieldCompress::compress8p(float *in, ushort *out, size_t n) {
  unsigned char *__restrict c = (unsigned char*)out;
  size_t i;
  unsigned int ui1, ui2;
  float v1, v2, v1max, v1min, v2max, v2min;
  float scale1, bias1, scale2, bias2;

  /* Read trace data and get absmax */
  if(n == 1) v1min = v1max = in[0], v2min = v2max = 0;
  else v1min = v1max = 0.5f * (in[0] + in[1]), v2min = v2max = 0.5f * (in[0] - in[1]);
  for(i = 0; i < n - 1; i += 2) {
    v1 = 0.5f * (in[i] + in[i + 1]);
    v2 = 0.5f * (in[i] - in[i + 1]);
    if(v1 > v1max) v1max = v1;
    if(v1 < v1min) v1min = v1;
    if(v2 > v2max) v2max = v2;
    if(v2 < v2min) v2min = v2;
  }
  if(i < n) { // still 1 number left
    v1 = in[i];
    if(v1 > v1max) v1max = v1;
    if(v1 < v1min) v1min = v1;
  }
  float range1 = 2 * max(fabsf(v1max), fabsf(v1min));
  float range2 = 2 * max(fabsf(v2max), fabsf(v2min));

  int bit1 = autobit8(range1, range2);

  int bit2 = 16 - bit1, bit3 = abs(bit1 - 8);
  int bit_range1 = (1 << bit1) - 1, bit_range2 = (1 << bit2) - 1, bit_range3 = (1 << bit3) - 1;
  // assert(bit1 >= 8); // no longer needed, bit3 takes the absolute value
  range1 = max(range1, FLT_MIN * bit_range1);
  bias1 = -0.5f * range1;
  range2 = max(range2, FLT_MIN * bit_range2);
  bias2 = -0.5f * range2;

  //  printf("#range1=%f, bias1=%f, range2=%f, bias2=%f, v1min=%f, v1max=%f\n", range1, bias1, range2, bias2,
  //      v1min, v1max);

  /* If max is zero, then put scale and unscale to zero too */
  scale1 = range1 > 0 ? bit_range1 / range1 : 0.0;
  scale2 = range2 > 0 ? bit_range2 / range2 : 0.0;

  /* Apply the scale and load in char data */
  for(i = 0; i < n - 1; i += 2, in += 2) {
    float x1 = 0.5f * (in[0] + in[1]) - bias1, x2 = 0.5f * (in[0] - in[1]) - bias2;
    x1 = scale1 * x1;
    x2 = scale2 * x2;
    ui1 = (unsigned int)(nearbyintf(bit1 < 8 ? x2 : x1)); // x1 range 0 to 0xFFF for 12bit
    ui2 = (unsigned int)(nearbyintf(bit1 < 8 ? x1 : x2)); // x2 range 0 to 0xF for 4bit
    *c++ = ui1 >> bit3; // first 8 bits for ui1 which is bit1
    *c++ = ((ui1 & bit_range3) << (bit1 < 8 ? bit1 : bit2)) + (ui2); // last bit3 bits of ui1, and ui2 (bit2)
  }
  if(i < n) { // still 1 number left
    float x1 = in[0] - bias1;
    x1 = scale1 * x1;
    ui1 = (unsigned int)(nearbyintf(x1)); // x1 range 0 to 0xFFF for 12bit
    *c++ = ui1 >> bit3; // remove the least bit3 off
    *c++ = ui1 & bit_range3; // last bit3 bits of ui1
  }

  ((float*)c)[0] = range1;
  ((float*)c)[1] = range2;

  return bit1;
}

int WaveFieldCompress::uncompress8p(ushort *in, float *out, size_t n) {
  size_t i;
  unsigned int ui1, ui2;
  float x1, x2;
  unsigned char *__restrict c = (unsigned char*)in;
  float unscale1, unscale2, bias1 = 0, bias2 = 0;

  /* extract the unscale factor */
  float *c2 = (float*)(c + (n + 1) / 2 * 2);
  float range1 = c2[0], range2 = c2[1];
  int bit1 = autobit8(range1, range2);

  int bit2 = 16 - bit1, bit3 = abs(bit1 - 8);
  int bit_range1 = (1 << bit1) - 1, bit_range2 = (1 << bit2) - 1, bit_range3 = (1 << bit3) - 1;

  unscale1 = bit_range1 ? range1 / bit_range1 : 0;
  unscale2 = bit_range2 ? range2 / bit_range2 : 0;
  bias1 = -0.5f * unscale1 * bit_range1;
  bias2 = -0.5f * unscale2 * bit_range2;

  for(i = 0; i < n - 1; i += 2, c += 2) {
    ui1 = (((unsigned int)c[0]) << bit3) + (c[1] >> (bit1 < 8 ? bit1 : bit2));
    ui2 = (unsigned int)(c[1] & (bit1 < 8 ? bit_range1 : bit_range2));
    x1 = (float)(bit1 < 8 ? ui2 : ui1);
    x2 = (float)(bit1 < 8 ? ui1 : ui2);
    x1 = x1 * unscale1 + bias1;
    x2 = x2 * unscale2 + bias2;
    *out++ = x1 + x2;
    *out++ = x1 - x2;
  }

  if(i < n) { // still 1 number left
    ui1 = (((unsigned int)c[0]) << bit3) + (c[1]);
    x1 = (float)ui1;
    *out++ = x1 * unscale1 + bias1;
  }

  return bit1;
}

int WaveFieldCompress::autobit16(float range1, float range2) {
  if(range1 > 0 && range2 == 0) return 32;
  if(range1 > 0 && range2 > 0) {
    float minerr = range1 + range2;
    int bitmin = 16;
    for(int bit1 = 0; bit1 <= 32; bit1++) {
      int bit2 = 32 - bit1;
      int bit_range1 = (1 << bit1), bit_range2 = (1 << bit2); // adding 1 to make sure it's not 0
      float totalerr = range1 / bit_range1 + range2 / bit_range2;
      if(totalerr < minerr) {
        minerr = totalerr;
        bitmin = bit1;
      }
    }
    return bitmin;
  }
  return 16;
}

int WaveFieldCompress::autobit8(float range1, float range2) {
  if(range1 > 0 && range2 == 0) return 16;
  if(range1 > 0 && range2 > 0) {
    float minerr = range1 + range2;
    int bitmin = 8;
    for(int bit1 = 0; bit1 <= 16; bit1++) {
      int bit2 = 16 - bit1;
      int bit_range1 = (1 << bit1), bit_range2 = (1 << bit2); // adding 1 to make sure it's not 0
      float totalerr = range1 / bit_range1 + range2 / bit_range2;
      if(totalerr < minerr) {
        minerr = totalerr;
        bitmin = bit1;
      }
    }
    return bitmin;
  }
  return 8;
}

int WaveFieldCompress::getCompression() {
  init_compression();
  return compression;
}

