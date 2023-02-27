#include "Derivative_cfd.h"
#include "timing.h"


Derivative_cfd::Derivative_cfd(Grid *inGrid, Boundary *inBnd, int modelType, int inNThreads)
  : Derivative(inGrid, inBnd, modelType, inNThreads), myCfd_x(nx, dx), myCfd_y(ny, dy), myCfd_z(nz, dz) {
  mywk = new float[nx * ny * nz]();
}

Derivative_cfd::~Derivative_cfd() {
  delete [] mywk;
}

void Derivative_cfd::dx1(float *pin, float *pout, int isign) {
  if(isign == 1)
    #pragma omp parallel for num_threads(nThreads) schedule(static)
    for(int iy = 0; iy < ny; ++iy)
      cfd_dev_xy_batch(myCfd_x.am1, myCfd_x.bm1, myCfd_x.bl1, nx, nz, nz, pin + iy * nxz, pout + iy * nxz);
  else
    #pragma omp parallel for num_threads(nThreads) schedule(static)
    for(int iy = 0; iy < ny; ++iy)
      cfd_dev_xy_batch(myCfd_x.am2, myCfd_x.bm2, myCfd_x.bl2, nx, nz, nz, pin + iy * nxz, pout + iy * nxz);

}
void Derivative_cfd::dy1(float *pin, float *pout, int isign) {
  if(isign == 1)
    #pragma omp parallel for num_threads(nThreads) schedule(static)
    for(int ix = 0; ix < nx; ix++)
      cfd_dev_xy_batch(myCfd_y.am1, myCfd_y.bm1, myCfd_y.bl1, ny, nz, nxz, pin + ix * nz, pout + ix * nz);
  else
    #pragma omp parallel for num_threads(nThreads) schedule(static)
    for(int ix = 0; ix < nx; ix++)
      cfd_dev_xy_batch(myCfd_y.am2, myCfd_y.bm2, myCfd_y.bl2, ny, nz, nxz, pin + ix * nz, pout + ix * nz);
}

void  Derivative_cfd::dz1(float *pin, float *pout, int isign) {
  if(isign == 1)
    #pragma omp parallel for num_threads(nThreads) schedule(static)
    for(int iy = 0; iy < ny; ++iy)
      cfd_dev_z_batch(myCfd_z.am1, myCfd_z.bm1, myCfd_z.bl1, nz, nx, pin + iy * nxz, pout + iy * nxz);
  else
    #pragma omp parallel for num_threads(nThreads) schedule(static)
    for(int iy = 0; iy < ny; ++iy)
      cfd_dev_z_batch(myCfd_z.am2, myCfd_z.bm2, myCfd_z.bl2, nz, nx, pin + iy * nxz, pout + iy * nxz);
}

void Derivative_cfd::cfd_dev_z_batch(float *am, float *bm, float *bl, int  n, int nbatch, float *win, float *wout) {
  banmul_dev_z_batch(am, n, nbatch, win, wout);
  banbks_dev_z_batch(bm, n, nbatch, bl, wout);
}

void Derivative_cfd::cfd_dev_xy_batch(float *am, float *bm, float *bl, int  n, int nbatch, int ldim, float *win, float *wout) {
  banmul_dev_xy_batch(am, n, nbatch, ldim, win, wout);
  banbks_dev_xy_batch(bm, n, nbatch, ldim, bl, wout);
}

void Derivative_cfd::banmul_dev_z_batch(float *am, int  n, int nbatch, float *win, float *wout) {
  // multi-thread here
  for(int ib = 0; ib < nbatch; ++ib)
    banmul_dev_z_c(am, n, &win[ib * n], &wout[ib * n]);
}

void Derivative_cfd::banmul_dev_xy_batch(float *a, int  n, int nbatch, int ldim, float *x, float *b) {
  for(int i = 0; i < 5; ++i) {
    for(int iz = 0; iz < nbatch; ++iz) b[i * ldim + iz] = 0.0;
    int k = i - 5;
    for(int j = MAX(0, -k); j < 11; ++j) {
      for(int ibatch = 0; ibatch < nbatch; ++ibatch) {
        b[i * ldim + ibatch] = b[i * ldim + ibatch] + a[j * 5 + i] * x[(j + k) * ldim + ibatch];
      }
    }
  }

  for(int i = 5; i < n - 5; ++i) {
    for(int ibatch = 0; ibatch < nbatch; ++ibatch) {
      b[i * ldim + ibatch] = a[55] * (x[(i + 1) * ldim + ibatch] - x[(i - 1) * ldim + ibatch])
                             + a[56] * (x[(i + 2) * ldim + ibatch] - x[(i - 2) * ldim + ibatch])
                             + a[57] * (x[(i + 3) * ldim + ibatch] - x[(i - 3) * ldim + ibatch])
                             + a[58] * (x[(i + 4) * ldim + ibatch] - x[(i - 4) * ldim + ibatch])
                             + a[59] * (x[(i + 5) * ldim + ibatch] - x[(i - 5) * ldim + ibatch]);
    }
  }

for(int i = n - 5; i < n; ++i) {
    for(int ibatch = 0; ibatch < nbatch; ++ibatch) b[i * ldim + ibatch] = 0.0;
    int k = i - 5;
    for(int j = 0; j < MIN(11, n - k); ++j) {
      for(int ibatch = 0; ibatch < nbatch; ++ibatch) {
        b[i * ldim + ibatch] = b[i * ldim + ibatch] - a[n - i + 49 - 5 * j] * x[(j + k) * ldim + ibatch];
      }
    }
  }
}

void Derivative_cfd::banbks_dev_z_batch(float *a, int n, int nbatch, float *al, float *b) {
  int band_size = 4;
  int nband = int (nbatch / band_size);
  int nbatch1 = nband * band_size;

  for(int ib = 0; ib < nband; ib++)
    banbks_dev_z_c_batch(a, n, al, &b[band_size * ib * n]);

  for(int ib = nbatch1; ib < nbatch; ib++)
    banbks_dev_z_c(a, n, al, &b[ib * n]);
}
void Derivative_cfd::banbks_dev_xy_batch(float *a, int n, int nbatch, int ldim, float *al, float *b) {
  for(int k = 0; k < n - 4; ++k) {
    for(int j = 0; j < 4; ++j) {
      for(int ibatch = 0; ibatch < nbatch; ++ibatch) {
        b[(j + k + 1)*ldim + ibatch] = b[(j + k + 1) * ldim + ibatch] - al[j + k * 4] * b[k * ldim + ibatch];
      }
    }
  }

  for(int k = n - 4; k < n; ++k) {
    for(int j = 0; j < n - k - 1; ++j) {
      for(int ibatch = 0; ibatch < nbatch; ++ibatch) {
        b[(j + k + 1)*ldim + ibatch] = b[(j + k + 1) * ldim + ibatch] - al[j + k * 4] * b[k * ldim + ibatch];
      }
    }
  }

  int l = 1;
  for(int i = n - 1; i >= n - 4 ; i--) {
    for(int k = 1; k < l ; k++) {
      for(int ibatch = 0; ibatch < nbatch; ++ibatch) {
        b[i * ldim + ibatch] +=  - a[i * 5 + k] * b[(k + i) * ldim + ibatch];
      }
    }
    for(int ibatch = 0; ibatch < nbatch; ++ibatch) b[i * ldim + ibatch] = b[i * ldim + ibatch] / a[i * 5];
    if(l < 5) l++;
  }
for(int i = n - 5; i >= 0 ; i--) {
    float *b0 = &b[i * ldim];
    float *b1 = &b[(i + 1) * ldim];
    float *b2 = &b[(i + 2) * ldim];
    float *b3 = &b[(i + 3) * ldim];
    float *b4 = &b[(i + 4) * ldim];
    for(int ibatch = 0; ibatch < nbatch; ++ibatch) {
      b0[ibatch] = (b0[ibatch] - a[i * 5 + 1] * b1[ibatch] - a[i * 5 + 2] * b2[ibatch] - a[i * 5 + 3] * b3[ibatch] - a[i * 5 + 4] * b4[ibatch]) /
                   a[i * 5];
    }
  }
}

void Derivative_cfd::banbks_dev_z_c_batch(float *a, int n, float *al, float *b) {
  float *b1 = &b[0];
  float *b2 = &b[n];
  float *b3 = &b[n + n];
  float *b4 = &b[n + n + n];

  for(int k = 0; k < n - 4; ++k) {
    for(int j = 0; j < 4; ++j) {
      b1[j + k + 1] = b1[j + k + 1] - al[j + k * 4] * b1[k];
      b2[j + k + 1] = b2[j + k + 1] - al[j + k * 4] * b2[k];
      b3[j + k + 1] = b3[j + k + 1] - al[j + k * 4] * b3[k];
      b4[j + k + 1] = b4[j + k + 1] - al[j + k * 4] * b4[k];
    }
  }
for(int k = n - 4; k < n; ++k) {
    for(int j = 0; j < n - k - 1; ++j) {
      b1[j + k + 1] = b1[j + k + 1] - al[j + k * 4] * b1[k];
      b2[j + k + 1] = b2[j + k + 1] - al[j + k * 4] * b2[k];
      b3[j + k + 1] = b3[j + k + 1] - al[j + k * 4] * b3[k];
      b4[j + k + 1] = b4[j + k + 1] - al[j + k * 4] * b4[k];
    }
  }

  int l = 1;
  for(int i = n - 1; i >= n - 4 ; i--) {
    for(int k = 1; k < l ; k++) {
      b1[i] +=  - a[i * 5 + k] * b1[k + i];
      b2[i] +=  - a[i * 5 + k] * b2[k + i];
      b3[i] +=  - a[i * 5 + k] * b3[k + i];
      b4[i] +=  - a[i * 5 + k] * b4[k + i];
    }
    b1[i] = b1[i] / a[i * 5];
    b2[i] = b2[i] / a[i * 5];
    b3[i] = b3[i] / a[i * 5];
    b4[i] = b4[i] / a[i * 5];

    if(l < 5) l++;
  }

  for(int i = n - 5; i >= 0 ; i--) {
    b1[i] = (b1[i] - a[i * 5 + 1] * b1[i + 1] - a[i * 5 + 2] * b1[i + 2] - a[i * 5 + 3] * b1[i + 3] - a[i * 5 + 4] * b1[i + 4]) / a[i * 5];
    b2[i] = (b2[i] - a[i * 5 + 1] * b2[i + 1] - a[i * 5 + 2] * b2[i + 2] - a[i * 5 + 3] * b2[i + 3] - a[i * 5 + 4] * b2[i + 4]) / a[i * 5];
    b3[i] = (b3[i] - a[i * 5 + 1] * b3[i + 1] - a[i * 5 + 2] * b3[i + 2] - a[i * 5 + 3] * b3[i + 3] - a[i * 5 + 4] * b3[i + 4]) / a[i * 5];
    b4[i] = (b4[i] - a[i * 5 + 1] * b4[i + 1] - a[i * 5 + 2] * b4[i + 2] - a[i * 5 + 3] * b4[i + 3] - a[i * 5 + 4] * b4[i + 4]) / a[i * 5];
  }
}
void Derivative_cfd::banmul_dev_z_c(float *a, int n, float *x, float *b) {

  for(int i = 0; i < 5; ++i) {
    b[i] = 0.0;
    int k = i - 5;
    for(int j = MAX(0, -k); j < 11; ++j) {
      b[i] = b[i] + a[j * 5 + i] * x[j + k];
    }
  }

  for(int i = 5; i < n - 5; ++i) {
    b[i] = a[55] * (x[i + 1] - x[i - 1]) + a[56] * (x[i + 2] - x[i - 2]) + a[57] * (x[i + 3] - x[i - 3]) + a[58] *
           (x[i + 4] - x[i - 4]) + a[59] * (x[i + 5] - x[i - 5]);
  }

  for(int i = n - 5; i < n; ++i) {
    b[i] = 0.0;
    int k = i - 5;
    for(int j = 0; j < MIN(11, n - k); ++j) {
      //          b[i] = b[i] - a[(n-i-1) + (10-j)*5]*x[j+k];
      b[i] = b[i] - a[n - i + 49 - 5 * j] * x[j + k];
    }
  }
}

void Derivative_cfd::banbks_dev_z_c(float *a, int n, float *al, float *b) {
  int mm = 5;

  for(int k = 0; k < n - 4; ++k) {
    for(int j = 0; j < 4; ++j) {
      b[j + k + 1] = b[j + k + 1] - al[j + k * 4] * b[k];
    }
  }

  for(int k = n - 4; k < n; ++k) {
    for(int j = 0; j < n - k - 1; ++j) {
      b[j + k + 1] = b[j + k + 1] - al[j + k * 4] * b[k];
    }
  }

  int l = 1;
  for(int i = n - 1; i >= n - 4 ; i--) {
    for(int k = 1; k < l ; k++) {
      b[i] +=  - a[i * 5 + k] * b[k + i];
    }
    b[i] = b[i] / a[i * 5];
    if(l < mm) l++;
  }

  for(int i = n - 5; i >= 0 ; i--) {
    b[i] = (b[i] - a[i * 5 + 1] * b[i + 1] - a[i * 5 + 2] * b[i + 2] - a[i * 5 + 3] * b[i + 3] - a[i * 5 + 4] * b[i + 4]) / a[i * 5];
  }

}


